import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from cadlib.macro import *
from src.unified_vocab.vocab import VOCAB_SIZE, MAX_ARGS_PER_CMD


class CADLoss(nn.Module):
    """
    适配13维CAD向量的损失函数。

    13维向量结构:
    - [0]: 命令类型 (Line, Arc, Circle, EOS, SOL, Ext)
    - [1:6]: 草图参数 (x, y, alpha, f, r)
    - [6]: 角度Token (theta, phi, gamma 编码为单一token)
    - [7]: 位置Token (px, py, pz 编码为单一token)
    - [8:13]: 拉伸参数 (e1, e2, b, u, s)
    """
    def __init__(self, cfg, weights=None):
        super().__init__()
        self.cfg = cfg
        self.n_commands = cfg.cad_n_commands
        self.n_args = cfg.cad_n_args
        self.args_dim = cfg.args_dim

        # 损失权重
        self.weights = weights or {
            'cmd': 1.0,
            'args': 1.0,
            'angle_token': 2.0,
            'pos_token': 2.0
        }

        # 适配13D向量的命令-参数掩码 (12个参数)
        # 参数索引: [0:5]=草图参数, [5]=角度token, [6]=位置token, [7:12]=拉伸参数
        cmd_args_mask_13d = np.array([
            [1, 1, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0],  # Line: x, y
            [1, 1, 1, 1, 0,  0, 0,  0, 0, 0, 0, 0],  # Arc: x, y, alpha, f
            [1, 1, 0, 0, 1,  0, 0,  0, 0, 0, 0, 0],  # Circle: x, y, r
            [0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0],  # EOS: 无参数
            [0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0],  # SOL: 无参数
            [0, 0, 0, 0, 0,  1, 1,  1, 1, 1, 1, 1],  # Ext: angle, pos, e1, e2, b, u, s
        ], dtype=np.float32)

        self.register_buffer("cmd_args_mask", torch.tensor(cmd_args_mask_13d))

    def forward(self, outputs, batch):
        """
        计算损失。

        Args:
            outputs: 模型输出字典
                - command_logits: [B, S, n_commands]
                - args_logits: [B, S, n_args, args_dim]
                - angle_logits: [B, S, n_angle_tokens]
                - pos_logits: [B, S, n_pos_tokens]
            batch: 数据批次
                - cad_sequence: [B, S, 13]
        """
        device = outputs['command_logits'].device

        # 目标序列
        cad_seq = batch['cad_sequence'].to(device)  # [B, S, 13]
        tgt_commands = cad_seq[:, :, 0]  # [B, S]
        tgt_args = cad_seq[:, :, 1:]  # [B, S, 12]

        # 获取有效位置掩码 (非EOS填充位置)
        valid_mask = self._get_valid_mask(tgt_commands)  # [B, S]

        # 1. 命令损失
        loss_cmd = self._compute_cmd_loss(outputs['command_logits'], tgt_commands, valid_mask)

        # 2. 参数损失 (使用 Gumbel soft label)
        loss_args = self._compute_args_loss(outputs['args_logits'], tgt_args, tgt_commands, valid_mask)

        # 3. 角度Token损失 (仅对Ext命令)
        loss_angle = self._compute_token_loss(
            outputs['angle_logits'],
            tgt_args[:, :, 5],  # 角度token在第6个参数位置(索引5)
            tgt_commands,
            valid_mask,
            cmd_type=EXT_IDX
        )

        # 4. 位置Token损失 (仅对Ext命令)
        loss_pos = self._compute_token_loss(
            outputs['pos_logits'],
            tgt_args[:, :, 6],  # 位置token在第7个参数位置(索引6)
            tgt_commands,
            valid_mask,
            cmd_type=EXT_IDX
        )

        # 加权总损失
        total_loss = (
            self.weights['cmd'] * loss_cmd +
            self.weights['args'] * loss_args +
            self.weights['angle_token'] * loss_angle +
            self.weights['pos_token'] * loss_pos
        )

        return {
            'loss': total_loss,
            'loss_cmd': loss_cmd,
            'loss_args': loss_args,
            'loss_angle': loss_angle,
            'loss_pos': loss_pos
        }

    def _get_valid_mask(self, commands):
        """获取有效序列位置的掩码 (到EOS之前的位置)"""
        # 累计EOS出现次数，第一个EOS之后都是填充
        eos_cumsum = (commands == EOS_IDX).cumsum(dim=-1)
        # 包含第一个EOS位置
        valid_mask = (eos_cumsum <= 1).float()
        return valid_mask

    def _compute_cmd_loss(self, logits, targets, valid_mask):
        """计算命令分类损失"""
        B, S, C = logits.shape

        # 转换为 float32 以提高数值稳定性
        logits = logits.float()

        # 展平计算 (使用 reshape 替代 view 以处理非连续张量)
        logits_flat = logits.reshape(-1, C)
        targets_flat = targets.reshape(-1).long()
        mask_flat = valid_mask.reshape(-1)

        # 交叉熵损失
        loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')

        # 检查 NaN 并替换
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

        loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)

        return loss

    def _compute_args_loss(self, logits, targets, commands, valid_mask, tolerance=3, alpha=2.0):
        """
        计算参数损失，使用 Gumbel soft label 允许一定容差。

        Args:
            logits: [B, S, n_args, args_dim]
            targets: [B, S, n_args] 参数值 (-1表示无效)
            commands: [B, S] 命令类型
            valid_mask: [B, S] 有效位置掩码
        """
        B, S, N_ARGS, N_CLASS = logits.shape
        device = logits.device

        # 转换为 float32 以提高数值稳定性
        logits = logits.float()

        # 获取每个命令对应的参数掩码
        cmd_mask = self.cmd_args_mask[commands.long()]  # [B, S, n_args]#type: ignore

        # 组合掩码: 有效位置 AND 该命令使用该参数
        combined_mask = valid_mask.unsqueeze(-1) * cmd_mask  # [B, S, n_args]

        if combined_mask.sum() < 1:
            return torch.tensor(0.0, device=device)

        # 目标值偏移 (+1 因为-1变成0作为padding类)
        targets_shifted = (targets + 1).clamp(0, N_CLASS - 1)  # [B, S, n_args]

        # 使用 log_softmax 提高数值稳定性
        log_probs = F.log_softmax(logits, dim=-1)  # [B, S, n_args, N_CLASS]

        # Gumbel soft label
        target_dist = torch.zeros(B, S, N_ARGS, N_CLASS, dtype=torch.float32, device=device)

        for shift in range(-tolerance, tolerance + 1):
            shifted_target = (targets_shifted + shift).clamp(0, N_CLASS - 1)
            weight = torch.exp(torch.tensor(-alpha * abs(shift), dtype=torch.float32, device=device))
            src = weight * torch.ones(B, S, N_ARGS, 1, dtype=torch.float32, device=device)
            target_dist.scatter_add_(
                3,
                shifted_target.unsqueeze(-1).long(),
                src
            )

        # 归一化
        target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True) + 1e-8)

        # 交叉熵损失 (使用 log_probs 而非 log(softmax))
        loss_per_pos = -torch.sum(target_dist * log_probs, dim=-1)  # [B, S, n_args]

        # 检查 NaN 并替换
        loss_per_pos = torch.where(torch.isnan(loss_per_pos), torch.zeros_like(loss_per_pos), loss_per_pos)

        loss = (loss_per_pos * combined_mask).sum() / (combined_mask.sum() + 1e-8)

        return loss

#弃用
    def _compute_token_loss(self, logits, targets, commands, valid_mask, cmd_type,
                             use_soft_label=True, tolerance=2, alpha=2.0):
        """
        计算特定Token的分类损失 (角度Token或位置Token)。

        Args:
            logits: [B, S, n_tokens]
            targets: [B, S] token目标值
            commands: [B, S] 命令类型
            valid_mask: [B, S] 有效位置掩码
            cmd_type: 仅对该命令类型计算损失
            use_soft_label: 是否使用软标签 (Gumbel-style)
            tolerance: 软标签容差范围
            alpha: 衰减系数
        """
        # 仅对指定命令类型计算
        cmd_mask = (commands == cmd_type).float()
        combined_mask = valid_mask * cmd_mask

        if combined_mask.sum() < 1:
            return torch.tensor(0.0, device=logits.device)

        B, S, C = logits.shape
        device = logits.device

        # 转换为 float32 以提高数值稳定性
        logits = logits.float()

        if use_soft_label:
            # 使用 log_softmax 提高数值稳定性
            log_probs = F.log_softmax(logits, dim=-1)  # [B, S, C]

            # Gumbel 软标签
            target_dist = torch.zeros(B, S, C, dtype=torch.float32, device=device)
            targets_clamped = targets.long().clamp(0, C - 1)

            for shift in range(-tolerance, tolerance + 1):
                shifted_target = (targets_clamped + shift).clamp(0, C - 1)
                weight = torch.exp(torch.tensor(-alpha * abs(shift), dtype=torch.float32, device=device))
                src = weight * torch.ones(B, S, 1, dtype=torch.float32, device=device)
                target_dist.scatter_add_(2, shifted_target.unsqueeze(-1), src)

            # 归一化
            target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True) + 1e-8)

            # 交叉熵损失
            loss_per_pos = -torch.sum(target_dist * log_probs, dim=-1)  # [B, S]

            # 检查 NaN 并替换
            loss_per_pos = torch.where(torch.isnan(loss_per_pos), torch.zeros_like(loss_per_pos), loss_per_pos)

            loss = (loss_per_pos * combined_mask).sum() / (combined_mask.sum() + 1e-8)
        else:
            # 硬标签交叉熵
            logits_flat = logits.reshape(-1, C)
            targets_flat = targets.reshape(-1).long().clamp(0, C - 1)
            mask_flat = combined_mask.reshape(-1)

            loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
            loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
            loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)

        return loss


class UnifiedCADLoss(nn.Module):
    """
    统一词表损失函数

    简洁设计：
    - loss_cmd: 命令分类损失 (交叉熵)
    - loss_args: 统一大词表参数损失 (Gumbel软标签)

    双解码器结构保持不变，参数预测改为大词汇表分类。
    """

    def __init__(self, cfg, weights=None, tolerance=3, alpha=2.0):
        super().__init__()
        self.cfg = cfg
        self.n_commands = cfg.cad_n_commands
        self.vocab_size = VOCAB_SIZE
        self.max_args = MAX_ARGS_PER_CMD

        # 损失权重
        self.weights = weights or {
            'cmd': 1.0,
            'args': 1.0,
        }

        # Gumbel软标签参数
        self.tolerance = tolerance
        self.alpha = alpha

        # 构建参数位置掩码: 哪些位置对哪些命令有效
        # [n_commands, MAX_ARGS_PER_CMD]
        # 使用完整参数序列模板长度（包含边界Token和SEP）。
        from src.unified_vocab.vocab import CMD_SEQUENCE_TEMPLATES
        args_mask = np.zeros((self.n_commands, MAX_ARGS_PER_CMD), dtype=np.float32)
        for cmd_idx in range(self.n_commands):
            template = CMD_SEQUENCE_TEMPLATES.get(cmd_idx, ['SEP'])
            seq_len = min(len(template), MAX_ARGS_PER_CMD)
            args_mask[cmd_idx, :seq_len] = 1.0
        self.register_buffer("args_mask", torch.tensor(args_mask))

    def forward(self, outputs, batch):
        """
        计算损失

        Args:
            outputs: 模型输出
                - command_logits: [B, S, n_commands]
                - unified_args_logits: [B, S, MAX_ARGS_PER_CMD, VOCAB_SIZE]
            batch: 数据批次
                - commands: [B, S] 命令目标
                - args_tokens: [B, S, MAX_ARGS_PER_CMD] 参数token目标
        """
        device = outputs['command_logits'].device

        tgt_commands = batch['commands'].to(device)
        tgt_args = batch['args_tokens'].to(device)

        # 有效位置掩码
        valid_mask = self._get_valid_mask(tgt_commands)

        # 1. 命令损失 (交叉熵)
        loss_cmd = self._compute_cmd_loss(
            outputs['command_logits'], tgt_commands, valid_mask
        )

        # 2. 参数损失 (Gumbel软标签)
        loss_args = self._compute_args_loss(
            outputs['unified_args_logits'], tgt_args, tgt_commands, valid_mask
        )

        # 总损失
        total_loss = (
            self.weights['cmd'] * loss_cmd +
            self.weights['args'] * loss_args
        )

        return {
            'loss': total_loss,
            'loss_cmd': loss_cmd,
            'loss_args': loss_args,
        }

    def _get_valid_mask(self, commands):
        """获取有效序列位置的掩码"""
        eos_cumsum = (commands == EOS_IDX).cumsum(dim=-1)
        valid_mask = (eos_cumsum <= 1).float()
        return valid_mask

    def _compute_cmd_loss(self, logits, targets, valid_mask):
        """计算命令分类损失 (交叉熵)"""
        B, S, C = logits.shape
        logits = logits.float()

        logits_flat = logits.reshape(-1, C)
        targets_flat = targets.reshape(-1).long()
        mask_flat = valid_mask.reshape(-1)

        loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
        loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)

        return loss

    def _compute_args_loss(self, logits, targets, commands, valid_mask):
        """
        计算统一大词表参数损失 (Gumbel软标签)

        Args:
            logits: [B, S, MAX_ARGS_PER_CMD, VOCAB_SIZE]
            targets: [B, S, MAX_ARGS_PER_CMD]
            commands: [B, S]
            valid_mask: [B, S]
        """
        B, S, N_ARGS, V = logits.shape
        device = logits.device
        logits = logits.float()

        # 获取每个命令对应的参数位置掩码
        cmd_mask = self.args_mask[commands.long()]  # [B, S, MAX_ARGS_PER_CMD]

        # 组合掩码
        combined_mask = valid_mask.unsqueeze(-1) * cmd_mask  # [B, S, MAX_ARGS_PER_CMD]

        if combined_mask.sum() < 1:
            return torch.tensor(0.0, device=device)

        # log_softmax 提高数值稳定性
        log_probs = F.log_softmax(logits, dim=-1)  # [B, S, N_ARGS, V]

        # 目标值
        targets_clamped = targets.long().clamp(0, V - 1)

        # 构建Gumbel软标签分布
        target_dist = torch.zeros(B, S, N_ARGS, V, dtype=torch.float32, device=device)

        for shift in range(-self.tolerance, self.tolerance + 1):
            shifted_target = (targets_clamped + shift).clamp(0, V - 1)
            weight = torch.exp(torch.tensor(-self.alpha * abs(shift), dtype=torch.float32, device=device))
            src = weight * torch.ones(B, S, N_ARGS, 1, dtype=torch.float32, device=device)
            target_dist.scatter_add_(3, shifted_target.unsqueeze(-1), src)

        # 归一化
        target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True) + 1e-8)

        # 交叉熵损失
        loss_per_pos = -torch.sum(target_dist * log_probs, dim=-1)  # [B, S, N_ARGS]
        loss_per_pos = torch.where(torch.isnan(loss_per_pos), torch.zeros_like(loss_per_pos), loss_per_pos)

        loss = (loss_per_pos * combined_mask).sum() / (combined_mask.sum() + 1e-8)

        return loss
