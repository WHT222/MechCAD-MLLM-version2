import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from cadlib.macro import *


class CADLoss(nn.Module):
    """
    适配17维CAD向量的损失函数（统一257级词表）。

    17维向量结构:
    - [0]: 命令类型 (Line, Arc, Circle, EOS, SOL, Ext)
    - [1:6]: 草图参数 (x, y, alpha, f, r)
    - [6:17]: 拉伸参数 (theta, phi, gamma, p_x, p_y, p_z, s, e1, e2, b, u)
    """
    def __init__(self, cfg, weights=None):
        super().__init__()
        self.cfg = cfg
        self.n_commands = cfg.cad_n_commands
        self.n_args = cfg.cad_n_args
        self.args_vocab_size = cfg.args_vocab_size

        # 损失权重
        self.weights = weights or {
            'cmd': 1.0,
            'args': 1.0,
        }

        # 使用 cadlib/macro.py 中定义的命令-参数掩码 (16个参数)
        self.register_buffer("cmd_args_mask", torch.tensor(CMD_ARGS_MASK, dtype=torch.float32))

    def forward(self, outputs, batch):
        """
        计算损失。

        Args:
            outputs: 模型输出字典
                - command_logits: [B, S, n_commands]
                - args_logits: [B, S, n_args, 257]
            batch: 数据批次
                - cad_sequence: [B, S, 17]
        """
        device = outputs['command_logits'].device

        # 目标序列
        cad_seq = batch['cad_sequence'].to(device)  # [B, S, 17]
        tgt_commands = cad_seq[:, :, 0]  # [B, S]
        tgt_args = cad_seq[:, :, 1:]  # [B, S, 16]

        # 获取有效位置掩码 (非EOS填充位置)
        valid_mask = self._get_valid_mask(tgt_commands)  # [B, S]

        # 1. 命令损失
        loss_cmd = self._compute_cmd_loss(outputs['command_logits'], tgt_commands, valid_mask)

        # 2. 参数损失 (使用 Gumbel soft label，统一257级词表)
        loss_args = self._compute_args_loss(outputs['args_logits'], tgt_args, tgt_commands, valid_mask)

        # 加权总损失
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
