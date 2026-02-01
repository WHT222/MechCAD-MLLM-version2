import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.model import MechCADModel, MechCADConfig
from src.trainer.loss import CADLoss
from src.trainer.scheduler import GradualWarmupScheduler
from src.trainer.base import BaseTrainer, TrainClock
from src.utils.chamfer_distance import ChamferDistanceEvaluator
from cadlib.macro import *


class MechCADTrainer(BaseTrainer):
    """
    MechCAD 模型训练器。

    支持:
    - Text-to-CAD (单视图或多视图)
    - 渐进式训练策略
    - 混合精度训练
    - 梯度累积
    """

    def __init__(self, cfg):
        """
        初始化训练器。

        Args:
            cfg: 配置对象，需包含以下属性:
                - log_dir: 日志目录
                - model_dir: 模型保存目录
                - batch_size: 批次大小
                - lr: 学习率
                - warmup_step: 预热步数
                - grad_clip: 梯度裁剪阈值
                - llava_model_name: LLaVA模型路径
                - text_only: 是否仅使用文本模态（第一阶段训练）
        """
        self.cfg = cfg
        self.log_dir = cfg.log_dir
        self.model_dir = cfg.model_dir
        self.clock = TrainClock()
        self.batch_size = cfg.batch_size
        self.text_only = getattr(cfg, 'text_only', False)

        # 确保目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # 构建模型
        self.build_net(cfg)

        # 设置损失函数
        self.set_loss_function()

        # 设置优化器
        self.set_optimizer(cfg)

        # TensorBoard (延迟导入以避免未安装时报错)
        try:
            from tensorboardX import SummaryWriter
            self.train_tb = SummaryWriter(os.path.join(self.log_dir, 'train.events'))
            self.val_tb = SummaryWriter(os.path.join(self.log_dir, 'val.events'))
            self.use_tensorboard = True
        except ImportError:
            print("警告: tensorboardX 未安装，跳过 TensorBoard 日志记录")
            self.use_tensorboard = False

    def build_net(self, cfg):
        """构建 MechCAD 模型"""
        model_cfg = MechCADConfig()

        # 从 cfg 覆盖配置 (如果存在)
        if hasattr(cfg, 'd_model'):
            model_cfg.d_model = cfg.d_model
        if hasattr(cfg, 'n_layers_decode'):
            model_cfg.n_layers_decode = cfg.n_layers_decode

        llava_path = getattr(cfg, 'llava_model_name', 'model_weights/llava-hf/llava-1.5-7b-hf')

        print("正在初始化 MechCADModel...")
        self.net = MechCADModel(model_cfg, llava_model_name=llava_path)
        print("模型初始化完成。")

        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.net.parameters())
        print(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,}")

        self.model_cfg = model_cfg

    def set_loss_function(self):
        """设置损失函数"""
        loss_weights = getattr(self.cfg, 'loss_weights', None)
        self.loss_func = CADLoss(self.model_cfg, weights=loss_weights)

    def set_optimizer(self, cfg):
        """设置优化器和学习率调度器"""
        # 仅优化解码器参数 (LLaVA 编码器已冻结)
        decoder_params = self.net.llm2cad_decoder.parameters()

        self.optimizer = optim.AdamW(
            decoder_params,
            lr=cfg.lr,
            weight_decay=getattr(cfg, 'weight_decay', 0.01)
        )

        warmup_steps = getattr(cfg, 'warmup_step', 1000)
        self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, warmup_steps)

    def forward(self, data):
        """
        前向传播。

        Args:
            data: 来自 DataLoader 的批次数据

        Returns:
            outputs: 模型输出字典
            loss_dict: 损失字典
        """
        outputs = self.net(data, text_only=self.text_only)

        # 将损失函数移到正确设备
        device = outputs['command_logits'].device
        self.loss_func.to(device)

        loss_dict = self.loss_func(outputs, data)

        return outputs, loss_dict

    def train_epoch(self, train_loader, epoch, val_loader=None, val_frequency=None):
        """
        训练一个 epoch。

        Args:
            train_loader: 训练数据加载器
            epoch: 当前 epoch 数
            val_loader: 可选的验证数据加载器 (用于step级验证)
            val_frequency: 每多少step验证一次 (None表示不进行step级验证)

        Returns:
            avg_loss: 平均损失
        """
        self.net.train()
        # 确保 LLaVA 保持 eval 模式
        self.net.llava_model.eval()

        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        total_loss = 0
        num_batches = 0

        # 创建验证数据迭代器 (用于step级验证)
        val_iter = iter(val_loader) if val_loader and val_frequency else None

        for batch_idx, data in enumerate(pbar):
            outputs, loss_dict = self.forward(data)

            # 反向传播
            self.update_network(loss_dict)

            # 每batch更新学习率 (warmup调度器需要per-step更新)
            self.scheduler.step()

            # 记录
            loss_val = loss_dict['loss'].item()
            total_loss += loss_val
            num_batches += 1

            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_val:.4f}",
                'cmd': f"{loss_dict['loss_cmd'].item():.4f}",
                'args': f"{loss_dict['loss_args'].item():.4f}",
                'lr': f"{current_lr:.2e}"
            })

            # 记录到 TensorBoard
            if self.use_tensorboard and self.clock.step % 10 == 0:
                self.record_losses(loss_dict, 'train')
                self.train_tb.add_scalar('learning_rate', current_lr, self.clock.step)

            # Step级验证
            if val_frequency and val_iter and self.clock.step % val_frequency == 0:
                try:
                    val_data = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_data = next(val_iter)
                self._step_validate(val_data)

            self.clock.tick()

        self.clock.tock()
        avg_loss = total_loss / max(num_batches, 1)

        return avg_loss

    def _step_validate(self, data):
        """单步验证 (用于训练过程中的快速验证)"""
        self.net.eval()
        with torch.no_grad():
            outputs, loss_dict = self.forward(data)
            if self.use_tensorboard:
                self.record_losses(loss_dict, 'validation')
        self.net.train()
        self.net.llava_model.eval()

    def validate(self, val_loader):
        """
        验证模型。

        Args:
            val_loader: 验证数据加载器

        Returns:
            avg_loss: 平均验证损失
            metrics: 评估指标字典
        """
        self.net.eval()

        pbar = tqdm(val_loader, desc=f"Validate Epoch {self.clock.epoch}")
        total_loss = 0
        num_batches = 0

        # 用于计算准确率的累积器
        all_cmd_correct = 0
        all_cmd_total = 0

        with torch.no_grad():
            for data in pbar:
                outputs, loss_dict = self.forward(data)

                loss_val = loss_dict['loss'].item()
                total_loss += loss_val
                num_batches += 1

                # 计算命令准确率
                cmd_acc = self._compute_cmd_accuracy(outputs, data)
                all_cmd_correct += cmd_acc['correct']
                all_cmd_total += cmd_acc['total']

                pbar.set_postfix({'val_loss': f"{loss_val:.4f}"})

        avg_loss = total_loss / max(num_batches, 1)
        cmd_accuracy = all_cmd_correct / max(all_cmd_total, 1)

        metrics = {
            'val_loss': avg_loss,
            'cmd_accuracy': cmd_accuracy
        }

        # 记录到 TensorBoard
        if self.use_tensorboard:
            self.val_tb.add_scalar('val_loss', avg_loss, self.clock.epoch)
            self.val_tb.add_scalar('cmd_accuracy', cmd_accuracy, self.clock.epoch)

        return avg_loss, metrics

    def _compute_cmd_accuracy(self, outputs, data):
        """计算命令预测准确率"""
        device = outputs['command_logits'].device
        cad_seq = data['cad_sequence'].to(device)
        tgt_commands = cad_seq[:, :, 0]

        pred_commands = outputs['command_logits'].argmax(dim=-1)

        # 有效掩码
        valid_mask = (tgt_commands != EOS_IDX).float()
        # 包含第一个 EOS
        eos_cumsum = (tgt_commands == EOS_IDX).cumsum(dim=-1)
        valid_mask = (eos_cumsum <= 1).float()

        correct = ((pred_commands == tgt_commands).float() * valid_mask).sum().item()
        total = valid_mask.sum().item()

        return {'correct': correct, 'total': total}

    def evaluate(self, val_loader, max_samples=None):
        """
        完整评估模型（生成CAD序列并计算指标）。

        Args:
            val_loader: 验证数据加载器
            max_samples: 最大评估样本数 (None表示全部)

        Returns:
            metrics: 评估指标字典
        """
        self.net.eval()

        all_pred_vecs = []
        all_gt_vecs = []
        total_cmd_correct = 0
        total_cmd_count = 0
        total_args_error = 0
        total_args_count = 0

        pbar = tqdm(val_loader, desc="Evaluate")
        sample_count = 0

        with torch.no_grad():
            for data in pbar:
                if max_samples and sample_count >= max_samples:
                    break

                outputs, loss_dict = self.forward(data)

                # 转换为CAD向量
                pred_vec = self.logits2vec(outputs)
                gt_vec = data['cad_sequence'].cpu().numpy()

                all_pred_vecs.append(pred_vec)
                all_gt_vecs.append(gt_vec)

                # 计算命令准确率
                cmd_acc = self._compute_cmd_accuracy(outputs, data)
                total_cmd_correct += cmd_acc['correct']
                total_cmd_count += cmd_acc['total']

                # 计算参数误差 (仅对命令正确的位置)
                device = outputs['command_logits'].device
                cad_seq = data['cad_sequence'].to(device)
                pred_cmd = outputs['command_logits'].argmax(dim=-1)
                gt_cmd = cad_seq[:, :, 0]

                cmd_match = (pred_cmd == gt_cmd)
                valid_mask = (gt_cmd != EOS_IDX) & (gt_cmd != SOL_IDX)
                eval_mask = cmd_match & valid_mask

                if eval_mask.sum() > 0:
                    pred_args = outputs['args_logits'].argmax(dim=-1) - 1
                    gt_args = cad_seq[:, :, 1:]
                    args_diff = torch.abs(pred_args - gt_args).float()
                    total_args_error += (args_diff * eval_mask.unsqueeze(-1)).sum().item()
                    total_args_count += eval_mask.sum().item() * 12

                sample_count += pred_vec.shape[0]

                pbar.set_postfix({
                    'samples': sample_count,
                    'cmd_acc': f"{total_cmd_correct / max(total_cmd_count, 1):.3f}"
                })

        # 汇总指标
        cmd_accuracy = total_cmd_correct / max(total_cmd_count, 1)
        args_mae = total_args_error / max(total_args_count, 1)

        metrics = {
            'cmd_accuracy': cmd_accuracy,
            'args_mae': args_mae,
            'num_samples': sample_count
        }

        # 计算 Chamfer Distance (几何相似度)
        print("\n计算 Chamfer Distance...")
        try:
            cd_evaluator = ChamferDistanceEvaluator(n_points=2048, normalize=True)
            # 合并所有预测和真实向量
            all_pred = np.concatenate(all_pred_vecs, axis=0)
            all_gt = np.concatenate(all_gt_vecs, axis=0)

            cd_metrics = cd_evaluator.evaluate(
                [all_pred[i] for i in range(min(100, len(all_pred)))],  # 限制数量避免太慢
                [all_gt[i] for i in range(min(100, len(all_gt)))]
            )

            metrics['chamfer_distance'] = cd_metrics['chamfer_distance']
            metrics['chamfer_valid_count'] = cd_metrics['valid_count']
            metrics['chamfer_failed_count'] = cd_metrics['failed_count']

            print(f"[Chamfer Distance] CD: {cd_metrics['chamfer_distance']:.6f}, "
                  f"有效: {cd_metrics['valid_count']}, 失败: {cd_metrics['failed_count']}")
        except Exception as e:
            print(f"Chamfer Distance 计算失败: {e}")
            metrics['chamfer_distance'] = -1.0

        print(f"\n[Evaluate] 命令准确率: {cmd_accuracy*100:.2f}%, 参数MAE: {args_mae:.4f}")

        # 记录到 TensorBoard
        if self.use_tensorboard:
            self.val_tb.add_scalar('eval/cmd_accuracy', cmd_accuracy, self.clock.epoch)
            self.val_tb.add_scalar('eval/args_mae', args_mae, self.clock.epoch)

        return metrics

    def logits2vec(self, outputs, refill_pad=True):
        """
        将模型输出转换为 CAD 向量。

        Args:
            outputs: 模型输出字典
            refill_pad: 是否将未使用的参数填充为 -1

        Returns:
            cad_vec: [B, S, 13] numpy 数组
        """
        cmd_logits = outputs['command_logits']
        args_logits = outputs['args_logits']
        angle_logits = outputs['angle_logits']
        pos_logits = outputs['pos_logits']

        # 命令预测
        pred_commands = cmd_logits.argmax(dim=-1)  # [B, S]

        # 参数预测 (减1恢复原始范围)
        pred_args = args_logits.argmax(dim=-1) - 1  # [B, S, 12]

        # 角度和位置 Token 预测
        pred_angle = angle_logits.argmax(dim=-1)  # [B, S]
        pred_pos = pos_logits.argmax(dim=-1)  # [B, S]

        # 对于 Ext 命令，用 token 预测覆盖对应参数位置
        ext_mask = (pred_commands == EXT_IDX)
        pred_args[:, :, 5] = torch.where(ext_mask, pred_angle, pred_args[:, :, 5])
        pred_args[:, :, 6] = torch.where(ext_mask, pred_pos, pred_args[:, :, 6])

        if refill_pad:
            # 根据命令类型填充未使用的参数为 -1
            # 适配13D向量的命令-参数掩码 (12个参数)
            cmd_args_mask_13d = torch.tensor([
                [1, 1, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0],  # Line: x, y
                [1, 1, 1, 1, 0,  0, 0,  0, 0, 0, 0, 0],  # Arc: x, y, alpha, f
                [1, 1, 0, 0, 1,  0, 0,  0, 0, 0, 0, 0],  # Circle: x, y, r
                [0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0],  # EOS: 无参数
                [0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0],  # SOL: 无参数
                [0, 0, 0, 0, 0,  1, 1,  1, 1, 1, 1, 1],  # Ext: angle, pos, e1-e2-b-u-s
            ], dtype=torch.float32, device=pred_commands.device)
            mask = cmd_args_mask_13d[pred_commands.long()]  # [B, S, 12]
            pred_args = torch.where(mask.bool(), pred_args, torch.tensor(-1, device=pred_args.device))

        # 组合为 13 维向量
        cad_vec = torch.cat([pred_commands.unsqueeze(-1), pred_args], dim=-1)

        return cad_vec.detach().cpu().numpy()

    def save_ckpt(self, name=None):
        """保存检查点"""
        if name is None:
            save_path = os.path.join(self.model_dir, f"ckpt_epoch{self.clock.epoch}.pth")
            print(f"保存检查点 epoch {self.clock.epoch}...")
        else:
            save_path = os.path.join(self.model_dir, f"{name}.pth")

        # 仅保存解码器权重 (LLaVA 太大且已冻结)
        decoder_state = self.net.llm2cad_decoder.cpu().state_dict()

        torch.save({
            'clock': self.clock.make_checkpoint(),
            'decoder_state_dict': decoder_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_cfg': self.model_cfg
        }, save_path)

        # 移回 GPU
        device = next(self.net.llava_model.parameters()).device
        self.net.llm2cad_decoder.to(device)

        print(f"检查点已保存到: {save_path}")

    def load_ckpt(self, name=None):
        """加载检查点"""
        if name is None or name == 'latest':
            # 查找最新的检查点
            ckpts = [f for f in os.listdir(self.model_dir) if f.startswith('ckpt_epoch')]
            if not ckpts:
                raise FileNotFoundError(f"未找到检查点: {self.model_dir}")
            ckpts.sort(key=lambda x: int(x.split('epoch')[1].split('.')[0]))
            name = ckpts[-1].replace('.pth', '')

        load_path = os.path.join(self.model_dir, f"{name}.pth")
        if not os.path.exists(load_path):
            load_path = os.path.join(self.model_dir, name)

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"检查点不存在: {load_path}")

        print(f"从 {load_path} 加载检查点...")
        # weights_only=False 用于加载包含自定义类 (MechCADConfig) 的检查点
        checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)

        self.net.llm2cad_decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.clock.restore_checkpoint(checkpoint['clock'])

        # 移到正确设备
        device = next(self.net.llava_model.parameters()).device
        self.net.llm2cad_decoder.to(device)

        # 将优化器状态也移到 GPU
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        print(f"检查点加载完成，恢复到 epoch {self.clock.epoch}")
