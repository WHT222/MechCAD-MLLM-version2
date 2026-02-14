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
from src.trainer.loss import CADLoss, UnifiedCADLoss
from src.trainer.scheduler import GradualWarmupScheduler, CosineAnnealingWarmupScheduler
from src.trainer.base import BaseTrainer, TrainClock
from src.utils.chamfer_distance import ChamferDistanceEvaluator
from src.utils.mesh_metrics import SegEDangELEvaluator
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

        # 混合精度训练
        self.use_amp = getattr(cfg, 'use_amp', True)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("启用混合精度训练 (AMP)")

        # 确保目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        # 构建模型
        self.build_net(cfg)

        # 设置损失函数
        self.set_loss_function()

        # 设置优化器
        self.set_optimizer(cfg)

        # 几何软目标配置（训练时启用）
        self._setup_geometry_guidance(cfg)

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

        # 多视图融合参数
        num_views = getattr(cfg, 'num_selected_views', 2)
        n_latents = getattr(cfg, 'n_latents', 64)

        print("正在初始化 MechCADModel...")
        self.net = MechCADModel(
            model_cfg,
            llava_model_name=llava_path,
            num_views=num_views,
            n_latents=n_latents
        )
        print(f"模型初始化完成。(num_views={num_views}, n_latents={n_latents})")

        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.net.parameters())
        print(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,}")

        self.model_cfg = model_cfg

    def set_loss_function(self):
        """设置损失函数"""
        raw_loss_weights = getattr(self.cfg, 'loss_weights', None)
        if isinstance(raw_loss_weights, dict):
            loss_weights = {
                'cmd': float(raw_loss_weights.get('cmd', 2.0)),
                'args': float(raw_loss_weights.get('args', 1.0)),
            }
        else:
            loss_weights = {
                'cmd': float(getattr(self.cfg, 'loss_cmd_weight', 2.0)),
                'args': float(getattr(self.cfg, 'loss_args_weight', 1.0)),
            }

        loss_tolerance = int(getattr(self.cfg, 'loss_tolerance', 3))
        loss_alpha = float(getattr(self.cfg, 'loss_alpha', 2.0))

        # 使用统一词表损失函数（类型感知参数损失）
        self.loss_func = UnifiedCADLoss(
            self.model_cfg,
            weights=loss_weights,
            tolerance=loss_tolerance,
            alpha=loss_alpha
        )
        print(
            f"损失配置: cmd={loss_weights['cmd']:.2f}, args={loss_weights['args']:.2f}, "
            f"tolerance={loss_tolerance}, alpha={loss_alpha:.2f}"
        )

    def _setup_geometry_guidance(self, cfg):
        """初始化几何软目标设置。"""
        self.enable_geo_soft_loss = bool(getattr(cfg, 'enable_geo_soft_loss', False))
        self.geo_probe_every = max(int(getattr(cfg, 'geo_probe_every', 100)), 1)
        self.geo_probe_samples = max(int(getattr(cfg, 'geo_probe_samples', 4)), 1)
        self.geo_probe_ema = float(np.clip(getattr(cfg, 'geo_probe_ema', 0.8), 0.0, 0.999))

        self.geo_lambda_warmup_ratio = float(
            np.clip(getattr(cfg, 'geo_lambda_warmup_ratio', 0.3), 0.0, 0.99)
        )
        self.geo_lambda_seg_start = float(getattr(cfg, 'geo_lambda_seg_start', 0.0))
        self.geo_lambda_seg_end = float(getattr(cfg, 'geo_lambda_seg_end', 0.15))
        self.geo_lambda_dang_start = float(getattr(cfg, 'geo_lambda_dang_start', 0.0))
        self.geo_lambda_dang_end = float(getattr(cfg, 'geo_lambda_dang_end', 0.15))
        self.geo_seg_clip = float(getattr(cfg, 'geo_seg_clip', 5.0))
        self.geo_dang_clip = float(getattr(cfg, 'geo_dang_clip', 5.0))

        self.geo_last_metrics = {
            'sege_rel': 0.0,
            'dangel_norm': 0.0,
            'valid_ratio': 0.0,
            'valid_count': 0.0,
            'failed_count': 0.0,
        }

        self.geo_evaluator = None
        if not self.enable_geo_soft_loss:
            return

        self.geo_evaluator = SegEDangELEvaluator()
        if not getattr(self.geo_evaluator, '_cad_available', False):
            print("警告: OCC/cadlib 不可用，禁用几何软目标训练")
            self.enable_geo_soft_loss = False
            self.geo_evaluator = None
            return

        print(
            "启用几何软目标: "
            f"probe_every={self.geo_probe_every}, probe_samples={self.geo_probe_samples}, "
            f"lambda_seg=[{self.geo_lambda_seg_start:.3f}->{self.geo_lambda_seg_end:.3f}], "
            f"lambda_dang=[{self.geo_lambda_dang_start:.3f}->{self.geo_lambda_dang_end:.3f}], "
            f"warmup_ratio={self.geo_lambda_warmup_ratio:.2f}"
        )

    def _tensor_scalar(self, value: float, ref_tensor: torch.Tensor) -> torch.Tensor:
        return torch.tensor(float(value), device=ref_tensor.device, dtype=ref_tensor.dtype)

    def _get_geo_lambdas(self):
        """按训练进度动态计算 λ_seg 与 λ_dang。"""
        total_steps = getattr(self.cfg, 'total_steps', None)
        if total_steps is not None and int(total_steps) > 0:
            progress = float(np.clip(self.clock.step / float(total_steps), 0.0, 1.0))
        else:
            total_epochs = max(int(getattr(self.cfg, 'epochs', 1)), 1)
            progress = float(np.clip((self.clock.epoch - 1) / float(total_epochs), 0.0, 1.0))

        if progress <= self.geo_lambda_warmup_ratio:
            phase = 0.0
        else:
            phase = (progress - self.geo_lambda_warmup_ratio) / (1.0 - self.geo_lambda_warmup_ratio + 1e-8)
            phase = float(np.clip(phase, 0.0, 1.0))

        lambda_seg = self.geo_lambda_seg_start + phase * (self.geo_lambda_seg_end - self.geo_lambda_seg_start)
        lambda_dang = self.geo_lambda_dang_start + phase * (self.geo_lambda_dang_end - self.geo_lambda_dang_start)
        return float(lambda_seg), float(lambda_dang)

    @torch.no_grad()
    def _run_geometry_probe(self, outputs, data):
        """在线几何探针：解析少量样本，更新几何质量统计。"""
        if (not self.enable_geo_soft_loss) or (self.geo_evaluator is None):
            return None

        try:
            pred_vec = self.logits2vec(outputs)
            gt_vec = data['cad_sequence'].detach().cpu().numpy()
            n_probe = min(self.geo_probe_samples, pred_vec.shape[0], gt_vec.shape[0])
            if n_probe <= 0:
                return None

            pred_list = [pred_vec[i] for i in range(n_probe)]
            gt_list = [gt_vec[i] for i in range(n_probe)]
            metrics = self.geo_evaluator.evaluate(pred_list, gt_list)

            valid_count = float(metrics.get('valid_count', 0))
            failed_count = float(metrics.get('failed_count', n_probe))
            total_count = max(valid_count + failed_count, 1.0)
            valid_ratio = valid_count / total_count

            sege_rel = float(metrics.get('sege_rel', -1.0))
            dangel_norm = float(metrics.get('dangel_norm', -1.0))
            if sege_rel < 0:
                sege_rel = self.geo_seg_clip
            if dangel_norm < 0:
                dangel_norm = self.geo_dang_clip

            sege_rel = float(np.clip(sege_rel, 0.0, self.geo_seg_clip))
            dangel_norm = float(np.clip(dangel_norm, 0.0, self.geo_dang_clip))

            m = self.geo_probe_ema
            self.geo_last_metrics['sege_rel'] = m * self.geo_last_metrics['sege_rel'] + (1.0 - m) * sege_rel
            self.geo_last_metrics['dangel_norm'] = m * self.geo_last_metrics['dangel_norm'] + (1.0 - m) * dangel_norm
            self.geo_last_metrics['valid_ratio'] = m * self.geo_last_metrics['valid_ratio'] + (1.0 - m) * valid_ratio
            self.geo_last_metrics['valid_count'] = valid_count
            self.geo_last_metrics['failed_count'] = failed_count

            return {
                'geo_probe_sege_rel': sege_rel,
                'geo_probe_dangel_norm': dangel_norm,
                'geo_probe_valid_ratio': valid_ratio,
                'geo_probe_valid_count': valid_count,
                'geo_probe_failed_count': failed_count,
            }
        except Exception as e:
            print(f"几何探针失败(step={self.clock.step}): {e}")
            return None

    def _apply_geometry_guidance(self, loss_dict):
        """
        几何软目标引导：
        使用 stop-gradient 的几何惩罚对 CE 总损失做重加权。
        """
        if not self.enable_geo_soft_loss:
            return loss_dict

        lambda_seg, lambda_dang = self._get_geo_lambdas()
        sege_rel = float(self.geo_last_metrics['sege_rel'])
        dangel_norm = float(self.geo_last_metrics['dangel_norm'])
        valid_ratio = float(self.geo_last_metrics['valid_ratio'])

        loss_ce = loss_dict['loss']
        lambda_seg_t = self._tensor_scalar(lambda_seg, loss_ce)
        lambda_dang_t = self._tensor_scalar(lambda_dang, loss_ce)
        sege_t = self._tensor_scalar(sege_rel, loss_ce)
        dangel_t = self._tensor_scalar(dangel_norm, loss_ce)
        valid_ratio_t = self._tensor_scalar(valid_ratio, loss_ce)

        # 几何惩罚（非可微，作为软目标信号）
        loss_geo = lambda_seg_t * sege_t + lambda_dang_t * dangel_t
        scaled_loss = loss_ce * (1.0 + loss_geo)

        loss_dict['loss_ce'] = loss_ce
        loss_dict['loss_geo'] = loss_geo
        loss_dict['loss_geo_proxy_total'] = loss_ce + loss_geo
        loss_dict['loss'] = scaled_loss

        loss_dict['lambda_seg'] = lambda_seg_t
        loss_dict['lambda_dang'] = lambda_dang_t
        loss_dict['geo_sege_rel'] = sege_t
        loss_dict['geo_dangel_norm'] = dangel_t
        loss_dict['geo_valid_ratio'] = valid_ratio_t
        return loss_dict

    def set_optimizer(self, cfg):
        """设置优化器和学习率调度器"""
        # 优化解码器和多视图融合模块的参数 (LLaVA 编码器已冻结)
        trainable_params = list(self.net.llm2cad_decoder.parameters())

        # 如果不是纯文本模式，添加多视图融合模块参数
        if not self.text_only:
            trainable_params += list(self.net.multiview_fusion.parameters())

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=cfg.lr,
            weight_decay=getattr(cfg, 'weight_decay', 0.01)
        )

        # 学习率调度器: Warmup + Cosine Decay
        warmup_steps = getattr(cfg, 'warmup_step', 500)
        total_steps = getattr(cfg, 'total_steps', None)

        if total_steps is not None:
            # 使用 Warmup + Cosine Decay
            min_lr = getattr(cfg, 'min_lr', cfg.lr * 0.01)
            self.scheduler = CosineAnnealingWarmupScheduler(
                self.optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr=min_lr
            )
            print(f"学习率调度: Warmup({warmup_steps}步) + Cosine Decay → {min_lr:.2e}")
        else:
            # 仅使用 Warmup（向后兼容）
            self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, warmup_steps)
            print(f"学习率调度: Warmup({warmup_steps}步)，无衰减")

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

            if self.enable_geo_soft_loss:
                if (self.clock.step + 1) % self.geo_probe_every == 0:
                    probe_stats = self._run_geometry_probe(outputs, data)
                    if probe_stats is not None:
                        for k, v in probe_stats.items():
                            loss_dict[k] = self._tensor_scalar(v, loss_dict['loss'])
                loss_dict = self._apply_geometry_guidance(loss_dict)

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
            postfix = {
                'loss': f"{loss_val:.4f}",
                'cmd': f"{loss_dict['loss_cmd'].item():.4f}",
                'args': f"{loss_dict['loss_args'].item():.4f}",
                'lr': f"{current_lr:.2e}"
            }
            if 'cmd_recall_ext' in loss_dict:
                postfix['ext_r'] = f"{loss_dict['cmd_recall_ext'].item():.3f}"
            if 'args_acc_param' in loss_dict:
                postfix['param_a'] = f"{loss_dict['args_acc_param'].item():.3f}"
            if 'loss_geo' in loss_dict:
                postfix['geo'] = f"{loss_dict['loss_geo'].item():.3f}"
            if 'geo_valid_ratio' in loss_dict:
                postfix['gvr'] = f"{loss_dict['geo_valid_ratio'].item():.2f}"
            pbar.set_postfix(postfix)

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
        tgt_commands = data['commands'].to(device)

        pred_commands = outputs['command_logits'].argmax(dim=-1)

        # 有效掩码
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
                    # 统一词表版本: 先解码为13维CAD向量，再计算参数误差
                    pred_vec_t = torch.from_numpy(pred_vec).to(device)
                    pred_args = pred_vec_t[:, :, 1:]
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

        # 计算几何指标 (Chamfer / SegE / DangEL)
        print("\n计算几何指标 (Chamfer / SegE / DangEL)...")
        try:
            if len(all_pred_vecs) == 0 or len(all_gt_vecs) == 0:
                raise ValueError("无可用样本用于几何评估")

            # 合并所有预测和真实向量
            all_pred = np.concatenate(all_pred_vecs, axis=0)
            all_gt = np.concatenate(all_gt_vecs, axis=0)
            geom_eval_count = min(100, len(all_pred), len(all_gt))  # 限制数量避免评估过慢

            eval_pred_list = [all_pred[i] for i in range(geom_eval_count)]
            eval_gt_list = [all_gt[i] for i in range(geom_eval_count)]

            cd_evaluator = ChamferDistanceEvaluator(n_points=2048, normalize=True)
            cd_metrics = cd_evaluator.evaluate(
                eval_pred_list,
                eval_gt_list
            )

            metrics['chamfer_distance'] = cd_metrics['chamfer_distance']
            metrics['chamfer_valid_count'] = cd_metrics['valid_count']
            metrics['chamfer_failed_count'] = cd_metrics['failed_count']

            seg_dangel_evaluator = SegEDangELEvaluator()
            topo_metrics = seg_dangel_evaluator.evaluate(eval_pred_list, eval_gt_list)
            metrics['sege'] = topo_metrics['sege']
            metrics['sege_rel'] = topo_metrics['sege_rel']
            metrics['dangel'] = topo_metrics['dangel']
            metrics['dangel_norm'] = topo_metrics['dangel_norm']
            metrics['mesh_valid_count'] = topo_metrics['valid_count']
            metrics['mesh_failed_count'] = topo_metrics['failed_count']

            print(
                f"[Chamfer] CD: {cd_metrics['chamfer_distance']:.6f}, "
                f"有效: {cd_metrics['valid_count']}, 失败: {cd_metrics['failed_count']}"
            )
            print(
                f"[SegE/DangEL] SegE: {topo_metrics['sege']:.4f}, "
                f"DangEL: {topo_metrics['dangel']:.4f}, "
                f"DangEL(norm): {topo_metrics['dangel_norm']:.6f}, "
                f"有效: {topo_metrics['valid_count']}, 失败: {topo_metrics['failed_count']}"
            )
        except Exception as e:
            print(f"几何指标计算失败: {e}")
            metrics['chamfer_distance'] = -1.0
            metrics['chamfer_valid_count'] = 0
            metrics['chamfer_failed_count'] = 0
            metrics['sege'] = -1.0
            metrics['sege_rel'] = -1.0
            metrics['dangel'] = -1.0
            metrics['dangel_norm'] = -1.0
            metrics['mesh_valid_count'] = 0
            metrics['mesh_failed_count'] = 0

        print(f"\n[Evaluate] 命令准确率: {cmd_accuracy*100:.2f}%, 参数MAE: {args_mae:.4f}")

        # 记录到 TensorBoard
        if self.use_tensorboard:
            self.val_tb.add_scalar('eval/cmd_accuracy', cmd_accuracy, self.clock.epoch)
            self.val_tb.add_scalar('eval/args_mae', args_mae, self.clock.epoch)
            if 'chamfer_distance' in metrics:
                self.val_tb.add_scalar('eval/chamfer_distance', metrics['chamfer_distance'], self.clock.epoch)
            if 'sege' in metrics:
                self.val_tb.add_scalar('eval/sege', metrics['sege'], self.clock.epoch)
            if 'dangel' in metrics:
                self.val_tb.add_scalar('eval/dangel', metrics['dangel'], self.clock.epoch)
            if 'dangel_norm' in metrics:
                self.val_tb.add_scalar('eval/dangel_norm', metrics['dangel_norm'], self.clock.epoch)

        return metrics

    def logits2vec(self, outputs, refill_pad=True):
        """
        将模型输出转换为 CAD 向量（统一词表版本）。

        Args:
            outputs: 模型输出字典
                - command_logits: [B, S, n_commands]
                - unified_args_logits: [B, S, MAX_ARGS_PER_CMD, VOCAB_SIZE]
            refill_pad: 是否将未使用的参数填充为 -1

        Returns:
            cad_vec: [B, S, 13] numpy 数组
        """
        from src.unified_vocab.converter import unified_tokens_to_13d

        cmd_logits = outputs['command_logits']
        unified_args_logits = outputs['unified_args_logits']

        # 命令预测
        pred_commands = cmd_logits.argmax(dim=-1)  # [B, S]

        # 参数token预测
        pred_args_tokens = unified_args_logits.argmax(dim=-1)  # [B, S, MAX_ARGS_PER_CMD]

        # 转换为numpy
        pred_commands_np = pred_commands.detach().cpu().numpy()
        pred_args_tokens_np = pred_args_tokens.detach().cpu().numpy()

        # 批量转换为13维CAD向量
        B, S = pred_commands_np.shape
        cad_vecs = []
        for b in range(B):
            cad_vec = unified_tokens_to_13d(pred_commands_np[b], pred_args_tokens_np[b])
            cad_vecs.append(cad_vec)

        return np.stack(cad_vecs, axis=0)  # [B, S, 13]

    def save_ckpt(self, name=None):
        """保存检查点"""
        if name is None:
            save_path = os.path.join(self.model_dir, f"ckpt_epoch{self.clock.epoch}.pth")
            print(f"保存检查点 epoch {self.clock.epoch}...")
        else:
            save_path = os.path.join(self.model_dir, f"{name}.pth")

        # 仅保存可训练模块权重 (LLaVA 太大且已冻结)
        decoder_state = self.net.llm2cad_decoder.cpu().state_dict()
        checkpoint_dict = {
            'clock': self.clock.make_checkpoint(),
            'decoder_state_dict': decoder_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_cfg': self.model_cfg
        }

        # 多模态阶段额外保存融合模块权重
        if not self.text_only:
            fusion_state = self.net.multiview_fusion.cpu().state_dict()
            checkpoint_dict['fusion_state_dict'] = fusion_state

        torch.save(checkpoint_dict, save_path)

        # 移回 GPU
        device = next(self.net.llava_model.parameters()).device
        self.net.llm2cad_decoder.to(device)
        self.net.multiview_fusion.to(device)

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
        elif os.path.isabs(name) or os.path.exists(name):
            # 如果是绝对路径或已存在的路径，直接使用
            load_path = name
        else:
            # 相对于 model_dir 的路径
            load_path = os.path.join(self.model_dir, f"{name}.pth")
            if not os.path.exists(load_path):
                load_path = os.path.join(self.model_dir, name)

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"检查点不存在: {load_path}")

        print(f"从 {load_path} 加载检查点...")
        # weights_only=False 用于加载包含自定义类 (MechCADConfig) 的检查点
        checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)

        # 加载模型权重
        self.net.llm2cad_decoder.load_state_dict(checkpoint['decoder_state_dict'])

        # 如果有多视图融合权重，也加载
        if 'fusion_state_dict' in checkpoint:
            self.net.multiview_fusion.load_state_dict(checkpoint['fusion_state_dict'])

        # 是否恢复优化器和调度器状态（跨阶段训练时需要重置）
        reset_scheduler = getattr(self.cfg, 'reset_scheduler', False)

        if not reset_scheduler:
            try:
                # 检查优化器参数组是否匹配
                saved_param_count = sum(
                    len(g['params']) for g in checkpoint['optimizer_state_dict']['param_groups']
                )
                current_param_count = sum(
                    len(list(g['params'])) for g in self.optimizer.param_groups
                )

                if saved_param_count != current_param_count:
                    print(f"警告: 优化器参数组大小不匹配 (保存:{saved_param_count} vs 当前:{current_param_count})")
                    print("跨阶段训练时参数组变化，将使用新的优化器和调度器")
                    reset_scheduler = True
                else:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    self.clock.restore_checkpoint(checkpoint['clock'])
                    print("已恢复优化器和调度器状态")
            except Exception as e:
                print(f"警告: 无法恢复优化器/调度器状态 ({e})")
                print("将使用新的调度器")
                reset_scheduler = True

        if reset_scheduler:
            print("学习率调度器已重置（从新的 warmup 开始）")
            self.clock.reset()  # 重置训练计数器

        # 移到正确设备
        device = next(self.net.llava_model.parameters()).device
        self.net.llm2cad_decoder.to(device)

        # 将优化器状态也移到 GPU
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        print(f"检查点加载完成，恢复到 epoch {self.clock.epoch}")
