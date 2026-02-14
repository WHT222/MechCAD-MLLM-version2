#!/usr/bin/env python3
"""
MechCAD-MLLM 训练脚本

用法:
    python train.py --epochs 50 --batch_size 4 --lr 1e-4
"""

import os
import sys
import argparse
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import torch
from torch.utils.data import DataLoader, random_split

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.dataset import OmniCADDataset
from src.trainer.trainer import MechCADTrainer


@dataclass
class TrainConfig:
    """训练配置"""
    # 数据路径
    cad_vec_dir: str = "data/Omni-CAD/cad_vec"
    text_dir: str = "data/Omni-CAD/txt"
    image_dir: str = "data/Omni-CAD/step_img"

    # 模型路径
    llava_model_name: str = "model_weights/llava-hf/llava-1.5-7b-hf"

    # 输出路径
    log_dir: str = "outputs/logs"
    model_dir: str = "outputs/checkpoints"

    # 训练超参数
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 50
    warmup_step: int = 500
    grad_clip: float = 1.0
    loss_cmd_weight: float = 2.0
    loss_args_weight: float = 1.0
    loss_tolerance: int = 3
    loss_alpha: float = 2.0
    enable_geo_soft_loss: bool = False
    geo_probe_every: int = 100
    geo_probe_samples: int = 4
    geo_probe_ema: float = 0.8
    geo_lambda_warmup_ratio: float = 0.3
    geo_lambda_seg_start: float = 0.0
    geo_lambda_seg_end: float = 0.15
    geo_lambda_dang_start: float = 0.0
    geo_lambda_dang_end: float = 0.15
    geo_seg_clip: float = 5.0
    geo_dang_clip: float = 5.0

    # 数据集
    sample_limit: Optional[int] = None  # None 表示使用全部数据
    val_split: float = 0.1
    test_split: float = 0.1  # 测试集比例
    num_workers: int = 4
    category_start: int = 0  # 起始类别 (0000)
    category_end: Optional[int] = None  # 结束类别 (None表示全部)
    clean_invalid_samples: bool = False
    clean_cache_path: Optional[str] = None
    clean_with_occ: bool = True

    # 保存策略
    save_every: int = 5
    eval_every: int = 1
    full_eval_every: int = 10  # 每N个epoch进行完整评估

    # Step级验证
    val_frequency: Optional[int] = None  # 每N步验证一次 (None表示不启用)

    # 随机种子
    seed: int = 42

    # 恢复训练
    resume: Optional[str] = None  # 检查点路径
    reset_scheduler: bool = False  # 恢复时重置调度器（跨阶段训练）

    # 训练模式
    text_only: bool = False  # 第一阶段：仅使用文本模态

    # 多视图融合
    num_selected_views: int = 2  # 随机选择的视图数量
    n_latents: int = 64  # PerceiverFusion 可学习查询数量

    # 学习率调度
    use_cosine_decay: bool = True  # 是否使用 Cosine Decay
    min_lr: float = 1e-6  # Cosine Decay 最小学习率
    total_steps: Optional[int] = None  # 总训练步数 (自动计算)
    metrics_output: Optional[str] = None  # 测试指标输出路径（默认 model_dir/test_metrics.json）


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MechCAD-MLLM 训练")

    # 数据路径
    parser.add_argument("--cad_vec_dir", type=str, default="data/Omni-CAD/cad_vec")
    parser.add_argument("--text_dir", type=str, default="data/Omni-CAD/txt")
    parser.add_argument("--image_dir", type=str, default="data/Omni-CAD/step_img")

    # 模型
    parser.add_argument("--llava_model_name", type=str,
                        default="model_weights/llava-hf/llava-1.5-7b-hf")

    # 输出
    parser.add_argument("--log_dir", type=str, default="outputs/logs")
    parser.add_argument("--model_dir", type=str, default="outputs/checkpoints")

    # 训练
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_step", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--loss_cmd_weight", type=float, default=2.0,
                        help="命令损失权重 (默认2.0)")
    parser.add_argument("--loss_args_weight", type=float, default=1.0,
                        help="参数损失权重 (默认1.0)")
    parser.add_argument("--loss_tolerance", type=int, default=3,
                        help="参数局部soft label容差半径 (默认3)")
    parser.add_argument("--loss_alpha", type=float, default=2.0,
                        help="参数局部soft label指数衰减系数 (默认2.0)")
    parser.add_argument("--enable_geo_soft_loss", action="store_true",
                        help="启用几何软目标训练（在线OCC探针 + 动态权重）")
    parser.add_argument("--geo_probe_every", type=int, default=100,
                        help="几何探针间隔步数 (默认100)")
    parser.add_argument("--geo_probe_samples", type=int, default=4,
                        help="每次几何探针抽样数量 (默认4)")
    parser.add_argument("--geo_probe_ema", type=float, default=0.8,
                        help="几何探针EMA平滑系数 (默认0.8)")
    parser.add_argument("--geo_lambda_warmup_ratio", type=float, default=0.3,
                        help="几何权重开始爬升前的训练占比 (默认0.3)")
    parser.add_argument("--geo_lambda_seg_start", type=float, default=0.0,
                        help="SegE权重起始值")
    parser.add_argument("--geo_lambda_seg_end", type=float, default=0.15,
                        help="SegE权重终值")
    parser.add_argument("--geo_lambda_dang_start", type=float, default=0.0,
                        help="DangEL权重起始值")
    parser.add_argument("--geo_lambda_dang_end", type=float, default=0.15,
                        help="DangEL权重终值")
    parser.add_argument("--geo_seg_clip", type=float, default=5.0,
                        help="SegE_rel裁剪上限 (默认5.0)")
    parser.add_argument("--geo_dang_clip", type=float, default=5.0,
                        help="DangEL_norm裁剪上限 (默认5.0)")

    # 数据集
    parser.add_argument("--sample_limit", type=int, default=None,
                        help="限制样本数量 (用于测试)")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="测试集比例 (用于最终评估)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--category_start", type=int, default=0,
                        help="起始类别编号 (默认0，即0000)")
    parser.add_argument("--category_end", type=int, default=None,
                        help="结束类别编号 (包含)，如 --category_end 9 表示加载0000-0009")
    parser.add_argument("--clean_invalid_samples", action="store_true",
                        help="数据集预处理阶段过滤非法样本")
    parser.add_argument("--clean_cache_path", type=str, default=None,
                        help="清洗缓存json路径（避免重复OCC检查）")
    parser.add_argument("--no_clean_with_occ", action="store_true",
                        help="清洗时不使用OCC，仅基础规则过滤")

    # 保存
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--full_eval_every", type=int, default=10,
                        help="每N个epoch进行完整评估")
    parser.add_argument("--val_frequency", type=int, default=None,
                        help="每N步进行验证 (None表示不启用step级验证)")

    # 随机种子
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (用于数据划分可复现)")

    # 恢复
    parser.add_argument("--resume", type=str, default=None,
                        help="从检查点恢复训练")
    parser.add_argument("--reset_scheduler", action="store_true",
                        help="恢复时重置学习率调度器（跨阶段训练时使用）")

    # 训练模式
    parser.add_argument("--text_only", action="store_true",
                        help="第一阶段：仅使用文本模态训练（跳过图像加载）")

    # 多视图融合
    parser.add_argument("--num_selected_views", type=int, default=2,
                        help="多视图融合时随机选择的视图数量 (默认2)")
    parser.add_argument("--n_latents", type=int, default=64,
                        help="PerceiverFusion 可学习查询数量 (默认64)")

    # 学习率调度
    parser.add_argument("--use_cosine_decay", action="store_true",
                        help="启用 Warmup + Cosine Decay 学习率调度")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                        help="Cosine Decay 最小学习率 (默认1e-6)")
    parser.add_argument("--metrics_output", type=str, default=None,
                        help="测试指标输出路径（默认 model_dir/test_metrics.json）")

    return parser.parse_args()


def main():
    args = parse_args()

    # 转换为配置对象
    cfg = TrainConfig(
        cad_vec_dir=args.cad_vec_dir,
        text_dir=args.text_dir,
        image_dir=args.image_dir,
        llava_model_name=args.llava_model_name,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        warmup_step=args.warmup_step,
        grad_clip=args.grad_clip,
        loss_cmd_weight=args.loss_cmd_weight,
        loss_args_weight=args.loss_args_weight,
        loss_tolerance=args.loss_tolerance,
        loss_alpha=args.loss_alpha,
        enable_geo_soft_loss=args.enable_geo_soft_loss,
        geo_probe_every=args.geo_probe_every,
        geo_probe_samples=args.geo_probe_samples,
        geo_probe_ema=args.geo_probe_ema,
        geo_lambda_warmup_ratio=args.geo_lambda_warmup_ratio,
        geo_lambda_seg_start=args.geo_lambda_seg_start,
        geo_lambda_seg_end=args.geo_lambda_seg_end,
        geo_lambda_dang_start=args.geo_lambda_dang_start,
        geo_lambda_dang_end=args.geo_lambda_dang_end,
        geo_seg_clip=args.geo_seg_clip,
        geo_dang_clip=args.geo_dang_clip,
        sample_limit=args.sample_limit,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        category_start=args.category_start,
        category_end=args.category_end,
        clean_invalid_samples=args.clean_invalid_samples,
        clean_cache_path=args.clean_cache_path,
        clean_with_occ=not args.no_clean_with_occ,
        save_every=args.save_every,
        eval_every=args.eval_every,
        full_eval_every=args.full_eval_every,
        val_frequency=args.val_frequency,
        seed=args.seed,
        resume=args.resume,
        reset_scheduler=args.reset_scheduler,
        text_only=args.text_only,
        num_selected_views=args.num_selected_views,
        n_latents=args.n_latents,
        use_cosine_decay=args.use_cosine_decay,
        min_lr=args.min_lr,
        metrics_output=args.metrics_output
    )

    print("=" * 60)
    print("MechCAD-MLLM 训练")
    print("=" * 60)
    print(f"训练模式: {'纯文本 (第一阶段)' if cfg.text_only else '多模态 (图像+文本)'}")
    print(f"数据目录: {cfg.cad_vec_dir}")
    print(f"批次大小: {cfg.batch_size}")
    print(f"学习率: {cfg.lr}")
    print(
        f"损失权重(cmd/args): {cfg.loss_cmd_weight:.2f}/{cfg.loss_args_weight:.2f}, "
        f"soft(tol={cfg.loss_tolerance}, alpha={cfg.loss_alpha:.2f})"
    )
    if cfg.enable_geo_soft_loss:
        print(
            f"几何软目标: on, probe_every={cfg.geo_probe_every}, probe_samples={cfg.geo_probe_samples}, "
            f"seg_lambda={cfg.geo_lambda_seg_start:.3f}->{cfg.geo_lambda_seg_end:.3f}, "
            f"dang_lambda={cfg.geo_lambda_dang_start:.3f}->{cfg.geo_lambda_dang_end:.3f}"
        )
    else:
        print("几何软目标: off")
    if cfg.clean_invalid_samples:
        print(
            f"数据清洗: on (occ={'on' if cfg.clean_with_occ else 'off'}, "
            f"cache={cfg.clean_cache_path if cfg.clean_cache_path else 'none'})"
        )
    print(f"Epochs: {cfg.epochs}")
    print(f"随机种子: {cfg.seed}")
    if cfg.category_end is not None:
        print(f"类别范围: {cfg.category_start:04d} - {cfg.category_end:04d}")
    else:
        print(f"类别范围: {cfg.category_start:04d} - 全部")
    print("=" * 60)

    # 设置随机种子
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # 1. 加载数据集
    print("\n[1/4] 加载数据集...")
    full_dataset = OmniCADDataset(
        cad_vec_dir=cfg.cad_vec_dir,
        text_dir=cfg.text_dir,
        image_dir=cfg.image_dir,
        sample_limit=cfg.sample_limit,
        category_start=cfg.category_start,
        category_end=cfg.category_end,
        text_only=cfg.text_only,
        num_selected_views=cfg.num_selected_views,
        clean_invalid_samples=cfg.clean_invalid_samples,
        clean_cache_path=cfg.clean_cache_path,
        clean_with_occ=cfg.clean_with_occ,
    )

    # 划分训练/验证/测试集 (三集划分)
    total_size = len(full_dataset)
    test_size = int(total_size * cfg.test_split)
    val_size = int(total_size * cfg.val_split)
    train_size = total_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(cfg.seed)
    )

    print(f"训练集: {len(train_dataset)} 样本 ({100*train_size/total_size:.1f}%)")
    print(f"验证集: {len(val_dataset)} 样本 ({100*val_size/total_size:.1f}%)")
    print(f"测试集: {len(test_dataset)} 样本 ({100*test_size/total_size:.1f}%)")

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    # 计算 total_steps (用于 Cosine Decay 学习率调度)
    steps_per_epoch = len(train_loader)
    total_steps = cfg.epochs * steps_per_epoch
    cfg.total_steps = total_steps if cfg.use_cosine_decay else None

    print(f"\n每 Epoch 步数: {steps_per_epoch}")
    print(f"总训练步数: {total_steps}")
    if cfg.use_cosine_decay:
        print(f"学习率调度: Warmup({cfg.warmup_step}步) + Cosine Decay → {cfg.min_lr:.2e}")

    # 2. 初始化训练器
    print("\n[2/4] 初始化训练器...")
    trainer = MechCADTrainer(cfg)

    # 恢复训练
    if cfg.resume:
        print(f"\n从检查点恢复: {cfg.resume}")
        trainer.load_ckpt(cfg.resume)

    # 3. 训练循环
    print("\n[3/4] 开始训练...")
    best_val_loss = float('inf')

    for epoch in range(trainer.clock.epoch, cfg.epochs + 1):
        # 训练 (支持step级验证)
        train_loss = trainer.train_epoch(
            train_loader, epoch,
            val_loader=val_loader if cfg.val_frequency else None,
            val_frequency=cfg.val_frequency
        )
        print(f"\nEpoch {epoch} 训练损失: {train_loss:.4f}")

        # 学习率在 train_epoch 内按 step 更新，这里不再重复 step

        # 验证
        if epoch % cfg.eval_every == 0:
            val_loss, metrics = trainer.validate(val_loader)
            print(f"Epoch {epoch} 验证损失: {val_loss:.4f}, "
                  f"命令准确率: {metrics['cmd_accuracy']:.4f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer.save_ckpt("best")
                print(f"保存最佳模型 (val_loss: {val_loss:.4f})")

        # 定期保存
        if epoch % cfg.save_every == 0:
            trainer.save_ckpt()

    # 4. 保存最终模型
    print("\n[4/4] 保存最终模型...")
    trainer.save_ckpt("final")

    # 5. 在测试集上进行最终评估
    print("\n" + "=" * 60)
    print("最终测试集评估")
    print("=" * 60)

    # 加载最佳模型进行测试
    print("加载最佳模型进行测试...")
    trainer.load_ckpt("best")

    test_loss, test_metrics = trainer.validate(test_loader)
    print(f"\n测试集结果:")
    print(f"  损失: {test_loss:.4f}")
    print(f"  命令准确率: {test_metrics['cmd_accuracy']*100:.2f}%")

    # 完整评估 (可选)
    if len(test_dataset) > 0:
        eval_metrics = trainer.evaluate(test_loader, max_samples=min(500, len(test_dataset)))
        print(f"  参数MAE: {eval_metrics['args_mae']:.4f}")
        print(f"  Chamfer Distance: {eval_metrics.get('chamfer_distance', -1.0):.6f}")
        print(f"  SegE: {eval_metrics.get('sege', -1.0):.4f}")
        print(f"  DangEL: {eval_metrics.get('dangel', -1.0):.4f}")
        print(f"  DangEL(norm): {eval_metrics.get('dangel_norm', -1.0):.6f}")

        # 保存测试集评估指标，供前端读取展示
        def _to_serializable(obj):
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_serializable(v) for v in obj]
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().tolist()
            # numpy scalar / python scalar
            if hasattr(obj, "item"):
                try:
                    return obj.item()
                except Exception:
                    pass
            return obj

        metrics_payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": os.path.join(cfg.model_dir, "best.pth"),
            "test_loss": float(test_loss),
            "test_metrics": _to_serializable(test_metrics),
            "eval_metrics": _to_serializable(eval_metrics),
            "config": {
                "text_only": bool(cfg.text_only),
                "batch_size": int(cfg.batch_size),
                "lr": float(cfg.lr),
                "epochs": int(cfg.epochs),
                "num_selected_views": int(cfg.num_selected_views),
                "n_latents": int(cfg.n_latents),
                "seed": int(cfg.seed),
                "test_split": float(cfg.test_split),
                "val_split": float(cfg.val_split),
            }
        }

        metrics_output_path = cfg.metrics_output or os.path.join(cfg.model_dir, "test_metrics.json")
        metrics_output_dir = os.path.dirname(metrics_output_path)
        if metrics_output_dir:
            os.makedirs(metrics_output_dir, exist_ok=True)
        with open(metrics_output_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2, ensure_ascii=False)
        print(f"  测试指标已保存: {metrics_output_path}")

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"测试集命令准确率: {test_metrics['cmd_accuracy']*100:.2f}%")
    print(f"模型保存于: {cfg.model_dir}")


if __name__ == "__main__":
    main()
