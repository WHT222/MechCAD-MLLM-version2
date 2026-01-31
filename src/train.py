#!/usr/bin/env python3
"""
MechCAD-MLLM 训练脚本

用法:
    python train.py --epochs 50 --batch_size 4 --lr 1e-4
"""

import os
import sys
import argparse
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

    # 数据集
    sample_limit: Optional[int] = None  # None 表示使用全部数据
    val_split: float = 0.1
    num_workers: int = 4
    category_start: int = 0  # 起始类别 (0000)
    category_end: Optional[int] = None  # 结束类别 (None表示全部)

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

    # 训练模式
    text_only: bool = False  # 第一阶段：仅使用文本模态


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

    # 数据集
    parser.add_argument("--sample_limit", type=int, default=None,
                        help="限制样本数量 (用于测试)")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--category_start", type=int, default=0,
                        help="起始类别编号 (默认0，即0000)")
    parser.add_argument("--category_end", type=int, default=None,
                        help="结束类别编号 (包含)，如 --category_end 9 表示加载0000-0009")

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

    # 训练模式
    parser.add_argument("--text_only", action="store_true",
                        help="第一阶段：仅使用文本模态训练（跳过图像加载）")

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
        sample_limit=args.sample_limit,
        val_split=args.val_split,
        num_workers=args.num_workers,
        category_start=args.category_start,
        category_end=args.category_end,
        save_every=args.save_every,
        eval_every=args.eval_every,
        full_eval_every=args.full_eval_every,
        val_frequency=args.val_frequency,
        seed=args.seed,
        resume=args.resume,
        text_only=args.text_only
    )

    print("=" * 60)
    print("MechCAD-MLLM 训练")
    print("=" * 60)
    print(f"训练模式: {'纯文本 (第一阶段)' if cfg.text_only else '多模态 (图像+文本)'}")
    print(f"数据目录: {cfg.cad_vec_dir}")
    print(f"批次大小: {cfg.batch_size}")
    print(f"学习率: {cfg.lr}")
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
        text_only=cfg.text_only
    )

    # 划分训练/验证集
    val_size = int(len(full_dataset) * cfg.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed)
    )

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")

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

        # 更新学习率
        trainer.update_learning_rate()

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

    print("\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存于: {cfg.model_dir}")


if __name__ == "__main__":
    main()
