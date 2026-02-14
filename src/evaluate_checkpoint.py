#!/usr/bin/env python3
"""
Checkpoint evaluation script for MechCAD-MLLM.

Usage example:
    python src/evaluate_checkpoint.py \
        --checkpoint outputs/stage2/best.pth \
        --text_only \
        --split test \
        --metrics_output outputs/stage2/test_metrics.json
"""

import os
import sys
import json
import argparse
from datetime import datetime
from types import SimpleNamespace
from typing import Optional

import torch
from torch.utils.data import DataLoader, random_split

# Add project root to PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.dataset import OmniCADDataset
from src.trainer.trainer import MechCADTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a MechCAD checkpoint on dataset split.")

    # Checkpoint / outputs
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path, e.g. outputs/stage2/best.pth")
    parser.add_argument("--model_dir", type=str, default=None, help="Model dir override (default: checkpoint dir)")
    parser.add_argument("--log_dir", type=str, default="outputs/eval_logs", help="Evaluation log dir")
    parser.add_argument("--metrics_output", type=str, default=None, help="Output JSON path")

    # Data
    parser.add_argument("--cad_vec_dir", type=str, default="data/Omni-CAD/cad_vec")
    parser.add_argument("--text_dir", type=str, default="data/Omni-CAD/txt")
    parser.add_argument("--image_dir", type=str, default="data/Omni-CAD/step_img")
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument("--category_start", type=int, default=0)
    parser.add_argument("--category_end", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)

    # Modal settings
    parser.add_argument("--text_only", action="store_true", help="Text-only mode (stage-1 checkpoints)")
    parser.add_argument("--num_selected_views", type=int, default=2)
    parser.add_argument("--n_latents", type=int, default=64)
    parser.add_argument("--deterministic_views", action="store_true", help="Disable random view sampling in dataset")

    # Model/runtime
    parser.add_argument("--llava_model_name", type=str, default="model_weights/llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP")

    # Optimizer/scheduler fields needed by trainer init
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_step", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # Evaluation controls
    parser.add_argument("--skip_full_eval", action="store_true", help="Only run validate()")
    parser.add_argument("--full_eval_max_samples", type=int, default=500, help="Max samples for full evaluate()")

    return parser.parse_args()


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def _build_trainer_cfg(args):
    ckpt_path = os.path.abspath(args.checkpoint)
    model_dir = os.path.abspath(args.model_dir) if args.model_dir else os.path.dirname(ckpt_path)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Minimal config required by MechCADTrainer
    return SimpleNamespace(
        log_dir=os.path.abspath(args.log_dir),
        model_dir=model_dir,
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        warmup_step=int(args.warmup_step),
        grad_clip=float(args.grad_clip),
        llava_model_name=args.llava_model_name,
        text_only=bool(args.text_only),
        num_selected_views=int(args.num_selected_views),
        n_latents=int(args.n_latents),
        use_amp=not bool(args.no_amp),
        total_steps=None,
        min_lr=float(args.min_lr),
        reset_scheduler=True,
    )


def _build_target_dataset(args):
    full_dataset = OmniCADDataset(
        cad_vec_dir=args.cad_vec_dir,
        text_dir=args.text_dir,
        image_dir=args.image_dir,
        sample_limit=args.sample_limit,
        category_start=args.category_start,
        category_end=args.category_end,
        text_only=args.text_only,
        num_selected_views=args.num_selected_views,
        random_select_views=not args.deterministic_views,
    )

    total_size = len(full_dataset)
    if total_size == 0:
        raise ValueError("Dataset is empty. Check data paths and category range.")

    if args.split == "all":
        return full_dataset, {
            "total_size": total_size,
            "train_size": None,
            "val_size": None,
            "test_size": None,
        }

    test_size = int(total_size * args.test_split)
    val_size = int(total_size * args.val_split)
    train_size = total_size - val_size - test_size
    if train_size <= 0:
        raise ValueError(
            f"Invalid split sizes: total={total_size}, train={train_size}, "
            f"val={val_size}, test={test_size}. Adjust val_split/test_split."
        )

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    if args.split == "train":
        target_dataset = train_dataset
    elif args.split == "val":
        target_dataset = val_dataset
    else:
        target_dataset = test_dataset

    return target_dataset, {
        "total_size": total_size,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
    }


def main():
    args = parse_args()
    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 72)
    print("MechCAD-MLLM Checkpoint Evaluation")
    print("=" * 72)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Split: {args.split}")
    print(f"Mode: {'text_only' if args.text_only else 'multimodal'}")

    target_dataset, split_stats = _build_target_dataset(args)
    print(
        f"Dataset sizes: total={split_stats['total_size']}, "
        f"train={split_stats['train_size']}, val={split_stats['val_size']}, "
        f"test={split_stats['test_size']}"
    )
    print(f"Target split size: {len(target_dataset)}")

    loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    cfg = _build_trainer_cfg(args)
    trainer = MechCADTrainer(cfg)
    trainer.load_ckpt(ckpt_path)

    val_loss, val_metrics = trainer.validate(loader)
    print(f"[validate] loss={val_loss:.6f}, cmd_accuracy={val_metrics['cmd_accuracy']:.6f}")

    eval_metrics = {}
    if not args.skip_full_eval:
        if args.full_eval_max_samples <= 0:
            max_samples: Optional[int] = None
        else:
            max_samples = min(args.full_eval_max_samples, len(target_dataset))
        eval_metrics = trainer.evaluate(loader, max_samples=max_samples)
        print(
            "[evaluate] "
            f"args_mae={eval_metrics.get('args_mae', -1):.6f}, "
            f"chamfer={eval_metrics.get('chamfer_distance', -1):.6f}, "
            f"sege={eval_metrics.get('sege', -1):.6f}, "
            f"dangel={eval_metrics.get('dangel', -1):.6f}"
        )

    model_dir = cfg.model_dir
    if args.metrics_output:
        metrics_output_path = os.path.abspath(args.metrics_output)
    else:
        default_name = "test_metrics.json" if args.split == "test" else f"{args.split}_metrics.json"
        metrics_output_path = os.path.join(model_dir, default_name)
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": ckpt_path,
        "split": args.split,
        "validate_loss": float(val_loss),
        "validate_metrics": _to_serializable(val_metrics),
        "eval_metrics": _to_serializable(eval_metrics),
        "config": {
            "text_only": bool(args.text_only),
            "batch_size": int(args.batch_size),
            "num_selected_views": int(args.num_selected_views),
            "n_latents": int(args.n_latents),
            "deterministic_views": bool(args.deterministic_views),
            "seed": int(args.seed),
            "val_split": float(args.val_split),
            "test_split": float(args.test_split),
            "category_start": int(args.category_start),
            "category_end": args.category_end,
            "sample_limit": args.sample_limit,
            "full_eval_max_samples": int(args.full_eval_max_samples),
        },
        "split_stats": split_stats,
    }

    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Metrics saved to: {metrics_output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
