#!/usr/bin/env python3
"""
MechCAD-MLLM 推理脚本

用于测试训练好的模型生成 CAD 序列。

用法:
    # 纯文本模式
    python src/inference.py --checkpoint outputs/checkpoints/best.pth --text "A cylinder with a hole"

    # 多模态模式 (带图像)
    python src/inference.py --checkpoint outputs/checkpoints/best.pth --text "A cylinder" --image path/to/image.png
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.model import MechCADModel, MechCADConfig
from src.unified_vocab.converter import unified_tokens_to_13d
from cadlib.macro import *


def load_model(checkpoint_path, llava_path="model_weights/llava-hf/llava-1.5-7b-hf",
               num_views=2, n_latents=64):
    """加载训练好的模型"""
    print(f"加载模型配置...")
    model_cfg = MechCADConfig()

    print(f"初始化 MechCADModel...")
    model = MechCADModel(
        model_cfg,
        llava_model_name=llava_path,
        num_views=num_views,
        n_latents=n_latents
    )

    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 加载解码器权重
    model.llm2cad_decoder.load_state_dict(checkpoint['decoder_state_dict'])

    # 如果有多视图融合权重，也加载
    if 'fusion_state_dict' in checkpoint:
        model.multiview_fusion.load_state_dict(checkpoint['fusion_state_dict'])

    # 移到正确设备
    device = next(model.llava_model.parameters()).device
    model.llm2cad_decoder.to(device)
    model.multiview_fusion.to(device)

    model.eval()
    print("模型加载完成!")

    return model


def generate_cad(model, text, image_path=None, text_only=True):
    """
    生成 CAD 序列

    Args:
        model: MechCADModel
        text: 文本描述
        image_path: 图像路径 (可选)
        text_only: 是否仅使用文本模态

    Returns:
        cad_vec: CAD 向量序列 [S, 13]
        outputs: 模型原始输出
    """
    device = next(model.llava_model.parameters()).device

    # 准备输入
    if text_only or image_path is None:
        # 纯文本模式
        batch = {
            'text_caption': [text],
            'images': torch.zeros(1, 2, 3, 224, 224)  # 占位
        }
    else:
        # 多模态模式
        from torchvision.transforms import Compose, Resize, ToTensor, Normalize
        transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
        # 复制到所需视图数
        image_tensor = image_tensor.expand(-1, model.num_views, -1, -1, -1)

        batch = {
            'text_caption': [text],
            'images': image_tensor
        }

    # 前向传播
    with torch.no_grad():
        outputs = model(batch, text_only=text_only)

    # 转换为 CAD 向量
    cmd_logits = outputs['command_logits']
    args_logits = outputs['unified_args_logits']

    pred_commands = cmd_logits.argmax(dim=-1).squeeze(0).cpu().numpy()  # [S]
    pred_args_tokens = args_logits.argmax(dim=-1).squeeze(0).cpu().numpy()  # [S, n_args]

    # 转换为 13D CAD 向量
    cad_vec = unified_tokens_to_13d(pred_commands, pred_args_tokens)

    return cad_vec, outputs


def decode_cad_sequence(cad_vec):
    """解析 CAD 向量序列为可读格式"""
    commands = []

    for i, vec in enumerate(cad_vec):
        cmd_idx = int(vec[0])

        if cmd_idx == EOS_IDX:
            commands.append(f"[{i}] EOS")
            break
        elif cmd_idx == SOL_IDX:
            commands.append(f"[{i}] SOL (Start of Loop)")
        elif cmd_idx == EXT_IDX:
            angle_token = int(vec[6])
            pos_token = int(vec[7])
            ext_params = vec[8:13]
            commands.append(f"[{i}] EXTRUDE: angle_token={angle_token}, pos_token={pos_token}, params={ext_params.tolist()}")
        elif cmd_idx == LINE_IDX:
            params = vec[1:6]
            commands.append(f"[{i}] LINE: params={params.tolist()}")
        elif cmd_idx == ARC_IDX:
            params = vec[1:6]
            commands.append(f"[{i}] ARC: params={params.tolist()}")
        elif cmd_idx == CIRCLE_IDX:
            params = vec[1:6]
            commands.append(f"[{i}] CIRCLE: params={params.tolist()}")
        else:
            commands.append(f"[{i}] UNKNOWN({cmd_idx}): {vec.tolist()}")

    return commands


def truncate_at_eos(cad_vec):
    """
    在第一个 EOS 之后截断序列，将后续命令都设为 EOS。

    Args:
        cad_vec: [S, 13] CAD 向量序列

    Returns:
        truncated_vec: 截断后的 CAD 向量
        valid_length: 有效命令数（包括第一个 EOS）
    """
    cad_vec = cad_vec.copy()
    eos_positions = np.where(cad_vec[:, 0] == EOS_IDX)[0]

    if len(eos_positions) > 0:
        first_eos = eos_positions[0]
        # 将 EOS 之后的所有命令设为 EOS
        cad_vec[first_eos + 1:, 0] = EOS_IDX
        cad_vec[first_eos + 1:, 1:] = -1  # 参数设为无效
        valid_length = first_eos + 1
    else:
        valid_length = len(cad_vec)

    return cad_vec, valid_length


def main():
    parser = argparse.ArgumentParser(description="MechCAD-MLLM 推理测试")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="检查点路径 (例如 outputs/checkpoints/best.pth)")
    parser.add_argument("--text", type=str, required=True,
                        help="CAD 模型的文本描述")
    parser.add_argument("--image", type=str, default=None,
                        help="图像路径 (可选，多模态模式)")
    parser.add_argument("--llava_path", type=str,
                        default="model_weights/llava-hf/llava-1.5-7b-hf",
                        help="LLaVA 模型路径")
    parser.add_argument("--output", type=str, default=None,
                        help="保存生成的 CAD 向量到文件 (可选)")
    parser.add_argument("--text_only", action="store_true",
                        help="强制使用纯文本模式")

    args = parser.parse_args()

    # 加载模型
    model = load_model(args.checkpoint, args.llava_path)

    # 确定模式
    text_only = args.text_only or args.image is None
    mode = "纯文本" if text_only else "多模态"

    print("\n" + "=" * 60)
    print(f"推理模式: {mode}")
    print(f"输入文本: {args.text}")
    if args.image:
        print(f"输入图像: {args.image}")
    print("=" * 60)

    # 生成 CAD 序列
    print("\n正在生成 CAD 序列...")
    cad_vec, outputs = generate_cad(model, args.text, args.image, text_only)

    # 在 EOS 处截断
    cad_vec, valid_length = truncate_at_eos(cad_vec)

    # 解析并显示结果
    print("\n" + "=" * 60)
    print("生成的 CAD 序列:")
    print("=" * 60)

    commands = decode_cad_sequence(cad_vec)
    for cmd in commands:
        print(cmd)

    # 统计
    print(f"\n有效命令数: {valid_length}")

    # 保存结果
    if args.output:
        np.save(args.output, cad_vec)
        print(f"\nCAD 向量已保存到: {args.output}")

    # 返回原始向量供进一步处理
    print("\n" + "=" * 60)
    print("原始 CAD 向量 (前10行):")
    print("=" * 60)
    print(cad_vec[:10])

    return cad_vec


if __name__ == "__main__":
    main()
