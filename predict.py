#!/usr/bin/env python3
"""
MechCAD-MLLM 预测脚本

用法:
    # 单个样本预测
    python predict.py --checkpoint outputs/checkpoints/best.pth --text "A cylinder with a hole"

    # 使用图像输入
    python predict.py --checkpoint outputs/checkpoints/best.pth --image path/to/image.png

    # 批量预测
    python predict.py --checkpoint outputs/checkpoints/best.pth --input_file inputs.json --output_dir outputs/predictions
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.model import MechCADModel, MechCADConfig
from cadlib.macro import *


class MechCADPredictor:
    """MechCAD 模型预测器"""

    def __init__(self, checkpoint_path, llava_model_name="model_weights/llava-hf/llava-1.5-7b-hf"):
        """
        初始化预测器。

        Args:
            checkpoint_path: 训练好的模型检查点路径
            llava_model_name: LLaVA模型路径
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 加载模型
        print("正在加载模型...")
        self.model = self._load_model(checkpoint_path, llava_model_name)
        self.model.eval()

        # 图像预处理
        self.image_transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("模型加载完成!")

    def _load_model(self, checkpoint_path, llava_model_name):
        """加载模型和权重"""
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 获取模型配置
        if 'model_cfg' in checkpoint:
            model_cfg = checkpoint['model_cfg']
        else:
            model_cfg = MechCADConfig()

        # 初始化模型
        model = MechCADModel(model_cfg, llava_model_name=llava_model_name)

        # 加载解码器权重
        if 'decoder_state_dict' in checkpoint:
            model.llm2cad_decoder.load_state_dict(checkpoint['decoder_state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        return model

    def predict(self, text=None, image_path=None, image_tensor=None):
        """
        执行预测。

        Args:
            text: 文本描述
            image_path: 图像文件路径
            image_tensor: 预处理后的图像张量 [1, C, H, W] 或 [1, N_views, C, H, W]

        Returns:
            cad_vec: CAD向量序列 [S, 13]
            cad_commands: 解析后的命令列表
        """
        # 准备输入
        batch = self._prepare_input(text, image_path, image_tensor)

        # 推理
        with torch.no_grad():
            outputs = self.model(batch)

        # 转换为CAD向量
        cad_vec = self._logits2vec(outputs)

        # 解析命令
        cad_commands = self._parse_cad_vec(cad_vec[0])  # 取第一个batch

        return cad_vec[0], cad_commands

    def _prepare_input(self, text, image_path, image_tensor):
        """准备模型输入"""
        batch = {}

        # 处理文本
        if text:
            batch['text_caption'] = [text]
        else:
            batch['text_caption'] = ["A 3D CAD model"]

        # 处理图像
        if image_tensor is not None:
            batch['images'] = image_tensor.to(self.device)
        elif image_path:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.image_transform(image).unsqueeze(0)  # [1, C, H, W]
            # 扩展为8视图格式 (复制同一张图)
            image_tensor = image_tensor.unsqueeze(1).expand(-1, 8, -1, -1, -1)  # [1, 8, C, H, W]
            batch['images'] = image_tensor.to(self.device)
        else:
            # 无图像输入，使用空白图像
            batch['images'] = torch.zeros(1, 8, 3, 224, 224).to(self.device)

        # 添加占位CAD序列 (用于Teacher Forcing，推理时可能不需要)
        batch['cad_sequence'] = torch.zeros(1, MAX_TOTAL_LEN, 13, dtype=torch.long).to(self.device)

        return batch

    def _logits2vec(self, outputs):
        """将模型输出转换为CAD向量"""
        cmd_logits = outputs['command_logits']
        args_logits = outputs['args_logits']
        angle_logits = outputs['angle_logits']
        pos_logits = outputs['pos_logits']

        # 命令预测
        pred_commands = cmd_logits.argmax(dim=-1)  # [B, S]

        # 参数预测 (减1恢复原始范围)
        pred_args = args_logits.argmax(dim=-1) - 1  # [B, S, 12]

        # 角度和位置Token预测
        pred_angle = angle_logits.argmax(dim=-1)  # [B, S]
        pred_pos = pos_logits.argmax(dim=-1)  # [B, S]

        # 对于Ext命令，用token预测覆盖对应参数位置
        ext_mask = (pred_commands == EXT_IDX)
        pred_args[:, :, 5] = torch.where(ext_mask, pred_angle, pred_args[:, :, 5])
        pred_args[:, :, 6] = torch.where(ext_mask, pred_pos, pred_args[:, :, 6])

        # 根据命令类型填充无效参数为-1
        cmd_args_mask = torch.tensor(CMD_ARGS_MASK, device=pred_commands.device)
        mask = cmd_args_mask[pred_commands.long()][:, :, :12]  # [B, S, 12]
        pred_args = torch.where(mask.bool(), pred_args, torch.tensor(-1, device=pred_args.device))

        # 组合为13维向量
        cad_vec = torch.cat([pred_commands.unsqueeze(-1), pred_args], dim=-1)

        return cad_vec.detach().cpu().numpy()

    def _parse_cad_vec(self, cad_vec):
        """解析CAD向量为可读格式"""
        commands = []
        cmd_names = {
            LINE_IDX: "Line",
            ARC_IDX: "Arc",
            CIRCLE_IDX: "Circle",
            EXT_IDX: "Extrude",
            SOL_IDX: "StartOfLoop",
            EOS_IDX: "EndOfSequence"
        }

        for i, vec in enumerate(cad_vec):
            cmd_idx = int(vec[0])
            if cmd_idx == EOS_IDX:
                commands.append({"index": i, "command": "EOS"})
                break

            cmd_name = cmd_names.get(cmd_idx, f"Unknown({cmd_idx})")
            params = vec[1:].tolist()

            commands.append({
                "index": i,
                "command": cmd_name,
                "parameters": params
            })

        return commands

    def save_prediction(self, cad_vec, output_path):
        """保存预测结果"""
        np.save(output_path, cad_vec)
        print(f"预测结果已保存到: {output_path}")

    def export_to_json(self, cad_commands, output_path):
        """导出CAD命令为JSON格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cad_commands, f, indent=2, ensure_ascii=False)
        print(f"CAD命令已导出到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MechCAD-MLLM 预测")

    # 模型参数
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型检查点路径")
    parser.add_argument("--llava_model_name", type=str,
                        default="model_weights/llava-hf/llava-1.5-7b-hf",
                        help="LLaVA模型路径")

    # 输入参数
    parser.add_argument("--text", type=str, default=None,
                        help="文本描述")
    parser.add_argument("--image", type=str, default=None,
                        help="输入图像路径")
    parser.add_argument("--input_file", type=str, default=None,
                        help="批量输入JSON文件")

    # 输出参数
    parser.add_argument("--output_dir", type=str, default="outputs/predictions",
                        help="输出目录")
    parser.add_argument("--save_npy", action="store_true",
                        help="保存为.npy格式")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化预测器
    predictor = MechCADPredictor(
        checkpoint_path=args.checkpoint,
        llava_model_name=args.llava_model_name
    )

    if args.input_file:
        # 批量预测模式
        with open(args.input_file, 'r', encoding='utf-8') as f:
            inputs = json.load(f)

        for i, item in enumerate(inputs):
            text = item.get('text')
            image = item.get('image')

            print(f"\n[{i+1}/{len(inputs)}] 预测中...")
            cad_vec, cad_commands = predictor.predict(text=text, image_path=image)

            # 保存结果
            output_name = item.get('id', f"sample_{i:04d}")
            predictor.export_to_json(cad_commands, os.path.join(args.output_dir, f"{output_name}.json"))

            if args.save_npy:
                predictor.save_prediction(cad_vec, os.path.join(args.output_dir, f"{output_name}.npy"))
    else:
        # 单样本预测模式
        if not args.text and not args.image:
            print("错误: 请提供 --text 或 --image 参数")
            return

        print("\n开始预测...")
        cad_vec, cad_commands = predictor.predict(text=args.text, image_path=args.image)

        # 显示结果
        print("\n" + "=" * 50)
        print("预测结果:")
        print("=" * 50)
        for cmd in cad_commands[:10]:  # 只显示前10个命令
            print(f"  [{cmd['index']:2d}] {cmd['command']}: {cmd.get('parameters', '')}")
        if len(cad_commands) > 10:
            print(f"  ... (共 {len(cad_commands)} 个命令)")

        # 保存结果
        predictor.export_to_json(cad_commands, os.path.join(args.output_dir, "prediction.json"))
        if args.save_npy:
            predictor.save_prediction(cad_vec, os.path.join(args.output_dir, "prediction.npy"))

    print("\n预测完成!")


if __name__ == "__main__":
    main()
