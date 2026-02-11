#!/usr/bin/env python3
"""
MechCAD-MLLM Web UI

基于 Gradio 的前端界面，支持：
1. 训练监控：实时查看训练曲线
2. 模型预测：加载模型生成 CAD 序列
3. 结果可视化：显示生成的 CAD 命令

用法:
    python src/app.py --checkpoint outputs/stage1/best.pth
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
from pathlib import Path

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    import gradio as gr
except ImportError:
    print("请先安装 Gradio: pip install gradio")
    sys.exit(1)

import torch
from PIL import Image

from src.model.model import MechCADModel, MechCADConfig
from src.unified_vocab.converter import unified_tokens_to_13d
from cadlib.macro import *


# ============== 全局变量 ==============
MODEL = None
MODEL_PATH = None


# ============== 模型加载 ==============
def load_model(checkpoint_path, llava_path="model_weights/llava-hf/llava-1.5-7b-hf"):
    """加载模型"""
    global MODEL, MODEL_PATH

    if MODEL is not None and MODEL_PATH == checkpoint_path:
        return "✅ 模型已加载"

    try:
        print(f"正在加载模型: {checkpoint_path}")
        model_cfg = MechCADConfig()

        model = MechCADModel(
            model_cfg,
            llava_model_name=llava_path,
            num_views=2,
            n_latents=64
        )

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.llm2cad_decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if 'fusion_state_dict' in checkpoint:
            model.multiview_fusion.load_state_dict(checkpoint['fusion_state_dict'])

        device = next(model.llava_model.parameters()).device
        model.llm2cad_decoder.to(device)
        model.multiview_fusion.to(device)
        model.eval()

        MODEL = model
        MODEL_PATH = checkpoint_path

        return f"✅ 模型加载成功: {checkpoint_path}"
    except Exception as e:
        return f"❌ 模型加载失败: {str(e)}"


# ============== CAD 生成 ==============
def generate_cad(text_input, image_input, use_image):
    """生成 CAD 序列"""
    global MODEL

    if MODEL is None:
        return "❌ 请先加载模型", "", None

    try:
        text_only = not use_image or image_input is None

        # 准备输入
        if text_only:
            batch = {
                'text_caption': [text_input],
                'images': torch.zeros(1, 2, 3, 224, 224)
            }
        else:
            from torchvision.transforms import Compose, Resize, ToTensor, Normalize
            transform = Compose([
                Resize((224, 224)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            if isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input)
            else:
                image = image_input

            image_tensor = transform(image.convert('RGB')).unsqueeze(0).unsqueeze(0)
            image_tensor = image_tensor.expand(-1, 2, -1, -1, -1)

            batch = {
                'text_caption': [text_input],
                'images': image_tensor
            }

        # 前向传播
        with torch.no_grad():
            outputs = MODEL(batch, text_only=text_only)

        # 转换为 CAD 向量
        cmd_logits = outputs['command_logits']
        args_logits = outputs['unified_args_logits']

        pred_commands = cmd_logits.argmax(dim=-1).squeeze(0).cpu().numpy()
        pred_args_tokens = args_logits.argmax(dim=-1).squeeze(0).cpu().numpy()

        cad_vec = unified_tokens_to_13d(pred_commands, pred_args_tokens)

        # 截断到 EOS
        cad_vec, valid_length = truncate_at_eos(cad_vec)

        # 格式化输出
        formatted_output = format_cad_sequence(cad_vec, valid_length)
        raw_output = format_raw_vector(cad_vec[:valid_length])

        return f"✅ 生成成功 (有效命令数: {valid_length})", formatted_output, raw_output

    except Exception as e:
        import traceback
        return f"❌ 生成失败: {str(e)}\n{traceback.format_exc()}", "", None


def truncate_at_eos(cad_vec):
    """在 EOS 处截断"""
    cad_vec = cad_vec.copy()
    eos_positions = np.where(cad_vec[:, 0] == EOS_IDX)[0]

    if len(eos_positions) > 0:
        first_eos = eos_positions[0]
        valid_length = first_eos + 1
    else:
        valid_length = len(cad_vec)

    return cad_vec, valid_length


def format_cad_sequence(cad_vec, valid_length):
    """格式化 CAD 序列为可读文本"""
    lines = []
    cmd_names = {
        LINE_IDX: "LINE",
        ARC_IDX: "ARC",
        CIRCLE_IDX: "CIRCLE",
        EOS_IDX: "EOS",
        SOL_IDX: "SOL",
        EXT_IDX: "EXTRUDE"
    }

    for i in range(valid_length):
        vec = cad_vec[i]
        cmd_idx = int(vec[0])
        cmd_name = cmd_names.get(cmd_idx, f"UNKNOWN({cmd_idx})")

        if cmd_idx == EOS_IDX:
            lines.append(f"[{i:2d}] {cmd_name}")
            break
        elif cmd_idx == SOL_IDX:
            lines.append(f"[{i:2d}] {cmd_name} (Start of Loop)")
        elif cmd_idx == EXT_IDX:
            angle = int(vec[6])
            pos = int(vec[7])
            params = vec[8:13].tolist()
            lines.append(f"[{i:2d}] {cmd_name}: angle={angle}, pos={pos}, params={params}")
        elif cmd_idx in [LINE_IDX, ARC_IDX, CIRCLE_IDX]:
            params = vec[1:6].tolist()
            lines.append(f"[{i:2d}] {cmd_name}: params={params}")
        else:
            lines.append(f"[{i:2d}] {cmd_name}")

    return "\n".join(lines)


def format_raw_vector(cad_vec):
    """格式化原始向量"""
    lines = ["[CMD, x, y, alpha, f, r, angle, pos, e1, e2, b, u, s]"]
    lines.append("-" * 60)
    for i, vec in enumerate(cad_vec):
        lines.append(f"[{i:2d}] {vec.tolist()}")
    return "\n".join(lines)


# ============== 训练监控 ==============
def load_training_logs(log_dir):
    """加载训练日志"""
    try:
        from tensorboard.backend.event_processing import event_accumulator

        train_dir = os.path.join(log_dir, 'train.events')
        val_dir = os.path.join(log_dir, 'val.events')

        data = {'train': {}, 'val': {}}

        # 加载事件文件 - 直接传入目录路径，EventAccumulator 会自动合并所有事件文件
        for mode, path in [('train', train_dir), ('val', val_dir)]:
            if os.path.isdir(path):
                ea = event_accumulator.EventAccumulator(path)
                ea.Reload()

                for tag in ea.Tags()['scalars']:
                    events = ea.Scalars(tag)
                    # 按 step 排序，确保曲线连续
                    sorted_events = sorted(events, key=lambda e: e.step)
                    data[mode][tag] = {
                        'steps': [e.step for e in sorted_events],
                        'values': [e.value for e in sorted_events]
                    }

        return data
    except Exception as e:
        return {'error': str(e)}


def plot_training_curves(log_dir):
    """绘制训练曲线"""
    import matplotlib.pyplot as plt

    data = load_training_logs(log_dir)

    if 'error' in data:
        return None, f"❌ 无法加载日志: {data['error']}"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss 曲线
    ax = axes[0, 0]
    if 'loss' in data['train']:
        ax.plot(data['train']['loss']['steps'], data['train']['loss']['values'], label='Train Loss')
    if 'val_loss' in data['val']:
        ax.plot(data['val']['val_loss']['steps'], data['val']['val_loss']['values'], label='Val Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True)

    # Command Loss
    ax = axes[0, 1]
    if 'loss_cmd' in data['train']:
        ax.plot(data['train']['loss_cmd']['steps'], data['train']['loss_cmd']['values'], label='Cmd Loss')
    if 'loss_args' in data['train']:
        ax.plot(data['train']['loss_args']['steps'], data['train']['loss_args']['values'], label='Args Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Component Losses')
    ax.legend()
    ax.grid(True)

    # Learning Rate
    ax = axes[1, 0]
    if 'learning_rate' in data['train']:
        ax.plot(data['train']['learning_rate']['steps'], data['train']['learning_rate']['values'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True)

    # Accuracy
    ax = axes[1, 1]
    if 'cmd_accuracy' in data['val']:
        ax.plot(data['val']['cmd_accuracy']['steps'], data['val']['cmd_accuracy']['values'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Command Accuracy')
    ax.grid(True)

    plt.tight_layout()

    return fig, "✅ 训练曲线已加载"


def list_checkpoints(model_dir):
    """列出可用的检查点"""
    if not os.path.exists(model_dir):
        return "目录不存在"

    ckpts = []
    for f in os.listdir(model_dir):
        if f.endswith('.pth'):
            path = os.path.join(model_dir, f)
            size = os.path.getsize(path) / (1024 * 1024)  # MB
            mtime = os.path.getmtime(path)
            ckpts.append(f"{f} ({size:.1f} MB)")

    return "\n".join(ckpts) if ckpts else "无检查点"


# ============== Gradio 界面 ==============
def create_ui():
    """创建 Gradio 界面"""

    with gr.Blocks(title="MechCAD-MLLM", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🔧 MechCAD-MLLM 控制台")
        gr.Markdown("多模态大语言模型驱动的 CAD 生成系统")

        with gr.Tabs():
            # ===== 模型预测 =====
            with gr.TabItem("🎯 模型预测"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 模型加载")
                        ckpt_path = gr.Textbox(
                            label="检查点路径",
                            placeholder="outputs/checkpoints/best.pth",
                            value="outputs/stage1/best.pth"
                        )
                        load_btn = gr.Button("加载模型", variant="primary")
                        load_status = gr.Textbox(label="状态", interactive=False)

                        gr.Markdown("### 输入")
                        text_input = gr.Textbox(
                            label="文本描述",
                            placeholder="A cylinder with a central hole",
                            lines=3
                        )
                        use_image = gr.Checkbox(label="使用图像（多模态模式）", value=False)
                        image_input = gr.Image(label="输入图像", visible=False)

                        generate_btn = gr.Button("生成 CAD 序列", variant="primary")

                    with gr.Column(scale=1):
                        gr.Markdown("### 输出")
                        gen_status = gr.Textbox(label="生成状态", interactive=False)
                        cad_output = gr.Textbox(
                            label="CAD 命令序列",
                            lines=15,
                            interactive=False
                        )
                        raw_output = gr.Textbox(
                            label="原始向量",
                            lines=10,
                            interactive=False
                        )

                # 事件绑定
                load_btn.click(load_model, inputs=[ckpt_path], outputs=[load_status])
                use_image.change(lambda x: gr.update(visible=x), inputs=[use_image], outputs=[image_input])
                generate_btn.click(
                    generate_cad,
                    inputs=[text_input, image_input, use_image],
                    outputs=[gen_status, cad_output, raw_output]
                )

            # ===== 训练监控 =====
            with gr.TabItem("📊 训练监控"):
                with gr.Row():
                    with gr.Column(scale=1):
                        log_dir = gr.Textbox(
                            label="日志目录",
                            placeholder="outputs/logs",
                            value="outputs/stage1_log"
                        )
                        refresh_btn = gr.Button("刷新训练曲线", variant="primary")
                        log_status = gr.Textbox(label="状态", interactive=False)

                    with gr.Column(scale=1):
                        model_dir = gr.Textbox(
                            label="模型目录",
                            placeholder="outputs/checkpoints",
                            value="outputs/checkpoints"
                        )
                        list_btn = gr.Button("列出检查点")
                        ckpt_list = gr.Textbox(label="可用检查点", lines=5, interactive=False)

                with gr.Row():
                    train_plot = gr.Plot(label="训练曲线")

                refresh_btn.click(plot_training_curves, inputs=[log_dir], outputs=[train_plot, log_status])
                list_btn.click(list_checkpoints, inputs=[model_dir], outputs=[ckpt_list])

            # ===== 示例 =====
            with gr.TabItem("📝 示例"):
                gr.Markdown("""
                ### 示例文本描述

                尝试以下描述来生成 CAD 模型：

                
                - `Generate a CAD model with a square base and a central circular hole`
                - `Generate a CAD model with a rectangular prism shape, featuring a uniform gray color and smooth surfaces`
                - `Generate a CAD model with a cylindrical shape featuring a hollow center and a split along one side, resembling a segmented ring`


                ### 使用提示

                1. **纯文本模式**: 适合第一阶段训练的模型
                2. **多模态模式**: 需要第二阶段训练的模型，可上传参考图像
                3. **检查点选择**: `best.pth` 通常是验证损失最低的模型
                """)

            # ===== 关于 =====
            with gr.TabItem("ℹ️ 关于"):
                gr.Markdown("""
                ### MechCAD-MLLM

                **多模态大语言模型驱动的 CAD 生成系统**

                #### 功能特点
                - 🔤 文本到 CAD：根据自然语言描述生成 CAD 命令序列
                - 🖼️ 多模态融合：结合图像和文本进行 CAD 生成
                - 🔄 渐进式训练：支持两阶段训练策略
                - 📊 训练监控：实时查看训练曲线

                #### 技术架构
                - 编码器：LLaVA-1.5-7B（冻结）
                - 多视图融合：PerceiverFusion
                - 解码器：Transformer Decoder
                - 词表：统一大词表（47651 tokens）

                #### 命令格式
                - `LINE`: 直线命令
                - `ARC`: 圆弧命令
                - `CIRCLE`: 圆命令
                - `EXTRUDE`: 拉伸命令
                - `SOL`: 循环开始
                - `EOS`: 序列结束
                """)

    return app


# ============== 主函数 ==============
def main():
    parser = argparse.ArgumentParser(description="MechCAD-MLLM Web UI")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="预加载的检查点路径")
    parser.add_argument("--port", type=int, default=7860,
                        help="服务端口")
    parser.add_argument("--share", action="store_true",
                        help="创建公共链接")

    args = parser.parse_args()

    # 预加载模型
    if args.checkpoint:
        print(load_model(args.checkpoint))

    # 创建并启动应用
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
