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
import subprocess
import numpy as np
from datetime import datetime
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
from src.utils.cad_export import export_from_cad13
from cadlib.macro import *


# ============== 全局变量 ==============
MODEL = None
MODEL_PATH = None
OUTPUTS_DIR = os.path.abspath(os.path.join(project_root, "outputs"))
UI_EXPORT_DIR = os.path.join(OUTPUTS_DIR, "ui_generated_models")


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
def generate_cad(text_input, image_input, use_image, export_stl, preview_mode):
    """生成 CAD 序列"""
    global MODEL

    if MODEL is None:
        return "❌ 请先加载模型", "", "", None, None, None

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

        # 导出模型文件和预览图
        status = f"✅ 生成成功 (有效命令数: {valid_length})"
        preview_path = None
        step_path = None
        stl_path = None

        try:
            out_dir = UI_EXPORT_DIR
            os.makedirs(out_dir, exist_ok=True)
            safe_text = "".join(c if c.isalnum() else "_" for c in text_input).strip("_")
            safe_text = safe_text[:40] if safe_text else "cad"
            stem = f"{safe_text}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            artifacts = export_from_cad13(
                cad_vec[:valid_length],
                output_dir=out_dir,
                stem=stem,
                export_step=True,
                export_stl=bool(export_stl),
                export_preview=True,
                preview_mode=preview_mode
            )
            preview_path = os.path.abspath(artifacts['preview_path']) if artifacts.get('preview_path') else None
            step_path = os.path.abspath(artifacts['step_path']) if artifacts.get('step_path') else None
            stl_path = os.path.abspath(artifacts['stl_path']) if artifacts.get('stl_path') else None
        except Exception as export_err:
            status += f"\n⚠️ 模型导出失败: {export_err}"

        return status, formatted_output, raw_output, preview_path, step_path, stl_path

    except Exception as e:
        import traceback
        return f"❌ 生成失败: {str(e)}\n{traceback.format_exc()}", "", "", None, None, None


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


def load_test_metrics_file(metrics_path):
    """
    读取训练后保存的测试指标 JSON。
    支持传入文件路径，或传入模型目录（自动读取 test_metrics.json）。
    """
    try:
        if not metrics_path:
            return {}, "❌ 请输入指标文件路径或模型目录"

        if os.path.isdir(metrics_path):
            metrics_path = os.path.join(metrics_path, "test_metrics.json")

        if not os.path.exists(metrics_path):
            return {}, f"❌ 文件不存在: {metrics_path}"

        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        status = f"✅ 已加载测试指标: {metrics_path}"
        return payload, status
    except Exception as e:
        return {}, f"❌ 读取失败: {e}"


def _get_metric(payload, paths, default=None):
    """从多个候选路径里读取第一个存在的指标值。"""
    for path in paths:
        cur = payload
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok:
            return cur
    return default


def summarize_metrics(payload):
    """生成评估摘要、表格和可视化图。"""
    import matplotlib.pyplot as plt

    if not isinstance(payload, dict) or len(payload) == 0:
        return "❌ 指标内容为空", [], None, None

    checkpoint = payload.get("checkpoint", "未知")
    split_name = payload.get("split", "test")
    timestamp = payload.get("timestamp", "未知")
    validate_loss = _get_metric(payload, [["validate_loss"], ["test_loss"]], default=None)

    cmd_acc = _get_metric(payload, [
        ["validate_metrics", "cmd_accuracy"],
        ["test_metrics", "cmd_accuracy"],
        ["eval_metrics", "cmd_accuracy"]
    ], default=-1.0)
    args_mae = _get_metric(payload, [["eval_metrics", "args_mae"]], default=-1.0)
    chamfer = _get_metric(payload, [["eval_metrics", "chamfer_distance"]], default=-1.0)
    sege = _get_metric(payload, [["eval_metrics", "sege"]], default=-1.0)
    dangel = _get_metric(payload, [["eval_metrics", "dangel"]], default=-1.0)
    dangel_norm = _get_metric(payload, [["eval_metrics", "dangel_norm"]], default=-1.0)

    c_valid = _get_metric(payload, [["eval_metrics", "chamfer_valid_count"]], default=-1)
    c_failed = _get_metric(payload, [["eval_metrics", "chamfer_failed_count"]], default=-1)
    m_valid = _get_metric(payload, [["eval_metrics", "mesh_valid_count"]], default=-1)
    m_failed = _get_metric(payload, [["eval_metrics", "mesh_failed_count"]], default=-1)

    summary_lines = [
        f"检查点: {checkpoint}",
        f"评估划分: {split_name}",
        f"时间戳: {timestamp}",
    ]
    if validate_loss is not None:
        summary_lines.append(f"验证损失: {float(validate_loss):.6f}")
    if cmd_acc is not None and float(cmd_acc) >= 0:
        summary_lines.append(f"命令准确率: {float(cmd_acc) * 100:.2f}%")
    if args_mae is not None and float(args_mae) >= 0:
        summary_lines.append(f"参数 MAE: {float(args_mae):.6f}")
    if chamfer is not None and float(chamfer) >= 0:
        summary_lines.append(f"Chamfer Distance: {float(chamfer):.6f}")
    if sege is not None and float(sege) >= 0:
        summary_lines.append(f"SegE: {float(sege):.6f}")
    if dangel is not None and float(dangel) >= 0:
        summary_lines.append(f"DangEL: {float(dangel):.6f}")
    if dangel_norm is not None and float(dangel_norm) >= 0:
        summary_lines.append(f"DangEL(norm): {float(dangel_norm):.6f}")

    summary_text = "\n".join(summary_lines)

    rows = []
    if cmd_acc is not None and float(cmd_acc) >= 0:
        rows.append(["cmd_accuracy", float(cmd_acc)])
    if args_mae is not None and float(args_mae) >= 0:
        rows.append(["args_mae", float(args_mae)])
    if chamfer is not None and float(chamfer) >= 0:
        rows.append(["chamfer_distance", float(chamfer)])
    if sege is not None and float(sege) >= 0:
        rows.append(["sege", float(sege)])
    if dangel is not None and float(dangel) >= 0:
        rows.append(["dangel", float(dangel)])
    if dangel_norm is not None and float(dangel_norm) >= 0:
        rows.append(["dangel_norm", float(dangel_norm)])
    if c_valid >= 0:
        rows.append(["chamfer_valid_count", int(c_valid)])
    if c_failed >= 0:
        rows.append(["chamfer_failed_count", int(c_failed)])
    if m_valid >= 0:
        rows.append(["mesh_valid_count", int(m_valid)])
    if m_failed >= 0:
        rows.append(["mesh_failed_count", int(m_failed)])

    metric_labels = []
    metric_values = []
    if cmd_acc is not None and float(cmd_acc) >= 0:
        metric_labels.append("cmd_acc(%)")
        metric_values.append(float(cmd_acc) * 100.0)
    if args_mae is not None and float(args_mae) >= 0:
        metric_labels.append("args_mae")
        metric_values.append(float(args_mae))
    if chamfer is not None and float(chamfer) >= 0:
        metric_labels.append("chamfer")
        metric_values.append(float(chamfer))
    if sege is not None and float(sege) >= 0:
        metric_labels.append("sege")
        metric_values.append(float(sege))
    if dangel is not None and float(dangel) >= 0:
        metric_labels.append("dangel")
        metric_values.append(float(dangel))
    if dangel_norm is not None and float(dangel_norm) >= 0:
        metric_labels.append("dangel_norm")
        metric_values.append(float(dangel_norm))

    metric_fig = None
    if len(metric_labels) > 0:
        metric_fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(metric_labels, metric_values, color="#4c78a8")
        ax.set_title("评估指标总览")
        ax.set_ylabel("Value")
        ax.grid(True, axis="y", alpha=0.3)
        ax.bar_label(bars, fmt="%.4f", padding=2, fontsize=8)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()

    count_fig = None
    count_labels = []
    count_values = []
    if c_valid >= 0:
        count_labels.append("chamfer_valid")
        count_values.append(int(c_valid))
    if c_failed >= 0:
        count_labels.append("chamfer_failed")
        count_values.append(int(c_failed))
    if m_valid >= 0:
        count_labels.append("mesh_valid")
        count_values.append(int(m_valid))
    if m_failed >= 0:
        count_labels.append("mesh_failed")
        count_values.append(int(m_failed))

    if len(count_labels) > 0:
        count_fig, ax2 = plt.subplots(figsize=(8, 3.5))
        bars2 = ax2.bar(count_labels, count_values, color="#72b7b2")
        ax2.set_title("几何评估样本统计")
        ax2.set_ylabel("Count")
        ax2.grid(True, axis="y", alpha=0.3)
        ax2.bar_label(bars2, padding=2, fontsize=8)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()

    return summary_text, rows, metric_fig, count_fig


def load_metrics_and_visualize(metrics_path):
    """读取指标并返回可视化结果。"""
    payload, status = load_test_metrics_file(metrics_path)
    if not payload:
        return {}, status, "❌ 无法生成可视化", [], None, None

    summary_text, rows, metric_fig, count_fig = summarize_metrics(payload)
    return payload, status, summary_text, rows, metric_fig, count_fig


def run_evaluation_and_visualize(
    checkpoint_path,
    text_only,
    split_name,
    batch_size,
    num_selected_views,
    n_latents,
    category_start,
    category_end,
    sample_limit,
    full_eval_max_samples,
    deterministic_views,
    skip_full_eval,
    metrics_output_path,
):
    """从前端触发评估脚本，然后加载并可视化评估结果。"""
    try:
        if not checkpoint_path:
            return "", {}, "❌ 请输入 checkpoint 路径", "❌ 无法生成可视化", [], None, None, ""

        ckpt_abs = os.path.abspath(checkpoint_path)
        if not os.path.exists(ckpt_abs):
            return "", {}, f"❌ checkpoint 不存在: {ckpt_abs}", "❌ 无法生成可视化", [], None, None, ""

        if metrics_output_path and metrics_output_path.strip():
            metrics_abs = os.path.abspath(metrics_output_path.strip())
        else:
            metrics_abs = os.path.join(os.path.dirname(ckpt_abs), f"{split_name}_metrics.json" if split_name != "test" else "test_metrics.json")

        log_dir = os.path.join(
            OUTPUTS_DIR,
            "eval_logs",
            f"ui_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(os.path.dirname(metrics_abs), exist_ok=True)

        eval_script = os.path.join(project_root, "src", "evaluate_checkpoint.py")
        cmd = [
            sys.executable,
            eval_script,
            "--checkpoint", ckpt_abs,
            "--split", str(split_name),
            "--batch_size", str(int(batch_size)),
            "--num_selected_views", str(int(num_selected_views)),
            "--n_latents", str(int(n_latents)),
            "--category_start", str(int(category_start) if category_start is not None else 0),
            "--full_eval_max_samples", str(int(full_eval_max_samples)),
            "--metrics_output", metrics_abs,
            "--log_dir", log_dir,
        ]
        if category_end is not None and str(category_end).strip() != "":
            cmd.extend(["--category_end", str(int(float(category_end)))])
        if sample_limit is not None and str(sample_limit).strip() != "":
            cmd.extend(["--sample_limit", str(int(float(sample_limit)))])
        if text_only:
            cmd.append("--text_only")
        if deterministic_views:
            cmd.append("--deterministic_views")
        if skip_full_eval:
            cmd.append("--skip_full_eval")

        proc = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        run_log = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")

        if proc.returncode != 0:
            status = f"❌ 评估执行失败 (exit={proc.returncode})"
            return metrics_abs, {}, status, "❌ 无法生成可视化", [], None, None, run_log

        payload, status = load_test_metrics_file(metrics_abs)
        if not payload:
            return metrics_abs, {}, f"⚠️ 评估完成但读取指标失败: {status}", "❌ 无法生成可视化", [], None, None, run_log

        summary_text, rows, metric_fig, count_fig = summarize_metrics(payload)
        status = f"✅ 评估完成并已加载: {metrics_abs}"
        return metrics_abs, payload, status, summary_text, rows, metric_fig, count_fig, run_log
    except Exception as e:
        return "", {}, f"❌ 评估异常: {e}", "❌ 无法生成可视化", [], None, None, ""


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
                        export_stl = gr.Checkbox(label="导出 STL 文件", value=False)
                        preview_mode = gr.Radio(
                            choices=[("点云预览（默认，稳定）", "pointcloud"), ("STEP渲染预览（OCC）", "occ_step")],
                            value="pointcloud",
                            label="预览转换方式"
                        )

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
                        preview_image = gr.Image(
                            label="模型预览图",
                            type="filepath",
                            interactive=False
                        )
                        step_file = gr.File(label="STEP 文件", interactive=False)
                        stl_file = gr.File(label="STL 文件", interactive=False)

                # 事件绑定
                load_btn.click(load_model, inputs=[ckpt_path], outputs=[load_status])
                use_image.change(lambda x: gr.update(visible=x), inputs=[use_image], outputs=[image_input])
                generate_btn.click(
                    generate_cad,
                    inputs=[text_input, image_input, use_image, export_stl, preview_mode],
                    outputs=[gen_status, cad_output, raw_output, preview_image, step_file, stl_file]
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

                with gr.Row():
                    with gr.Column(scale=1):
                        metrics_path = gr.Textbox(
                            label="测试指标文件/目录",
                            placeholder="outputs/checkpoints/test_metrics.json 或 outputs/checkpoints",
                            value="outputs/checkpoints/test_metrics.json"
                        )
                        load_metrics_btn = gr.Button("读取测试指标", variant="primary")
                        metrics_status = gr.Textbox(label="指标读取状态", interactive=False)
                    with gr.Column(scale=1):
                        metrics_json = gr.JSON(label="测试集评估指标")

                refresh_btn.click(plot_training_curves, inputs=[log_dir], outputs=[train_plot, log_status])
                list_btn.click(list_checkpoints, inputs=[model_dir], outputs=[ckpt_list])
                load_metrics_btn.click(load_test_metrics_file, inputs=[metrics_path], outputs=[metrics_json, metrics_status])

            # ===== 指标评估 =====
            with gr.TabItem("📈 指标评估"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 方式1：读取已有指标文件")
                        eval_metrics_path = gr.Textbox(
                            label="指标文件/目录",
                            placeholder="outputs/stage2/test_metrics.json 或 outputs/stage2",
                            value="outputs/checkpoints/test_metrics.json"
                        )
                        eval_load_btn = gr.Button("加载并可视化", variant="primary")
                        eval_status = gr.Textbox(label="状态", interactive=False)
                        eval_summary = gr.Textbox(label="评估摘要", lines=9, interactive=False)
                    with gr.Column(scale=1):
                        eval_json = gr.JSON(label="原始指标 JSON")
                        eval_table = gr.Dataframe(
                            label="关键指标表",
                            headers=["Metric", "Value"],
                            datatype=["str", "number"],
                            interactive=False
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 方式2：一键运行评估并可视化")
                        run_ckpt_path = gr.Textbox(
                            label="Checkpoint 路径",
                            placeholder="outputs/stage2/best.pth",
                            value="outputs/stage2/best.pth"
                        )
                        with gr.Row():
                            run_text_only = gr.Checkbox(label="text_only（阶段1）", value=False)
                            run_deterministic_views = gr.Checkbox(label="固定视图采样", value=True)
                            run_skip_full_eval = gr.Checkbox(label="仅validate（跳过几何指标）", value=False)
                        with gr.Row():
                            run_split = gr.Dropdown(
                                choices=["test", "val", "train", "all"],
                                value="test",
                                label="评估数据划分"
                            )
                            run_batch_size = gr.Number(label="batch_size", value=4, precision=0)
                            run_eval_max_samples = gr.Number(label="full_eval_max_samples", value=500, precision=0)
                        with gr.Row():
                            run_num_views = gr.Number(label="num_selected_views", value=2, precision=0)
                            run_n_latents = gr.Number(label="n_latents", value=64, precision=0)
                        with gr.Row():
                            run_category_start = gr.Number(label="category_start", value=0, precision=0)
                            run_category_end = gr.Textbox(
                                label="category_end（可选）",
                                placeholder="例如 9，留空表示全部",
                                value=""
                            )
                            run_sample_limit = gr.Textbox(
                                label="sample_limit（可选）",
                                placeholder="例如 200，留空表示全部",
                                value=""
                            )
                        run_metrics_output = gr.Textbox(
                            label="输出指标路径（可选）",
                            placeholder="留空则默认写到 checkpoint 同目录",
                            value=""
                        )
                        run_eval_btn = gr.Button("运行评估并加载结果", variant="primary")
                    with gr.Column(scale=1):
                        run_log = gr.Textbox(
                            label="评估执行日志",
                            lines=14,
                            interactive=False
                        )

                with gr.Row():
                    eval_metric_plot = gr.Plot(label="指标可视化")
                    eval_count_plot = gr.Plot(label="有效/失败样本统计")

                eval_load_btn.click(
                    load_metrics_and_visualize,
                    inputs=[eval_metrics_path],
                    outputs=[eval_json, eval_status, eval_summary, eval_table, eval_metric_plot, eval_count_plot]
                )
                run_eval_btn.click(
                    run_evaluation_and_visualize,
                    inputs=[
                        run_ckpt_path,
                        run_text_only,
                        run_split,
                        run_batch_size,
                        run_num_views,
                        run_n_latents,
                        run_category_start,
                        run_category_end,
                        run_sample_limit,
                        run_eval_max_samples,
                        run_deterministic_views,
                        run_skip_full_eval,
                        run_metrics_output,
                    ],
                    outputs=[
                        eval_metrics_path,
                        eval_json,
                        eval_status,
                        eval_summary,
                        eval_table,
                        eval_metric_plot,
                        eval_count_plot,
                        run_log,
                    ]
                )

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
        share=args.share,
        allowed_paths=[OUTPUTS_DIR]
    )


if __name__ == "__main__":
    main()
