import os
import tempfile
from typing import Dict, Optional

import numpy as np

from cadlib.macro import ARC_IDX, CIRCLE_IDX, EOS_IDX, EXT_IDX, LINE_IDX, PAD_VAL, SOL_IDX


_OCC_DISPLAY = None
_OCC_VDISPLAY = None


def _bin_to_uint8(bin_idx: int, n_bins: int) -> int:
    """使用区间中心将离散的 bin 索引映射到量化的 [0, 255] 值。"""
    bin_idx = int(np.clip(bin_idx, 0, n_bins - 1))
    return int(round(((bin_idx + 0.5) / n_bins) * 255))


def cad13_to_cad17_numerical(cad_vec_13d: np.ndarray) -> np.ndarray:
    """
    将 [S, 13] CAD 向量转换回 cadlib 期望的 [S, 17] 量化向量。

    13D 格式：
      [CMD, x, y, alpha, f, r, angle_tok, pos_tok, p12, p13, p14, p15, p16]

    cadlib 使用的 17D 格式：
      [CMD, x, y, alpha, f, r, theta, phi, gamma, px, py, pz, p12, p13, p14, p15, p16]
    """
    vec13 = np.asarray(cad_vec_13d, dtype=np.int32)
    if vec13.ndim != 2 or vec13.shape[1] != 13:
        raise ValueError(f"期望 cad_vec_13d 形状为 [S, 13]，得到 {vec13.shape}")

    s = vec13.shape[0]
    vec17 = np.full((s, 17), PAD_VAL, dtype=np.int32)
    vec17[:, 0] = vec13[:, 0]

    for i in range(s):
        cmd = int(vec13[i, 0])

        if cmd in (LINE_IDX, ARC_IDX, CIRCLE_IDX):
            vec17[i, 1:6] = np.clip(vec13[i, 1:6], 0, 255)
            continue

        if cmd == EXT_IDX:
            angle_idx = int(max(vec13[i, 6], 0))
            i_theta = angle_idx // 81
            rem = angle_idx % 81
            i_phi = rem // 9
            i_gamma = rem % 9

            # 9-bin 方向三元组 -> 量化到 [0,255]
            vec17[i, 6] = _bin_to_uint8(i_theta, 9)
            vec17[i, 7] = _bin_to_uint8(i_phi, 9)
            vec17[i, 8] = _bin_to_uint8(i_gamma, 9)

            pos_idx = int(max(vec13[i, 7], 0))
            i_z = pos_idx // 1296
            rem = pos_idx % 1296
            i_y = rem // 36
            i_x = rem % 36

            # 36-bin 位置三元组 -> 量化到 [0,255]
            vec17[i, 9] = _bin_to_uint8(i_x, 36)
            vec17[i, 10] = _bin_to_uint8(i_y, 36)
            vec17[i, 11] = _bin_to_uint8(i_z, 36)

            # 剩余的 ext 参数保持为量化值
            vec17[i, 12:17] = np.clip(vec13[i, 8:13], 0, 255)

    return vec17


def _save_pointcloud_preview(points: np.ndarray, preview_path: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    # 稀疏采样以加快渲染速度，同时保持形状感知能力。
    if len(points) > 5000:
        idx = np.random.choice(len(points), 5000, replace=False)
        points = points[idx]

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.5, c=points[:, 2], cmap="viridis")
    ax.set_axis_off()

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0 + 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))

    fig.tight_layout(pad=0.0)
    fig.savefig(preview_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return preview_path


def _get_occ_display():
    """
    延迟初始化 pythonOCC 离屏显示。
    遵循 `occ_render.py` 中使用的相同策略。
    """
    global _OCC_DISPLAY, _OCC_VDISPLAY
    if _OCC_DISPLAY is not None:
        return _OCC_DISPLAY

    os.environ.setdefault("PYTHONOCC_OFFSCREEN_RENDERER", "1")

    if os.environ.get("DISPLAY") is None:
        try:
            from pyvirtualdisplay import Display

            _OCC_VDISPLAY = Display(visible=False, size=(1024, 768))
            _OCC_VDISPLAY.start()
            os.environ["DISPLAY"] = _OCC_VDISPLAY.new_display_var
        except Exception:
            # 尽力而为：继续执行，如果需要，让 OCC 初始化报告具体错误。
            pass

    try:
        from OCC.Display.SimpleGui import init_display

        display, _, _, _ = init_display()
        try:
            display.View.TriedronErase()  # type: ignore[attr-defined]
        except Exception:
            pass
        _OCC_DISPLAY = display
        return display
    except Exception as e:
        raise RuntimeError(f"OCC 显示初始化失败: {e}") from e


def _save_occ_preview_from_step(step_path: str, preview_path: str) -> str:
    """
    通过 pythonOCC 离屏显示将 STEP 文件渲染为 PNG。
    """
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB

    display = _get_occ_display()

    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != 1:
        raise RuntimeError(f"读取 STEP 文件失败: {step_path}, status={status}")
    reader.TransferRoot()
    shape = reader.Shape()

    try:
        color = Quantity_Color(0.1, 0.1, 0.1, Quantity_TOC_RGB)  # type: ignore[arg-type]
        display.DisplayColoredShape(shape, color, update=True)  # type: ignore[attr-defined]
    except Exception:
        display.DisplayShape(shape, update=True)  # type: ignore[attr-defined]

    try:
        try:
            display.View.SetProj(1.0, -1.0, 1.0)  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            display.FitAll()  # type: ignore[attr-defined]
        except Exception:
            pass
        display.View.Dump(preview_path)  # type: ignore[attr-defined]
    finally:
        try:
            display.EraseAll()  # type: ignore[attr-defined]
            display.View.Reset()  # type: ignore[attr-defined]
        except Exception:
            pass

    return preview_path


def export_from_cad13(
    cad_vec_13d: np.ndarray,
    output_dir: str,
    stem: str,
    export_step: bool = True,
    export_stl: bool = False,
    export_preview: bool = True,
    preview_points: int = 4096,
    preview_mode: str = "pointcloud",
) -> Dict[str, str]:
    """
    从 13D 向量构建 CAD 实体并导出各种文件。

    返回一个包含可选键的字典：
      - step_path
      - stl_path
      - preview_path
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        from OCC.Extend.DataExchange import write_step_file, write_stl_file
        from cadlib.visualize import CADsolid2pc, vec2CADsolid
    except Exception as e:  # pragma: no cover - 依赖本地 OCC 运行时
        raise RuntimeError(
            "CAD 导出需要 pythonocc-core 和 cadlib 依赖项。"
            f"导入失败: {e}"
        ) from e

    vec17 = cad13_to_cad17_numerical(cad_vec_13d)
    shape = vec2CADsolid(vec17, is_numerical=True, n=256)

    artifacts: Dict[str, str] = {}
    if export_step:
        step_path = os.path.join(output_dir, f"{stem}.step")
        write_step_file(shape, step_path)
        artifacts["step_path"] = step_path

    if export_stl:
        stl_path = os.path.join(output_dir, f"{stem}.stl")
        write_stl_file(shape, stl_path)
        artifacts["stl_path"] = stl_path

    if export_preview:
        preview_path = os.path.join(output_dir, f"{stem}.png")
        if preview_mode == "pointcloud":
            points = CADsolid2pc(shape, n_points=preview_points, name=stem)
            _save_pointcloud_preview(points, preview_path)
            artifacts["preview_path"] = preview_path
        elif preview_mode == "occ_step":
            # 如果可用，重用生成的 STEP；否则为渲染编写临时 STEP。
            temp_step_path = None
            render_step_path = artifacts.get("step_path")
            if render_step_path is None:
                with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
                    temp_step_path = f.name
                write_step_file(shape, temp_step_path)
                render_step_path = temp_step_path
            try:
                _save_occ_preview_from_step(render_step_path, preview_path)
                artifacts["preview_path"] = preview_path
            finally:
                if temp_step_path and os.path.exists(temp_step_path):
                    os.remove(temp_step_path)
        else:
            raise ValueError(f"未知的 preview_mode: {preview_mode}")

    return artifacts
