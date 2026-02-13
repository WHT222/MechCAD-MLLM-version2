import os
from typing import Dict, Optional

import numpy as np

from cadlib.macro import ARC_IDX, CIRCLE_IDX, EOS_IDX, EXT_IDX, LINE_IDX, PAD_VAL, SOL_IDX


def _bin_to_uint8(bin_idx: int, n_bins: int) -> int:
    """Map a discrete bin index to a quantized [0, 255] value using bin center."""
    bin_idx = int(np.clip(bin_idx, 0, n_bins - 1))
    return int(round(((bin_idx + 0.5) / n_bins) * 255))


def cad13_to_cad17_numerical(cad_vec_13d: np.ndarray) -> np.ndarray:
    """
    Convert [S, 13] CAD vectors back to [S, 17] quantized vectors expected by cadlib.

    13D format:
      [CMD, x, y, alpha, f, r, angle_tok, pos_tok, p12, p13, p14, p15, p16]

    17D format used by cadlib:
      [CMD, x, y, alpha, f, r, theta, phi, gamma, px, py, pz, p12, p13, p14, p15, p16]
    """
    vec13 = np.asarray(cad_vec_13d, dtype=np.int32)
    if vec13.ndim != 2 or vec13.shape[1] != 13:
        raise ValueError(f"Expected cad_vec_13d shape [S, 13], got {vec13.shape}")

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

            # 9-bin orientation triplet -> quantized [0,255]
            vec17[i, 6] = _bin_to_uint8(i_theta, 9)
            vec17[i, 7] = _bin_to_uint8(i_phi, 9)
            vec17[i, 8] = _bin_to_uint8(i_gamma, 9)

            pos_idx = int(max(vec13[i, 7], 0))
            i_z = pos_idx // 1296
            rem = pos_idx % 1296
            i_y = rem // 36
            i_x = rem % 36

            # 36-bin position triplet -> quantized [0,255]
            vec17[i, 9] = _bin_to_uint8(i_x, 36)
            vec17[i, 10] = _bin_to_uint8(i_y, 36)
            vec17[i, 11] = _bin_to_uint8(i_z, 36)

            # Remaining ext params are kept as quantized values
            vec17[i, 12:17] = np.clip(vec13[i, 8:13], 0, 255)

    return vec17


def _save_pointcloud_preview(points: np.ndarray, preview_path: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    # Sparse scatter for faster rendering while keeping shape perception.
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


def export_from_cad13(
    cad_vec_13d: np.ndarray,
    output_dir: str,
    stem: str,
    export_step: bool = True,
    export_stl: bool = False,
    export_preview: bool = True,
    preview_points: int = 4096,
) -> Dict[str, str]:
    """
    Build CAD solid from 13D vectors and export artifacts.

    Returns a dict with optional keys:
      - step_path
      - stl_path
      - preview_path
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        from OCC.Extend.DataExchange import write_step_file, write_stl_file
        from cadlib.visualize import CADsolid2pc, vec2CADsolid
    except Exception as e:  # pragma: no cover - depends on local OCC runtime
        raise RuntimeError(
            "CAD export requires pythonocc-core and cadlib dependencies. "
            f"Import failed: {e}"
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
        points = CADsolid2pc(shape, n_points=preview_points, name=stem)
        _save_pointcloud_preview(points, preview_path)
        artifacts["preview_path"] = preview_path

    return artifacts

