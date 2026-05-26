#!/usr/bin/env python3
"""
OCC pipeline diagnostic for MechCAD cad_vec dataset.

Checks, in order:
1) h5 loading / key existence (vec or out_vec)
2) basic command-sequence validity
3) OCC shape construction via vec2CADsolid
4) optional strict BRepCheck validation
"""

import argparse
import glob
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import h5py
import numpy as np

import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from cadlib.macro import ARC_IDX, CIRCLE_IDX, EOS_IDX, EXT_IDX, LINE_IDX, PAD_VAL, SOL_IDX


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose OCC failures on cad_vec h5 files.")
    parser.add_argument("--cad_vec_dir", type=str, default="data/Omni-CAD/cad_vec")
    parser.add_argument("--category_start", type=int, default=0)
    parser.add_argument("--category_end", type=int, default=None)
    parser.add_argument("--limit", type=int, default=200, help="Max files to test. <=0 means all.")
    parser.add_argument("--strict", action="store_true", help="Enable strict BRepCheck validation.")
    parser.add_argument("--dump_failures", type=str, default=None, help="Optional JSON path.")
    return parser.parse_args()


def collect_h5_paths(cad_vec_dir: str, category_start: int, category_end: int | None, limit: int) -> List[str]:
    if not os.path.isdir(cad_vec_dir):
        raise FileNotFoundError(f"cad_vec_dir not found: {cad_vec_dir}")
    categories = sorted([d for d in os.listdir(cad_vec_dir) if os.path.isdir(os.path.join(cad_vec_dir, d))])
    if category_end is None:
        selected = [c for c in categories if int(c) >= category_start]
    else:
        selected = [c for c in categories if category_start <= int(c) <= category_end]

    paths: List[str] = []
    for c in selected:
        paths.extend(sorted(glob.glob(os.path.join(cad_vec_dir, c, "*.h5"))))
    if limit > 0:
        paths = paths[:limit]
    return paths


def load_vec17(path: str) -> Tuple[np.ndarray | None, str | None]:
    try:
        with h5py.File(path, "r") as fp:
            if "vec" in fp:
                arr = fp["vec"][:]  # type: ignore
            elif "out_vec" in fp:
                arr = fp["out_vec"][:]  # type: ignore
            else:
                return None, "h5_missing_vec_and_out_vec"
    except Exception:
        return None, "h5_open_failed"

    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] != 17:
        return None, f"bad_shape_{arr.shape}"
    return arr.astype(np.int32), None


def basic_trim_and_check(vec17: np.ndarray) -> Tuple[np.ndarray | None, str | None]:
    cmds = vec17[:, 0].astype(np.int32)
    valid_cmds = {PAD_VAL, LINE_IDX, ARC_IDX, CIRCLE_IDX, EOS_IDX, SOL_IDX, EXT_IDX}
    bad_cmd = [int(c) for c in cmds if int(c) not in valid_cmds]
    if bad_cmd:
        return None, "invalid_command_value"

    non_pad = cmds != PAD_VAL
    trimmed = vec17[non_pad]
    if trimmed.shape[0] == 0:
        return None, "empty_after_pad_trim"

    eos_positions = np.where(trimmed[:, 0].astype(np.int32) == EOS_IDX)[0]
    if len(eos_positions) > 0:
        trimmed = trimmed[: int(eos_positions[0]) + 1]

    geom_cmd_mask = np.isin(trimmed[:, 0].astype(np.int32), [LINE_IDX, ARC_IDX, CIRCLE_IDX, EXT_IDX])
    if not np.any(geom_cmd_mask):
        return None, "no_geometry_command"
    return trimmed, None


def try_occ_build(trimmed: np.ndarray, strict: bool):
    try:
        from cadlib.visualize import vec2CADsolid
    except Exception as e:
        return None, f"occ_import_failed:{type(e).__name__}:{e}"

    try:
        shape = vec2CADsolid(trimmed.astype(np.float32), is_numerical=True, n=256)
    except Exception as e:
        return None, f"vec2CADsolid_exception:{type(e).__name__}:{e}"
    if shape is None:
        return None, "vec2CADsolid_none"

    try:
        if hasattr(shape, "IsNull") and shape.IsNull():  # type: ignore[attr-defined]
            return None, "shape_is_null"
    except Exception:
        pass

    if not strict:
        return True, None

    try:
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
    except Exception:
        return None, "brepcheck_import_failed"

    try:
        analyzer = BRepCheck_Analyzer(shape)
        if not analyzer.IsValid():
            return None, "brepcheck_invalid"
    except Exception as e:
        return None, f"brepcheck_exception:{type(e).__name__}:{e}"

    return True, None


def main():
    args = parse_args()

    paths = collect_h5_paths(args.cad_vec_dir, args.category_start, args.category_end, args.limit)
    print(f"[diag] files_to_test={len(paths)}")
    if len(paths) == 0:
        print("[diag] no files found for the given category range.")
        return

    reason_counter: Counter[str] = Counter()
    failures: List[Dict[str, str]] = []
    ok_count = 0

    for p in paths:
        vec17, err = load_vec17(p)
        if err is not None:
            reason_counter[err] += 1
            failures.append({"path": p, "reason": err})
            continue

        trimmed, err = basic_trim_and_check(vec17)
        if err is not None:
            reason_counter[err] += 1
            failures.append({"path": p, "reason": err})
            continue

        _, err = try_occ_build(trimmed, strict=args.strict)
        if err is not None:
            reason_counter[err] += 1
            failures.append({"path": p, "reason": err})
            continue

        ok_count += 1

    total = len(paths)
    fail_count = total - ok_count
    print(f"[diag] ok={ok_count} fail={fail_count} pass_rate={ok_count / max(total, 1):.3f}")
    print("[diag] top failure reasons:")
    for reason, cnt in reason_counter.most_common(10):
        print(f"  - {reason}: {cnt}")

    if args.dump_failures:
        payload = {
            "strict": bool(args.strict),
            "total": total,
            "ok": ok_count,
            "fail": fail_count,
            "reasons": dict(reason_counter),
            "failures": failures,
        }
        out_dir = os.path.dirname(args.dump_failures)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.dump_failures, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[diag] failure dump saved: {args.dump_failures}")


if __name__ == "__main__":
    main()
