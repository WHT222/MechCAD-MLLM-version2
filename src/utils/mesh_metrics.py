"""
Mesh-based evaluation metrics for CAD generation:
- SegE: Segment Error (component-count error against GT)
- DangEL: Dangling Edge Length (sum of boundary edge lengths)
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional

import numpy as np


def _count_mesh_components(mesh) -> int:
    if mesh is None or mesh.is_empty:
        return 0
    comps = mesh.split(only_watertight=False)
    return len(comps) if comps is not None else 0


def _dangling_edge_length(mesh) -> float:
    """
    Dangling edges are edges incident to exactly one face.
    Return the sum of their Euclidean lengths.
    """
    if mesh is None or mesh.is_empty:
        return 0.0

    faces = np.asarray(mesh.faces, dtype=np.int64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if faces.size == 0 or verts.size == 0:
        return 0.0

    # Collect all triangle edges and canonicalize by sorting vertex ids.
    e01 = faces[:, [0, 1]]
    e12 = faces[:, [1, 2]]
    e20 = faces[:, [2, 0]]
    edges = np.vstack([e01, e12, e20])
    edges = np.sort(edges, axis=1)

    edge_view = np.ascontiguousarray(edges).view(
        dtype=[("v0", edges.dtype), ("v1", edges.dtype)]
    )
    unique_edges, counts = np.unique(edge_view, return_counts=True)
    dangling_view = unique_edges[counts == 1]
    if len(dangling_view) == 0:
        return 0.0

    dangling = dangling_view.view(edges.dtype).reshape(-1, 2)
    seg = verts[dangling[:, 0]] - verts[dangling[:, 1]]
    return float(np.linalg.norm(seg, axis=1).sum())


def _bbox_diag(mesh) -> float:
    if mesh is None or mesh.is_empty:
        return 0.0
    bounds = np.asarray(mesh.bounds)
    if bounds.shape != (2, 3):
        return 0.0
    return float(np.linalg.norm(bounds[1] - bounds[0]))


class SegEDangELEvaluator:
    """
    Evaluate mesh-topology quality metrics from CAD vectors.

    Notes:
    - SegE is implemented as component-count error:
      abs(n_components(pred) - n_components(gt)).
    - DangEL is computed on predicted mesh only.
    """

    def __init__(self):
        self._cad_available = False
        try:
            import trimesh
            from OCC.Extend.DataExchange import write_stl_file
            from cadlib.visualize import vec2CADsolid
            from src.utils.cad_export import cad13_to_cad17_numerical

            self._trimesh = trimesh
            self._write_stl_file = write_stl_file
            self._vec2CADsolid = vec2CADsolid
            self._cad13_to_cad17_numerical = cad13_to_cad17_numerical
            self._cad_available = True
        except Exception:
            self._cad_available = False

    def _vec_to_mesh(self, cad_vec: np.ndarray):
        if not self._cad_available:
            return None

        try:
            vec = np.asarray(cad_vec)
            if vec.ndim != 2:
                return None

            if vec.shape[1] == 13:
                vec = self._cad13_to_cad17_numerical(vec)
            elif vec.shape[1] != 17:
                return None

            shape = self._vec2CADsolid(vec.astype(np.float32), is_numerical=True, n=256)
            if shape is None:
                return None

            # Convert CAD shape to mesh via temporary STL.
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
                stl_path = f.name
            try:
                self._write_stl_file(shape, stl_path)
                mesh = self._trimesh.load(stl_path, force="mesh")
            finally:
                if os.path.exists(stl_path):
                    os.remove(stl_path)
            return mesh
        except Exception:
            return None

    def evaluate(self, pred_vecs: List[np.ndarray], gt_vecs: List[np.ndarray]) -> Dict[str, float]:
        if not self._cad_available:
            return {
                "sege": -1.0,
                "sege_rel": -1.0,
                "dangel": -1.0,
                "dangel_norm": -1.0,
                "valid_count": 0,
                "failed_count": len(pred_vecs),
            }

        sege_list: List[float] = []
        sege_rel_list: List[float] = []
        dangel_list: List[float] = []
        dangel_norm_list: List[float] = []
        failed = 0

        for pred_vec, gt_vec in zip(pred_vecs, gt_vecs):
            pred_mesh = self._vec_to_mesh(pred_vec)
            gt_mesh = self._vec_to_mesh(gt_vec)
            if pred_mesh is None or gt_mesh is None:
                failed += 1
                continue

            n_pred = _count_mesh_components(pred_mesh)
            n_gt = _count_mesh_components(gt_mesh)
            seg_e = abs(float(n_pred - n_gt))
            seg_e_rel = seg_e / max(float(n_gt), 1.0)

            dang_len = _dangling_edge_length(pred_mesh)
            diag = _bbox_diag(pred_mesh)
            dang_norm = dang_len / (diag + 1e-8) if diag > 0 else 0.0

            sege_list.append(seg_e)
            sege_rel_list.append(seg_e_rel)
            dangel_list.append(dang_len)
            dangel_norm_list.append(dang_norm)

        valid = len(sege_list)
        if valid == 0:
            return {
                "sege": -1.0,
                "sege_rel": -1.0,
                "dangel": -1.0,
                "dangel_norm": -1.0,
                "valid_count": 0,
                "failed_count": failed,
            }

        return {
            "sege": float(np.mean(sege_list)),
            "sege_rel": float(np.mean(sege_rel_list)),
            "dangel": float(np.mean(dangel_list)),
            "dangel_norm": float(np.mean(dangel_norm_list)),
            "valid_count": valid,
            "failed_count": failed,
        }

