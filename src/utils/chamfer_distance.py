"""
Chamfer Distance (倒角距离) 计算工具

用于评估生成的 CAD 模型与真实模型之间的几何相似度。

Chamfer Distance 定义:
CD(P, Q) = (1/|P|) * sum_{p in P} min_{q in Q} ||p - q||^2
         + (1/|Q|) * sum_{q in Q} min_{p in P} ||q - p||^2
"""

import os
import sys
import numpy as np
import torch
from typing import Optional, Tuple, Union

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def chamfer_distance_numpy(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    sqrt: bool = False
) -> Tuple[float, float, float]:
    """
    计算两个点云之间的 Chamfer Distance (NumPy 版本)。

    Args:
        pred_points: 预测点云 [N, 3]
        gt_points: 真实点云 [M, 3]
        sqrt: 是否对距离开根号 (默认 False，使用平方距离)

    Returns:
        cd: 总 Chamfer Distance
        cd_pred: 预测到真实的单向距离
        cd_gt: 真实到预测的单向距离
    """
    # 计算距离矩阵 [N, M]
    diff = pred_points[:, np.newaxis, :] - gt_points[np.newaxis, :, :]  # [N, M, 3]
    dist_matrix = np.sum(diff ** 2, axis=-1)  # [N, M]

    if sqrt:
        dist_matrix = np.sqrt(dist_matrix)

    # 预测 -> 真实: 每个预测点到最近真实点的距离
    min_dist_pred = np.min(dist_matrix, axis=1)  # [N]
    cd_pred = np.mean(min_dist_pred)

    # 真实 -> 预测: 每个真实点到最近预测点的距离
    min_dist_gt = np.min(dist_matrix, axis=0)  # [M]
    cd_gt = np.mean(min_dist_gt)

    # 总 Chamfer Distance
    cd = cd_pred + cd_gt

    return cd, cd_pred, cd_gt


def chamfer_distance_torch(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    sqrt: bool = False,
    batch_reduction: str = 'mean'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算两个点云之间的 Chamfer Distance (PyTorch 版本，支持批处理)。

    Args:
        pred_points: 预测点云 [B, N, 3] 或 [N, 3]
        gt_points: 真实点云 [B, M, 3] 或 [M, 3]
        sqrt: 是否对距离开根号
        batch_reduction: 批处理归约方式 ('mean', 'sum', 'none')

    Returns:
        cd: 总 Chamfer Distance
        cd_pred: 预测到真实的单向距离
        cd_gt: 真实到预测的单向距离
    """
    # 确保是3D张量 [B, N, 3]
    if pred_points.dim() == 2:
        pred_points = pred_points.unsqueeze(0)
        gt_points = gt_points.unsqueeze(0)

    B, N, _ = pred_points.shape
    _, M, _ = gt_points.shape

    # 计算距离矩阵 [B, N, M]
    pred_expand = pred_points.unsqueeze(2)  # [B, N, 1, 3]
    gt_expand = gt_points.unsqueeze(1)  # [B, 1, M, 3]
    dist_matrix = torch.sum((pred_expand - gt_expand) ** 2, dim=-1)  # [B, N, M]

    if sqrt:
        dist_matrix = torch.sqrt(dist_matrix + 1e-8)

    # 预测 -> 真实
    min_dist_pred, _ = torch.min(dist_matrix, dim=2)  # [B, N]
    cd_pred = torch.mean(min_dist_pred, dim=1)  # [B]

    # 真实 -> 预测
    min_dist_gt, _ = torch.min(dist_matrix, dim=1)  # [B, M]
    cd_gt = torch.mean(min_dist_gt, dim=1)  # [B]

    # 总 Chamfer Distance
    cd = cd_pred + cd_gt  # [B]

    # 批处理归约
    if batch_reduction == 'mean':
        cd, cd_pred, cd_gt = cd.mean(), cd_pred.mean(), cd_gt.mean()
    elif batch_reduction == 'sum':
        cd, cd_pred, cd_gt = cd.sum(), cd_pred.sum(), cd_gt.sum()

    return cd, cd_pred, cd_gt


def chamfer_distance_batch(
    pred_points_list: list,
    gt_points_list: list,
    sqrt: bool = False
) -> dict:
    """
    批量计算 Chamfer Distance (处理不同大小的点云)。

    Args:
        pred_points_list: 预测点云列表，每个元素 [N_i, 3]
        gt_points_list: 真实点云列表，每个元素 [M_i, 3]
        sqrt: 是否使用欧氏距离

    Returns:
        metrics: 包含统计信息的字典
    """
    cd_list = []
    cd_pred_list = []
    cd_gt_list = []
    valid_count = 0

    for pred_pc, gt_pc in zip(pred_points_list, gt_points_list):
        if pred_pc is None or gt_pc is None:
            continue
        if len(pred_pc) == 0 or len(gt_pc) == 0:
            continue

        cd, cd_pred, cd_gt = chamfer_distance_numpy(pred_pc, gt_pc, sqrt=sqrt)
        cd_list.append(cd)
        cd_pred_list.append(cd_pred)
        cd_gt_list.append(cd_gt)
        valid_count += 1

    if valid_count == 0:
        return {
            'chamfer_distance': 0.0,
            'chamfer_pred': 0.0,
            'chamfer_gt': 0.0,
            'valid_count': 0
        }

    return {
        'chamfer_distance': np.mean(cd_list),
        'chamfer_distance_std': np.std(cd_list),
        'chamfer_pred': np.mean(cd_pred_list),
        'chamfer_gt': np.mean(cd_gt_list),
        'valid_count': valid_count
    }


class ChamferDistanceEvaluator:
    """
    Chamfer Distance 评估器，整合 CAD 向量到点云的转换和距离计算。
    """

    def __init__(self, n_points: int = 2048, normalize: bool = True):
        """
        Args:
            n_points: 从每个模型采样的点数
            normalize: 是否归一化点云到 [-1, 1]
        """
        self.n_points = n_points
        self.normalize = normalize
        self._cad13_to_cad17_numerical = None

        # 尝试导入 CAD 相关模块
        self._cad_available = False
        try:
            from cadlib.visualize import vec2CADsolid, CADsolid2pc
            from src.utils.cad_export import cad13_to_cad17_numerical
            self._vec2CADsolid = vec2CADsolid
            self._CADsolid2pc = CADsolid2pc
            self._cad13_to_cad17_numerical = cad13_to_cad17_numerical
            self._cad_available = True
        except ImportError:
            print("警告: cadlib 不可用，Chamfer Distance 评估将使用简化模式")

    def _normalize_points(self, points: np.ndarray) -> np.ndarray:
        """归一化点云到 [-1, 1]"""
        if len(points) == 0:
            return points
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        scale = np.max(max_vals - min_vals)
        if scale < 1e-8:
            return points
        center = (min_vals + max_vals) / 2
        return (points - center) / (scale / 2)

    def cad_vec_to_pointcloud(self, cad_vec: np.ndarray) -> Optional[np.ndarray]:
        """
        将 CAD 向量转换为点云。

        Args:
            cad_vec: CAD 向量序列 [S, 13] 或 [S, 17]

        Returns:
            points: 点云 [N, 3]，失败返回 None
        """
        if not self._cad_available:
            return None

        try:
            cad_vec = np.asarray(cad_vec)
            if cad_vec.ndim != 2:
                return None
            if cad_vec.shape[1] == 13:
                if self._cad13_to_cad17_numerical is None:
                    return None
                cad_vec = self._cad13_to_cad17_numerical(cad_vec)
            elif cad_vec.shape[1] != 17:
                return None

            # 转换为 CAD 实体
            shape = self._vec2CADsolid(cad_vec.astype(np.float32), is_numerical=True, n=256)
            if shape is None:
                return None

            # 从表面采样点云
            points = self._CADsolid2pc(shape, self.n_points)
            if points is None or len(points) == 0:
                return None

            if self.normalize:
                points = self._normalize_points(points)

            return points

        except Exception as e:
            # CAD 构建或采样失败
            return None

    def evaluate(
        self,
        pred_vecs: list,
        gt_vecs: list,
        sqrt: bool = False
    ) -> dict:
        """
        评估预测 CAD 向量与真实 CAD 向量之间的 Chamfer Distance。

        Args:
            pred_vecs: 预测 CAD 向量列表
            gt_vecs: 真实 CAD 向量列表
            sqrt: 是否使用欧氏距离 (默认使用平方距离)

        Returns:
            metrics: 评估指标字典
        """
        pred_pcs = []
        gt_pcs = []
        failed_count = 0

        for pred_vec, gt_vec in zip(pred_vecs, gt_vecs):
            pred_pc = self.cad_vec_to_pointcloud(pred_vec)
            gt_pc = self.cad_vec_to_pointcloud(gt_vec)

            if pred_pc is None or gt_pc is None:
                failed_count += 1
                continue

            pred_pcs.append(pred_pc)
            gt_pcs.append(gt_pc)

        metrics = chamfer_distance_batch(pred_pcs, gt_pcs, sqrt=sqrt)
        metrics['failed_count'] = failed_count
        metrics['total_count'] = len(pred_vecs)

        return metrics


# --- 测试代码 ---
if __name__ == '__main__':
    print("测试 Chamfer Distance 计算...")

    # 创建测试点云
    np.random.seed(42)
    pred_pc = np.random.randn(1024, 3).astype(np.float32)
    gt_pc = pred_pc + np.random.randn(1024, 3).astype(np.float32) * 0.1

    # NumPy 版本
    cd, cd_pred, cd_gt = chamfer_distance_numpy(pred_pc, gt_pc)
    print(f"NumPy - CD: {cd:.6f}, CD_pred: {cd_pred:.6f}, CD_gt: {cd_gt:.6f}")

    # PyTorch 版本
    pred_tensor = torch.from_numpy(pred_pc)
    gt_tensor = torch.from_numpy(gt_pc)
    cd_t, cd_pred_t, cd_gt_t = chamfer_distance_torch(pred_tensor, gt_tensor)
    print(f"PyTorch - CD: {cd_t:.6f}, CD_pred: {cd_pred_t:.6f}, CD_gt: {cd_gt_t:.6f}")

    # 批量计算
    metrics = chamfer_distance_batch(
        [pred_pc, pred_pc * 0.5],
        [gt_pc, gt_pc * 0.5]
    )
    print(f"Batch - CD: {metrics['chamfer_distance']:.6f}, Valid: {metrics['valid_count']}")
