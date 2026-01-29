import math
import numpy as np


def rads_to_degs(rads):
    """将弧度制角转换为角度制"""
    return 180 * rads / math.pi


def angle_from_vector_to_x(vec):
    """计算单位向量与 x 正轴之间的角度 (0~2pi)"""
    angle = 0.0
    # 2 | 1
    # -------
    # 3 | 4
    if vec[0] >= 0:
        if vec[1] >= 0:
            # Qadrant 1
            angle = math.asin(vec[1])
        else:
            # Qadrant 4
            angle = 2.0 * math.pi - math.asin(-vec[1])
    else:
        if vec[1] >= 0:
            # Qadrant 2
            angle = math.pi - math.asin(vec[1])
        else:
            # Qadrant 3
            angle = math.pi + math.asin(-vec[1])
    return angle


def cartesian2polar(vec, with_radius=False):
    """将笛卡尔坐标向量转换为极坐标（球坐标）"""
    vec = vec.round(6)
    norm = np.linalg.norm(vec)
    theta = np.arccos(vec[2] / norm) # (0, pi)
    phi = np.arctan(vec[1] / (vec[0] + 1e-15)) # (-pi, pi) # FIXME: -0.0 cannot be identified here
    if not with_radius:
        return np.array([theta, phi])
    else:
        return np.array([theta, phi, norm])


def polar2cartesian(vec):
    """将极坐标（球坐标）向量转换为笛卡尔坐标"""
    r = 1 if len(vec) == 2 else vec[2]
    theta, phi = vec[0], vec[1]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def rotate_by_x(vec, theta):
    mat = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    return np.dot(mat, vec)


def rotate_by_y(vec, theta):
    mat = np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(mat, vec)


def rotate_by_z(vec, phi):
    mat = np.array([[np.cos(phi), -np.sin(phi), 0],
                    [np.sin(phi), np.cos(phi), 0],
                    [0, 0, 1]])
    return np.dot(mat, vec)


def polar_parameterization(normal_3d, x_axis_3d):
    """用相对于标准三维坐标系的旋转来表示一个坐标系

    Args:
        normal_3d (np.array): 法向（z 轴）的单位向量
        x_axis_3d (np.array): x 轴的单位向量

    Returns:
        theta, phi, gamma: 轴角表示的旋转
    """
    normal_polar = cartesian2polar(normal_3d)
    theta = normal_polar[0]
    phi = normal_polar[1]

    ref_x = rotate_by_z(rotate_by_y(np.array([1, 0, 0]), theta), phi)

    gamma = np.arccos(np.dot(x_axis_3d, ref_x).round(6))
    if np.dot(np.cross(ref_x, x_axis_3d), normal_3d) < 0:
        gamma = -gamma
    return theta, phi, gamma


def polar_parameterization_inverse(theta, phi, gamma):
    """根据给定的旋转（相对标准三维坐标系）构建坐标系"""
    normal_3d = polar2cartesian([theta, phi])
    ref_x = rotate_by_z(rotate_by_y(np.array([1, 0, 0]), theta), phi)
    ref_y = np.cross(normal_3d, ref_x)
    x_axis_3d = ref_x * np.cos(gamma) + ref_y * np.sin(gamma)
    return normal_3d, x_axis_3d
