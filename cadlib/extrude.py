import numpy as np
import random
from .sketch import Profile
from .macro import *
from .math_utils import cartesian2polar, polar2cartesian, polar_parameterization, polar_parameterization_inverse

class CoordSystem(object):
    """草图平面的局部坐标系。"""
    def __init__(self, origin, theta, phi, gamma, y_axis=None, is_numerical=False):
        self.origin = origin
        self._theta = theta # 0~pi
        self._phi = phi     # -pi~pi
        self._gamma = gamma # -pi~pi
        self._y_axis = y_axis # (theta, phi)
        self.is_numerical = is_numerical

    @property
    def normal(self):
        return polar2cartesian([self._theta, self._phi])# 3D 法向量（z 轴）

    @property
    def x_axis(self):
        normal_3d, x_axis_3d = polar_parameterization_inverse(self._theta, self._phi, self._gamma)
        return x_axis_3d# 3D x 轴

    @property
    def y_axis(self):
        if self._y_axis is None:
            return np.cross(self.normal, self.x_axis)
        return polar2cartesian(self._y_axis)# 3D y 轴

    @staticmethod
    def from_dict(stat):
        origin = np.array([stat["origin"]["x"], stat["origin"]["y"], stat["origin"]["z"]])
        normal_3d = np.array([stat["z_axis"]["x"], stat["z_axis"]["y"], stat["z_axis"]["z"]])
        x_axis_3d = np.array([stat["x_axis"]["x"], stat["x_axis"]["y"], stat["x_axis"]["z"]])
        y_axis_3d = np.array([stat["y_axis"]["x"], stat["y_axis"]["y"], stat["y_axis"]["z"]])
        theta, phi, gamma = polar_parameterization(normal_3d, x_axis_3d)
        return CoordSystem(origin, theta, phi, gamma, y_axis=cartesian2polar(y_axis_3d))

    @staticmethod
    def from_vector(vec, is_numerical=False, n=256):
        origin = vec[:3]
        theta, phi, gamma = vec[3:]
        system = CoordSystem(origin, theta, phi, gamma)
        if is_numerical:
            system.denumericalize(n)
        return system

    def __str__(self):
        return "origin: {}, normal: {}, x_axis: {}, y_axis: {}".format(
            self.origin.round(4), self.normal.round(4), self.x_axis.round(4), self.y_axis.round(4))

    def transform(self, translation, scale):
        self.origin = (self.origin + translation) * scale

    def numericalize(self, n=256):
        """注意：仅应在归一化后调用。"""
        # assert np.max(self.origin) <= 1.0 and np.min(self.origin) >= -1.0 # 待办：origin 可能越界
        self.origin = ((self.origin + 1.0) / 2 * n).round().clip(min=0, max=n-1).astype(int)
        tmp = np.array([self._theta, self._phi, self._gamma])
        self._theta, self._phi, self._gamma = ((tmp / np.pi + 1.0) / 2 * n).round().clip(
            min=0, max=n-1).astype(int)
        self.is_numerical = True

    def denumericalize(self, n=256):
        self.origin = self.origin / n * 2 - 1.0
        tmp = np.array([self._theta, self._phi, self._gamma])
        self._theta, self._phi, self._gamma = (tmp / n * 2 - 1.0) * np.pi
        self.is_numerical = False

    def to_vector(self):
        return np.array([*self.origin, self._theta, self._phi, self._gamma])


class Extrude(object):
    """单次拉伸操作及其对应的草图轮廓。
    注意：仅支持单个草图轮廓，多个轮廓的拉伸会被拆分。"""
    def __init__(self, profile: Profile, sketch_plane: CoordSystem,
                 operation, extent_type, extent_one, extent_two, sketch_pos, sketch_size):
        """
        Args:
            profile (Profile): 归一化后的草图轮廓
            sketch_plane (CoordSystem): 草图平面的坐标系
            operation (int): EXTRUDE_OPERATIONS 的索引，见 macro.py
            extent_type (int): EXTENT_TYPE 的索引，见 macro.py
            extent_one (float): 法向拉伸距离（注意：某些数据中为负）
            extent_two (float): 反向拉伸距离
            sketch_pos (np.array): 草图起点的全局三维坐标
            sketch_size (float): 草图尺寸
        """
        self.profile = profile # 归一化后的草图
        self.sketch_plane = sketch_plane
        self.operation = operation
        self.extent_type = extent_type
        self.extent_one = extent_one
        self.extent_two = extent_two

        self.sketch_pos = sketch_pos
        self.sketch_size = sketch_size

    @staticmethod
    def from_dict(all_stat, extrude_id, sketch_dim=256):
        """从 JSON 数据构造 Extrude

        Args:
            all_stat (dict): 全部 JSON 数据
            extrude_id (str): 此拉伸的实体 ID
            sketch_dim (int, optional): 草图归一化尺寸，默认 256。

        Returns:
            list: 一个或多个 Extrude 实例
        """
        extrude_entity = all_stat["entities"][extrude_id]
        assert extrude_entity["start_extent"]["type"] == "ProfilePlaneStartDefinition"

        all_skets = []
        n = len(extrude_entity["profiles"])
        for i in range(len(extrude_entity["profiles"])):
            sket_id, profile_id = extrude_entity["profiles"][i]["sketch"], extrude_entity["profiles"][i]["profile"]
            sket_entity = all_stat["entities"][sket_id]
            sket_profile = Profile.from_dict(sket_entity["profiles"][profile_id])
            sket_plane = CoordSystem.from_dict(sket_entity["transform"])
            # 归一化草图
            point = sket_profile.start_point
            sket_pos = point[0] * sket_plane.x_axis + point[1] * sket_plane.y_axis + sket_plane.origin
            sket_size = sket_profile.bbox_size
            sket_profile.normalize(sketch_dim)
            all_skets.append((sket_profile, sket_plane, sket_pos, sket_size))

        operation = EXTRUDE_OPERATIONS.index(extrude_entity["operation"])
        extent_type = EXTENT_TYPE.index(extrude_entity["extent_type"])
        extent_one = extrude_entity["extent_one"]["distance"]["value"]
        extent_two = 0.0
        if extrude_entity["extent_type"] == "TwoSidesFeatureExtentType":
            extent_two = extrude_entity["extent_two"]["distance"]["value"]

        if operation == EXTRUDE_OPERATIONS.index("NewBodyFeatureOperation"):
            all_operations = [operation] + [EXTRUDE_OPERATIONS.index("JoinFeatureOperation")] * (n - 1)
        else:
            all_operations = [operation] * n

        return [Extrude(all_skets[i][0], all_skets[i][1], all_operations[i], extent_type, extent_one, extent_two,
                        all_skets[i][2], all_skets[i][3]) for i in range(n)]

    @staticmethod
    def from_vector(vec, is_numerical=False, n=256):
        """向量表示：命令 [SOL, ..., SOL, ..., EXT]"""
        assert vec[-1][0] == EXT_IDX and vec[0][0] == SOL_IDX
        profile_vec = np.concatenate([vec[:-1], EOS_VEC[np.newaxis]])
        profile = Profile.from_vector(profile_vec, is_numerical=is_numerical)
        ext_vec = vec[-1][-N_ARGS_EXT:]

        sket_pos = ext_vec[N_ARGS_PLANE:N_ARGS_PLANE + 3]
        sket_size = ext_vec[N_ARGS_PLANE + N_ARGS_TRANS - 1]
        sket_plane = CoordSystem.from_vector(np.concatenate([sket_pos, ext_vec[:N_ARGS_PLANE]]))
        ext_param = ext_vec[-N_ARGS_EXT_PARAM:]

        res = Extrude(profile, sket_plane, int(ext_param[2]), int(ext_param[3]), ext_param[0], ext_param[1],
                      sket_pos, sket_size)
        if is_numerical:
            res.denumericalize(n)
        return res

    def __str__(self):
        s = "Sketch-Extrude pair:"
        s += "\n  -" + str(self.sketch_plane)
        s += "\n  -sketch position: {}, sketch size: {}".format(self.sketch_pos.round(4), self.sketch_size.round(4))
        s += "\n  -operation:{}, type:{}, extent_one:{}, extent_two:{}".format(
            self.operation, self.extent_type, self.extent_one.round(4), self.extent_two.round(4))
        s += "\n  -" + str(self.profile)
        return s

    def transform(self, translation, scale):
        """线性变换"""
        # self.profile.transform(np.array([0, 0]), scale)
        self.sketch_plane.transform(translation, scale)
        self.extent_one *= scale
        self.extent_two *= scale
        self.sketch_pos = (self.sketch_pos + translation) * scale
        self.sketch_size *= scale

    def numericalize(self, n=256):
        """量化当前表示。
        注意：仅应在 CADSequence.normalize 之后调用（形状已落在 -1~1 的单位立方体内）"""
        assert -2.0 <= self.extent_one <= 2.0 and -2.0 <= self.extent_two <= 2.0
        self.profile.numericalize(n)
        self.sketch_plane.numericalize(n)
        self.extent_one = ((self.extent_one + 1.0) / 2 * n).round().clip(min=0, max=n-1).astype(int) 
        self.extent_two = ((self.extent_two + 1.0) / 2 * n).round().clip(min=0, max=n-1).astype(int) 
        self.operation = int(self.operation)
        self.extent_type = int(self.extent_type)

        self.sketch_pos = ((self.sketch_pos + 1.0) / 2 * n).round().clip(min=0, max=n-1).astype(int) 
        self.sketch_size = (self.sketch_size / 2 * n).round().clip(min=0, max=n-1).astype(int) 

    def denumericalize(self, n=256):
        """反量化表示。"""
        self.extent_one = self.extent_one / n * 2 - 1.0
        self.extent_two = self.extent_two / n * 2 - 1.0
        self.sketch_plane.denumericalize(n)
        self.sketch_pos = self.sketch_pos / n * 2 - 1.0
        self.sketch_size = self.sketch_size / n * 2

        self.operation = self.operation
        self.extent_type = self.extent_type

    def flip_sketch(self, axis):
        self.profile.flip(axis)
        self.profile.normalize()

    def to_vector(self, max_n_loops=6, max_len_loop=15, pad=True):
        """向量表示：命令 [SOL, ..., SOL, ..., EXT]"""
        profile_vec = self.profile.to_vector(max_n_loops, max_len_loop, pad=False)
        if profile_vec is None:
            return None
        sket_plane_orientation = self.sketch_plane.to_vector()[3:]
        ext_param = list(sket_plane_orientation) + list(self.sketch_pos) + [self.sketch_size] + \
                    [self.extent_one, self.extent_two, self.operation, self.extent_type]
        ext_vec = np.array([EXT_IDX, *[PAD_VAL] * N_ARGS_SKETCH, *ext_param])
        vec = np.concatenate([profile_vec[:-1], ext_vec[np.newaxis], profile_vec[-1:]], axis=0) # 注意：最后一个是 EOS
        if pad:
            pad_len = max_n_loops * max_len_loop - vec.shape[0]
            vec = np.concatenate([vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        return vec


class CADSequence(object):
    """CAD 建模序列，由一系列拉伸操作组成。"""
    def __init__(self, extrude_seq, bbox=None):
        self.seq = extrude_seq
        self.bbox = bbox

    @staticmethod  # 静态方法
    def from_dict(all_stat):
        """从 JSON 数据构造 CADSequence"""
        seq = []
        for item in all_stat["sequence"]:
            if item["type"] == "ExtrudeFeature":
                extrude_ops = Extrude.from_dict(all_stat, item["entity"])
                seq.extend(extrude_ops)
        bbox_info = all_stat["properties"]["bounding_box"]
        max_point = np.array([bbox_info["max_point"]["x"], bbox_info["max_point"]["y"], bbox_info["max_point"]["z"]])
        min_point = np.array([bbox_info["min_point"]["x"], bbox_info["min_point"]["y"], bbox_info["min_point"]["z"]])
        bbox = np.stack([max_point, min_point], axis=0)
        return CADSequence(seq, bbox)

    @staticmethod
    def from_vector(vec, is_numerical=False, n=256):
        commands = vec[:, 0]
        ext_indices = [-1] + np.where(commands == EXT_IDX)[0].tolist()
        ext_seq = []
        for i in range(len(ext_indices) - 1):
            start, end = ext_indices[i], ext_indices[i + 1]
            ext_seq.append(Extrude.from_vector(vec[start+1:end+1], is_numerical, n))
        cad_seq = CADSequence(ext_seq)
        return cad_seq

    def __str__(self):
        return  "" + "\n".join(["({})".format(i) + str(ext) for i, ext in enumerate(self.seq)])

    def to_vector(self, max_n_ext=10, max_n_loops=6, max_len_loop=15, max_total_len=60, pad=False):
        if len(self.seq) > max_n_ext:
            return None
        vec_seq = []
        for item in self.seq:
            vec = item.to_vector(max_n_loops, max_len_loop, pad=False)
            if vec is None:
                return None
            vec = vec[:-1] # 最后一个是 EOS，已删除
            vec_seq.append(vec)

        vec_seq = np.concatenate(vec_seq, axis=0)
        vec_seq = np.concatenate([vec_seq, EOS_VEC[np.newaxis]], axis=0)

        # 添加 EOS 填充
        if pad and vec_seq.shape[0] < max_total_len:
            pad_len = max_total_len - vec_seq.shape[0]
            vec_seq = np.concatenate([vec_seq, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        return vec_seq

    def transform(self, translation, scale):
        """线性变换"""
        for item in self.seq:
            item.transform(translation, scale)

    def normalize(self, size=1.0):
        """将形状归一化到单位立方体 (-1~1)。"""
        if self.bbox is not None:
            scale = size * NORM_FACTOR / np.max(np.abs(self.bbox))
            self.transform(0.0, scale)

    def numericalize(self, n=256):
        for item in self.seq:
            item.numericalize(n)

    def flip_sketch(self, axis):
        for item in self.seq:
            item.flip_sketch(axis)

    def random_transform(self):
        for item in self.seq:
            # 随机变换草图
            scale = random.uniform(0.8, 1.2)
            item.profile.transform(-np.array([128, 128]), scale)
            translate = np.array([random.randint(-5, 5), random.randint(-5, 5)], dtype=int) + 128
            item.profile.transform(translate, 1)

            # 随机变换和缩放拉伸
            t = 0.05
            translate = np.array([random.uniform(-t, t), random.uniform(-t, t), random.uniform(-t, t)])
            scale = random.uniform(0.8, 1.2)
            # item.sketch_plane.transform(translate, scale)
            item.sketch_pos = (item.sketch_pos + translate) * scale
            item.extent_one *= random.uniform(0.8, 1.2)
            item.extent_two *= random.uniform(0.8, 1.2)

    def random_flip_sketch(self):
        for item in self.seq:
            flip_idx = random.randint(0, 3)
            if flip_idx > 0:
                item.flip_sketch(['x', 'y', 'xy'][flip_idx - 1])