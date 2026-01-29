import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用无界面后端，兼容 WSL/headless 环境
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .curves import *
from .macro import *


##########################   base  ###########################
class SketchBase(object):
    """草图基类（曲线集合）。"""
    def __init__(self, children, reorder=True):
        self.children = children

        if reorder:
            self.reorder()

    @staticmethod
    def from_dict(stat):
        """从 JSON 数据构造草图

        Args:
            stat (dict): 来自 JSON 数据的字典
        """
        raise NotImplementedError

    @staticmethod
    def from_vector(vec, start_point, is_numerical=True):
        """从向量表示构造草图

        Args:
            vec (np.array): (seq_len, n_args)
            start_point (np.array): (2, ). 如果为 None，则隐式定义为最后一个结束点。
        """
        raise NotImplementedError

    def reorder(self):
        """重新排列曲线以遵循逆时针方向"""
        raise NotImplementedError

    @property
    def start_point(self):
        return self.children[0].start_point

    @property
    def end_point(self):
        return self.children[-1].end_point

    @property
    def bbox(self):
        """计算草图的边界框（最小/最大点）"""
        all_points = np.concatenate([child.bbox for child in self.children], axis=0)
        return np.stack([np.min(all_points, axis=0), np.max(all_points, axis=0)], axis=0)

    @property
    def bbox_size(self):
        """计算边界框尺寸（高度和宽度的最大值）"""
        bbox_min, bbox_max = self.bbox[0], self.bbox[1]
        bbox_size = np.max(np.abs(np.concatenate([bbox_max - self.start_point, bbox_min - self.start_point])))
        return bbox_size

    @property
    def global_trans(self):
        """起始点 + 草图尺寸（bbox_size）"""
        return np.concatenate([self.start_point, np.array([self.bbox_size])])

    def transform(self, translate, scale):
        """线性变换"""
        for child in self.children:
            child.transform(translate, scale)

    def flip(self, axis):
        for child in self.children:
            child.flip(axis)
        self.reorder()

    def numericalize(self, n=256):
        """将曲线参数量化为整数"""
        for child in self.children:
            child.numericalize(n)

    def normalize(self, size=256):
        """在给定尺寸内归一化，起始点位于中心"""
        cur_size = self.bbox_size
        scale = (size / 2 * NORM_FACTOR - 1) / cur_size # 防止应用数据增强时潜在溢出
        self.transform(-self.start_point, scale)
        self.transform(np.array((size / 2, size / 2)), 1)

    def denormalize(self, bbox_size, size=256):
        """归一化的逆过程"""
        scale = bbox_size / (size / 2 * NORM_FACTOR - 1)
        self.transform(-np.array((size / 2, size / 2)), scale)

    def to_vector(self):
        """转换为向量表示"""
        raise NotImplementedError

    def draw(self, ax):
        """在 matplotlib ax 上绘制草图"""
        raise NotImplementedError

    def to_image(self):
        """转换为图像"""
        fig, ax = plt.subplots()
        canvas = FigureCanvasAgg(fig)  # 明确使用 Agg 画布，避免后端兼容问题
        self.draw(ax)
        ax.axis('equal')
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        X = buf[:, :, :3]
        plt.close(fig)
        return X

    def sample_points(self, n=32):
        """从草图上均匀采样点"""
        raise NotImplementedError


####################### loop & profile #######################
class Loop(SketchBase):
    """草图环，一系列连接的曲线。"""
    def __init__(self, children, reorder=True):
        super().__init__(children, reorder)
        self.is_outer = False  # 是否为最外层环
        

    @staticmethod
    def from_dict(stat):
        all_curves = [construct_curve_from_dict(item) for item in stat['profile_curves']]
        this_loop = Loop(all_curves)
        this_loop.is_outer = stat['is_outer']
        return this_loop

    def __str__(self):
        return "Loop:" + "\n      -" + "\n      -".join([str(curve) for curve in self.children])

    @staticmethod
    def from_vector(vec, start_point=None, is_numerical=True):
        all_curves = []
        if start_point is None:
            # FIXME: 这里可以避免显式 for 循环
            for i in range(vec.shape[0]):
                if vec[i][0] == EOS_IDX:
                    start_point = vec[i - 1][1:3]
                    break
        for i in range(vec.shape[0]):
            type = vec[i][0]
            if type == SOL_IDX:
                continue
            elif type == EOS_IDX:
                break
            else:
                curve = construct_curve_from_vector(vec[i], start_point, is_numerical=is_numerical)
                start_point = vec[i][1:3] # current curve's end_point serves as next curve's start_point
            all_curves.append(curve)
        return Loop(all_curves)

    def reorder(self):
        """按最左侧开始并逆时针重新排序"""
        if len(self.children) <= 1:
            return

        start_curve_idx = -1
        sx, sy = 10000, 10000

        # 校正起始-结束点顺序
        if np.allclose(self.children[0].start_point, self.children[1].start_point) or \
            np.allclose(self.children[0].start_point, self.children[1].end_point):
            self.children[0].reverse()

        # 校正起始-结束点顺序并找到最左侧点
        for i, curve in enumerate(self.children):
            if i < len(self.children) - 1 and np.allclose(curve.end_point, self.children[i + 1].end_point):
                self.children[i + 1].reverse()
            if round(curve.start_point[0], 6) < round(sx, 6) or \
                    (round(curve.start_point[0], 6) == round(sx, 6) and round(curve.start_point[1], 6) < round(sy, 6)):
                start_curve_idx = i
                sx, sy = curve.start_point

        self.children = self.children[start_curve_idx:] + self.children[:start_curve_idx]

        # 确保大体上为逆时针
        if isinstance(self.children[0], Circle) or isinstance(self.children[-1], Circle): # FIXME: 硬编码
            return
        start_vec = self.children[0].direction()
        end_vec = self.children[-1].direction(from_start=False)
        if np.cross(end_vec, start_vec) <= 0:
            for curve in self.children:
                curve.reverse()
            self.children.reverse()

    def to_vector(self, max_len=None, add_sol=True, add_eos=True):
        loop_vec = np.stack([curve.to_vector() for curve in self.children], axis=0)
        if add_sol:
            loop_vec = np.concatenate([SOL_VEC[np.newaxis], loop_vec], axis=0)
        if add_eos:
            loop_vec = np.concatenate([loop_vec, EOS_VEC[np.newaxis]], axis=0)
        if max_len is None:
            return loop_vec

        if loop_vec.shape[0] > max_len:
            return None
        elif loop_vec.shape[0] < max_len:
            pad_vec = np.tile(EOS_VEC, max_len - loop_vec.shape[0]).reshape((-1, len(EOS_VEC)))
            loop_vec = np.concatenate([loop_vec, pad_vec], axis=0) # (max_len, 1 + N_ARGS)
        return loop_vec

    def draw(self, ax):
        colors = ['red', 'blue', 'green', 'brown', 'pink', 'yellow', 'purple', 'black'] * 10
        for i, curve in enumerate(self.children):
            curve.draw(ax, colors[i])

    def sample_points(self, n=32):
        points = np.stack([curve.sample_points(n) for curve in self.children], axis=0) # (n_curves, n, 2)
        return points


class Profile(SketchBase):
    """草图轮廓，由一个或多个环形成的闭合区域。
    最外层的环放在最前面。"""
    @staticmethod
    def from_dict(stat):
        all_loops = [Loop.from_dict(item) for item in stat['loops']]
        return Profile(all_loops)

    def __str__(self):
        return "Profile:" + "\n    -".join([str(loop) for loop in self.children])

    @staticmethod
    def from_vector(vec, start_point=None, is_numerical=True):
        all_loops = []
        command = vec[:, 0]
        end_idx = command.tolist().index(EOS_IDX)
        indices = np.where(command[:end_idx] == SOL_IDX)[0].tolist() + [end_idx]
        for i in range(len(indices) - 1):
            loop_vec = vec[indices[i]:indices[i + 1]]
            loop_vec = np.concatenate([loop_vec, EOS_VEC[np.newaxis]], axis=0)
            if loop_vec[0][0] == SOL_IDX and loop_vec[1][0] not in [SOL_IDX, EOS_IDX]:
                all_loops.append(Loop.from_vector(loop_vec, is_numerical=is_numerical))
        return Profile(all_loops)

    def reorder(self):
        if len(self.children) <= 1:
            return
        all_loops_bbox_min = np.stack([loop.bbox[0] for loop in self.children], axis=0).round(6)
        ind = np.lexsort(all_loops_bbox_min.transpose()[[1, 0]])
        self.children = [self.children[i] for i in ind]

    def draw(self, ax):
        for i, loop in enumerate(self.children):
            loop.draw(ax)
            ax.text(loop.start_point[0], loop.start_point[1], str(i))

    def to_vector(self, max_n_loops=None, max_len_loop=None, pad=True):
        loop_vecs = [loop.to_vector(None, add_eos=False) for loop in self.children]
        if max_n_loops is not None and len(loop_vecs) > max_n_loops:
            return None
        for vec in loop_vecs:
            if max_len_loop is not None and vec.shape[0] > max_len_loop:
                return None
        profile_vec = np.concatenate(loop_vecs, axis=0)
        profile_vec = np.concatenate([profile_vec, EOS_VEC[np.newaxis]], axis=0)
        if pad:
            assert max_n_loops is not None and max_len_loop is not None
            pad_len = max_n_loops * max_len_loop - profile_vec.shape[0]
            profile_vec = np.concatenate([profile_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)
        return profile_vec

    def sample_points(self, n=32):
        points = np.concatenate([loop.sample_points(n) for loop in self.children], axis=0)
        return points
