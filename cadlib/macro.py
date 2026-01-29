import numpy as np

ALL_COMMANDS = ['Line', 'Arc', 'Circle', 'EOS', 'SOL', 'Ext']
LINE_IDX = ALL_COMMANDS.index('Line')
ARC_IDX = ALL_COMMANDS.index('Arc')
CIRCLE_IDX = ALL_COMMANDS.index('Circle')
EOS_IDX = ALL_COMMANDS.index('EOS')
SOL_IDX = ALL_COMMANDS.index('SOL')
EXT_IDX = ALL_COMMANDS.index('Ext')

EXTRUDE_OPERATIONS = ["NewBodyFeatureOperation", "JoinFeatureOperation",
                      "CutFeatureOperation", "IntersectFeatureOperation"]
EXTENT_TYPE = ["OneSideFeatureExtentType", "SymmetricFeatureExtentType",
               "TwoSidesFeatureExtentType"]

PAD_VAL = -1
N_ARGS_SKETCH = 5 # 草图参数: x, y, alpha, f, r
N_ARGS_PLANE = 3 # 草图平面朝向: theta, phi, gamma
N_ARGS_TRANS = 4 # 草图平面原点 + 草图 bbox 尺寸: p_x, p_y, p_z, s
N_ARGS_EXT_PARAM = 4 # 拉伸参数: e1, e2, b, u
N_ARGS_EXT = N_ARGS_PLANE + N_ARGS_TRANS + N_ARGS_EXT_PARAM
N_ARGS = N_ARGS_SKETCH + N_ARGS_EXT

SOL_VEC = np.array([SOL_IDX, *([PAD_VAL] * N_ARGS)])
EOS_VEC = np.array([EOS_IDX, *([PAD_VAL] * N_ARGS)])

CMD_ARGS_MASK = np.array([[1, 1, 0, 0, 0, *[0]*N_ARGS_EXT],  # 线段
                          [1, 1, 1, 1, 0, *[0]*N_ARGS_EXT],  # 圆弧
                          [1, 1, 0, 0, 1, *[0]*N_ARGS_EXT],  # 圆
                          [0, 0, 0, 0, 0, *[0]*N_ARGS_EXT],  # EOS
                          [0, 0, 0, 0, 0, *[0]*N_ARGS_EXT],  # SOL
                          [*[0]*N_ARGS_SKETCH, *[1]*N_ARGS_EXT]]) # 拉伸

NORM_FACTOR = 0.75 # 归一化的缩放因子，防止数据增强时溢出

MAX_N_EXT = 10 # 最大拉伸次数
MAX_N_LOOPS = 6 # 每个草图的最大环数
MAX_N_CURVES = 15 # 每个环的最大曲线数
MAX_TOTAL_LEN = 60 # 最大 CAD 序列长度
ARGS_DIM = 256
