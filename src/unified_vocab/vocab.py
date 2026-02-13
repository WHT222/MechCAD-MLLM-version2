"""
统一词表定义 (Enhanced Version)

基于CAD-GPT的边界Token设计，将所有参数类型合并到一个大词表中。

词表结构:
==========
特殊Token (0-9):
  [0]: PAD - 填充
  [1]: SEP - 参数序列结束/分隔符
  [2]: <angles> - 角度参数开始
  [3]: </angles> - 角度参数结束
  [4]: <spatial_position> - 空间位置开始
  [5]: </spatial_position> - 空间位置结束
  [6]: <sketch_x> - 草图X坐标开始
  [7]: </sketch_x> - 草图X坐标结束
  [8]: <sketch_y> - 草图Y坐标开始
  [9]: </sketch_y> - 草图Y坐标结束

参数值Token (10-265):
  v_0 到 v_255 - 量化后的参数值

角度Token (266-994):
  angle_0 到 angle_728 - 9^3 = 729个离散角度组合

位置Token (995-47650):
  pos_0 到 pos_46655 - 36^3 = 46656个离散位置组合

总词表大小: 47651
"""

import numpy as np
from typing import Tuple, List, Optional

# === 特殊Token定义 ===
PAD_TOKEN = 0
SEP_TOKEN = 1

# 边界Token (CAD-GPT风格)
BT_ANGLES_START = 2
BT_ANGLES_END = 3
BT_SPATIAL_START = 4
BT_SPATIAL_END = 5
BT_SKETCH_X_START = 6
BT_SKETCH_X_END = 7
BT_SKETCH_Y_START = 8
BT_SKETCH_Y_END = 9

# 边界Token名称映射
BOUNDARY_TOKENS = {
    BT_ANGLES_START: '<angles>',
    BT_ANGLES_END: '</angles>',
    BT_SPATIAL_START: '<spatial_position>',
    BT_SPATIAL_END: '</spatial_position>',
    BT_SKETCH_X_START: '<sketch_x>',
    BT_SKETCH_X_END: '</sketch_x>',
    BT_SKETCH_Y_START: '<sketch_y>',
    BT_SKETCH_Y_END: '</sketch_y>',
}

N_SPECIAL_TOKENS = 10

# === 参数值Token ===
PARAM_OFFSET = N_SPECIAL_TOKENS  # 10
PARAM_SIZE = 256  # 0-255

# === 角度Token ===
ANGLE_OFFSET = PARAM_OFFSET + PARAM_SIZE  # 266
ANGLE_BINS = 9
ANGLE_SIZE = ANGLE_BINS ** 3  # 729

# === 位置Token ===
POS_OFFSET = ANGLE_OFFSET + ANGLE_SIZE  # 995
POS_GRID_SIZE = 36
POS_SIZE = POS_GRID_SIZE ** 3  # 46656

# === 总词表大小 ===
VOCAB_SIZE = POS_OFFSET + POS_SIZE  # 47651


# === Token类型枚举 ===
class TokenType:
    PAD = "PAD"
    SEP = "SEP"
    BOUNDARY = "BOUNDARY"
    PARAM = "PARAM"
    ANGLE = "ANGLE"
    POS = "POS"
    UNKNOWN = "UNKNOWN"


# === Token转换函数 ===

def param_to_token(value: int) -> int:
    """将参数值(0-255)转换为词表token"""
    if value < 0:  # PAD_VAL = -1
        return PAD_TOKEN
    return PARAM_OFFSET + int(np.clip(value, 0, 255))


def token_to_param(token: int) -> int:
    """将词表token转换回参数值"""
    if token == PAD_TOKEN:
        return -1
    if token == SEP_TOKEN:
        return -1
    if PARAM_OFFSET <= token < ANGLE_OFFSET:
        return token - PARAM_OFFSET
    return -1


def angle_to_token(angle_idx: int) -> int:
    """将角度索引(0-728)转换为词表token"""
    if angle_idx < 0:
        return PAD_TOKEN
    return ANGLE_OFFSET + int(np.clip(angle_idx, 0, ANGLE_SIZE - 1))


def token_to_angle(token: int) -> int:
    """将词表token转换回角度索引"""
    if ANGLE_OFFSET <= token < POS_OFFSET:
        return token - ANGLE_OFFSET
    return -1


def pos_to_token(pos_idx: int) -> int:
    """将位置索引(0-46655)转换为词表token"""
    if pos_idx < 0:
        return PAD_TOKEN
    return POS_OFFSET + int(np.clip(pos_idx, 0, POS_SIZE - 1))


def token_to_pos(token: int) -> int:
    """将词表token转换回位置索引"""
    if POS_OFFSET <= token < VOCAB_SIZE:
        return token - POS_OFFSET
    return -1


def get_token_type(token: int) -> str:
    """获取token类型"""
    if token == PAD_TOKEN:
        return TokenType.PAD
    if token == SEP_TOKEN:
        return TokenType.SEP
    if token in BOUNDARY_TOKENS:
        return TokenType.BOUNDARY
    if PARAM_OFFSET <= token < ANGLE_OFFSET:
        return TokenType.PARAM
    if ANGLE_OFFSET <= token < POS_OFFSET:
        return TokenType.ANGLE
    if POS_OFFSET <= token < VOCAB_SIZE:
        return TokenType.POS
    return TokenType.UNKNOWN


def token_to_string(token: int) -> str:
    """将token转换为可读字符串"""
    if token == PAD_TOKEN:
        return "[PAD]"
    if token == SEP_TOKEN:
        return "[SEP]"
    if token in BOUNDARY_TOKENS:
        return BOUNDARY_TOKENS[token]
    if PARAM_OFFSET <= token < ANGLE_OFFSET:
        return f"v_{token - PARAM_OFFSET}"
    if ANGLE_OFFSET <= token < POS_OFFSET:
        return f"<A{token - ANGLE_OFFSET}>"
    if POS_OFFSET <= token < VOCAB_SIZE:
        return f"<P{token - POS_OFFSET}>"
    return f"[UNK:{token}]"


# === 角度索引与三元组转换 ===

def angle_triplet_to_idx(theta: int, phi: int, gamma: int) -> int:
    """
    将角度三元组转换为单一索引

    Args:
        theta, phi, gamma: 各自量化到 0-8 范围

    Returns:
        索引值 0-728
    """
    theta = int(np.clip(theta, 0, ANGLE_BINS - 1))
    phi = int(np.clip(phi, 0, ANGLE_BINS - 1))
    gamma = int(np.clip(gamma, 0, ANGLE_BINS - 1))
    return theta * ANGLE_BINS * ANGLE_BINS + phi * ANGLE_BINS + gamma


def idx_to_angle_triplet(idx: int) -> Tuple[int, int, int]:
    """
    将索引转换回角度三元组

    Returns:
        (theta, phi, gamma)
    """
    idx = int(np.clip(idx, 0, ANGLE_SIZE - 1))
    theta = idx // (ANGLE_BINS * ANGLE_BINS)
    remainder = idx % (ANGLE_BINS * ANGLE_BINS)
    phi = remainder // ANGLE_BINS
    gamma = remainder % ANGLE_BINS
    return theta, phi, gamma


# === 位置索引与三元组转换 ===

def pos_triplet_to_idx(px: int, py: int, pz: int) -> int:
    """
    将位置三元组转换为单一索引

    Args:
        px, py, pz: 各自量化到 0-35 范围

    Returns:
        索引值 0-46655
    """
    px = int(np.clip(px, 0, POS_GRID_SIZE - 1))
    py = int(np.clip(py, 0, POS_GRID_SIZE - 1))
    pz = int(np.clip(pz, 0, POS_GRID_SIZE - 1))
    return px * POS_GRID_SIZE * POS_GRID_SIZE + py * POS_GRID_SIZE + pz


def idx_to_pos_triplet(idx: int) -> Tuple[int, int, int]:
    """
    将索引转换回位置三元组

    Returns:
        (px, py, pz)
    """
    idx = int(np.clip(idx, 0, POS_SIZE - 1))
    px = idx // (POS_GRID_SIZE * POS_GRID_SIZE)
    remainder = idx % (POS_GRID_SIZE * POS_GRID_SIZE)
    py = remainder // POS_GRID_SIZE
    pz = remainder % POS_GRID_SIZE
    return px, py, pz


# === 命令参数定义 ===

# 每种命令的参数数量(不含边界Token和SEP)
CMD_ARG_COUNTS = {
    0: 2,   # Line: x, y
    1: 4,   # Arc: x, y, alpha, f
    2: 3,   # Circle: x, y, r
    3: 0,   # EOS: 无参数
    4: 0,   # SOL: 无参数
    5: 7,   # Ext: angle, pos, e1, e2, b, u, s
}

# 13维向量中各参数的索引映射
# 13维结构: [cmd, x, y, alpha, f, r, angle_tok, pos_tok, e1, e2, b, u, s]
#           [0,   1, 2, 3,     4, 5, 6,         7,       8,  9,  10,11,12]
PARAM_INDICES = {
    0: [1, 2],              # Line: x, y
    1: [1, 2, 3, 4],        # Arc: x, y, alpha, f
    2: [1, 2, 5],           # Circle: x, y, r
    3: [],                  # EOS
    4: [],                  # SOL
    5: [6, 7, 8, 9, 10, 11, 12],  # Ext: angle, pos, e1, e2, b, u, s
}

# Ext命令中哪些是特殊token（角度和位置）
EXT_SPECIAL_INDICES = {6: 'angle', 7: 'pos'}


# === 带边界Token的序列长度 ===

# 每种命令的完整token序列长度（包含边界Token）
# Line: <sketch_x> x </sketch_x> <sketch_y> y </sketch_y> SEP = 7
# Arc: <sketch_x> x </sketch_x> <sketch_y> y </sketch_y> v_alpha v_f SEP = 9
# Circle: <sketch_x> x </sketch_x> <sketch_y> y </sketch_y> v_r SEP = 8
# EOS/SOL: SEP = 1
# Ext: <angles> angle </angles> <spatial_position> pos </spatial_position> e1 e2 b u s SEP = 12

MAX_ARGS_PER_CMD = 12  # 最大参数序列长度(含边界Token和SEP)

# 命令到参数序列模板的映射
CMD_SEQUENCE_TEMPLATES = {
    0: ['sketch_x', 'x', '/sketch_x', 'sketch_y', 'y', '/sketch_y', 'SEP'],  # Line
    1: ['sketch_x', 'x', '/sketch_x', 'sketch_y', 'y', '/sketch_y', 'alpha', 'f', 'SEP'],  # Arc
    2: ['sketch_x', 'x', '/sketch_x', 'sketch_y', 'y', '/sketch_y', 'r', 'SEP'],  # Circle
    3: ['SEP'],  # EOS
    4: ['SEP'],  # SOL
    5: ['angles', 'angle', '/angles', 'spatial', 'pos', '/spatial', 'e1', 'e2', 'b', 'u', 's', 'SEP'],  # Ext
}


# === 参数类型掩码（用于损失计算）===

def get_param_type_masks(n_commands: int = 6) -> dict:
    """
    获取参数类型掩码，用于损失函数中区分不同参数类型

    Returns:
        dict: 包含各类型参数的位置掩码
            - 'param_mask': 普通参数位置 (使用PARAM范围)
            - 'angle_mask': 角度参数位置 (使用ANGLE范围)
            - 'pos_mask': 位置参数位置 (使用POS范围)
            - 'boundary_mask': 边界Token位置 (固定值)
            - 'sep_mask': SEP位置 (固定值)
    """
    masks = {
        'param_mask': np.zeros((n_commands, MAX_ARGS_PER_CMD), dtype=np.float32),
        'angle_mask': np.zeros((n_commands, MAX_ARGS_PER_CMD), dtype=np.float32),
        'pos_mask': np.zeros((n_commands, MAX_ARGS_PER_CMD), dtype=np.float32),
        'boundary_mask': np.zeros((n_commands, MAX_ARGS_PER_CMD), dtype=np.float32),
        'sep_mask': np.zeros((n_commands, MAX_ARGS_PER_CMD), dtype=np.float32),
    }

    # Line: [<sketch_x>, x, </sketch_x>, <sketch_y>, y, </sketch_y>, SEP]
    #       [boundary,   param, boundary,   boundary, param, boundary, sep]
    masks['boundary_mask'][0, [0, 2, 3, 5]] = 1.0
    masks['param_mask'][0, [1, 4]] = 1.0
    masks['sep_mask'][0, 6] = 1.0

    # Arc: [<sketch_x>, x, </sketch_x>, <sketch_y>, y, </sketch_y>, alpha, f, SEP]
    masks['boundary_mask'][1, [0, 2, 3, 5]] = 1.0
    masks['param_mask'][1, [1, 4, 6, 7]] = 1.0
    masks['sep_mask'][1, 8] = 1.0

    # Circle: [<sketch_x>, x, </sketch_x>, <sketch_y>, y, </sketch_y>, r, SEP]
    masks['boundary_mask'][2, [0, 2, 3, 5]] = 1.0
    masks['param_mask'][2, [1, 4, 6]] = 1.0
    masks['sep_mask'][2, 7] = 1.0

    # EOS: [SEP]
    masks['sep_mask'][3, 0] = 1.0

    # SOL: [SEP]
    masks['sep_mask'][4, 0] = 1.0

    # Ext: [<angles>, angle, </angles>, <spatial>, pos, </spatial>, e1, e2, b, u, s, SEP]
    masks['boundary_mask'][5, [0, 2, 3, 5]] = 1.0
    masks['angle_mask'][5, 1] = 1.0
    masks['pos_mask'][5, 4] = 1.0
    masks['param_mask'][5, [6, 7, 8, 9, 10]] = 1.0
    masks['sep_mask'][5, 11] = 1.0

    return masks


# === 目标token序列模板 ===

def get_boundary_token_targets(cmd: int) -> List[Optional[int]]:
    """
    获取指定命令的边界Token目标值

    Args:
        cmd: 命令类型索引

    Returns:
        边界Token的目标值列表，非边界位置为None
    """
    templates = {
        0: [BT_SKETCH_X_START, None, BT_SKETCH_X_END,
            BT_SKETCH_Y_START, None, BT_SKETCH_Y_END, SEP_TOKEN],
        1: [BT_SKETCH_X_START, None, BT_SKETCH_X_END,
            BT_SKETCH_Y_START, None, BT_SKETCH_Y_END, None, None, SEP_TOKEN],
        2: [BT_SKETCH_X_START, None, BT_SKETCH_X_END,
            BT_SKETCH_Y_START, None, BT_SKETCH_Y_END, None, SEP_TOKEN],
        3: [SEP_TOKEN],
        4: [SEP_TOKEN],
        5: [BT_ANGLES_START, None, BT_ANGLES_END,
            BT_SPATIAL_START, None, BT_SPATIAL_END,
            None, None, None, None, None, SEP_TOKEN],
    }
    return templates.get(cmd, [SEP_TOKEN])


if __name__ == '__main__':
    # 测试词表
    print("=== 统一词表信息 ===")
    print(f"总词表大小: {VOCAB_SIZE}")
    print(f"特殊Token数量: {N_SPECIAL_TOKENS}")
    print(f"参数Token范围: [{PARAM_OFFSET}, {ANGLE_OFFSET})")
    print(f"角度Token范围: [{ANGLE_OFFSET}, {POS_OFFSET})")
    print(f"位置Token范围: [{POS_OFFSET}, {VOCAB_SIZE})")

    print("\n=== 边界Token ===")
    for token, name in BOUNDARY_TOKENS.items():
        print(f"  {token}: {name}")

    print("\n=== Token转换测试 ===")
    test_param = 128
    print(f"参数值 {test_param} -> Token {param_to_token(test_param)} -> {token_to_string(param_to_token(test_param))}")

    test_angle = angle_triplet_to_idx(4, 4, 4)
    print(f"角度 (4,4,4) -> 索引 {test_angle} -> Token {angle_to_token(test_angle)} -> {token_to_string(angle_to_token(test_angle))}")

    test_pos = pos_triplet_to_idx(18, 18, 18)
    print(f"位置 (18,18,18) -> 索引 {test_pos} -> Token {pos_to_token(test_pos)} -> {token_to_string(pos_to_token(test_pos))}")

    print("\n=== 参数类型掩码 ===")
    masks = get_param_type_masks()
    for name, mask in masks.items():
        print(f"{name}: shape={mask.shape}")
