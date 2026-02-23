"""
统一词表数据转换器

将13维CAD向量转换为统一词表的token序列，包含边界Token
"""

import numpy as np
from typing import Tuple

from .vocab import (
    PAD_TOKEN, SEP_TOKEN,
    BT_ANGLES_START, BT_ANGLES_END,
    BT_SPATIAL_START, BT_SPATIAL_END,
    BT_SKETCH_X_START, BT_SKETCH_X_END,
    BT_SKETCH_Y_START, BT_SKETCH_Y_END,
    param_to_token, angle_to_token, pos_to_token,
    MAX_ARGS_PER_CMD, VOCAB_SIZE,
    PARAM_OFFSET, ANGLE_OFFSET, POS_OFFSET,
    get_param_type_masks, get_boundary_token_targets,
)


_TYPE_MASKS = get_param_type_masks()
_PARAM_MASK = _TYPE_MASKS['param_mask'].astype(bool)
_ANGLE_MASK = _TYPE_MASKS['angle_mask'].astype(bool)
_POS_MASK = _TYPE_MASKS['pos_mask'].astype(bool)
_SEP_MASK = _TYPE_MASKS['sep_mask'].astype(bool)


def _argmax_in_range(logits_1d: np.ndarray, low: int, high: int) -> int:
    """在指定 token 范围内做 argmax，避免跨类型误解码。"""
    low = int(max(0, low))
    high = int(min(VOCAB_SIZE, high))
    if high <= low:
        return low
    return int(low + np.argmax(logits_1d[low:high]))


def constrained_argmax_args_tokens(commands: np.ndarray, args_logits: np.ndarray) -> np.ndarray:
    """
    对 unified args logits 进行类型约束解码（按命令与槽位类型）。

    Args:
        commands: [S] 命令序列
        args_logits: [S, MAX_ARGS_PER_CMD, VOCAB_SIZE] 参数 logits

    Returns:
        args_tokens: [S, MAX_ARGS_PER_CMD] 约束后的参数 token
    """
    commands = np.asarray(commands, dtype=np.int32)
    args_logits = np.asarray(args_logits)

    if args_logits.ndim != 3:
        raise ValueError(f"args_logits 应为 [S, N_ARGS, V]，得到 {args_logits.shape}")
    s, n_args, vocab_size = args_logits.shape
    if commands.shape[0] != s:
        raise ValueError(f"commands 与 args_logits 长度不一致: {commands.shape[0]} vs {s}")
    if n_args != MAX_ARGS_PER_CMD:
        raise ValueError(f"N_ARGS 应为 {MAX_ARGS_PER_CMD}，得到 {n_args}")
    if vocab_size != VOCAB_SIZE:
        raise ValueError(f"VOCAB_SIZE 应为 {VOCAB_SIZE}，得到 {vocab_size}")

    out = np.full((s, n_args), PAD_TOKEN, dtype=np.int32)

    for i in range(s):
        cmd = int(np.clip(commands[i], 0, _PARAM_MASK.shape[0] - 1))
        boundary_targets = get_boundary_token_targets(cmd)

        for j in range(n_args):
            # 固定边界 token 优先
            if j < len(boundary_targets) and boundary_targets[j] is not None:
                out[i, j] = int(boundary_targets[j])
                continue

            if _SEP_MASK[cmd, j]:
                out[i, j] = SEP_TOKEN
            elif _PARAM_MASK[cmd, j]:
                out[i, j] = _argmax_in_range(args_logits[i, j], PARAM_OFFSET, ANGLE_OFFSET)
            elif _ANGLE_MASK[cmd, j]:
                out[i, j] = _argmax_in_range(args_logits[i, j], ANGLE_OFFSET, POS_OFFSET)
            elif _POS_MASK[cmd, j]:
                out[i, j] = _argmax_in_range(args_logits[i, j], POS_OFFSET, VOCAB_SIZE)
            else:
                out[i, j] = PAD_TOKEN

    return out


def constrained_logits_to_13d(command_logits: np.ndarray, args_logits: np.ndarray) -> np.ndarray:
    """
    从 logits 直接解码为 13D CAD 向量，包含命令与参数类型约束。

    Args:
        command_logits: [S, n_commands]
        args_logits: [S, MAX_ARGS_PER_CMD, VOCAB_SIZE]

    Returns:
        cad_vec_13d: [S, 13]
    """
    command_logits = np.asarray(command_logits)
    if command_logits.ndim != 2:
        raise ValueError(f"command_logits 应为 [S, n_commands]，得到 {command_logits.shape}")

    commands = np.argmax(command_logits, axis=-1).astype(np.int32)
    args_tokens = constrained_argmax_args_tokens(commands, args_logits)
    return unified_tokens_to_13d(commands, args_tokens)


def convert_13d_to_unified_tokens(cad_vec_13d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将13维CAD向量序列转换为统一词表token序列（含边界Token）

    13维结构: [cmd, x, y, alpha, f, r, angle_tok, pos_tok, e1, e2, b, u, s]
              [0,   1, 2, 3,     4, 5, 6,         7,       8,  9,  10,11,12]

    输出格式（含边界Token）:
    - Line:   [<sketch_x>, x, </sketch_x>, <sketch_y>, y, </sketch_y>, SEP, PAD...]
    - Arc:    [<sketch_x>, x, </sketch_x>, <sketch_y>, y, </sketch_y>, alpha, f, SEP, PAD...]
    - Circle: [<sketch_x>, x, </sketch_x>, <sketch_y>, y, </sketch_y>, r, SEP, PAD...]
    - EOS/SOL: [SEP, PAD...]
    - Ext:    [<angles>, angle, </angles>, <spatial>, pos, </spatial>, e1, e2, b, u, s, SEP]

    Args:
        cad_vec_13d: [S, 13] 的CAD向量，S为序列长度

    Returns:
        commands: [S] 命令序列 (保持不变)
        args_tokens: [S, MAX_ARGS_PER_CMD] 统一词表的参数token序列
    """
    S = cad_vec_13d.shape[0]
    commands = cad_vec_13d[:, 0].astype(np.int32)
    args_tokens = np.full((S, MAX_ARGS_PER_CMD), PAD_TOKEN, dtype=np.int32)

    for i in range(S):
        cmd = int(commands[i])
        vec = cad_vec_13d[i]

        if cmd == 0:  # Line: x, y
            # [<sketch_x>, x, </sketch_x>, <sketch_y>, y, </sketch_y>, SEP]
            args_tokens[i, 0] = BT_SKETCH_X_START
            args_tokens[i, 1] = param_to_token(int(vec[1]))  # x
            args_tokens[i, 2] = BT_SKETCH_X_END
            args_tokens[i, 3] = BT_SKETCH_Y_START
            args_tokens[i, 4] = param_to_token(int(vec[2]))  # y
            args_tokens[i, 5] = BT_SKETCH_Y_END
            args_tokens[i, 6] = SEP_TOKEN

        elif cmd == 1:  # Arc: x, y, alpha, f
            # [<sketch_x>, x, </sketch_x>, <sketch_y>, y, </sketch_y>, alpha, f, SEP]
            args_tokens[i, 0] = BT_SKETCH_X_START
            args_tokens[i, 1] = param_to_token(int(vec[1]))  # x
            args_tokens[i, 2] = BT_SKETCH_X_END
            args_tokens[i, 3] = BT_SKETCH_Y_START
            args_tokens[i, 4] = param_to_token(int(vec[2]))  # y
            args_tokens[i, 5] = BT_SKETCH_Y_END
            args_tokens[i, 6] = param_to_token(int(vec[3]))  # alpha
            args_tokens[i, 7] = param_to_token(int(vec[4]))  # f
            args_tokens[i, 8] = SEP_TOKEN

        elif cmd == 2:  # Circle: x, y, r
            # [<sketch_x>, x, </sketch_x>, <sketch_y>, y, </sketch_y>, r, SEP]
            args_tokens[i, 0] = BT_SKETCH_X_START
            args_tokens[i, 1] = param_to_token(int(vec[1]))  # x
            args_tokens[i, 2] = BT_SKETCH_X_END
            args_tokens[i, 3] = BT_SKETCH_Y_START
            args_tokens[i, 4] = param_to_token(int(vec[2]))  # y
            args_tokens[i, 5] = BT_SKETCH_Y_END
            args_tokens[i, 6] = param_to_token(int(vec[5]))  # r
            args_tokens[i, 7] = SEP_TOKEN

        elif cmd == 3 or cmd == 4:  # EOS / SOL: 无参数
            # [SEP]
            args_tokens[i, 0] = SEP_TOKEN

        elif cmd == 5:  # Ext: angle, pos, e1, e2, b, u, s
            # [<angles>, angle, </angles>, <spatial>, pos, </spatial>, e1, e2, b, u, s, SEP]
            args_tokens[i, 0] = BT_ANGLES_START
            args_tokens[i, 1] = angle_to_token(int(vec[6]))  # angle token
            args_tokens[i, 2] = BT_ANGLES_END
            args_tokens[i, 3] = BT_SPATIAL_START
            args_tokens[i, 4] = pos_to_token(int(vec[7]))    # pos token
            args_tokens[i, 5] = BT_SPATIAL_END
            args_tokens[i, 6] = param_to_token(int(vec[8]))  # e1
            args_tokens[i, 7] = param_to_token(int(vec[9]))  # e2
            args_tokens[i, 8] = param_to_token(int(vec[10])) # b
            args_tokens[i, 9] = param_to_token(int(vec[11])) # u
            args_tokens[i, 10] = param_to_token(int(vec[12])) # s
            args_tokens[i, 11] = SEP_TOKEN

        else:
            # 未知命令，只放SEP
            args_tokens[i, 0] = SEP_TOKEN

    return commands, args_tokens


def unified_tokens_to_13d(commands: np.ndarray, args_tokens: np.ndarray) -> np.ndarray:
    """
    将统一词表token序列转换回13维CAD向量

    Args:
        commands: [S] 命令序列
        args_tokens: [S, MAX_ARGS_PER_CMD] 参数token序列

    Returns:
        cad_vec_13d: [S, 13] CAD向量
    """
    from .vocab import token_to_param, token_to_angle, token_to_pos, PAD_TOKEN

    S = commands.shape[0]
    cad_vec_13d = np.full((S, 13), -1, dtype=np.int32)
    cad_vec_13d[:, 0] = commands

    for i in range(S):
        cmd = int(commands[i])
        tokens = args_tokens[i]

        if cmd == 0:  # Line
            # tokens: [<sketch_x>, x, </sketch_x>, <sketch_y>, y, </sketch_y>, SEP, ...]
            cad_vec_13d[i, 1] = token_to_param(int(tokens[1]))  # x
            cad_vec_13d[i, 2] = token_to_param(int(tokens[4]))  # y

        elif cmd == 1:  # Arc
            cad_vec_13d[i, 1] = token_to_param(int(tokens[1]))  # x
            cad_vec_13d[i, 2] = token_to_param(int(tokens[4]))  # y
            cad_vec_13d[i, 3] = token_to_param(int(tokens[6]))  # alpha
            cad_vec_13d[i, 4] = token_to_param(int(tokens[7]))  # f

        elif cmd == 2:  # Circle
            cad_vec_13d[i, 1] = token_to_param(int(tokens[1]))  # x
            cad_vec_13d[i, 2] = token_to_param(int(tokens[4]))  # y
            cad_vec_13d[i, 5] = token_to_param(int(tokens[6]))  # r

        elif cmd == 5:  # Ext
            cad_vec_13d[i, 6] = token_to_angle(int(tokens[1]))   # angle
            cad_vec_13d[i, 7] = token_to_pos(int(tokens[4]))     # pos
            cad_vec_13d[i, 8] = token_to_param(int(tokens[6]))   # e1
            cad_vec_13d[i, 9] = token_to_param(int(tokens[7]))   # e2
            cad_vec_13d[i, 10] = token_to_param(int(tokens[8]))  # b
            cad_vec_13d[i, 11] = token_to_param(int(tokens[9]))  # u
            cad_vec_13d[i, 12] = token_to_param(int(tokens[10])) # s

    return cad_vec_13d
