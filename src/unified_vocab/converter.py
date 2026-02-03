"""
统一词表数据转换器

将13维CAD向量转换为统一词表的token序列
"""

import numpy as np
from typing import Tuple

from .vocab import (
    PAD_TOKEN, SEP_TOKEN,
    param_to_token, angle_to_token, pos_to_token,
    MAX_ARGS_PER_CMD, CMD_ARG_COUNTS, PARAM_INDICES, EXT_SPECIAL_INDICES
)


def convert_13d_to_unified_tokens(cad_vec_13d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将13维CAD向量序列转换为统一词表token序列

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

        if cmd not in PARAM_INDICES:
            # 未知命令，只放SEP
            args_tokens[i, 0] = SEP_TOKEN
            continue

        param_indices = PARAM_INDICES[cmd]
        token_pos = 0

        for idx in param_indices:
            value = int(vec[idx])

            if cmd == 5 and idx in EXT_SPECIAL_INDICES:
                # Ext命令的特殊token
                if EXT_SPECIAL_INDICES[idx] == 'angle':
                    args_tokens[i, token_pos] = angle_to_token(value)
                else:  # pos
                    args_tokens[i, token_pos] = pos_to_token(value)
            else:
                # 普通参数
                args_tokens[i, token_pos] = param_to_token(value)

            token_pos += 1

        # 添加SEP结束符
        args_tokens[i, token_pos] = SEP_TOKEN

    return commands, args_tokens
