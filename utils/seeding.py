"""
随机种子管理
确保实验可复现
"""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """固定所有随机源，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """本实验统一使用CPU"""
    return torch.device('cpu')