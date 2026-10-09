import copy
import numpy as np

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    if isinstance(input, np.ndarray):
        return torch.from_numpy(input)
    elif isinstance(input, (int, float)):
        return torch.tensor([input])  # 保证 batch 维度
    elif isinstance(input, list):
        return torch.tensor(input)
    else:
        return input