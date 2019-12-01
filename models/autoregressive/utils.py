import numpy as np
import torch
from torch.functional import F
from torch.nn import Parameter, init
import matplotlib.pyplot as plt
import pickle, torch, math, random

def build_maskA(in_channels, out_channels, kernal_size, group=3):
    mask = np.ones((in_channels, out_channels, *kernal_size))
    mid_row = kernal_size[0] // 2
    mid_col = kernal_size[1] // 2

    # mask context
    mask[:, :, (mid_row+1):, :] = 0
    mask[:, :, mid_row, (mid_col+1):] = 0

    if group == 3:
        in_group = in_channels // 3
        out_group = out_channels // 3
        # mask channels
        mask[:in_group, :out_group, mid_row, mid_col] = 0 # R
        mask[in_group:2*in_group, :2*out_group, mid_row, mid_col] = 0 # G
        mask[2*in_group:, :, mid_row, mid_col] = 0 # B
    elif group == 1: # for mnist
        mask[:, :, mid_row, mid_col] = 0

    mask = np.transpose(mask, (1, 0, 2, 3))

    return mask

def build_maskB(in_channels, out_channels, kernal_size, group=3):
    mask = np.ones((in_channels, out_channels, *kernal_size))
    mid_row = kernal_size[0] // 2
    mid_col = kernal_size[1] // 2

    # mask context
    mask[:, :, (mid_row+1):, :] = 0
    mask[:, :, mid_row, (mid_col+1):] = 0

    if group == 3:
        in_group = in_channels // 3
        out_group = out_channels // 3
        # mask channels
        mask[in_group:2*in_group, :out_group, mid_row, mid_col] = 0 # G
        mask[2*in_group:, :2*out_group, mid_row, mid_col] = 0 # B
    elif group == 1: # for mnist
        pass

    mask = np.transpose(mask, (1, 0, 2, 3))

    return mask

def shuffle(h):
    # NOTE: This is a wired operation!
    #|           r           |           g           |           b           |
    #| o_r | f_r | i_r | g_r | o_g | f_g | i_g | g_g | o_b | f_b | i_b | g_b |
    #| o_r | o_g | o_b | f_r | f_g | f_b | i_r | i_g | i_b | g_r | g_g | g_b |
    #|        o        |        f        |        i        |        g        |
    chunks = torch.chunk(h, 12, dim=1)
    o = torch.cat([chunks[i] for i in range(12) if i % 4 == 0], dim=1)
    f = torch.cat([chunks[i] for i in range(12) if i % 4 == 1], dim=1)
    i = torch.cat([chunks[i] for i in range(12) if i % 4 == 2], dim=1)
    g = torch.cat([chunks[i] for i in range(12) if i % 4 == 3], dim=1)
    h = torch.cat([o, f, i, g], dim=1)

    return h

def stew(x):
    b, c, h, w = x.shape # h == w

    left = torch.zeros(b, c, h, 2 * w - 1, device=x.device, dtype=x.dtype)
    right = torch.zeros(b, c, h, 2 * w - 1, device=x.device, dtype=x.dtype)

    for i in range(h):
        left[:, :, i, i : (i + w)] = x[:, :, i, :]
        right[:, :, i, (w - 1 - i) : (2 * w - 1 - i)] = x[:, :, i, :]

    return left, right

def unstew(left, right):
    b, c, h, w = left.shape

    w = (w + 1) // 2

    left_out = torch.zeros(b, c, h, w, device=left.device, dtype=left.dtype)
    right_out = torch.zeros(b, c, h, w, device=right.device, dtype=right.dtype)

    for i in range(h):
        left_out[:, :, i, :] = left[:, :, i, i : (i + w)]
        right_out[:, :, i, :] = right[:, :, i, (w - 1 - i) : (2 * w - 1 - i)]

    return left_out, right_out
