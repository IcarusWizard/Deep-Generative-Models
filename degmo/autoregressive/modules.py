import torch
from torch.functional import F

from ..modules import MaskConv
from .utils import build_maskA, build_maskB, shuffle, stew, unstew
from .utils import build_horizontal_mask, build_vertical_mask, gate_activation

class RowLSTM(torch.nn.Module):
    def __init__(self, features, height, width, group):
        super().__init__()

        self.height = height
        self.width = width
        self.group = group
        
        self.i2s_conv = MaskConv(2 * features, 4 * features, (1, 3),
            build_maskB(2 * features, 4 * features, (1, 3), group=group), padding=(0, 1), bias=False)

        self.s2s_conv = torch.nn.Conv2d(features, 4 * features, (1, 3), 1, padding=(0, 1), bias=False)

        self.skip_conv = MaskConv(features, 2 * features, (1, 1), 
            build_maskB(features, 2 * features, (1, 1), group=group))


    def forward(self, x):
        h = self.i2s_conv(x)

        # chunks = torch.chunk(h, 12, dim=1)
        # o = torch.cat([chunks[i] for i in range(12) if i % 4 == 0], dim=1)
        # f = torch.cat([chunks[i] for i in range(12) if i % 4 == 1], dim=1)
        # i = torch.cat([chunks[i] for i in range(12) if i % 4 == 2], dim=1)
        # g = torch.cat([chunks[i] for i in range(12) if i % 4 == 3], dim=1)
        # h = torch.cat([o, f, i, g], dim=1)
        if self.group == 3:
            h = shuffle(h)

        out_h = []
        _h = torch.zeros(h.shape[0], h.shape[1] // 4, 1, h.shape[3], device=x.device, dtype=x.dtype)
        c = torch.zeros_like(_h)
        for i in range(self.height):
            o, f, i, g = torch.chunk(torch.sigmoid(self.s2s_conv(_h) + h[:, :, i:i+1, :]), 4, dim=1)
            c = f * c + i * g
            _h = o * torch.tanh(c)

            out_h.append(_h)

        out_h = torch.cat(out_h, dim=2)

        return x + self.skip_conv(out_h)

class BiLSTM(torch.nn.Module):
    def __init__(self, features, height, width, group):
        super().__init__()

        self.height = height
        self.width = width
        self.group = group
        
        self.i2s_conv = MaskConv(2 * features, 4 * features, (1, 1),
            build_maskB(2 * features, 4 * features, (1, 1), group=group), padding=(0, 0), bias=False)

        self.left_s2s_conv = torch.nn.Conv2d(features, 4 * features, (2, 1), 1, padding=0)
        self.right_s2s_conv = torch.nn.Conv2d(features, 4 * features, (2, 1), 1, padding=0)

        self.skip_conv = MaskConv(features, 2 * features, (1, 1), 
            build_maskB(features, 2 * features, (1, 1), group=group))


    def forward(self, x):
        h = self.i2s_conv(x)

        if self.group == 3:
            h = shuffle(h) # maintain the dependency

        left_h, right_h = stew(h)

        left_c = torch.zeros(h.shape[0], h.shape[1] // 4, left_h.shape[2], left_h.shape[3], device=x.device, dtype=x.dtype)
        right_c = torch.zeros_like(left_c)
        _left_h = torch.zeros_like(left_c)
        _right_h = torch.zeros_like(left_c)

        for i in range(self.height):
            # perform for the left stew
            _left_h = F.pad(_left_h, (1, 0, 1, 0))
            _left_h = self.left_s2s_conv(_left_h)[:, :, :, :-1]
            o, f, i, g = torch.chunk(torch.sigmoid(_left_h + left_h), 4, dim=1)
            left_c = f * left_c + i * g
            _left_h = o * torch.tanh(left_c)

            # perform for the right stew
            _right_h = F.pad(_right_h, (0, 1, 1, 0))
            _right_h = self.right_s2s_conv(_right_h)[:, :, :, 1:]
            o, f, i, g = torch.chunk(torch.sigmoid(_right_h + right_h), 4, dim=1)
            right_c = f * right_c + i * g
            _right_h = o * torch.tanh(right_c)            

        left_h, right_h = unstew(_left_h, _right_h)

        # shift down one row for the right
        right_h = F.pad(right_h, (0, 0, 1, 0))
        right_h = right_h[:, :, :-1, :]

        out_h = left_h + right_h

        return x + self.skip_conv(out_h)

class MaskRes(torch.nn.Module):
    def __init__(self, h, w, features, group):
        # NOTE: You cannot use normalization that use a statistical mean and variance over space or channel,
        # for space statistic will break the dependency of spacial order, while channel statistic will break 
        # the dependency relationship of RGB pixels.
        super().__init__()
        self.skip_conv = torch.nn.Sequential(
            # torch.nn.LayerNorm((2 * features, h, w)),
            torch.nn.ReLU(True),
            MaskConv(2 * features, features, (1, 1),
                build_maskB(2 * features, features, (1, 1), group=group), padding=(0, 0)),
            # torch.nn.LayerNorm((features, h, w)),
            torch.nn.ReLU(True),
            MaskConv(features, features, (3, 3),
                build_maskB(features, features, (3, 3), group=group), padding=(1, 1)),
            # torch.nn.LayerNorm((features, h, w)),
            torch.nn.ReLU(True),
            MaskConv(features, 2 * features, (1, 1),
                build_maskB(features, 2 * features, (1, 1), group=group), padding=(0, 0)),
        )

    def forward(self, x):
        return x + self.skip_conv(x)

class GatePixelCNNBlock(torch.nn.Module):
    def __init__(self, in_dim, features, filter_size=3):
        super().__init__()

        self.is_first = in_dim != features

        padding = filter_size // 2

        self.horizontal = MaskConv(in_dim, 2 * features, (1, filter_size),
            build_horizontal_mask(in_dim, 2 * features, filter_size, self.is_first), padding=(0, padding))

        self.horizontal_out = torch.nn.Conv2d(features, features, 1)

        self.vertical = MaskConv(in_dim, 2 * features, filter_size,
            build_vertical_mask(in_dim, 2 * features, filter_size, self.is_first), padding=padding)

        self.trans = torch.nn.Conv2d(2 * features, 2 * features, 1)

    def forward(self, horizontal_x, vertical_x):
        vertical_mid = self.vertical(vertical_x)

        horizontial_mid = self.trans(vertical_mid) + self.horizontal(horizontal_x)

        horizontal_out = self.horizontal_out(gate_activation(horizontial_mid))
        vertical_out = gate_activation(vertical_mid)

        if not self.is_first:
            horizontal_out += horizontal_x

        return horizontal_out, vertical_out