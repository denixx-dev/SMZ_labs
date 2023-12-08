import numpy as np
from torch import nn
import torch
import pytest

def upsample_bilinear(input_tensor, size):
    N, C, H, W = input_tensor.shape
    new_H, new_W = size

    scale_factor_H = new_H / H
    scale_factor_W = new_W / W

    output_np = np.empty((N, C, new_H, new_W))

    for n in range(N):
        for c in range(C):
            for i in range(new_H):
                for j in range(new_W):
                    x = (i + 0.5) / scale_factor_H - 0.5
                    y = (j + 0.5) / scale_factor_W - 0.5
                    x0 = int(np.floor(x))
                    x1 = min(x0 + 1, H - 1)
                    y0 = int(np.floor(y))
                    y1 = min(y0 + 1, W - 1)

                    output_np[n, c, i, j] = (input_tensor[n, c, x0, y0] * (x1 - x) * (y1 - y) +
                                             input_tensor[n, c, x0, y1] * (x1 - x) * (y - y0) +
                                             input_tensor[n, c, x1, y0] * (x - x0) * (y1 - y) +
                                             input_tensor[n, c, x1, y1] * (x - x0) * (y - y0))

    return torch.from_numpy(output_np).double()
