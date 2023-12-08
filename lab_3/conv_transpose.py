import numpy as np
from torch import nn
import torch
import pytest

def conv_transpose(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, in_channels, kernel_height, kernel_width = weight.shape
    out_height = (in_height - 1) * stride - 2 * padding + kernel_height + output_padding
    out_width = (in_width - 1) * stride - 2 * padding + kernel_width + output_padding
    output = np.zeros((batch_size, out_channels, out_height, out_width))
    for b in range(batch_size):
        for c in range(out_channels):
            for i in range(out_height):
                for j in range(out_width):
                    for k in range(in_channels):
                        for s in range(kernel_height):
                            for t in range(kernel_width):
                                ii = i + padding - s * dilation
                                jj = j + padding - t * dilation
                                if ii >= 0 and jj >= 0 and ii < in_height * stride and jj < in_width * stride and (ii % stride == 0) and (jj % stride == 0):
                                    ii //= stride
                                    jj //= stride
                                    output[b, c, i, j] += input[b, k, ii, jj] * weight[c, k, s, t]
            if bias is not None:
                output[b, c, :, :] += bias[c]
    return output
input_data = torch.randn(1, 1, 3, 3)
weight = torch.randn(1, 1, 3, 3)
