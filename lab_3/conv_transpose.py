
import numpy as np
from torch import nn
import torch
from torch.nn.functional import conv_transpose2d
import pytest
from conv2D import conv2d

def generate_zeros(input, weight, stride, padding, output_padding):
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, in_channels, kernel_height, kernel_width = weight.shape
    out_height = (in_height - 1) * stride - 2 * padding + kernel_height + output_padding
    out_width = (in_width - 1) * stride - 2 * padding + kernel_width + output_padding
    output = np.zeros((batch_size, out_channels, out_height, out_width))
    return (batch_size, out_channels, out_height, out_width, 
            in_channels, kernel_height, 
            kernel_width, in_height, in_width, output)

# def 

def conv_transpose(input_data, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    (batch_size, out_channels, out_height, 
    out_width, in_channels,
    kernel_height, kernel_width,
    in_height, in_width, output) = generate_zeros(input_data, weight,
                                                       stride, padding,
                                                       output_padding)
    # for b in range(batch_size):
    #     for c in range(out_channels):
    #         for i in range(out_height):
    #             for j in range(out_width):
    #                 for k in range(in_channels):
    #                     for s in range(kernel_height):
    #                         for t in range(kernel_width):
    #                             ii = i + padding - s * dilation
    #                             jj = j + padding - t * dilation
    #                             if ii >= 0 and jj >= 0 and ii < in_height * stride and jj < in_width * stride and (ii % stride == 0) and (jj % stride == 0):
    #                                 ii //= stride
    #                                 jj //= stride
    #                                 output[b, c, i, j] += input[b, k, ii, jj] * weight[c, k, s, t]
    #         if bias is not None:
    #             output[b, c, :, :] += bias[c]

    output = conv2d(input_data, weight, padding=0, dilation=1, stride=1, groups = 1)

    return output
