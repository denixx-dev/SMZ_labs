import numpy as np
import torch

def generate_zeros(input, weight, stride, padding, output_padding):
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, in_channels, kernel_height, kernel_width = weight.shape
    H_out = (in_height - 1) * stride - 2 * padding + kernel_height + output_padding
    W_out = (in_width - 1) * stride - 2 * padding + kernel_width + output_padding
    output = np.zeros((batch_size, out_channels, H_out, W_out))
    return (batch_size, out_channels, H_out, W_out, 
            in_channels, kernel_height, 
            kernel_width, in_height, in_width, output)

def conv2d(input_data, weight_tensor, padding=0, dilation=1, stride=1, groups = 1):
    (batch_size, out_channels, H_out, 
    W_out, in_channels,
    kernel_height, kernel_width,
    in_height, in_width, result) = generate_zeros(input_data, weight_tensor,
                                                       stride, padding,
                                                       padding)

    if padding > 0:
        input_data = np.pad(input_data, padding, mode='constant')

    result = np.zeros((batch_size, out_channels, H_out, W_out))

    for b in range(batch_size):
        for c in range(out_channels):
            for i in range(H_out):
                for j in range(W_out):
                    for k in range(in_channels):
                        for s in range(kernel_height):
                            for t in range(kernel_width):
                                ii = i + padding - s * dilation
                                jj = j + padding - t * dilation
                                if ii >= 0 and jj >= 0 and ii < in_height * stride and jj < in_width * stride and (ii % stride == 0) and (jj % stride == 0):
                                    ii //= stride
                                    jj //= stride
                                    result[b, c, i, j] += np.multiply(input_data[b, k, ii, jj], weight_tensor[c, k, s, t])
    return result
