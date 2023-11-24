import numpy as np

def conv2D(input_tensor, weight_tensor, padding=0, dilation=1, stride=1, groups = 1):
    # Get input dimensions
    image_height, image_width  = input_tensor.shape
    weight_height, weight_width  = weight_tensor.shape

    # Calculate output dimensions
    H_out = int((image_height - weight_height + 2 * padding) / stride) + 1
    W_out = int((image_width - weight_width + 2 * padding) / stride) + 1

    #создание результирующего изображения
    result = np.zeros((H_out, W_out))
    #добавление padding (добавление нулей вокруг входной матрицы)
    if padding > 0:
        input_tensor = np.pad(input_tensor, padding, mode='constant')
    #проход ядром по изображению
    for y in range(H_out):
        for x in range(W_out):
            result[y, x] = np.sum(input_tensor[y*stride:y*stride+weight_height, x*stride:x*stride+weight_width] * weight_tensor)
    return result
