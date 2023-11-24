from conv2D import conv2D
import torch
from torch.nn.functional import conv2d as libConv2d
import pytest

def test_1():
    image = torch.randn(1, 1, 5, 5) 
    kernel = torch.randn(1, 1, 3, 3)
    
    test1_output1 = torch.from_numpy(conv2D(image[0, 0].numpy(), kernel[0, 0].numpy()))

    test1_output2 = libConv2d(image, kernel)

    test1_output1 = test1_output1.to(test1_output2.dtype)

    assert torch.allclose(test1_output1, test1_output2)

def test_2():
    image = torch.randn(1, 1, 4, 4) 
    kernel = torch.randn(1, 1, 2, 2) 
    
    test1_output1 = torch.from_numpy(conv2D(image[0, 0].numpy(), kernel[0, 0].numpy(), stride = 2))

    test1_output2 = libConv2d(image, kernel, stride = 2)

    test1_output1 = test1_output1.to(test1_output2.dtype)

    assert torch.allclose(test1_output1, test1_output2)

def test_3():
    image = torch.randn(1, 1, 6, 6) 
    kernel = torch.randn(1, 1, 3, 3) 
    
    test1_output1 = torch.from_numpy(conv2D(image[0, 0].numpy(), kernel[0, 0].numpy(), stride = 2))

    test1_output2 = libConv2d(image, kernel, stride = 2)

    test1_output1 = test1_output1.to(test1_output2.dtype)

    assert torch.allclose(test1_output1, test1_output2)

def test_3():
    image = torch.randn(1, 1, 6, 6) 
    kernel = torch.randn(1, 1, 3, 3) 
    
    test1_output1 = torch.from_numpy(conv2D(image[0, 0].numpy(), kernel[0, 0].numpy(), stride = 2))

    test1_output2 = libConv2d(image, kernel, stride = 2)

    test1_output1 = test1_output1.to(test1_output2.dtype)

    assert torch.allclose(test1_output1, test1_output2)

def test_4():
    image = torch.randn(1, 1, 8, 8) 
    kernel = torch.randn(1, 1, 2, 2) 
    
    test1_output1 = torch.from_numpy(conv2D(image[0, 0].numpy(), kernel[0, 0].numpy(), stride = 2))

    test1_output2 = libConv2d(image, kernel, stride = 2)

    test1_output1 = test1_output1.to(test1_output2.dtype)

    assert torch.allclose(test1_output1, test1_output2)