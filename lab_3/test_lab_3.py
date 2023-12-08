from conv_transpose import conv_transpose
import torch
from torch.nn.functional import conv_transpose2d
import pytest

def test_1():
    image = torch.randn(1, 1, 3, 3) 
    weight = torch.randn(1, 1, 3, 3)
    
    myConvT = torch.from_numpy(conv_transpose(image.numpy(), weight.numpy()))

    torchConvT = conv_transpose2d(image, weight)

    myConvT = myConvT.to(torchConvT.dtype)

    assert torch.allclose(myConvT, torchConvT)

def test_2():
    image = torch.randn(1, 1, 5, 5) 
    weight = torch.randn(1, 1, 5, 5)
    
    myConvT = torch.from_numpy(conv_transpose(image.numpy(), weight.numpy()))

    torchConvT = conv_transpose2d(image, weight)

    myConvT = myConvT.to(torchConvT.dtype)

    assert torch.allclose(myConvT, torchConvT)

def test_3():
    image = torch.randn(1, 1, 4, 4) 
    weight = torch.randn(1, 1, 4, 4)
    
    myConvT = torch.from_numpy(conv_transpose(image.numpy(), weight.numpy()))

    torchConvT = conv_transpose2d(image, weight)

    myConvT = myConvT.to(torchConvT.dtype)

    assert torch.allclose(myConvT, torchConvT)

def test_4():
    image = torch.randn(1, 1, 6, 6) 
    weight = torch.randn(1, 1, 6, 6)
    
    myConvT = torch.from_numpy(conv_transpose(image.numpy(), weight.numpy()))

    torchConvT = conv_transpose2d(image, weight)

    myConvT = myConvT.to(torchConvT.dtype)

    assert torch.allclose(myConvT, torchConvT)