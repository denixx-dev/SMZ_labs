import numpy as np
from torch import nn
import torch
import pytest
from upsample_bilinear import upsample_bilinear

def test_1():
    input_tensor = torch.rand(1, 3, 4, 4)
    output_size = (1, 1)
    my_upsample = upsample_bilinear(input_tensor, output_size)
    built_in_upsample = torch.nn.functional.interpolate(input_tensor, size=output_size, mode='bilinear').double()
    assert torch.allclose(my_upsample, built_in_upsample)

def test_2():
    input_tensor = torch.rand(1, 3, 4, 4)
    output_size = (2, 2)
    my_upsample = upsample_bilinear(input_tensor, output_size)
    built_in_upsample = torch.nn.functional.interpolate(input_tensor, size=output_size, mode='bilinear').doubsle()
    assert torch.allclose(my_upsample, built_in_upsample)

def test_3():
    input_tensor = torch.rand(1, 3, 6, 6)
    output_size = (2, 2)
    my_upsample = upsample_bilinear(input_tensor, output_size)
    built_in_upsample = torch.nn.functional.interpolate(input_tensor, size=output_size, mode='bilinear').double()
    assert torch.allclose(my_upsample, built_in_upsample)

def test_4():
    input_tensor = torch.rand(1, 3, 6, 6)
    output_size = (1, 1)
    my_upsample = upsample_bilinear(input_tensor, output_size)
    built_in_upsample = torch.nn.functional.interpolate(input_tensor, size=output_size, mode='bilinear').double()
    assert torch.allclose(my_upsample, built_in_upsample)
