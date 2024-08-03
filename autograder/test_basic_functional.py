"""
NOTE: These test cases do not check the correctness of your solution,
      only whether anything has been implemented in functional.py.
      You are free to add your own test cases for checking correctness
"""

import  numpy as np
import  torch
from    mytorch.functional_hw1 import *

def test_add_backward():
    grad_output = np.zeros((5, 5))
    a = np.zeros_like(grad_output)
    b = np.zeros_like(grad_output)
    a_torch = torch.nn.Parameter(a)
    b_torch = torch.nn.Parameter(b)
    grad_output_torch = torch.nn.Parameter(grad_output)
    c = a_torch + b_torch
    c.backward(grad_output_torch)
    da,db = add_backward(grad_output, a, b)
    if np.allclose(da, a_torch.grad) and np.allclose(db, b_torch.grad):
        return True
    else: 
        return False

def test_sub_backward():

    grad_output = np.array(torch.randn((5, 5)))
    a = np.array(torch.randn(grad_output.shape))
    b = np.array(torch.randn(grad_output.shape))
    a_torch = torch.nn.Parameter(torch.as_tensor(a))
    b_torch = torch.nn.Parameter(torch.as_tensor(b))
    grad_output_torch = torch.nn.Parameter(torch.as_tensor(grad_output))
    c = a_torch - b_torch
    c.backward(grad_output_torch)
    da,db = sub_backward(grad_output, a, b)
    if np.allclose(da, a_torch.grad) and np.allclose(db, b_torch.grad):
        return True
    else: 
        return False

def test_matmul_backward():
    grad_output = np.array(torch.randn((5, 5)))
    a = np.array(torch.randn(grad_output.shape))
    b = np.array(torch.randn(grad_output.shape))
    a_torch = torch.nn.Parameter(torch.as_tensor(a))
    b_torch = torch.nn.Parameter(torch.as_tensor(b))
    grad_output_torch = torch.nn.Parameter(torch.as_tensor(grad_output))
    c_torch = torch.matmul(a_torch,b_torch)
    c_torch.backward(grad_output_torch)
    da,db = matmul_backward(grad_output, a, b)
    if np.allclose(da, a_torch.grad) and np.allclose(db, b_torch.grad):
        return True
    else: 
        return False

def test_mul_backward():
    grad_output = np.array(torch.randn((5, 5)))
    a = np.array(torch.randn(grad_output.shape))
    b = np.array(torch.randn(grad_output.shape))
    a_torch = torch.nn.Parameter(torch.as_tensor(a))
    b_torch = torch.nn.Parameter(torch.as_tensor(b))
    grad_output_torch = torch.nn.Parameter(torch.as_tensor(grad_output))
    c_torch = a_torch*b_torch
    c_torch.backward(grad_output_torch)
    da,db = mul_backward(grad_output, a, b)
    if np.allclose(da, a_torch.grad) and np.allclose(db, b_torch.grad):
        return True
    else: 
        return False

def test_div_backward():
    grad_output = np.array(torch.randn((5, 5)))
    a = np.array(torch.randn(grad_output.shape))
    b = np.array(torch.randn(grad_output.shape))
    a_torch = torch.nn.Parameter(torch.as_tensor(a))
    b_torch = torch.nn.Parameter(torch.as_tensor(b))
    grad_output_torch = torch.nn.Parameter(torch.as_tensor(grad_output))
    c_torch = a_torch/b_torch
    c_torch.backward(grad_output_torch)
    da,db = div_backward(grad_output, a, b)
    if np.allclose(da, a_torch.grad) and np.allclose(db, b_torch.grad):
        return True
    else: 
        return False

def test_log_backward():
    grad_output = np.array(torch.randn((5, 5)))
    a = np.array(torch.randn(grad_output.shape))
    a_torch = torch.nn.Parameter(torch.as_tensor(a))
    grad_output_torch = torch.nn.Parameter(torch.as_tensor(grad_output))
    c_torch = torch.log(a_torch)
    c_torch.backward(grad_output_torch)
    da = log_backward(grad_output, a)
    if np.allclose(da, a_torch.grad):
        return True
    else: 
        return False

def test_exp_backward():
    grad_output = np.array(torch.randn((5, 5)))
    a = np.array(torch.randn(grad_output.shape))
    a_torch = torch.nn.Parameter(torch.as_tensor(a))
    grad_output_torch = torch.nn.Parameter(torch.as_tensor(grad_output))
    c_torch = torch.exp(a_torch)
    c_torch.backward(grad_output_torch)
    da = exp_backward(grad_output, a)
    if np.allclose(da, a_torch.grad):
        return True
    else: 
        return False
