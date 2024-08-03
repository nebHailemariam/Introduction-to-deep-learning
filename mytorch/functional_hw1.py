import numpy as np
from mytorch.autograd_engine import Autograd

"""
Mathematical Functionalities
    These are some IMPORTANT things to keep in mind:
    - Make sure grad of inputs are exact same shape as inputs.
    - Make sure the input and output order of each function is consistent with
        your other code.
    Optional:
    - You can account for broadcasting, but it is not required 
        in the first bonus.
"""

def identity_backward(grad_output, a):
    """Backward for identity. Already implemented."""

    return grad_output

def add_backward(grad_output, a, b):
    """Backward for addition. Already implemented."""
    
    a_grad = grad_output * np.ones(a.shape)
    b_grad = grad_output * np.ones(b.shape)

    return a_grad, b_grad


def sub_backward(grad_output, a, b):
    """Backward for subtraction"""

    return NotImplementedError


def matmul_backward(grad_output, a, b):
    """Backward for matrix multiplication"""

    return NotImplementedError


def mul_backward(grad_output, a, b):
    """Backward for multiplication"""

    return NotImplementedError


def div_backward(grad_output, a, b):
    """Backward for division"""

    return NotImplementedError


def log_backward(grad_output, a):
    """Backward for log"""

    return NotImplementedError


def exp_backward(grad_output, a):
    """Backward of exponential"""

    return NotImplementedError


def max_backward(grad_output, a):
    """Backward of max"""

    return NotImplementedError


def sum_backward(grad_output, a):
    """Backward of sum"""

    return NotImplementedError


def SoftmaxCrossEntropy_backward(grad_output, pred, ground_truth):
    """
    TODO: implement Softmax CrossEntropy Loss here. You may want to
    modify the function signature to include more inputs.
    NOTE: Since the gradient of the Softmax CrossEntropy Loss is
          is straightforward to compute, you may choose to implement
          this directly rather than rely on the backward functions of
          more primitive operations.
    """

    return NotImplementedError


