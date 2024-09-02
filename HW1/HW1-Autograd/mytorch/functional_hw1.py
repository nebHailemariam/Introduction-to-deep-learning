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
    a_grad = grad_output * np.ones(a.shape)
    b_grad = -grad_output * np.ones(b.shape)

    return a_grad, b_grad


def matmul_backward(grad_output, a, b):
    """Backward for matrix multiplication"""

    a_grad = grad_output @ b.T
    b_grad = a.T @ grad_output

    return a_grad, b_grad


def mul_backward(grad_output, a, b):
    """Backward for multiplication"""

    a_grad = grad_output * b
    b_grad = grad_output * a

    return a_grad, b_grad


def div_backward(grad_output, a, b):
    """Backward for division"""

    a_grad = grad_output * (1 / b)
    b_grad = -grad_output * (1 / np.square(b)) * a

    return a_grad, b_grad


def log_backward(grad_output, a):
    """Backward for log"""

    return grad_output / a


def exp_backward(grad_output, a):
    """Backward of exponential"""

    return grad_output * np.exp(a)


def max_backward(grad_output, a):
    """Backward of max"""
    a[a > 0] = 1
    a[a < 0] = 0
    return grad_output * a


def sum_backward(grad_output, a):
    """Backward of sum"""

    return grad_output * np.ones(a.shape)


def tanh_backward(grad_output, a):
    """Backward of tanh"""
    return grad_output * (1 - np.tanh(a) ** 2)


def SoftmaxCrossEntropy_backward(grad_output, pred, ground_truth):
    """
    TODO: implement Softmax CrossEntropy Loss here. You may want to
    modify the function signature to include more inputs.
    NOTE: Since the gradient of the Softmax CrossEntropy Loss is
          is straightforward to compute, you may choose to implement
          this directly rather than rely on the backward functions of
          more primitive operations.
    """
    softmax = np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)
    a = (softmax - ground_truth) / pred.shape[0]
    return a, None
