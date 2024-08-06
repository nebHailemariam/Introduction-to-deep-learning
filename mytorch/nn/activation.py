import numpy as np
from mytorch.functional_hw1 import *


class Activation(object):
    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result,
    i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an
    # abstract base class for the others

    # Note that these activation functions are scalar operations. I.e, they
    # shouldn't change the shape of the input.

    def __init__(self, autograd_engine):
        self.state = None
        self.autograd_engine = autograd_engine

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError


class Identity(Activation):
    """
    Identity function (already implemented).
    This class is a gimme as it is already implemented for you as an example.
    Just complete the forward by returning self.state.
    """

    def __init__(self, autograd_engine):
        super(Identity, self).__init__(autograd_engine)

    def forward(self, x):

        self.autograd_engine.add_operation(
            inputs=[x],
            output=x,
            gradients_to_update=[None],
            backward_operation=identity_backward,
        )

        return x


class Sigmoid(Activation):
    """
    Sigmoid activation.
    Feel free to enumerate primitive operations here (you may find it helpful).
    Feel free to refer back to your implementation from part 1 of the HW for equations.
    """

    def __init__(self, autograd_engine):
        super(Sigmoid, self).__init__(autograd_engine)

    def forward(self, x):

        # TODO Compute forward with primitive operations
        # TODO Add operations to the autograd engine as you go
        ones = -1.0 * np.ones(shape=x.shape)
        neg = ones * x
        self.autograd_engine.add_operation(
            inputs=[ones, x],
            output=neg,
            gradients_to_update=[None, None],
            backward_operation=mul_backward,
        )

        exp = np.exp(neg)
        self.autograd_engine.add_operation(
            inputs=[neg],
            output=exp,
            gradients_to_update=[None],
            backward_operation=exp_backward,
        )
        ones = np.ones(shape=exp.shape)
        sum = ones + exp

        self.autograd_engine.add_operation(
            inputs=[ones, exp],
            output=sum,
            gradients_to_update=[None, None],
            backward_operation=add_backward,
        )
        ones = np.ones(shape=exp.shape)
        div = ones / sum

        self.autograd_engine.add_operation(
            inputs=[ones, sum],
            output=div,
            gradients_to_update=[None, None],
            backward_operation=div_backward,
        )
        return div


class Tanh(Activation):
    """
    Tanh activation.
    Feel free to enumerate primitive operations here (you may find it helpful).
    Feel free to refer back to your implementation from part 1 of the HW for equations.
    """

    def __init__(self, autograd_engine):
        super(Tanh, self).__init__(autograd_engine)

    def forward(self, x):

        # TODO Compute forward with primitive operations
        # TODO Add operations to the autograd engine as you go
        a = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        self.autograd_engine.add_operation(
            inputs=[x],
            output=a,
            gradients_to_update=[None],
            backward_operation=tanh_backward,
        )
        return np.tanh(x)


class ReLU(Activation):
    """
    ReLU activation.
    Feel free to enumerate primitive operations here (you may find it helpful).
    Feel free to refer back to your implementation from part 1 of the HW for equations.
    """

    def __init__(self, autograd_engine):
        super(ReLU, self).__init__(autograd_engine)

    def forward(self, x):

        # TODO Compute forward with primitive operations
        # TODO Add operations to the autograd engine as you go
        max_x = np.maximum(x, 0)

        self.autograd_engine.add_operation(
            inputs=[x],
            output=max_x,
            gradients_to_update=[None],
            backward_operation=max_backward,
        )
        return max_x
