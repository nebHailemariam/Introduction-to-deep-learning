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

        self.state = x

        raise NotImplementedError


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

        raise NotImplementedError


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
        
        raise NotImplementedError


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
        
        raise NotImplementedError
