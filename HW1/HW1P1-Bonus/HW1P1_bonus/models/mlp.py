import numpy as np

from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU


class MLP0:

    def __init__(self, debug=False):
        """
        Initialize a single linear layer of shape (2,3).
        Use Relu activations for the layer.
        """

        self.layers = [Linear(2, 3), ReLU()]
        self.debug = debug

    def forward(self, A0):
        """
        Pass the input through the linear layer followed by the activation layer to get the model output.
        """

        Z0 = None  # TODO
        A1 = None  # TODO

        if self.debug:

            self.Z0 = Z0
            self.A1 = A1

        return NotImplemented

    def backward(self, dLdA1):
        """
        Refer to the pseudo code outlined in the writeup to implement backpropogation through the model.
        """

        dLdZ0 = None  # TODO
        dLdA0 = None  # TODO

        if self.debug:

            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return NotImplemented


class MLP1:

    def __init__(self, debug=False):
        """
        Initialize 2 linear layers. Layer 1 of shape (2,3) and Layer 2 of shape (3, 2).
        Use Relu activations for both the layers.
        Implement it on the same lines(in a list) as MLP0
        """

        self.layers = None  # TODO
        self.debug = debug

    def forward(self, A0):
        """
        Pass the input through the linear layers and corresponding activation layer alternately to get the model output.
        """

        Z0 = None  # TODO
        A1 = None  # TODO

        Z1 = None  # TODO
        A2 = None  # TODO

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1
            self.Z1 = Z1
            self.A2 = A2

        return NotImplemented

    def backward(self, dLdA2):
        """
        Refer to the pseudo code outlined in the writeup to implement backpropogation through the model.
        """

        dLdZ1 = None  # TODO
        dLdA1 = None  # TODO

        dLdZ0 = None  # TODO
        dLdA0 = None  # TODO

        if self.debug:

            self.dLdZ1 = dLdZ1
            self.dLdA1 = dLdA1

            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return NotImplemented


class MLP4:
    def __init__(self, debug=False):
        """
        Initialize 4 hidden layers and an output layer of shape below:
        Layer1 (2, 4),
        Layer2 (4, 8),
        Layer3 (8, 8),
        Layer4 (8, 4),
        Output Layer (4, 2)

        Refer the diagramatic view in the writeup for better understanding.
        Use ReLU activation function for all the linear layers.)
        """

        # List of Hidden and activation Layers in the correct order
        self.layers = None  # TODO

        self.debug = debug

    def forward(self, A):
        """
        Pass the input through the linear layers and corresponding activation layer alternately to get the model output.
        """

        if self.debug:

            self.A = [A]

        L = len(self.layers)

        for i in range(L):

            A = None  # TODO

            if self.debug:

                self.A.append(A)

        return NotImplemented

    def backward(self, dLdA):
        """
        Refer to the pseudo code outlined in the writeup to implement backpropogation through the model.
        """

        if self.debug:

            self.dLdA = [dLdA]

        L = len(self.layers)

        for i in reversed(range(L)):

            dLdA = None  # TODO

            if self.debug:

                self.dLdA = [dLdA] + self.dLdA

        return NotImplemented

