import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features, in_features ))
        self.b = np.ones((out_features, 1))  # TODO

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = A  # TODO
        self.N = self.A.shape[0]  # TODO store the batch size of input
        # Think how will self.Ones helps in the calculations and uncomment below
        self.Ones = np.ones((self.N, 1))
        Z = self.A @ self.W.T + (np.tile(self.b.T, self.Ones))

        return np.squeeze(Z)

    def backward(self, dLdZ):

        dLdA = dLdZ @ self.W  # TODO
        self.dLdW = dLdZ.T @ self.A  # TODO
        self.dLdb = dLdZ.T @ np.ones((dLdZ.T.shape[1], 1))  # TODO

        if self.debug:
            
            self.dLdA = dLdA

        return np.squeeze(dLdA)
