import numpy as np

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """

        # Get the shapes of the input array
        self.batch_size, self.in_channels, self.in_width = A.shape

        # Reshape A to flatten along the specified dimensions
        Z = A.reshape(self.batch_size, -1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """
        dLdA = dLdZ. reshape(self.batch_size, self.in_channels, self.in_width)

        return dLdA
