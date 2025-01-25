import numpy as np


class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        batch_size, in_channels, input_width = A.shape
        output_width = self.upsampling_factor * (input_width - 1) + 1
        # Todo
        Z = np.zeros((batch_size, in_channels, output_width), dtype=A.dtype) 

        # Iterate over the input array A and fill the upsampled array Z
        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(input_width):
                    Z[b, c, i * self.upsampling_factor] = A[b, c, i]


        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        batch_size, in_channels, out_width = dLdZ.shape
        input_width = int(((out_width - 1) / self.upsampling_factor) + 1)
        dLdA =  np.zeros((batch_size, in_channels, input_width), dtype=dLdZ.dtype)   # TODO

        # Iterate over the output array dLdZ and fill the input array dLdA
        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(input_width):
                    dLdA[b, c, i] = dLdZ[b, c, i * self.upsampling_factor]

        return dLdA


class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.input_width = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        batch_size, in_channels, input_width = A.shape
        self.input_width = input_width
        output_width = int((((input_width - 1) / self.downsampling_factor)) + 1)
        Z =  np.zeros((batch_size, in_channels, output_width), dtype=A.dtype)   # TODO

        # Iterate over the output array dLdZ and fill the input array dLdA
        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(output_width):
                    Z[b, c, i] = A[b, c, i * self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        batch_size, in_channels, output_width = dLdZ.shape
        # Todo
        dLdA = np.zeros((batch_size, in_channels, self.input_width), dtype=dLdZ.dtype) # TODO

        # Iterate over the input array A and fill the upsampled array Z
        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(output_width):
                    dLdA[b, c, i * self.downsampling_factor] = dLdZ[b, c, i]
        return dLdA


class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        batch_size, in_channels, input_height, input_width = A.shape

        output_height = self.upsampling_factor * (input_height - 1) + 1
        output_width = self.upsampling_factor * (input_width - 1) + 1
        # Todo
        Z = np.zeros((batch_size, in_channels, output_height, output_width), dtype=A.dtype) 

        # Iterate over the input array A and fill the upsampled array Z
        for b in range(batch_size):
            for c in range(in_channels):
                for h in range(input_width):
                    for i in range(input_width):
                        Z[b, c, h * self.upsampling_factor, i * self.upsampling_factor] = A[b, c, h, i]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        batch_size, in_channels, output_height, out_width = dLdZ.shape
        input_height = int(((output_height - 1) / self.upsampling_factor) + 1)
        input_width = int(((out_width - 1) / self.upsampling_factor) + 1)

        dLdA =  np.zeros((batch_size, in_channels, input_height, input_width), dtype=dLdZ.dtype)   # TODO

        # Iterate over the output array dLdZ and fill the input array dLdA
        for b in range(batch_size):
            for c in range(in_channels):
                for h in range(input_height):
                    for i in range(input_width):
                        dLdA[b, c, h, i] = dLdZ[b, c, h * self.upsampling_factor, i * self.upsampling_factor]
        return dLdA


class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor
        self.input_height = None
        self.input_width = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """

        batch_size, in_channels, input_height, input_width = A.shape
        self.input_height = input_height
        self.input_width = input_width
        output_height = int((((input_height - 1) / self.downsampling_factor)) + 1)
        output_width = int((((input_width - 1) / self.downsampling_factor)) + 1)
        Z =  np.zeros((batch_size, in_channels, output_height, output_width), dtype=A.dtype)   # TODO

        # Iterate over the output array dLdZ and fill the input array dLdA
        for b in range(batch_size):
            for c in range(in_channels):
                for h in range(output_height):
                    for i in range(output_width):
                        Z[b, c, h, i] = A[b, c, h * self.downsampling_factor, i * self.downsampling_factor]

        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        batch_size, in_channels, output_height, output_width = dLdZ.shape
        dLdA = np.zeros((batch_size, in_channels, self.input_height, self.input_width), dtype=dLdZ.dtype) # TODO

        # Iterate over the input array A and fill the upsampled array Z
        for b in range(batch_size):
            for c in range(in_channels):
                for h in range(output_height):
                    for i in range(output_width):
                        dLdA[b, c, h * self.downsampling_factor, i * self.downsampling_factor] = dLdZ[b, c, h, i]
        return dLdA
