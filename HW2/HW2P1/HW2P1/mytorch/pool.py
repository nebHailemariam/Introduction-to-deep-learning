import numpy as np
# from HW2.HW2P1.HW2P1.mytorch.resampling import *
from resampling import *

class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel
        self.A = None

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1
        Z = np.zeros((batch_size, in_channels, output_width, output_height ))


        for b in range(batch_size):
            for c in range(in_channels):
                for o in range(output_width):
                    for h in range(output_height):
                        Z[b,c,o,h] = np.max(A[b,c, o:o+self.kernel, h:h+self.kernel]) 

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, out_channels, output_width, output_height = dLdZ.shape
        batch_size, in_channels, input_width, input_height = self.A.shape
        
        dLdA = np.zeros((batch_size, in_channels, input_width, input_height ))
        # print("dLdA: ",dLdA.shape)
        # print("dLdZ: ",dLdZ.shape)

        for b in range(batch_size):
            for c in range(out_channels):
                for o in range(output_width):
                    for h in range(output_height):
                        x, y = np.unravel_index(np.argmax(self.A[b,c, o:o+self.kernel, h:h+self.kernel]), self.A[b,c, o:o+self.kernel, h:h+self.kernel].shape)
                        # print(f"X: {x}, Y: {y} value:", self.A[b, c, x,y])
                        dLdA[b, c, o:o+self.kernel, h:h+self.kernel][x, y] += dLdZ[b,c,o,h]


        return dLdA


class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1
        Z = np.zeros((batch_size, in_channels, output_width, output_height ))


        for b in range(batch_size):
            for c in range(in_channels):
                for o in range(output_width):
                    for h in range(output_height):
                        # print(A[b,c, o:o+output_width, h:h+output_height])
                        Z[b,c,o,h] = np.mean(A[b,c, o:o+self.kernel, h:h+self.kernel]) 

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, out_channels, output_width, output_height = dLdZ.shape
        batch_size, in_channels, input_width, input_height = self.A.shape
        
        dLdA = np.zeros((batch_size, in_channels, input_width, input_height ))
        # print("dLdA: ",dLdA.shape)
        # print("dLdZ: ",dLdZ.shape)

        for b in range(batch_size):
            for c in range(out_channels):
                for o in range(output_width):
                    for h in range(output_height):
                        dLdA[b, c, o:o+self.kernel, h:h+self.kernel] += dLdZ[b,c,o,h]/(self.kernel**2)


        return dLdA


class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        pool = self.maxpool2d_stride1.forward(A)

        Z = self.downsample2d.forward(pool)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.downsample2d.backward(dLdZ)

        dLdA = self.maxpool2d_stride1.backward(dLdA)

        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        pooled_A = self.meanpool2d_stride1.forward(A)

        Z = self.downsample2d.forward(pooled_A)

        return Z
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = self.downsample2d.backward(dLdZ)

        dLdA = self.meanpool2d_stride1.backward(dLdA)

        return dLdA
