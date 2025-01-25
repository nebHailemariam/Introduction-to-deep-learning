# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
            W (np.array): (out_channels, in_channels, kernel_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        batch_size, in_channels, input_size = A.shape
        output_size = (input_size - self.kernel_size) + 1
        Z = np.zeros((batch_size, self.out_channels, output_size), dtype=A.dtype) # TODO  
        # A: batch_size, in_channels, input_size
        # W: out_channels, in_channels, kernel_size
        # Z: batch_size, out_channels, output_size)

        for b in range(batch_size):
            for c in range(self.out_channels):
                for o in range(output_size):
                    filter = self.W[c, :, ]
                    window = self.A[b, :, o:o + self.kernel_size]
                    result = np.sum(np.sum(filter * window, axis=1), axis=0) + self.b[c]
                    Z[b, c, o] = result

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch_size, in_channels, input_size = self.A.shape
        batch_size, out_channels, output_size = dLdZ.shape
        out_channels, in_channels, kernel_size = self.dLdW.shape

        # Calculate gradients with respect to weights (dLdW)
        for o in range(out_channels):
            for i in range(in_channels):
                result = np.zeros((kernel_size), dtype=float)
                for k in range(kernel_size):
                    for b in range(batch_size):
                        window = self.A[b, i, k:k + output_size]
                        filter = dLdZ[b, o, :]
                        result[k] += np.sum(np.sum(window * filter, axis=0),axis=0)

                self.dLdW[o,i,:] += result 

        # Calculate gradient with respect to biases (dLdb)
        self.dLdb = np.sum(np.sum(dLdZ, axis=2), axis=0)  # TODO

        # Calculate gradient of loss with respect to input (dLdA)

        W_flipped = np.flip(self.W, axis=2)
         # Pad the gradient of loss with respect to Z (dLdZ)
        padded_dLdZ = np.pad(dLdZ, ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1)), mode='constant', constant_values=0)
        dLdA = np.zeros((batch_size, in_channels, input_size), dtype=dLdZ.dtype)  # TODO
        padded_dLdZ_output_size = padded_dLdZ.shape[2]
        padded_dLdZ_number_of_moves = (padded_dLdZ_output_size - self.kernel_size) + 1
        
        # Calculate gradient of loss with respect to input (dLdA)
        for b in range(batch_size):
            for c in range(in_channels):
                results = np.zeros((padded_dLdZ_number_of_moves), dtype=float)
                for s in range(padded_dLdZ_number_of_moves):
                    window = padded_dLdZ[b, :, s:s + self.kernel_size]
                    flipped_filter = W_flipped[:, c, :]
                    results[s] = np.sum(np.sum(window * flipped_filter, axis=0), axis=0)
                dLdA[b, c, :] = results

        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride
        self.padding = padding

        # Initialize Conv1d_stride1() and Downsample1d() instance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn) # TODO
        self.downsample1d = Downsample1d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        
        # Pad the input appropriately using np.pad() function
        # TODO
        padded_A = np.pad(A, ((0, 0), (0, 0), (int(self.padding), int(self.padding))), constant_values=0)
        # Call conv1d_stride1
        # TODO
        conv = self.conv1d_stride1.forward(padded_A)

        # Call downsample1d

        Z = self.downsample1d.forward(conv)  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        # TODO
        dLdA = self.downsample1d.backward(dLdZ)

        # Call conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdA)  # TODO
        
        # Unpad the gradient
        if self.padding != 0:
            dLdA = dLdA[:, :, int(self.padding):-int(self.padding)] # TODO

        return dLdA
