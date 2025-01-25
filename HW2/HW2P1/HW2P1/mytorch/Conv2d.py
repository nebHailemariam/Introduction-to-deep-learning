import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batch_size, in_channels, input_height, input_width = A.shape
        output_height_size = (input_height - self.kernel_size) + 1
        output_width_size = (input_width - self.kernel_size) + 1

        Z = np.zeros((batch_size, self.out_channels, output_height_size, output_width_size), dtype=A.dtype)   # TODO
        
        
        for b in range(batch_size):
            for c in range(self.out_channels):
                for o_h in range(output_height_size):
                    for o_w in range(output_width_size):
                        filter = self.W[c, :, ]
                        window = self.A[b, :, o_h:o_h + self.kernel_size, o_w:o_w + self.kernel_size]
                        # print("Filter:", filter.shape)
                        # print("window:", window.shape)
                        result = np.sum(np.sum(np.sum(filter * window, axis=1), axis=0)) + self.b[c]
                        Z[b, c, o_h, o_w] = result
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        batch_size, in_channels, input_height_size, input_width_size = self.A.shape
        batch_size, out_channels, output_height_size, output_width_size = dLdZ.shape
        out_channels, in_channels, kernel_size, kernel_size = self.dLdW.shape
        
        # Calculate gradients with respect to weights (dLdW)
        for o in range(out_channels):
            for i in range(in_channels):
                result = np.zeros((kernel_size, kernel_size), dtype=float)
                for k_1 in range(kernel_size):
                    for k_2 in range(kernel_size):
                        for b in range(batch_size):
                            window = self.A[b, i, k_1:k_1 + output_height_size, k_2:k_2 + output_width_size]
                            filter = dLdZ[b, o, :]
                            result[k_1, k_2] += np.sum(np.sum(window * filter, axis=0),axis=0)

                self.dLdW[o,i,:] += result 

        # Calculate gradient with respect to biases (dLdb)
        self.dLdb = np.sum(np.sum(np.sum(dLdZ, axis=3), axis=2), axis=0)  # TODO

        # Calculate gradient of loss with respect to input (dLdA)
        # W_flipped = np.flip(self.W, axis=3)
         # Pad the gradient of loss with respect to Z (dLdZ)

        dLdA = np.zeros((batch_size, in_channels, input_height_size, input_width_size), dtype=dLdZ.dtype)   # TODO  # TODO
        
        padded_dLdZ = np.pad(dLdZ, ((0, 0), (0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1)), constant_values=0)
        padded_dLdZ = np.pad(padded_dLdZ, ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1), (0, 0)), constant_values=0)
        
        padded_dLdZ_output_height_size = padded_dLdZ.shape[2]
        padded_dLdZ_number_of_moves_height = (padded_dLdZ_output_height_size - self.kernel_size) + 1
        padded_dLdZ_output_width_size = padded_dLdZ.shape[3]
        padded_dLdZ_number_of_moves_width = (padded_dLdZ_output_width_size - self.kernel_size) + 1
        
        # Calculate gradient of loss with respect to input (dLdA)
        for b in range(batch_size):
            for c in range(in_channels):
                results = np.zeros((padded_dLdZ_number_of_moves_height, padded_dLdZ_number_of_moves_width), dtype=float)
                for s_1 in range(padded_dLdZ_number_of_moves_height):
                    for s_2 in range(padded_dLdZ_number_of_moves_width):
                        window = padded_dLdZ[b, :, s_1:s_1 + self.kernel_size, s_2:s_2 + self.kernel_size]
                        flipped_filter = np.flip(np.flip(self.W[:, c, :], axis=2), axis=1)
                        results[s_1, s_2] = np.sum(np.sum(np.sum(window * flipped_filter, axis=0), axis=0))
                dLdA[b, c, :] = results
        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.padding = padding

        # Initialize Conv2d_stride1() and Downsample2d() instance
        self.conv2d_stride1 = Conv2d_stride1(in_channels,out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        
        # Pad the input appropriately using np.pad() function
        # TODO
        padded_A = np.pad(A, ((0, 0), (0, 0), (0, 0), (self.padding, self.padding)), constant_values=0)
        padded_A = np.pad(padded_A, ((0, 0), (0, 0), (self.padding, self.padding), (0, 0)), constant_values=0)
        
        # Call conv2d_stride1
        # TODO
        conv = self.conv2d_stride1.forward(padded_A)

        # Call downsample2d
        Z = self.downsample2d.forward(conv)   # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample2d backward
        # TODO
        dLdA = self.downsample2d.backward(dLdZ)

        # Call conv2d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdA)  # TODO
        
        
        # Unpad the gradient
        dLdA = dLdA[:, :, int(self.padding):-int(self.padding), int(self.padding):-int(self.padding)] # TODO

        return dLdA
