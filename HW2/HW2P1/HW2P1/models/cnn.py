# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)​

from flatten import *
from Conv1d import *
from linear import *
from activation import *
from loss import *
import numpy as np
import os
import sys

sys.path.append('mytorch')


class CNN(object):

    """
    A simple convolutional neural network
    """

    def __init__(self, input_width, num_input_channels, num_channels, kernel_sizes, strides,
                 num_linear_neurons, activations, conv_weight_init_fn, bias_init_fn,
                 linear_weight_init_fn, criterion, lr):
        """
        input_width           : int    : The width of the input to the first convolutional layer
        num_input_channels    : int    : Number of channels for the input layer
        num_channels          : [int]  : List containing number of (output) channels for each conv layer
        kernel_sizes          : [int]  : List containing kernel width for each conv layer
        strides               : [int]  : List containing stride size for each conv layer
        num_linear_neurons    : int    : Number of neurons in the linear layer
        activations           : [obj]  : List of objects corresponding to the activation fn for each conv layer
        conv_weight_init_fn   : fn     : Function to init each conv layers weights
        bias_init_fn          : fn     : Function to initialize each conv layers AND the linear layers bias to 0
        linear_weight_init_fn : fn     : Function to initialize the linear layers weights
        criterion             : obj    : Object to the criterion (SoftMaxCrossEntropy) to be used
        lr                    : float  : The learning rate for the class

        You can be sure that len(activations) == len(num_channels) == len(kernel_sizes) == len(strides)
        """

        # Don't change this -->
        self.train_mode = True
        self.nlayers = len(num_channels)

        self.activations = activations
        self.criterion = criterion

        self.lr = lr
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Your code goes here -->
        # self.convolutional_layers (list Conv1d) = []
        # self.flatten              (Flatten)     = Flatten()
        # self.linear_layer         (Linear)      = Linear(???)
        
        # Initialize the convolutional layers
        self.convolutional_layers = []    # TODO        
        # <---------------------

        out_channel = num_input_channels
        input_size = input_width

        for i in range(self.nlayers):
            self.convolutional_layers.append(
                Conv1d(
                    in_channels=out_channel,
                    out_channels=num_channels[i],
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    weight_init_fn=conv_weight_init_fn,
                    bias_init_fn=bias_init_fn,
                )
            )
            out_channel = num_channels[i]
            input_size = (input_size - kernel_sizes[i]) // strides[i] + 1

        # Initialize the flatten layer
        self.flatten = Flatten() # TODO

        # Initialize the linear layer
        self.linear_layer = Linear(
            in_features=out_channel * input_size, out_features=num_linear_neurons
        )

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, num_input_channels, input_width)
        Return:
            Z (np.array): (batch_size, num_linear_neurons)
        """

        # Your code goes here -->
        # Iterate through each layer
        # <---------------------
        for i in range(self.nlayers):
            A = self.activations[i].forward(self.convolutional_layers[i].forward(A))
        
        # Flatten the output
        A = self.flatten.forward(A)
        
        # Forward pass through the linear layer
        self.Z = self.linear_layer.forward(A)

        # Reshape the output
        # Save output (necessary for error and loss)
        self.Z = self.Z.reshape(1, -1)
        
        return self.Z

    def backward(self, labels):
        """
        Argument:
            labels (np.array): (batch_size, num_linear_neurons)
        Return:
            grad (np.array): (batch size, num_input_channels, input_width)
        """
        # Compute the loss and gradient using the specified criterion
        self.loss = self.criterion.forward(self.Z, labels).sum()
        grad = self.criterion.backward()

        # Backward pass through layers in reverse order
        grad = self.linear_layer.backward(grad)
        grad = self.flatten.backward(grad)
        
        for conv, activation in zip(self.convolutional_layers[::-1], self.activations[::-1]):
            grad = conv.backward(activation.backward(grad))
        
        return grad

    def zero_grads(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].conv1d_stride1.dLdW.fill(0.0)
            self.convolutional_layers[i].conv1d_stride1.dLdb.fill(0.0)

        self.linear_layer.dLdW = np.zeros(self.linear_layer.W.shape)
        self.linear_layer.dLdb = np.zeros(self.linear_layer.b.shape)

    def step(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].conv1d_stride1.W = (self.convolutional_layers[i].conv1d_stride1.W -
                                                             self.lr * self.convolutional_layers[i].conv1d_stride1.dLdW)
            self.convolutional_layers[i].conv1d_stride1.b = (self.convolutional_layers[i].conv1d_stride1.b -
                                                             self.lr * self.convolutional_layers[i].conv1d_stride1.dLdb)

        self.linear_layer.W = (
            self.linear_layer.W -
            self.lr *
            self.linear_layer.dLdW)
        self.linear_layer.b = (
            self.linear_layer.b -
            self.lr *
            self.linear_layer.dLdb)

    def train(self):
        # Do not modify this method
        self.train_mode = True

    def eval(self):
        # Do not modify this method
        self.train_mode = False
