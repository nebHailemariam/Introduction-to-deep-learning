import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = A.shape[0]  # TODO
        self.C = A.shape[1]  # TODO
        se = (A - Y) * (A - Y)  # TODO
        sse = np.sum(se)  # TODO
        mse = sse/(self.N*self.C)  # TODO

        return mse

    def backward(self):

        dLdA = 2*(self.A - self.Y)/((self.N*self.C))

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0]  # TODO
        C = A.shape[1]  # TODO

        Ones_C = np.ones((C, 1))  # TODO
        Ones_N = np.ones((self.N, 1))  # TODO

        self.softmax = (np.exp(A))/np.sum(np.exp(A), axis=1)[:, np.newaxis]   # TODO
        crossentropy = (-Y * np.log(self.softmax)) @ Ones_C  # TODO
        sum_crossentropy = Ones_N.T @ crossentropy  # TODO
        L = sum_crossentropy / self.N

        return L

    def backward(self):

        dLdA = (self.softmax - self.Y)/self.N  # TODO

        return dLdA
