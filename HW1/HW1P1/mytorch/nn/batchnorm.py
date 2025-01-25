import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        """
        self.Z = Z
        self.N = Z.shape[0]  # TODO
        self.C = Z.shape[1]
        self.M = np.mean(self.Z, axis=0).reshape (1, self.C) # TODO
        self.V = (np.std(self.Z, axis=0)**2).reshape(1, self.C)  # TODO

        if eval == False:
            # training mode
            self.NZ = (self.Z - self.M)/(self.V + self.eps)**0.5  # TODO
            self.BZ = self.BW * self.NZ + self.Bb  # TODO

            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.NZ # TODO
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.BZ  # TODO
        else:
            # inference mode
            self.NZ = (self.Z - self.running_M)/(self.running_V + self.eps)**0.5  # TODO
            self.BZ = self.BW * self.NZ + self.Bb  # TODO

        return self.BZ

    def backward(self, dLdBZ):

        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0)  # TODO
        self.dLdBb = np.sum(dLdBZ, axis=0)  # TODO

        dLdNZ = dLdBZ * self.BW  # TODO
        dLdV = -0.5 * np.sum((dLdNZ * (self.Z-self.M) * ((self.V + self.eps)**(-3/2))), axis=0)  # TODO

        # dNZdM = -((self.V + self.eps)**(-0.5)) - 0.5*(self.Z - self.M ) * (self.V + self.eps)**(-3/2) \
            # @ ((-2/self.N) * np.sum(self.Z - self.M, axis=0).reshape(1, self.C)).T # TODO

        # dLdM = np.sum(dLdNZ * dNZdM)

        dLdM = -np.sum(dLdNZ *(self.V + self.eps)**(-0.5), axis=0) \
            + (-2*(1/self.N) * np.sum(self.Z - self.M, axis=0).reshape(1, self.C))

        
        dLdZ = dLdNZ * ((self.V + self.eps)**(-0.5)) + dLdV*((2/self.N)*(self.Z - self.M)) \
            + ((1/self.N)*dLdM)   # TODO

        return dLdZ
