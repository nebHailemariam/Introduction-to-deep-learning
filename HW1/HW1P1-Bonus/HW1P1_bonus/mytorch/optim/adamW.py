# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class AdamW:
    def __init__(self, model, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.l = model.layers[::2]  # every second layer is activation function
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.lr = lr
        self.t = 0
        self.weight_decay = weight_decay

        self.m_W = [np.zeros(l.W.shape, dtype="f") for l in self.l]
        self.v_W = [np.zeros(l.W.shape, dtype="f") for l in self.l]

        self.m_b = [np.zeros(l.b.shape, dtype="f") for l in self.l]
        self.v_b = [np.zeros(l.b.shape, dtype="f") for l in self.l]

    def step(self):

        self.t += 1
        for layer_id, layer in enumerate(self.l):

            # TODO: Calculate updates for weight
            dLdW = layer.dLdW
            self.m_W[layer_id] = (
                self.beta1 * self.m_W[layer_id] + (1 - self.beta1) * dLdW
            )

            self.v_W[layer_id] = (
                self.beta2 * self.v_W[layer_id] + (1 - self.beta2) * dLdW**2
            )
            m_w_hat = self.m_W[layer_id] / (1 - self.beta1**self.t)
            v_w_hat = self.v_W[layer_id] / (1 - self.beta2**self.t)

            # TODO: calculate updates for bias
            dLdb = layer.dLdb
            self.m_b[layer_id] = (
                self.beta1 * self.m_b[layer_id] + (1 - self.beta1) * dLdb
            )
            self.v_b[layer_id] = (
                self.beta2 * self.v_b[layer_id] + (1 - self.beta2) * dLdb**2
            )

            m_b_hat = self.m_b[layer_id] / (1 - self.beta1**self.t)
            v_b_hat = self.v_b[layer_id] / (1 - self.beta2**self.t)

            # TODO: Perform weight and bias updates

            layer.W = layer.W - self.lr * self.weight_decay * layer.W
            layer.b = layer.b - self.lr * self.weight_decay * layer.b

            layer.W = layer.W - self.lr * (m_w_hat / (np.sqrt(v_w_hat) + self.eps))
            layer.b = layer.b - self.lr * (m_b_hat / (np.sqrt(v_b_hat) + self.eps))
