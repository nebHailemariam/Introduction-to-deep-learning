import torch


class Softmax:
    """
    DO NOT MODIFY! AN INSTANCE IS ALREADY SET IN THE Attention CLASS' CONSTRUCTOR. USE IT!
    Performs softmax along the last dimension
    """

    def forward(self, Z):

        z_original_shape = Z.shape

        self.N = Z.shape[0] * Z.shape[1]
        self.C = Z.shape[2]
        Z = Z.reshape(self.N, self.C)

        Ones_C = torch.ones((self.C, 1))
        self.A = torch.exp(Z) / (torch.exp(Z) @ Ones_C)

        return self.A.reshape(z_original_shape)

    def backward(self, dLdA):

        dLdA_original_shape = dLdA.shape

        dLdA = dLdA.reshape(self.N, self.C)

        dLdZ = torch.zeros((self.N, self.C))

        for i in range(self.N):

            J = torch.zeros((self.C, self.C))

            for m in range(self.C):
                for n in range(self.C):
                    if n == m:
                        J[m, n] = self.A[i][m] * (1 - self.A[i][m])
                    else:
                        J[m, n] = -self.A[i][m] * self.A[i][n]

            dLdZ[i, :] = dLdA[i, :] @ J

        return dLdZ.reshape(dLdA_original_shape)


class Attention:

    def __init__(self, weights_keys, weights_queries, weights_values):
        """
        Initialize instance variables. Refer to writeup for notation.
        input_dim = D, key_dim = query_dim = D_k, value_dim = D_v

        Argument(s)
        -----------

        weights_keys (torch.tensor, dim = (D X D_k)): weight matrix for keys
        weights_queries (torch.tensor, dim = (D X D_k)): weight matrix for queries
        weights_values (torch.tensor, dim = (D X D_v)): weight matrix for values

        """

        # Store the given weights as parameters of the class.
        self.D_K = weights_keys.shape[1]
        self.W_k = weights_keys  # TODO
        self.W_q = weights_queries  # TODO
        self.W_v = weights_values  # TODO

        # Use this object to perform softmax related operations.
        # It performs softmax over the last dimension which is what you'll need.
        self.softmax = Softmax()

    def forward(self, X):
        """
        Compute outputs of the self-attention layer.
        Stores keys, queries, values, raw and normalized attention weights.
        Refer to writeup for notation.
        batch_size = B, seq_len = T, input_dim = D, value_dim = D_v

        Note that input to this method is a batch not a single sequence, so doing a transpose using .T can yield unexpected results.
        You should permute only the required axes.

        Input
        -----
        X (torch.tensor, dim = (B, T, D)): Input batch

        Return
        ------
        X_new (torch.tensor, dim = (B, T, D_v)): Output batch

        """

        self.X = X

        # Compute the values of Key, Query and Value

        self.Q = torch.matmul(X, self.W_q)  # TODO
        self.K = torch.matmul(X, self.W_k)  # TODO
        self.V = torch.matmul(X, self.W_v)  # TODO

        # Calculate unormalized Attention Scores (logits)

        self.A_w = torch.matmul(self.Q, self.K.transpose(1, 2))  # TODO

        # Create additive causal attention mask and apply mask
        # Hint: Look into torch.tril/torch.triu and account for batch dimension

        attn_mask = torch.tril(self.A_w)  # TODO
        attn_mask[attn_mask == 0] = -float("inf")
        attn_mask[attn_mask != -float("inf")] = 0

        # Calculate/normalize Attention Scores

        self.A_sig = self.softmax.forward(
            (attn_mask + self.A_w) / torch.sqrt(torch.tensor(self.D_K))
        )  # TODO

        # Calculate Attention context

        X_new = torch.matmul(self.A_sig, self.V)  # TODO

        return X_new

    def backward(self, dLdXnew):
        """
        Backpropogate derivatives through the self-attention layer.
        Stores derivatives wrt keys, queries, values, and weight matrices.
        Refer to writeup for notation.
        batch_size = B, seq_len = T, input_dim = D, value_dim = D_v

        Note that input to this method is a batch not a single sequence, so doing a transpose using .T can yield unexpected results.
        You should permute only the required axes.

        Input
        -----
        dLdXnew (torch.tensor, dim = (B, T, D_v)): Derivative of the divergence wrt attention layer outputs

        Return
        ------
        dLdX (torch.tensor, dim = (B, T, D)): Derivative of the divergence wrt attention layer inputs

        """

        # Derivatives wrt attention weights (raw and normalized)

        dLdA_sig       = # TODO
        dLdA_w         = # TODO

        # Derivatives wrt keys, queries, and value

        self.dLdV      = # TODO
        self.dLdK      = # TODO
        self.dLdQ      = # TODO

        # Dervatives wrt weight matrices
        # Remember that you need to sum the derivatives along the batch dimension.

        self.dLdWq     = # TODO
        self.dLdWv     = # TODO
        self.dLdWk     = # TODO

        # Derivative wrt input

        dLdX      = # TODO

        return dLdX
