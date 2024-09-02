import numpy as np
import sys

sys.path.append("./")
from mytorch.autograd_engine import *
from mytorch.functional import *


class CTC(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------

        BLANK (int, optional): blank label index. Default 0.

        """

        # No need to modify
        self.BLANK = BLANK

    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """

        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)

        N = len(extended_symbols)

        # -------------------------------------------->
        # TODO
        # <---------------------------------------------

        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))

        # return extended_symbols, skip_connect
        raise NotImplementedError

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO: Intialize alpha[0][0]
        # TODO: Intialize alpha[0][1]
        # TODO: Compute all values for alpha[t][sym] where 1 <= t < T and 1 <= sym < S (assuming zero-indexing)
        # IMP: Remember to check for skipConnect when calculating alpha
        # <---------------------------------------------

        # return alpha
        raise NotImplementedError

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO
        # <--------------------------------------------

        # return beta
        raise NotImplementedError

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """

        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))

        # -------------------------------------------->
        # TODO
        # <---------------------------------------------

        # return gamma
        raise NotImplementedError


class CTCLoss(object):

    def __init__(self, autograd_engine, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.

        """
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()
        self.autograd_engine = autograd_engine

        self.BLANK = BLANK
        self.ctc = CTC()

        # NOTE: Toggle using ctc_loss_backward version
        # or a version using more primitive operations
        self.USE_PRIMITIVE = True
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):
        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

        Computes the CTC Loss by calculating forward, backward, and
        posterior proabilites, and then calculating the avg. loss between
        targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        # IMP:
        # Output losses should be the mean loss over the batch
        # Dont track div operation to match test values!

        # No need to modify
        B, _ = target.shape
        self.gammas = np.empty(B, dtype=object)
        self.extended_symbols = np.empty(B, dtype=object)

        # NOTE: Arrays with the same views cannot be added to the gradient buffer
        # So, in-place operations are not allowed.
        # Each loss update must be accounted for as a seperate np.array
        # to ensure that there is are no breaks in the computation graph
        # during gradient backpropagation.
        # We will initialize loss as a dict of key'd by batch_size,
        # where each value will be a variable length list of (seq_len + 1) x singleton np.ndarray's
        tmp_loss = {k: [np.array([0.0], dtype=np.float64)] for k in range(B)}

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # NOTE: Remember to add the Slice operation as you index into things.
            # See slice_backward in functional.py and learn about the np.index_exp object
            # which you will use to pass index expressions to the backward function
            # Remember to wrap the np.index_exp object in a np.array(..., dtype=object)
            # <---------------------------------------------
            pass

        """ DO NOT MODIFY """
        # NOTE: tmp_loss[b][-1] should contain the most recently updated
        # loss value for batch b.
        # We will iterate over all batches and update each element of total_loss
        # as a sum of tmp_loss[b][-1] and the previous total_loss elem.
        # This is equivalent to doing np.sum() but preserves the integrity
        # of the computational graph
        total_loss = [np.array([0.0], dtype=np.float64) for _ in range(B + 1)]
        for i in range(B):
            total_loss[i + 1] = total_loss[i] + tmp_loss[i][-1]
            self.autograd_engine.add_operation(
                inputs=[total_loss[i], tmp_loss[i][-1]],
                output=total_loss[i + 1],
                gradients_to_update=[None, None],
                backward_operation=add_backward,
            )

        # NOTE: Dont add div operation to match test values
        t_loss = total_loss[-1] / B
        return t_loss
