# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class Dropout(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):

        if train:
            # TODO: Generate mask and apply to x

            raise NotImplementedError("Dropout Forward (Train) Not Implemented")
            
        else:
            # TODO: Return x as is

            raise NotImplementedError("Dropout Forward (Inference) Not Implemented")
		
    def backward(self, delta):
        # TODO: Multiply mask with delta and return

        raise NotImplementedError("Dropout Backward Not Implemented")