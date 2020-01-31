import numpy as np

class HMM:
    def __init__(self, observe_state_num, hidden_state_num, dtype=np.float32):
        # setting
        self.dtype = dtype

        # set number of each states
        self.observe_state_num = observe_state_num
        self.hidden_state_num = hidden_state_num

        # allocate memories of probability model
        self.A = np.ones(shape=[hidden_state_num, hidden_state_num], dtype=dtype)
        self.B = np.ones(shape=[hidden_state_num, observe_state_num], dtype=dtype)
        self.Pi = np.ones(shape=[hidden_state_num], dtype=dtype)

        # initialize probability
        for i in range(hidden_state_num):
            self.A[i, :] /= hidden_state_num
            self.B[i, :] /= observe_state_num
            self.Pi[i] /= hidden_state_num

    def initialize(self, A, B, Pi):
        """ initialize model with given A, B and Pi """
        if self.A.shape != A.shape:
            raise TypeError("Shape of transition probability is not equal!")

        if self.B.shape != B.shape:
            raise TypeError("Shape of observation probability is not equal!")

        if self.Pi.shape != Pi.shape:
            raise TypeError("Shape of initial probability is not equal!")

        self.A = A
        self.B = B
        self.Pi = Pi

    def forward(self, O):
        """ Calculate the probability of observing the observed sequence O using forward algorithm """
        T = len(O)  # length of observation sequence

        # allocate memories
        alpha = np.zeros(shape=[T, self.hidden_state_num], dtype=self.dtype)

        # step 1. initialize alpha

        # step 2. calculate alpha[t, i] inductively

        return alpha

    def backward(self, O):
        """ Calculate the probability of observing the observed sequence O using backward algorithm """
        T = len(O)  # length of observation sequence

        # allocate memories
        beta = np.zeros(shape=[T, self.hidden_state_num], dtype=self.dtype)

        # step 1. initialize alpha

        # step 2. calculate alpha[t, i] inductively

        return beta

    def decode(self, O):
        """ Find 'optimal' hidden state sequence by using Viterbi algorithm """
        T = len(O)  # length of observation sequence

        #
        states = list()

        return states