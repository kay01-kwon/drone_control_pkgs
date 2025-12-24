import numpy as np

class LowPassFilter(object):

    def __init__(self, cutoff_freq=10.0):
        self.cutoff_freq = cutoff_freq
        self.output = np.zeros((6,))

    def filter(self, input, dt):
        beta = np.exp(-2.0*np.pi*self.cutoff_freq * dt)
        self.output = beta*self.output + (1-beta)*input
        return self.output