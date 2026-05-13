import numpy as np

class LowPassFilter(object):
    """First-order discrete LPF: y[k] = β y[k-1] + (1-β) x[k],  β = exp(-2π fc dt).

    The output vector is initialized lazily on the first filter() call to match
    the shape of the input, so a single class works for scalar, ω (3-d), full
    odom (6/13-d), etc.
    """

    def __init__(self, cutoff_freq=10.0):
        self.cutoff_freq = cutoff_freq
        self.output = None    # initialized on first filter() call

    def reset(self, value):
        self.output = np.asarray(value, dtype=float).copy()

    def filter(self, input, dt):
        x = np.asarray(input, dtype=float)
        if self.output is None or self.output.shape != x.shape:
            self.output = x.copy()
            return self.output
        if self.cutoff_freq <= 0.0:
            self.output = x.copy()    # bypass: cutoff disabled
            return self.output
        beta = np.exp(-2.0 * np.pi * self.cutoff_freq * dt)
        self.output = beta * self.output + (1.0 - beta) * x
        return self.output