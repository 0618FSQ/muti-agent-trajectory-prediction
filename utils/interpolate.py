import numpy as np
import scipy.interpolate as spi

DELTA = 1e-10


class Interpolate(object):
    def __init__(self, x, y):
        if np.any(x[1:] == x[:-1]):
            x[np.where(x[1:] == x[:-1])] = x[np.where(x[1:] == x[:-1])]+DELTA
        self.ipo = spi.interp1d(x, y, kind='quadratic')

    def __call__(self, x):
        iy = self.ipo(x)
        return iy

