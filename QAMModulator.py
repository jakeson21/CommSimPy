import numpy as np
import utils

__author__ = 'Fuguru'
__all__ = ['QAMModulator']

class QAMModulator():
    """

    """

    def __init__(self, c, sps=1):
        """

        :param c: Constellation data (numpy.array) or filename (string)
        :param sps: output sample rate
        :return:
        """
        # test if c is numpy array, otherwise treat as filename
        if type(c).__module__ == np.__name__:
            self.constellation = c
        else:
            self.constellation = utils.read_point_data_file(filename=c)
        c_shape = self.constellation.shape
        # Check dimension, enforce shape = (M,1)
        if len(c_shape) == 2 and c_shape[1] == 1:
            pass
        elif len(c_shape) == 1:
            self.constellation = np.reshape(self.constellation, newshape=(np.product(c_shape), 1))
        elif len(c_shape) == 2 and not c_shape[1] == 1:
            self.constellation = np.reshape(self.constellation, newshape=(np.product(c_shape), 1))
        elif len(c_shape) > 2:
            print "Dimensions of constellation are not (Mx1)"
            raise ValueError
        else:
            print "Dimensions of constellation are not (Mx1)"
            raise ValueError

        self.order = self.constellation.size
        self.sample_rate = sps

    def execute(self, symbols):
        """execute()

        :param symbols: input symbols to modulate
        :return: modulated impulses, up sampled to sps
        """
        s = self.__get_next(symbols)
        x = np.hstack((s, np.zeros((s.size, self.sample_rate-1), dtype=complex)))
        x = np.reshape(x, (x.size, 1))
        return x

    def __get_next(self, symbols):
        if type(symbols).__module__ == np.__name__:
            out = self.constellation[symbols[:, 0]]
        else:
            out = self.constellation[np.array(symbols)]

        return out
