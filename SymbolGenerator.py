import numpy as _np

__author__ = 'Fuguru'
__all__ = ['SymbolGenerator']

class SymbolGenerator():
    """Symbol Generator class

    """

    def __init__(self, m=2, size=(1, 1), seed=1):
        self.order = m
        self.block_size = size
        self.seed = seed
        self.last = []
        _np.random.seed(seed)
        self._state = _np.random.get_state()

    def execute(self):
        return self.__get_next(self.block_size)

    def __get_next(self, n=(1, 1)):
        """
        get_next(n)
        :param n: number of symbols to generate
        :return: nx1 numpy.array of random symbols
        """
        _np.random.set_state(self._state)
        s = _np.random.randint(low=0, high=self.order, size=n)
        self.last = s
        self._state = _np.random.get_state()
        return s

