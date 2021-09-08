import numpy as np
import utils
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel
from pyfft.cl import Plan
import numexpr as ne

__author__ = 'Fuguru'
__all__ = ['point_data_filter']


class PointDataFilter:
    """
    Point data filter class
    Init
    PointDataFilter(filename, fs=1, nfft=4096, fsf=1, gdos=0)
    """

    def __init__(self, filename, fs=1, nfft=4096, fsf=1, gdos=0, memory=0, use_cl=False):
        if type(filename).__module__ == np.__name__:
            h = filename
            self.filename = ''
        else:
            # assume filename is text, read in point data file
            self.filename = filename
            h = utils.read_point_data_file(filename=self.filename)
        self.filtered = []
        self.fs = fs
        self.nfft = nfft
        self.fsf = fsf
        self.group_delay = gdos
        self.memory = memory
        # interpolate to desired frequency range and resolution
        self.h_imp, self.h_fgp = utils.fgp_to_impulse_response(h, fs=self.fs, length=self.nfft,
                                                               gdelay=self.group_delay*self.fs)

        self.H = np.fft.fft(self.h_imp, n=self.nfft, axis=0)
        self._buffer = np.zeros((self.nfft, 1), dtype='complex')
        self.block_size = self.nfft - np.ceil(self.memory*self.fs)

        self.use_cl = use_cl
        self.context = []
        if self.use_cl:
            platforms = cl.get_platforms()
            for found_platform in platforms:
                if found_platform.name == 'AMD Accelerated Parallel Processing':
                    platform = found_platform
            for found_device in platform.get_devices():
                if cl.device_type.to_string(found_device.type) == 'GPU':
                    device = found_device
                    self.context = cl.Context([device])
                    self.complex_prod = ElementwiseKernel(self.context,
                        "float2 *x, float2 *y, float2 *z",
                        "z[i] = complex_mul(x[i], y[i])",
                        "complex_prod",
                        preamble="""
                        #define complex_ctr(x, y) (float2)(x, y)
                        #define complex_mul(a, b) complex_ctr(mad(-(a).y, (b).y, (a).x * (b).x), mad((a).y, (b).x, (a).x * (b).y))
                        #define complex_div_scalar(a, b) complex_ctr((a).x / (b), (a).y / (b))
                        #define conj(a) complex_ctr((a).x, -(a).y)
                        #define conj_transp(a) complex_ctr(-(a).y, (a).x)
                        #define conj_transp_and_mul(a, b) complex_ctr(-(a).y * (b), (a).x * (b))
                        """)


    def execute(self, x):
        if self.use_cl:
            return self.filter_cl(x)
        else:
            return self.filter(x)

    def filter(self, x):
        """
        .filter(x) filters x by h_fgp using overlap-and-save method
        :param x:
        :return: filtered signal
        """
        carry = self.nfft - self.block_size
        if not x.size == self.block_size:
            print "Input dimensions are incorrect"
            raise
        # put input signal into end of buffer
        self._buffer[carry:] = x
        x_fft = np.fft.fft(self._buffer, n=self.nfft, axis=0)
        x_fft = x_fft*self.H
        y = np.fft.ifft(x_fft, n=self.nfft, axis=0)
        # put tail end of filtered signal into front of buffer
        self._buffer[0:carry] = x[-carry:]
        y = y[carry:]
        return y

    def filter_cl(self, x):
        """
        .filter_cl(x) filters x by h_fgp using overlap-and-save method
        :param x:
        :return: filtered signal
        """
        carry = self.nfft - self.block_size
        if not x.size == self.block_size:
            print "Input dimensions are incorrect"
            raise

        # ctx = cl.create_some_context(interactive=False)
        queue = cl.CommandQueue(self.context)
        plan = Plan(self.nfft, queue=queue)
        # put input signal into end of buffer
        self._buffer[carry:] = x
        gpu_data = cl_array.to_device(queue, self._buffer.astype(dtype=np.complex64))
        # Perform fft
        plan.execute(gpu_data.data)
        # Perform filtering with element-wise product
        gpu_H = cl_array.to_device(queue, self.H.astype(dtype=np.complex64))
        x_filt = cl_array.empty_like(gpu_data)
        self.complex_prod(gpu_data, gpu_H, x_filt)
        # Perform ifft
        plan.execute(x_filt.data, inverse=True)
        y = x_filt.get()
        # put tail end of filtered signal into front of buffer
        self._buffer[0:carry] = x[-carry:]
        y = y[carry:]
        return y
