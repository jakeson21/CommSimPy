import numpy as np
import utils
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.tools
from pyopencl.elementwise import ElementwiseKernel
import numexpr as ne

__author__ = 'Fuguru'

class AdditiveWhiteGaussianNoise():
    def __init__(self, n=1, sigma=1, use_cl=False):
        self.block_size = n
        self.noise_pwr = sigma ** 2
        self.use_cl = use_cl

        if self.use_cl:

            self.queue = []
            self.context = []
            self.kernels = {}
            if self.use_cl:
                platforms = cl.get_platforms()
                for found_platform in platforms:
                    if found_platform.name == 'AMD Accelerated Parallel Processing':
                        platform = found_platform
                for found_device in platform.get_devices():
                    if cl.device_type.to_string(found_device.type) == 'GPU':
                        device = found_device
                        self.context = cl.Context([device])

                        self.kernels['lin_comb'] = ElementwiseKernel(self.context,
                                                                     "float2* x_in, float2* noise, float2* y_out",
                                                                     "y_out[i] = complex_add(x_in[i], noise[i])",
                                                                     "lin_comb",
                                                                     preamble="""
                                                          #define complex_ctr(x, y) (float2)(x, y)
                                                          #define complex_add(a, b) complex_ctr((a.x + b.x), (a.y + b.y))
                                                          """)
                        self.kernels['complex_add'] = ElementwiseKernel(self.context, "float2 *x, float2 *y, float2 *z",
                                                                        "z[i] = x[i] + y[i]",
                                                                        "complex_add")
                        self.kernels['add'] = ElementwiseKernel(self.context,
                                                                "cfloat_t* x, cfloat_t* y, cfloat_t* z",
                                                                "z[i] = x[i] + y[i]",
                                                                "add")

    def execute(self, x):
        if not self.use_cl:
            sd = np.sqrt(self.noise_pwr)
            noise = np.random.normal(loc=0.0, scale=sd, size=x.shape) + 1j * np.random.normal(loc=0.0, scale=sd,
                                                                                              size=x.shape)
            x += noise
            # x = ne.evaluate("x + noise")
            return x

        else:
            sd = np.sqrt(self.noise_pwr)
            noise = np.random.normal(loc=0.0, scale=sd, size=x.shape) + 1j * np.random.normal(loc=0.0, scale=sd,
                                                                                              size=x.shape)
            # self.A_host = np.real(x)
            # self.B_host = np.real(noise)

            self.queue = cl.CommandQueue(self.context)

            # create device side vectors and copy values from host to device memory
            # A_dev = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.A_host)
            # B_dev = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.B_host)
            # C_dev = cl.Buffer(self.context, cl.mem_flags.COPY_HOST_PTR, hostbuf=self.C_host)
            x_gpu = cl_array.to_device(self.queue, x.astype(np.complex64))
            noise_gpu = cl_array.to_device(self.queue, noise.astype(np.complex64))
            y_gpu = cl_array.empty_like(x_gpu)
            # complex_add(x_gpu, noise_gpu, y_gpu)
            self.kernels['add'](x_gpu, noise_gpu, y_gpu)

            # result = (x_gpu + noise_gpu).get()
            # run the kernel (see string sourceCode above)
            # self.program.complex_vector_add(self.queue, self.A_host.shape, None, A_dev, B_dev, C_dev)
            # enqueue data transfer from device to host memory
            # cl.enqueue_read_buffer(self.queue, C_dev, self.C_host).wait()

            # y = x + noise
            y = y_gpu.get()
            return y

