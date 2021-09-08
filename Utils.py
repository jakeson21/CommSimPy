from pylab import arange, newaxis
from scipy.interpolate import interp1d
from numpy import fft, roll, floor, abs, isinf
import numpy as np
import os

__author__ = 'Fuguru'
__all__ = ['read_point_data_file', 'fgp_to_impulse_response', 'make_rrc', 'pwr']

def read_point_data_file(filename):
    """
    read_point_data_file(filename)
    :param filename: file name of point data file, formatted as
        3-column text file (csv or tab) [Freq, Gain (dB), Phase (deg)]
    :return h_fgp: ndarray of [Freq, Gain (dB), Phase (deg)]
    """
    if os.path.exists(filename):
        try:
            h_fgp = np.loadtxt(filename, comments='#', skiprows=0, ndmin=2)
            # return h_fgp
        except ValueError:
            h_fgp = np.loadtxt(filename, comments='#', delimiter=',', skiprows=0, ndmin=2)
        return h_fgp
    else:
        print "Filename", filename, "does not exist"
        raise ValueError


def fgp_to_impulse_response(h_fgp, fs=1, length=16, gdelay=0):
    """
    fgp_to_impulse_response(h_fgp, fs=1, length=11, gdelay=0)
    :param h_fgp: 3-column text file (csv or tab) [Freq, Gain (dB), Phase (deg)]
    :param fs: target sample rate
    :param length: desired length of impulse response
    :param gdelay: delay
    :return: (h, h_new_fgp)
    """
    f_new = arange(-floor(length/2), floor(length/2))[:, newaxis] / length * fs
    # if length % 2:  # odd
    #     f_new = arange(-np.floor(length/2), np.floor(length/2))[:, newaxis] / length * fs
    # else:  # even
    #     f_new = arange(-np.floor(length/2), np.floor(length/2))[:, newaxis] / length * fs
    #     f_new = f_new[0:-1]

    fc_gain = interp1d(h_fgp[:, 0], h_fgp[:, 1], kind='linear')
    fc_phase = interp1d(h_fgp[:, 0], h_fgp[:, 2], kind='linear')
    new_gain = fc_gain(f_new)
    new_phase = fc_phase(f_new)
    h_new_fgp = np.hstack((f_new, new_gain, new_phase))
    h = (10**(new_gain/20)) * np.exp(1j*new_phase*np.pi/180)
    h = fft.ifftshift(h, axes=0)
    # f_new = fft.ifftshift(f_new)
    h = fft.ifft(h, n=length, axis=0)
    h = roll(h, shift=np.int32(gdelay), axis=0)
    return h, h_new_fgp


def pwr(x):
    p = np.mean(abs(x)**2.0)
    return p


def make_rrc(bw, beta, n=501):
    f = np.array(np.linspace(-2.0*bw, stop=2.0*bw, num=n))
    f = np.reshape(f, newshape=(n, 1))
    phase = np.zeros_like(f)
    g = np.zeros_like(f)
    fn = bw/2.0
    f_low = (1.0-beta) * fn
    f_high = (1.0+beta) * fn
    f_roll_off = np.logical_and(abs(f) >= f_low, abs(f) <= f_high)

    peak = np.abs(f) <= (1.0-beta)*fn
    g[np.nonzero(peak)[0]] = 1.0
    if beta > 0:
        rt = np.sqrt(0.5 + 0.5*np.sin(np.pi/2.0/fn*(fn-abs(f[f_roll_off[:, 0]]))/beta))
        if f_roll_off.size > 0:
            g[f_roll_off[:, 0]] = rt

    g[g <= 1e-50] = 1e-50
    g = 20.0*np.log10(g)
    g[isinf(g)[:, 0]] = -1000.0
    rrc_fgp = np.hstack((f, g, phase))
    rrc_fgp = np.vstack(([-50.0, -1000.0, 0.0], rrc_fgp, [50.0, -1000.0, 0.0]))
    return rrc_fgp


def makerrc(sps, memory, beta):
    Ts = 1.0
    if memory*sps == 0.0:
        t = np.array(np.linspace(-memory*Ts/2.0, stop=memory*Ts/2.0, num=memory*sps))
    else:
        t = np.array(np.linspace(-memory*Ts/2.0, stop=memory*Ts/2.0, num=memory*sps+1.0))
    rrc = 1.0/np.sqrt(Ts)*(np.sin(np.pi*t/Ts*(1.0-beta))+4*beta*t/Ts*np.cos(np.pi*t*(1.0+beta)/Ts)) / \
          (np.pi*t*(1.0-((4.0*beta*t/Ts)**2.0)/Ts))
    rrc[t == 0] = 1.0/np.sqrt(Ts)*(1.0+beta+4.0*beta/np.pi)
    rrc[abs(t) == Ts/4.0/beta] = beta/np.sqrt(2.0*Ts)*((1.0+2.0/np.pi)*np.sin(np.pi/4/beta)+(1-2.0/np.pi)*np.cos(np.pi/4.0/beta))
    h = rrc/sum(rrc)
    return h