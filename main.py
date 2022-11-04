__author__ = 'Jacob Miller'
from datetime import datetime
import PointDataFilter as pdfilt
import AdditiveWhiteGaussianNoise as awgn
import numpy as np
import numexpr as ne
import SymbolGenerator
import QAMModulator
from utils import make_rrc, pwr
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import sys
# from pylab import plot, show, xlim, ylim, gca
# from matplotlib import pyplot
# import seaborn as sns

import pyaudio
CHUNK = 2**12
WIDTH = 2
CHANNELS = 2
RATE = 48000
RECORD_SECONDS = 10.0
DTYPE = np.int16
MAX_INT = 32768.0
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK)

print ne.detect_number_of_cores()

# Create constellation
cr = np.linspace(0, 3, 4)
cs = np.linspace(-1, 1, 4)
(Cx, Cy) = np.meshgrid(cr, cr, indexing='ij')
C = cs[Cx.astype('int')] + 1j*cs[Cy.astype('int')]

# C = np.array([1+1j, 1-1j, -1+1j, -1-1j])
C = np.reshape(C, newshape=(C.size, 1))
# plot(np.real(C), np.imag(C), '.')
# show()

# Symbols to generate per loop
MEMORY = 20
NFFT = 2**12
sps = 4.0
Rs = 1.0
fs = sps*Rs
N_syms = np.floor((NFFT - MEMORY*sps)/sps)  # make integer result
print N_syms, "symbols per block"
print N_syms*sps, "samples per block"
print "NFFT =", NFFT

# Instantiate a SymbolGenerator
sg = SymbolGenerator.SymbolGenerator(m=C.size, size=(N_syms, 1))
# Instantiate a QAMModulator
qam_modulator = QAMModulator.QAMModulator(C, sps)
# Generate a RRC Filter
rrc = make_rrc(1.0, 0.3, 1801)
np.savetxt("c:\\temp\\rrc.txt", rrc, fmt='%.8e', delimiter='\t')
# filename = "c:\\temp\\rrc.txt"

use_cl = False
tx_rrc = pdfilt.PointDataFilter(filename=rrc, fs=fs, nfft=NFFT, fsf=1, gdos=8, memory=MEMORY, use_cl=use_cl)
rx_rrc = pdfilt.PointDataFilter(filename=rrc, fs=fs, nfft=NFFT, fsf=1, gdos=8, memory=MEMORY, use_cl=use_cl)
chan = np.array([[-50, -00, -000], [-5, 00, -00], [0, 0, 00], [5, 0, 00], [50, 00, -0]], dtype=np.float64)
channel_filters = pdfilt.PointDataFilter(filename=chan, fs=fs, nfft=NFFT, fsf=1, gdos=8, memory=MEMORY, use_cl=use_cl)
block_size = tx_rrc.block_size
AWGN = awgn.AdditiveWhiteGaussianNoise(n=block_size, sigma=0.00001, use_cl=False)


win = pg.GraphicsWindow(title='Waveform Plotting')
win.resize(900, 600)
win.setWindowTitle('Waveform Plotting')
pg.setConfigOptions(antialias=True)
# Waveform plot
this_plot = [win.addPlot(title='Waveform plot', name='waveform')]
curve = [this_plot[0].plot(pen='y'), this_plot[0].plot(pen='m')]
this_plot[0].setLabels(bottom='Time (s)', left='Amplitude (V)')
this_plot[0].showGrid(x=True, y=True)
# FFT plot
this_plot.append(win.addPlot(title='FFT plot', name='FFT'))
curve.append(this_plot[1].plot(pen='w', antialias=False))
this_plot[1].setLabels(bottom='Frequency (Hz)', left='|H|^2 (dB)')
this_plot[1].showGrid(x=True, y=True)
win.nextRow()
# Eye diagrams
this_plot.append(win.addPlot(title='Eye Diagram', name='REeye'))
curve.append(this_plot[2].plot())
this_plot[2].setLabels(bottom='Time (s)', left='Real')
this_plot[2].showGrid(x=True, y=True)
this_plot.append(win.addPlot(title='Eye Diagram', name='IMeye'))
curve.append(this_plot[3].plot())
this_plot[3].setLabels(bottom='Time (s)', left='Imaginary')
this_plot[3].showGrid(x=True, y=True)
# Constellation scatter plot
curve.append(pg.ScatterPlotItem(size=2, pen=pg.mkPen('r'), pxMode=True))
this_plot.append(win.addPlot())
this_plot[4].addItem(curve[-1])
this_plot[4].showGrid(x=True, y=True)
this_plot[4].setLabels(bottom='I', left='Q')

# GLOBALS
completed_passes = 0
samples_processed = 0
eye_plot_handles = []


def make_eye_data(data, ns, spns, nr=2):
    # _ns = 3
    ts = spns*ns
    max_nr = np.floor(data.size/ts)
    if nr > max_nr:
        nr = max_nr
    y_eye = data[0:ts*(nr-1), 0]
    y_eye = np.reshape(y_eye, (nr-1, ts))
    y_eye = np.real(np.hstack((y_eye, data[ts:ts*nr:ts])))
    # Make time axis
    t_eye = np.linspace(0, ts, int(ts)+1)
    t_eye = np.tile(t_eye, (y_eye.shape[0], 1)) / spns
    return y_eye, t_eye


def update_plot(c, d, p):
    global completed_passes, sps, block_size, eye_plot_handles, NFFT
    N = NFFT/2
    # update waveform plot
    p[0].setTitle('Pass %d' % completed_passes)
    c[0].setData(np.real(d[0:N, 0]))
    c[1].setData(np.imag(d[0:N, 0]))
    # update FFT plot
    D = np.fft.fftshift(np.fft.fft(d[0:N, 0]))
    f = np.fft.fftshift(np.fft.fftfreq(N, d=1/sps))
    c[2].setData(f, 20*np.log10(abs(D)))
    # update eye diagram
    yre_eye = np.real(d[0:sps*2+1, 0])
    yim_eye = np.imag(d[0:sps*2+1, 0])
    t_eye = np.linspace(0, sps*2, num=sps*2+1) / sps
    pen_y = pg.mkPen(color=(255, 255, 0, 256*2/4), width=2.0)
    pen_m = pg.mkPen(color=(255, 0, 255, 256*2/4), width=2.0)
    if len(eye_plot_handles) < 50*2:
        eye_plot_handles.append(p[2].plot(x=t_eye, y=yre_eye, pen=pen_y, antialias=False))
        eye_plot_handles.append(p[3].plot(x=t_eye, y=yim_eye, pen=pen_m, antialias=False))
        # eye_plot_handles.append(pg.PlotDataItem(parent=p[2], x=t_eye, y=yre_eye))
        # eye_plot_handles.append(pg.PlotDataItem(parent=p[3], x=t_eye, y=yim_eye))
    else:
        # remove first element
        cre_eye = eye_plot_handles.pop(0)
        cim_eye = eye_plot_handles.pop(0)
        # set new data
        cre_eye.setData(t_eye, yre_eye)
        cim_eye.setData(t_eye, yim_eye)
        # put at end of list
        eye_plot_handles.append(cre_eye)
        eye_plot_handles.append(cim_eye)

    # update constellation scatter plot
    c[5].setData(np.real(d[:2*N:sps, 0]), np.imag(d[:2*N:sps, 0]))
    if completed_passes == 30:
        for myp in p:
            myp.enableAutoRange('xy', False)  # stop auto-scaling after the first data set is plotted


# main simulation loop
def RunSimulation():
    global samples_processed, y_rx, completed_passes, curve, y_rx, this_plot
    # Main simulation loop
    # for n in range(0, 100):
    # Begin Simulation
    s = sg.execute()
    tx = qam_modulator.execute(s)
    # Transmit pulse shaping
    y_tx = tx_rrc.execute(tx)
    # Channel Filtering
    y_chan = channel_filters.execute(y_tx)
    y_noise = AWGN.execute(y_chan)
    # Receive pulse shaping
    y_rx = rx_rrc.execute(y_noise)
    y_pwr_scale = 1.0/np.sqrt(pwr(y_rx[0::sps]))
    y_rx = ne.evaluate("y_rx * y_pwr_scale")

    update_plot(c=curve, d=y_rx, p=this_plot)

    data = np.zeros((2*NFFT, 1))
    data[0:2*y_rx.size:2, 0] = np.real(y_rx[:, 0]) / max(abs(y_rx))
    data[1:2*y_rx.size:2, 0] = np.imag(y_rx[:, 0]) / max(abs(y_rx))
    # Convert to char
    audio_data = np.array(np.round(data * MAX_INT/4), dtype=DTYPE)
    # write audio
    string_audio_data = audio_data.tostring()
    stream.write(string_audio_data, CHUNK)

    samples_processed += y_rx.size
    completed_passes += 1
    if completed_passes % 10 == 0:
        print completed_passes


def main():
    global samples_processed, y_rx, completed_passes, sps

    start_time = datetime.now()
    print start_time
    samples_processed = 0
    completed_passes = 0

    # setup simulation event timer
    timer = QtCore.QTimer()
    timer.timeout.connect(RunSimulation)
    timer.start(0)
    # start application
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
        timer.stop()

        # Run time metrics
        stop_time = datetime.now()
        elapsed_time = stop_time - start_time
        #
        print stop_time
        print "Elapsed time: ", elapsed_time
        print "Processed ", samples_processed, "samples"
        print "Processed ", samples_processed/sps, "symbols"
        print "Processed ", samples_processed/sps*np.log2(C.size), "bits"

        # y_rx = np.trim_zeros(y_rx, 'b')
        # (y_eye, t_eye) = make_eye_data(data=y_rx, ns=3, spns=sps, nr=100)
        # plot(np.transpose(t_eye), np.transpose(y_eye), 'y', alpha=0.5)
        # ax = gca()
        # ax.set_axis_bgcolor('black')
        # pyplot.grid(False)
        # xlim((0, t_eye[0, -1]))
        # ylim((-2, 2))
        # show()

        # ks = np.floor(y_rx.size / NFFT)
        # Pxx, freqs, bins, im = specgram(y_rx, NFFT=NFFT, Fs=fs, noverlap=NFFT/2, cmap=cm.gist_heat)
        # show()

        print "Finished!"

if __name__ == "__main__":
    main()