# Re-sample to decimate input I-Q samples

import numpy as np
import matplotlib.pyplot as plt
import rtlsdr
import scipy.signal

sdr = rtlsdr.RtlSdr()

# configure device
sdr.sample_rate = 2.048e6
sdr.center_freq = 80.0e6
#sdr.agc = True
sdr.gain = 50
decimation_factor = 8

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_xlabel('Frequency (MHz)')
ax1.set_ylabel('Relative power (dB)')
ax2.set_xlabel('Time (sec)')
ax2.set_ylabel('Frequency (MHz)')
while(True):
    samples = sdr.read_samples(128*2048)
    # decimated = samples[::decimation_factor]
    decimated = scipy.signal.decimate(samples, decimation_factor)
    # use matplotlib to estimate and plot the PSD
    ax1.psd(decimated, NFFT=1024, Fs=sdr.sample_rate/1e6/decimation_factor, Fc=sdr.center_freq/1e6)
    ax2.specgram(decimated, Fs=sdr.sample_rate/1e6/decimation_factor)
    plt.pause(0.1)
sdr.close()