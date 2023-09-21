# Continuously display spectrum graph for rtlsdr

import numpy as np
import matplotlib.pyplot as plt
import rtlsdr

sdr = rtlsdr.RtlSdr()

# configure device
sdr.sample_rate = 2.4e6
sdr.center_freq = 80.0e6
#sdr.agc = True
sdr.gain = 50

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_xlabel('Frequency (MHz)')
ax1.set_ylabel('Relative power (dB)')
ax2.set_xlabel('Time (sec)')
ax2.set_ylabel('Frequency (MHz)')
while(True):
    samples = sdr.read_samples(256*1024)
    # use matplotlib to estimate and plot the PSD
    ax1.psd(samples, NFFT=1024, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6)
    ax2.specgram(samples, Fs=sdr.sample_rate/1e6)
    plt.pause(0.1)
    print(samples)
sdr.close()