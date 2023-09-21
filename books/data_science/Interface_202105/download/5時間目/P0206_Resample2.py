# Display time plot graph for decimated samples

from pylab import *
from rtlsdr import *
import numpy as np
import scipy.signal

sdr = RtlSdr()

# configure device
sdr.sample_rate = 2.048e6
sdr.center_freq = 80.0e6
#sdr.agc = True
sdr.gain = 50
decimation_factor = 8

samples = sdr.read_samples(256*1024)
sdr.close()
decimated1 = samples[::decimation_factor]
decimated2 = scipy.signal.decimate(samples, decimation_factor)

xs = np.linspace(0, 256*1024, decimation_factor*1024)
xd = xs[::decimation_factor]
plt.plot(xs, np.real(samples[:len(xs)]),'b', xd, np.real(decimated1[:len(xd)]),'or-', xd, np.real(decimated2[:len(xd)]), '*g-')
xlabel('Time')
ylabel('Signal Level')

show()