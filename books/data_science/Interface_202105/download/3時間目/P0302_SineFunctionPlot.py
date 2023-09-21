# Plot sin function

import numpy as np
import matplotlib.pyplot as plt
import math

sample_rate = 1.0e5
carrier_freq = 1.0e3
duration = 10.0e-3
#xs, step = np.linspace(0, duration, num=duration*sample_rate)
xs = np.arange(0, duration, 1/sample_rate)
ys1 = np.sin(2.0*np.pi*carrier_freq*xs)
ys2 = np.cos(2.0*np.pi*carrier_freq*xs)

plt.plot(xs, ys1, 'b', xs, ys2, 'r')
plt.xlabel('Time(sec)')
plt.ylabel('sin(2*pi*f*t)')

plt.show()