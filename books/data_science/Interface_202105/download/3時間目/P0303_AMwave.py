# Plot sin function

import numpy as np
import matplotlib.pyplot as plt
import math

sample_rate = 1.0e5
carrier_freq = 1.0e2
signal_freq = 0.5e1
duration = 1.0
#xs, step = np.linspace(0, duration, num=duration*sample_rate)
xs = np.arange(0, duration, 1/sample_rate)
ys1 = np.sin(2.0*np.pi*carrier_freq*xs)
ys2 = np.sin(2.0*np.pi*signal_freq*xs)*0.5

ys3 = ys1*(ys2+1.0)

plt.plot(xs, ys3, 'g')
plt.xlabel('Time(sec)')
plt.ylabel('sin(2*pi*f*t)')

plt.show()