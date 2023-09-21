import numpy as np
import matplotlib.pyplot as plt
import math

sample_rate = 1.0e5
carrier_freq = 5.0e2
duration = 10.0e-3

xs = np.arange(0, duration, 1/sample_rate)
ys1 = np.sin(2.0*np.pi*carrier_freq*xs)             # 直接波

# 遅延プロファイル
delayp = [10.0, 23.0, 28.0, 31.0, 45.0, 71.0]
lossp  = [-10, -18, -33, -41, -47, -51, -55]
phasep = [30.0, 170.0, -210.0, -35.0, -14.0, 137.0]

ys2 = np.sin(2.0*np.pi*carrier_freq*xs+np.pi)*0.8   # 逆相のマルチパス波
ys3 = ys1 + ys2 # 合成波

fig = plt.figure()
ax = fig.gca()
plt.plot(xs, ys1, 'b', label=r'$\alpha=\sin(2\pi ft)$')
plt.plot(xs, ys2, 'g', label=r'$\beta=0.8*\sin(2\pi ft+\lambda/2)$')
plt.plot(xs, ys3, 'r', label=r'$\alpha+\beta$')
plt.xlabel('Time(sec)')
plt.ylabel('sin(2*pi*f*t)')
ax.legend(loc='upper right')
ax.set_title('Multipaths synthesized signal')

plt.show()