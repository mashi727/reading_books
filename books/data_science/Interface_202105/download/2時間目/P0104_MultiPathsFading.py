import numpy as np
import matplotlib.pyplot as plt
import math

sample_rate = 1.0e5
carrier_freq = 5.0e2
duration = 10.0e-3

# マルチパス波の位相差(deg)
print("マルチパス波の位相差(deg<360):", end='')
mp_ph = float(input())/180.0*np.pi
# マルチパス波の減衰度(真値)
print("マルチパス波の減衰(真値<1.0):", end='')
mp_att = float(input())

xs = np.arange(0, duration, 1/sample_rate)
ys1 = np.sin(2.0*np.pi*carrier_freq*xs)                # 直接波
ys2 = np.sin(2.0*np.pi*carrier_freq*xs+mp_ph)*mp_att   # マルチパス波
ys3 = ys1 + ys2 # 合成波

fig = plt.figure()
ax = fig.gca()
plt.plot(xs, ys1, 'b', label=r'$\alpha=\sin(2\pi ft)$')
plt.plot(xs, ys2, 'g', label=r'$\beta=%.2f*\sin(2\pi ft+%.1f\lambda)$'%(mp_att, mp_ph/2/np.pi))
plt.plot(xs, ys3, 'r', label=r'$\alpha+\beta$')
plt.xlabel('Time(sec)')
plt.ylabel('sin(2*pi*f*t)')
ax.legend(loc='upper right')
ax.set_title('Multipaths synthesized signal')

plt.show()