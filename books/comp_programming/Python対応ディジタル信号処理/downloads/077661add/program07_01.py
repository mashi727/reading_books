import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

h = np.array([1, 1, 1, 1, 1]) / 5                     # インパルス応答
w = np.linspace(-np.pi, np.pi, 1024, endpoint=False)  # 周波数の範囲と刻み
_, H = signal.freqz(h, 1, w)                          # 周波数応答
magH = np.abs(H)                                      # 振幅特性
argH = np.angle(H)                                    # 位相特性
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(w, magH)                                     # 振幅特性の図示
ax1.set_xlim(-np.pi, np.pi); ax1.set_ylim(0, 1.2); ax1.grid()
ax1.set_xlabel('Frequency $\omega$ [rad]')
ax1.set_ylabel('$|H(e^{j\omega})|$')
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(w, argH)                                     # 位相特性の図示
ax2.set_xlim(-np.pi, np.pi); ax2.set_ylim(-np.pi, np.pi); ax2.grid()
ax2.set_xlabel('Frequency $\omega$ [rad]')
ax2.set_ylabel('$\\theta$ [rad]')
fig.tight_layout()
