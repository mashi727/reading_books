import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

h = np.array([1, 3, 3, 1]) / 8                        # 単位インパルス応答
w = np.linspace(-np.pi, np.pi, 1024, endpoint=False)  # 周波数の範囲と刻み
_, H = signal.freqz(h, 1, w)                          # 周波数応答の計算
magH = np.abs(H)                                      # 振幅特性
argH = np.angle(H)                                    # 位相特性
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(w, magH)                                     # 振幅特性の図示
ax1.set_xlim(-np.pi, np.pi); ax1.set_ylim(0, 1.5); ax1.grid()
ax1.set_xlabel('Frequency $\omega$ [rad]')
ax1.set_ylabel('$|H(e^{j\omega})|$')
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(w, argH)                                     # 位相特性の図示
ax2.set_xlim(-np.pi, np.pi); ax2.set_ylim(-np.pi, np.pi); ax2.grid()
ax2.set_xlabel('Frequency $\omega$ [rad]')
ax2.set_ylabel('$\\theta$ [rad]')
