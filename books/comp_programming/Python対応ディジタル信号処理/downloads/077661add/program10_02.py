import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

Wc = np.pi / 8                                     # 遮断周波数
Ws = 6 * np.pi / 8                                 # 阻止域端周波数
As = 30                                            # 阻止域減衰量
N, _ = signal.buttord(Wc, Ws, 3, As, analog=True)  # バタワースフィルタの次数の決定
print('N =\n', N)
b, a = signal.butter(N, Wc, analog=True)           # バタワースフィルタの設計
print('b =\n', b)
print('a =\n', a)
w, Ha = signal.freqs(b, a)                         # 周波数応答の計算
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(w, 20 * np.log10(np.abs(Ha)))             # 振幅特性の図示
ax1.set_xlim(0, np.pi); ax1.set_ylim(-40, 5); ax1.grid()
ax1.set_xlabel('Frequency $\Omega$ [rad/sec]')
ax1.set_ylabel('$|H_\mathrm{a}(j\Omega)|$ [dB]')
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(w, np.angle(Ha))                          # 位相特性の図示
ax2.set_xlim(0, np.pi); ax2.set_ylim(-4, 0); ax2.grid()
ax2.set_xlabel('Frequency $\Omega$ [rad/sec]')
ax2.set_ylabel('$\\angle H_\mathrm{a}(j\Omega)$ [rad]')
