import numpy as np
from scipy import signal
from mydirect import mydirect
import matplotlib.pyplot as plt

wc = 0.25 * np.pi                                       # 遮断周波数
ws = 0.75 * np.pi                                       # 阻止域端周波数
As = 30                                                 # 阻止域減衰量
b, a = mydirect(wc, ws, As)                             # 直接設計されたフィルタの分子・分母係数
print('b =\n', b)
print('a =\n', a)
zr = np.roots(b)                                        # 零点
print('zr =\n', zr)
pl = np.roots(a)                                        # 極
print('pl =\n', pl)
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
theta = np.linspace(-np.pi, np.pi, 1024, endpoint=False)
uc = np.exp(1j * theta)                                 # 単位円の計算
ax1.plot(np.real(uc), np.imag(uc), '-')                 # 単位円の図示
ax1.plot(np.real(zr), np.imag(zr), 'o', label='Zeros')  # 零点の図示
ax1.plot(np.real(pl), np.imag(pl), 'x', label='Poles')  # 極の図示
ax1.axis('equal'); ax1.grid()
ax1.set_xlabel('Real part')
ax1.set_ylabel('Imaginary part')
ax1.legend(loc='upper left')
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
w = np.linspace(0, np.pi, 512, endpoint=False)
_, H = signal.freqz(b, a, w)                            # 周波数応答の計算
ax2.plot(w, 20 * np.log10(np.abs(H)))                   # 振幅特性の図示
ax2.set_xlim(0, np.pi); ax2.set_ylim(-60, 5); ax2.grid()
ax2.set_xlabel('Frequency $\omega$ [rad]')
ax2.set_ylabel('$|H(e^{j\omega})|$ [dB]')
