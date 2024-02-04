import numpy as np
from scipy import signal
from myfreqztrans import myfreqztrans
import matplotlib.pyplot as plt

# プロトタイプフィルタ
b = 0.0976 * np.array([1, 2, 1])                        # プロトタイプフィルタの分子係数
a = np.array([1, -0.9428, 0.3333])                      # プロトタイプフィルタの分母係数
thetac = 0.25 * np.pi                                   # プロトタイプフィルタの遮断周波数

# フィルタの設計
wc = 0.75 * np.pi                                       # 所望のフィルタの遮断周波数
B, A = myfreqztrans(b, a, 'lp', thetac, wc)             # 低域-低域変換
print('B =\n', B)
print('A =\n', A)
zr = np.roots(B)                                        # 設計されたフィルタの零点
print('zr =\n', zr)
pl = np.roots(A)                                        # 設計されたフィルタの極
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
ax1.legend(loc='upper right')
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
w = np.linspace(0, np.pi, 512, endpoint=False)
_, H = signal.freqz(B, A, w)                            # 周波数応答の計算
ax2.plot(w, 20 * np.log10(np.abs(H)))                   # 振幅特性の図示
ax2.set_xlim(0, np.pi); ax2.set_ylim(-60, 5); ax2.grid()
ax2.set_xlabel('Frequency $\omega$ [rad]')
ax2.set_ylabel('$|H(e^{j\omega})|$ [dB]')
