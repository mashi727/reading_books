import numpy as np
from scipy import signal
from myfreqztrans import myfreqztrans
import matplotlib.pyplot as plt
import sys
eps = sys.float_info.epsilon                      # np.log10(0) の計算を避けるための計算機イプシロン(2.22044e-16)

# プロトタイプフィルタ
b = 0.0976 * np.array([1, 2, 1])                  # プロトタイプフィルタの分子係数
a = np.array([1, -0.9428, 0.3333])                # プロトタイプフィルタの分母係数
thetac = 0.25 * np.pi                             # プロトタイプフィルタの遮断周波数
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
theta = np.linspace(0, np.pi, 512, endpoint=False)
_, Hl = signal.freqz(b, a, theta)                 # 周波数応答
ax1.plot(theta, 20 * np.log10(np.abs(Hl) + eps))  # 振幅特性の図示
ax1.set_xlim(0, np.pi); ax1.set_ylim(-60, 5); ax1.grid()
ax1.set_xlabel('Frequency $\\theta$ [rad]')
ax1.set_ylabel('$|H(e^{j\\theta})|$ [dB]')

# 高域フィルタの設計
wc = 0.6 * np.pi                                  # 高域フィルタの遮断周波数
Bh, Ah = myfreqztrans(b, a, 'hp', thetac, wc)     # 低域-高域変換
print('Bh =\n', Bh)
print('Ah =\n', Ah)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
w = np.linspace(0, np.pi, 512, endpoint=False)
_, Hh = signal.freqz(Bh, Ah, w)                   # 高域フィルタの周波数応答
ax2.plot(w, 20 * np.log10(np.abs(Hh) + eps))      # 振幅特性の図示
ax2.set_xlim(0, np.pi); ax2.set_ylim(-60, 5); ax2.grid()
ax2.set_xlabel('Frequency $\omega$ [rad]')
ax2.set_ylabel('$|H_\mathrm{h}(e^{j\omega})|$ [dB]')

# 帯域フィルタの設計
wc = np.array([0.25 * np.pi, 0.5 * np.pi])        # 帯域フィルタの遮断周波数
Bbp, Abp = myfreqztrans(b, a, 'bp', thetac, wc)   # 低域-帯域変換
print('Bbp =\n', Bbp)
print('Abp =\n', Abp)
fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)
w = np.linspace(0, np.pi, 512, endpoint=False)
_, Hbp = signal.freqz(Bbp, Abp, w)                # 帯域フィルタの周波数応答
ax3.plot(w, 20 * np.log10(np.abs(Hbp) + eps))     # 振幅特性の図示
ax3.set_xlim(0, np.pi); ax3.set_ylim(-60, 5); ax3.grid()
ax3.set_xlabel('Frequency $\omega$ [rad]')
ax3.set_ylabel('$|H_\mathrm{bp}(e^{j\omega})|$ [dB]')

# 帯域阻止フィルタの設計
wc = np.array([0.25 * np.pi, 0.5 * np.pi])        # 帯域阻止フィルタの遮断周波数
Bbs, Abs = myfreqztrans(b, a, 'bs', thetac, wc)   # 低域-帯域阻止変換
print('Bbs =\n', Bbs)
print('Abs =\n', Abs)
fig4 = plt.figure()
ax4 = fig4.add_subplot(1, 1, 1)
w = np.linspace(0, np.pi, 512, endpoint=False)
_, Hbs = signal.freqz(Bbs, Abs, w)                # 帯域阻止フィルタの周波数応答
ax4.plot(w, 20 * np.log10(np.abs(Hbs) + eps))     # 振幅特性の図示
ax4.set_xlim(0, np.pi); ax4.set_ylim(-60, 5); ax4.grid()
ax4.set_xlabel('Frequency $\omega$ [rad]')
ax4.set_ylabel('$|H_\mathrm{bs}(e^{j\omega})|$ [dB]')
