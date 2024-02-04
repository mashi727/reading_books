import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 理想低域フィルタhd(n)
L = 25; n = np.arange(-L, L + 1)                           # 信号の時間の範囲
wc = np.pi / 2                                             # 遮断周波数
hd = (wc / np.pi) * np.sinc(n * wc / np.pi)                # 理想的単位インパルス応答
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.stem(n, hd)                                            # 理想的単位インパルス応答を図示
ax1.set_xlim(-L, L); ax1.set_ylim(-0.2, 0.6); ax1.grid()
ax1.set_xlabel('Time $n$')
ax1.set_ylabel('$h_\mathrm{d}[n]$')

# 窓関数による設計h(n)
M = 10
N = 2 * M + 1                                              # タップ数
win = signal.boxcar(N)                                     # 方形窓の選択
winz = np.hstack([np.zeros(L - M), win, np.zeros(L - M)])  # ゼロづめ（hdの切り出しのため）
h = hd * winz                                              # 窓関数をかけられた単位インパルス応答
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.stem(n, h)                                             # 窓関数をかけられた単位インパルス応答を図示
ax2.set_xlim(-L, L); ax2.set_ylim(-0.2, 0.6); ax2.grid()
ax2.set_xlabel('Time $n$')
ax2.set_ylabel('$h[n]$')

# 周波数スペクトルWと周波数応答Hの計算と図示
w = np.linspace(-np.pi, np.pi, 1024, endpoint=False)       # 周波数の範囲と刻み
_, Win = signal.freqz(win, 1, w)                           # 窓の周波数スペクトル
maxWin = np.max(np.abs(Win))
fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)
ax3.plot(w, 20 * np.log10(np.abs(Win) / maxWin))           # 窓の振幅特性を図示
ax3.set_xlim(-np.pi, np.pi); ax3.set_ylim(-80, 5); ax3.grid()
ax3.set_xlabel('Frequency $\omega$ [rad]')
ax3.set_ylabel('$|W(e^{j\omega})|$ [dB]')
_, H = signal.freqz(h, 1, w)                               # 設計されたFIRフィルタの周波数応答
fig4 = plt.figure()
ax4 = fig4.add_subplot(1, 1, 1)
ax4.plot(w, 20 * np.log10(np.abs(H)))                      # 設計されたFIRフィルタの振幅特性を図示
ax4.set_xlim(-np.pi, np.pi); ax4.set_ylim(-80, 5); ax4.grid()
ax4.set_xlabel('Frequency $\omega$ [rad]')
ax4.set_ylabel('$|H(e^{j\omega})|$ [dB]')
