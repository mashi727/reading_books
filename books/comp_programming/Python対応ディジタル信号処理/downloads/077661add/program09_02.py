import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 窓の選択（選択しない窓をコメントアウトせよ）
N = 21                                                # 窓の長さ（フィルタのタップ長（奇数））
#win = signal.boxcar(N)                               # 方形窓
#win = signal.hann(N)                                 # ハニング窓
win = signal.hamming(N)                               # ハミング窓
#win = signal.blackman(N)                             # ブラックマン窓
#win = signal.kaiser(N, 2 * np.pi)                    # カイザー窓

# 窓の周波数応答
w = np.linspace(-np.pi, np.pi, 1024, endpoint=False)  # 周波数の範囲と刻み
_, Win = signal.freqz(win, 1, w)                      # 周波数応答
maxWin = np.max(np.abs(Win))
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(w, 20 * np.log10(np.abs(Win) / maxWin))      # 窓の振幅特性を図示
ax1.set_xlim(-np.pi, np.pi); ax1.set_ylim(-80, 5); ax1.grid()
ax1.set_xlabel('Frequency $\omega$ [rad]')
ax1.set_ylabel('$|W(e^{j\omega})|$ [dB]')

# 窓関数によるFIR フィルタの設計
wc = np.pi / 2                                        # 遮断周波数
# 単位インパルス応答（窓関数による設計）
# 窓の選択（選択しない窓をコメントアウトする）
#h = signal.firwin(N, wc / np.pi, window='boxcar')
#h = signal.firwin(N, wc / np.pi, window='hann')
h = signal.firwin(N, wc / np.pi, window='hamming')
#h = signal.firwin(N, wc / np.pi, window='blackman')
#h = signal.firwin(N, wc / np.pi, window=('kaiser', 2 * np.pi))
_, H = signal.freqz(h, 1, w)                          # 設計されたフィルタの周波数応答
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(w, 20 * np.log10(np.abs(H)))                 # 設計されたフィルタの振幅特性を図示
ax2.set_xlim(-np.pi, np.pi); ax2.set_ylim(-80, 5); ax2.grid()
ax2.set_xlabel('Frequency $\omega$ [rad]')
ax2.set_ylabel('$|H(e^{j\omega})|$ [dB]')
