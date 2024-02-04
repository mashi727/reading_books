import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
N = 64                                                           # 窓の長さ（2のべき乗とせよ）
Nb = -N // 4; Ne = N + N // 4                                    # 信号の開始時刻と終了時刻
n = np.arange(Nb, Ne + 1)                                        # 信号の継続時間の設定

# 信号x[n]
alpha = np.pi / 8                                                # 余弦波の周波数
x = np.cos(alpha * n)                                            # 余弦波
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.stem(n, x)                                                   # 原信号の図示
ax1.set_xlim(Nb, Ne); ax1.set_ylim(-2, 2); ax1.grid()
ax1.set_xlabel('Time $n$')
ax1.set_ylabel('$x[n]$')

# 信号の窓かけ（利用しない窓はコメントアウトせよ）
win = signal.boxcar(N)                                           # 方形窓の選択
#win = signal.hamming(N)                                          # ハミング窓の選択
winzero = np.hstack([np.zeros(-Nb), win, np.zeros(Ne + 1 - N)])  # 図示のため窓の前後にゼロづめ
xw = x * winzero                                                 # 窓による切り出し

# 窓をかけられた信号の図示（原信号も図示）
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.stem(n, xw)                                                  # 窓をかけられた信号
ax2.plot(n, x, ':')                                              # 原信号
ax2.set_xlim(Nb, Ne); ax2.set_ylim(-2, 2); ax2.grid()
ax2.set_xlabel('Time $n$')
ax2.set_ylabel('$x_{w}[n]$')

# 窓関数の周波数スペクトル
w = np.linspace(-np.pi, np.pi, 1024, endpoint=False)             # 周波数の範囲と刻み
_, Win = signal.freqz(win, 1, w)                                 # 周波数スペクトル
magWin = np.abs(Win)                                             # 周波数スペクトルの振幅
maxWin = np.max(magWin)                                          # 正規化のための最大値
fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)
ax3.plot(w, 20 * np.log10(magWin / maxWin))                      # 正規化振幅スペクトルの図示
ax3.set_xlim(-np.pi, np.pi); ax3.set_ylim(-60, 0); ax3.grid()
ax3.set_xlabel('Frequency $\omega$ [rad]')
ax3.set_ylabel('$|W(e^{j\omega})|$ [dB]')

# 窓をかけられた信号の周波数スペクトル
_, Xw = signal.freqz(xw, 1, w)                                   # 周波数スペクトル
magXw = np.abs(Xw)                                               # 周波数スペクトルの振幅
maxXw = np.max(magXw)                                            # 正規化のための最大値
fig4 = plt.figure()
ax4 = fig4.add_subplot(1, 1, 1)
ax4.plot(w, 20 * np.log10(magXw / maxXw))                        # 正規化振幅スペクトルの図示
ax4.set_xlim(-np.pi, np.pi); ax4.set_ylim(-60, 0); ax4.grid()
ax4.set_xlabel('Frequency $\omega$ [rad]')
ax4.set_ylabel('$|X_{w}(e^{j\omega})|$ [dB]')
