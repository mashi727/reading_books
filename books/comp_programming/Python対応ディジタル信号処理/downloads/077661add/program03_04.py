import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
N = 64                                                         # 窓の長さ（2のべき乗とせよ）
Nb = -N // 4; Ne = N + N // 4                                  # 信号の開始時刻と終了時刻
n = np.arange(Nb, Ne + 1)                                      # 信号の継続時間の設定

# 信号x[n]の生成
wa = 0.15 * np.pi; wb = 0.25 * np.pi; wc = 0.5 * np.pi         # 余弦波の周波数
x = np.cos(wa * n) + 0.8 * np.cos(wb * n) \
    + 0.01 * np.cos(wc * n)                                    # 余弦波の線形結合
	
# 信号の窓かけ
wr = signal.boxcar(N)                                          # 方形窓
wrzero = np.hstack([np.zeros(-Nb), wr, np.zeros(Ne + 1 - N)])  # 方形窓にゼロづめ
xr = x * wrzero                                                # 方形窓による切り出し
wh = signal.hamming(N)                                         # ハミング窓
whzero = np.hstack([np.zeros(-Nb), wh, np.zeros(Ne + 1 - N)])  # ハミング窓にゼロづめ
xh = x * whzero                                                # ハミング窓による切り出し

# 窓をかけられた信号の図示（原信号も図示）
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.stem(n, xr)                                                # 方形窓による信号の図示
ax1.plot(n, x, ':')                                            # 原信号の図示
ax1.set_xlim(Nb, Ne); ax1.set_ylim(-2, 2); ax1.grid()
ax1.set_xlabel('Time $n$')
ax1.set_ylabel('$x_\mathrm{r}[n]$')
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.stem(n, xh)                                                # ハミング窓による信号の図示
ax2.plot(n, x, ':')                                            # 原信号の図示
ax2.set_xlim(Nb, Ne); ax2.set_ylim(-2, 2); ax2.grid()
ax2.set_xlabel('Time $n$')
ax2.set_ylabel('$x_\mathrm{h}[n]$')

# 方形窓をかけられた信号の周波数スペクトルの計算
w = np.linspace(-np.pi, np.pi, 1024, endpoint=False)           # 周波数の範囲と刻み
_, Xr = signal.freqz(xr, 1, w)                                 # 周波数スペクトル
magXr = np.abs(Xr)                                             # 周波数スペクトルの振幅
maxXr = np.max(magXr)                                          # 正規化のための最大値
fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)
ax3.plot(w, 20 * np.log10(magXr / maxXr))                      # 正規化振幅スペクトルの図示
ax3.set_xlim(-np.pi, np.pi); ax3.set_ylim(-60, 0); ax3.grid()
ax3.set_xlabel('Frequency $\omega$ [rad]')
ax3.set_ylabel('$|X_\mathrm{r}(e^{j\omega})|$ [dB]')

# ハミング窓をかけられた信号の周波数スペクトルの計算
_, Xh = signal.freqz(xh, 1, w)                                 # 周波数スペクトル
magXh = np.abs(Xh)                                             # 周波数スペクトルの振幅
maxXh = np.max(magXh)                                          # 正規化のための最大値
fig4 = plt.figure()
ax4 = fig4.add_subplot(1, 1, 1)
ax4.plot(w, 20 * np.log10(magXh / maxXh))                      # 正規化振幅スペクトルの図示
ax4.set_xlim(-np.pi, np.pi); ax4.set_ylim(-60, 0); ax4.grid()
ax4.set_xlabel('Frequency $\omega$ [rad]')
ax4.set_ylabel('$|X_\mathrm{h}(e^{j\omega})|$ [dB]')
