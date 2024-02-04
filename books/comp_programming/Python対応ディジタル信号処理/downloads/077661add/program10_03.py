import numpy as np
from scipy import signal
from myimpinvar import myimpinvar
import matplotlib.pyplot as plt

Wc = np.pi / 8                                           # 遮断周波数
print('Wc =\n', Wc)
N = 2                                                    # プロトタイプフィルタの次数
bp, ap = signal.butter(N, Wc, analog=True)               # プロトタイプフィルタの分子・分母係数
print('bp =\n', bp)
print('ap =\n', ap)
bz, az = myimpinvar(bp, ap, 1)                           # インパルス不変変換法による設計
print('bz =\n', bz)
print('az =\n', az)
w = np.linspace(0, np.pi, 512, endpoint=False)           # 周波数の範囲と刻み
_, Ha = signal.freqs(bp, ap, w)                          # プロトタイプフィルタの周波数応答
_, Hz = signal.freqz(bz, az, w)                          # ディジタルフィルタの周波数応答
maxHz = np.max(np.abs(Hz))
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(w, 20 * np.log10(np.abs(Ha)), ':', label='$H_\mathrm{a}(j\Omega)$')
ax1.plot(w, 20 * np.log10(np.abs(Hz) / maxHz), label='$H(e^{j\omega})$')
ax1.set_xlim(0, np.pi); ax1.set_ylim(-40, 5); ax1.grid()
ax1.set_xlabel('Frequency $\Omega, \omega$')
ax1.set_ylabel('$|H_\mathrm{a}(j\Omega)|, |H(e^{j\omega})|$ [dB]')
ax1.legend(loc='upper right')

# 単位インパルス応答の比較
tend = 25; t = np.arange(0, tend + 0.1, 0.1)             # 時刻の範囲
alpha = Wc / np.sqrt(2)
ha = 2 * alpha * np.exp(-alpha * t) * np.sin(alpha * t)  # プロトタイプフィルタのインパルス応答
n = np.arange(0, tend + 1)                               # 時間の範囲
x = np.zeros(tend + 1); x[0] = 1                         # 単位インパルス入力
h = signal.lfilter(bz, az, x)                            # ディジタルフィルタのインパルス応答
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(t, ha, label='$h_\mathrm{a}(t)$')               # プロトタイプフィルタのインパルス応答を図示
ax2.stem(n, h, label='$h[n]$')                           # ディジタルフィルタのインパルス応答を図示
ax2.set_xlim(0, tend); ax2.set_ylim(-0.05, 0.2); ax2.grid()
ax2.set_xlabel('Time $t, n$')
ax2.set_ylabel('$h_\mathrm{a}(t), h[n]$')
ax2.legend(loc='upper right')
