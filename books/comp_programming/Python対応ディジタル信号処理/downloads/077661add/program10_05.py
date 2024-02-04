import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

wc = 0.25 * np.pi                                  # ディジタルフィルタの遮断周波数
ws = 0.75 * np.pi                                  # ディジタルフィルタの阻止域端周波数
As = 30                                            # ディジタルフィルタの減衰量
Wc = 2 * np.tan(wc / 2)                            # 遮断周波数のプリワーピング
print('Wc =\n', Wc)
Ws = 2 * np.tan(ws / 2)                            # 阻止域端周波数のプリワーピング
print('Ws =\n', Ws)
N, _ = signal.buttord(Wc, Ws, 3, As, analog=True)  # プロトタイプフィルタの次数
print('N =\n', N)
bs1, as1 = signal.butter(N, Wc, analog=True)       # プロトタイプフィルタの設計
print('bs1 =\n', bs1)
print('as1 =\n', as1)
bz, az = signal.bilinear(bs1, as1, 1)              # 双1次z変換によるディジタルフィルタの設計
print('bz =\n', bz)
print('az =\n', az)
W = np.linspace(0, 5, 512, endpoint=False)
_, Ha = signal.freqs(bs1, as1, W)                  # プロトタイプフィルタの周波数応答
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(W, 20 * np.log10(np.abs(Ha)))             # 振幅特性の図示
ax1.set_xlim(0, 5); ax1.set_ylim(-60, 5); ax1.grid()
ax1.set_xlabel('Frequency $\Omega$ [rad/sec]')
ax1.set_ylabel('$|H_\mathrm{a}(j\Omega)|$ [dB]')
w = np.linspace(0, np.pi, 512, endpoint=False)
_, Hz = signal.freqz(bz, az, w)                    # ディジタルフィルタの周波数応答
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(w, 20 * np.log10(np.abs(Hz)))             # 振幅特性の図示
ax2.set_xlim(0, np.pi); ax2.set_ylim(-60, 5); ax2.grid()
ax2.set_xlabel('Frequency $\omega$ [rad]')
ax2.set_ylabel('$|H(e^{j\omega})|$ [dB]')
