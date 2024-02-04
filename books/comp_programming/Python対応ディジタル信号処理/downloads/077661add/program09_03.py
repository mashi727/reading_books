import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 窓関数の長さ（FIRフィルタのタップ長）の決定
wp = 0.2 * np.pi                                    # 通過域端周波数
ws = 0.3 * np.pi                                    # 阻止域端周波数
trwidth = ws - wp                                   # 遷移帯域幅
N = math.ceil(6.6 * np.pi / trwidth)                # ハミング窓の長さ（タップ長）
if N % 2 == 0:                                      # Nが偶数ならば
    N = N + 1                                       # 次の奇数へ修正
print('N = %d' % N)

# 窓関数によるFIRフィルタの設計（単位インパルス応答）
wc = (wp + ws) / 2                                  # 遮断周波数
h = signal.firwin(N, wc / np.pi, window='hamming')  # 設計された単位インパルス応答
n = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1)        # 時刻の範囲
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.stem(n, h)                                      # 単位インパルス応答の図示
ax1.set_xlim(-(N - 1) / 2, (N - 1) / 2); ax1.set_ylim(-0.1, 0.3);
ax1.grid()
ax1.set_xlabel('Time $n$')
ax1.set_ylabel('$h[n]$')

# 設計されたFIRフィルタの周波数応答
w = np.linspace(0, np.pi, 512, endpoint=False)      # 周波数の範囲と刻み
dw = w[1] - w[0]                                    # 周波数の刻み
_, H = signal.freqz(h, 1, w)                        # 周波数応答の計算
maxH = np.max(np.abs(H))                            # 周波数応答の最大値
dBH = 20 * np.log10(np.abs(H) / maxH)               # 振幅特性の正規化
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(w, dBH)                                    # 振幅特性の図示
ax2.set_xlim(0, np.pi); ax2.set_ylim(-80, 5); ax2.grid()
ax2.set_xlabel('Frequency $\omega$ [rad]')
ax2.set_ylabel('$|H(e^{j\omega})|$ [dB]')
Rp = -np.min(dBH[0:math.floor(wp/dw)+1])            # 通過域リップルの確認
print('Rp = %7.4f' % Rp)
As = -np.max(dBH[math.ceil(ws/dw):])                # 阻止域減衰量の確認
print('As = %7.4f' % As)
