import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 共通の準備事項
zr = np.array([0, 0])                  # 二つの零点の位置
nend = 50; n = np.arange(0, nend + 1)  # 時刻の範囲
x = np.zeros(nend + 1); x[0] = 1       # 単位インパルスの入力

# 異なる実数の極の場合
pl1 = np.array([0.4, 0.9])             # 極の位置
b1, a1 = signal.zpk2tf(zr, pl1, 1)     # 零点・極から2次フィルタの分子・分母係数
h1 = signal.lfilter(b1, a1, x)         # フィルタリングによる単位インパルス応答
fig1 = plt.figure()
ax1 = fig1.add_subplot(3, 1, 1)
ax1.stem(n, h1)                        # 単位インパルス応答の図示
ax1.set_xlim(0, nend); ax1.set_ylim(0, 1.5); ax1.grid()
ax1.set_xlabel('Time $n$'); ax1.set_ylabel('$h_1[n]$')

# 重根の極の場合
pl2 = np.array([0.9, 0.9])             # 以下同じ
b2, a2 = signal.zpk2tf(zr, pl2, 1)
h2 = signal.lfilter(b2, a2, x)
fig2 = plt.figure()
ax2 = fig2.add_subplot(3, 1, 1)
ax2.stem(n, h2)
ax2.set_xlim(0, nend); ax2.set_ylim(0, 4); ax2.grid()
ax2.set_xlabel('Time $n$'); ax2.set_ylabel('$h_2[n]$')

# 複素共役の極の場合
pl3 = np.array([0.9 * np.exp( 1j * np.pi / 6),
                0.9 * np.exp(-1j * np.pi / 6)])
b3, a3 = signal.zpk2tf(zr, pl3, 1)
h3 = signal.lfilter(b3, a3, x)
fig3 = plt.figure()
ax3 = fig3.add_subplot(3, 1, 1)
ax3.stem(n, h3)
ax3.set_xlim(0, nend); ax3.set_ylim(-1, 2); ax3.grid()
ax3.set_xlabel('Time $n$'); ax3.set_ylabel('$h_3[n]$')
