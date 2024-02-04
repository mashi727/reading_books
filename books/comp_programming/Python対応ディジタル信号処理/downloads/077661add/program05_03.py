import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

nend = 10
n = np.arange(0, nend + 1)   # 時刻の範囲
x = np.ones(nend + 1)        # 単位ステップ入力

# alpha = 0.5 のときの単位ステップ応答
alpha1 = 0.5
h1 = alpha1**n               # 単位インパルス応答
y1 = signal.convolve(h1, x)  # たたみこみ
y1 = y1[0:nend+1]            # 出力の図示範囲の制限
print('y1 = \n', y1)
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.stem(n, y1)              # 出力の図示
ax1.set_xlim(0, nend); ax1.set_ylim(0, 2.5); ax1.grid()
ax1.set_xlabel('Time $n$'); ax1.set_ylabel('$y[n]$')

# 以下同様にalpha = -0.5 のときの単位ステップ応答
alpha2 = -0.5
h2 = alpha2**n
y2 = signal.convolve(h2, x)
y2 = y2[0:nend+1]
print('y2 = \n', y2)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.stem(n, y2)
ax2.set_xlim(0, nend); ax2.set_ylim(0, 2.5); ax2.grid()
ax2.set_xlabel('Time $n$'); ax2.set_ylabel('$y[n]$')
