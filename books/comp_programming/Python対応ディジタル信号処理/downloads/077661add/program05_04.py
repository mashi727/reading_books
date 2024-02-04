import numpy as np
from mylfilter import mylfilter
import matplotlib.pyplot as plt

b = np.array([0.5]); a = np.array([1, -0.5])  # 係数
nend = 10; n = np.arange(0, nend + 1)         # 時刻の範囲

# 単位インパルス応答
xi = np.hstack([1, np.zeros(nend)])           # 単位インパルス入力
yi = mylfilter(b, a, xi)                      # フィルタリング
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.stem(n, yi)                               # 単位インパルス応答の図示
ax1.set_xlim(0, nend); ax1.set_ylim(0, 1.2); ax1.grid()
ax1.set_xlabel('Time $n$'); ax1.set_ylabel('$y_\mathrm{i}[n]$')
print('yi = \n', yi)

# 単位ステップ応答
xs = np.ones(nend + 1)                        # 単位ステップ入力
ys = mylfilter(b, a, xs)                      # フィルタリング
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.stem(n, ys)                               # 単位ステップ応答の図示
ax2.set_xlim(0, nend); ax2.set_ylim(0, 1.2); ax2.grid()
ax2.set_xlabel('Time $n$'); ax2.set_ylabel('$y_\mathrm{s}[n]$')
print('ys = \n', ys)
