import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-20, 20 + 1)                  # 信号の時刻の範囲
wc = np.pi / 4                              # 信号の帯域幅
x = (wc / np.pi) * np.sinc(n * wc / np.pi)  # 信号の計算
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.stem(n, x)                               # 信号の図示
ax.set_xlim(-20, 20); ax.set_ylim(-0.1, 0.3); ax.grid()
ax.set_xlabel('Time $n$'); ax.set_ylabel('$x[n]$')
