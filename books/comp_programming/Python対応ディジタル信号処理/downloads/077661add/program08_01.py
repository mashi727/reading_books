import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 二つの三角関数波の合成
wl = np.pi / 8; wh = 7 * np.pi / 8                    # 周波数の値の設定
nend = 50; n = np.arange(0, nend + 1)                 # 信号の区間
s = np.sin(wl * n)                                    # 望ましい信号s
v = np.cos(wh * n)                                    # 雑音v
x = s + v	# 信号x = s + v の合成
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(n, s, label='$s[n]$')                        # s の図示
ax1.plot(n, v, ':', label='$v[n]$')                   # v の図示
ymax = 1.5
ax1.set_xlim(0, nend); ax1.set_ylim(-ymax, ymax); ax1.grid()
ax1.set_xlabel('Time $n$'); ax1.set_ylabel('$s[n]$ and $v[n]$')
ax1.legend(loc='upper right')
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(n, x)	# x = s + v の図示
ax2.set_xlim(0, nend); ax2.set_ylim(-ymax, ymax); ax2.grid()
ax2.set_xlabel('Time $n$'); ax2.set_ylabel('$x[n]=s[n]+v[n]$')

# FIR フィルタの振幅特性
h = np.array([1, 4, 6, 4, 1]) / 16                    # 単位インパルス応答
w = np.linspace(-np.pi, np.pi, 1024, endpoint=False)  # 周波数の範囲と刻み
_, H = signal.freqz(h, 1, w)                          # 周波数応答
fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)
ax3.plot(w, np.abs(H))                                # 振幅特性の図示
ax3.set_xlim(-np.pi, np.pi); ax3.set_ylim(0, 1.2); ax3.grid()
ax3.set_xlabel('Frequency $\omega$ [rad]')
ax3.set_ylabel('$|H(e^{j\omega})|$')

# FIR フィルタリング
y = signal.lfilter(h, 1, x)                           # フィルタリング
fig4 = plt.figure()
ax4 = fig4.add_subplot(1, 1, 1)
ax4.plot(n, y)                                        # フィルタ出力の図示
ax4.set_xlim(0, nend); ax4.set_ylim(-ymax, ymax); ax4.grid()
ax4.set_xlabel('Time $n$'); ax4.set_ylabel('$y[n]$')
