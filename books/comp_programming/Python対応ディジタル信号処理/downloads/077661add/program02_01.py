import numpy as np
import matplotlib.pyplot as plt

nbegin = -5; nend = 10
n = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])   # 時刻の範囲

# 単位インパルス
delta = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])     # n=0で1
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.stem(n, delta)                                                     # 単位インパルス関数を図示
ax1.set_xlim(nbegin, nend); ax1.set_ylim(-0.2, 1.2); ax1.grid()
ax1.set_xlabel('Time $n$'); ax1.set_ylabel('$\delta[n]$')

# 単位ステップ
unitstep = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # n>=0で1
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.stem(n, unitstep)                                                  # 単位ステップ関数を図示
ax2.set_xlim(nbegin, nend); ax2.set_ylim(-0.2, 1.2); ax2.grid()
ax2.set_xlabel('Time $n$'); ax2.set_ylabel('$u_0[n]$')

# ランプ関数
ramp = n * unitstep                                                    # 時刻n>=0でランプの値
fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)
ax3.stem(n, ramp)                                                      # ランプ関数を図示
ax3.set_xlim(nbegin, nend); ax3.set_ylim(-2, 12); ax3.grid()
ax3.set_xlabel('Time $n$'); ax3.set_ylabel('$r[n]$')

# 指数関数
alpha = 0.75
a = (alpha**n) * unitstep                                              # 時刻n>=0で指数関数
fig4 = plt.figure()
ax4 = fig4.add_subplot(1, 1, 1)
ax4.stem(n, a)                                                         # 指数関数を図示
ax4.set_xlim(nbegin, nend); ax4.set_ylim(-0.2, 1.2); ax4.grid()
ax4.set_xlabel('Time $n$'); ax4.set_ylabel('$a[n]$')
