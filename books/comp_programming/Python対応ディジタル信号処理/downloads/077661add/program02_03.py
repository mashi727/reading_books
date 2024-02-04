import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 矩形波
w = np.linspace(-np.pi, np.pi, 1024, endpoint=False)  # 周波数の範囲と刻み
rc = np.ones(8)                                       # 矩形波
_, Rc = signal.freqz(rc, 1, w)                        # 矩形波の離散時間フーリエ変換
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(w, np.abs(Rc))                               # 矩形波の振幅スペクトルの図示
ax1.set_xlim(-np.pi, np.pi); ax1.set_ylim(0, 10); ax1.grid()
ax1.set_xlabel('Frequency $\omega$ [rad]')
ax1.set_ylabel('$|R_\mathrm{c}(e^{j\omega})|$')

# 指数関数
n = np.arange(0, 2**12)                               # 時刻を十分大きな値までとる
a = 0.5**n                                            # 指数関数
_, A = signal.freqz(a, 1, w)                          # 指数関数の離散時間フーリエ変換
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(w, np.abs(A))                                # 指数関数の振幅スペクトルの図示
ax2.set_xlim(-np.pi, np.pi); ax2.set_ylim(0, 3); ax2.grid()
ax2.set_xlabel('Frequency $\omega$ [rad]')
ax2.set_ylabel('$|A(e^{j\omega})|$')
