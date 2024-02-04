import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 線形位相FIRフィルタ
hlin = np.array([1, 4, 6, 4, 1]) / 16           # 線形位相フィルタの単位インパルス応答
w = np.linspace(0, np.pi, 512, endpoint=False)  # 周波数の範囲と刻み
_, Hlin = signal.freqz(hlin, 1, w)              # 周波数応答の計算
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax1.plot(w, np.abs(Hlin))                       # 振幅特性の図示
ax1.set_xlim(0, np.pi); ax1.set_ylim(0, 1.2); ax1.grid()
ax1.set_xlabel('Frequency $\omega$ [rad]')
ax1.set_ylabel('$|H_\mathrm{l}(e^{j\omega})|$')
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(w, np.unwrap(np.angle(Hlin)))          # 位相特性の図示
ax2.set_xlim(0, np.pi); ax2.set_ylim(-10, 0); ax2.grid()
ax2.set_xlabel('Frequency $\omega$ [rad]')
ax2.set_ylabel('$\\theta_\mathrm{l}(\omega)$ [rad]')

# 線形位相フィルタリング
ones = np.ones(8)
x = np.hstack([ones, -ones, ones, -ones])       # 入力信号
n = np.arange(0, np.size(x))                    # 時刻の範囲
fig3 = plt.figure()
ax3 = fig3.add_subplot(1, 1, 1)
ax3.stem(n, x)                                  # 入力の図示
ax3.set_xlim(0, np.size(x) - 1); ax3.set_ylim(-2, 2); ax3.grid()
ax3.set_xlabel('Time $n$')
ax3.set_ylabel('$x[n]$')
ylin = signal.lfilter(hlin, 1, x)               # 線形位相フィルタリング
fig4 = plt.figure()
ax4 = fig4.add_subplot(1, 1, 1)
ax4.stem(n, ylin)                               # 出力の図示
ax4.set_xlim(0, np.size(x) - 1); ax4.set_ylim(-2, 2); ax4.grid()
ax4.set_xlabel('Time $n$')
ax4.set_ylabel('$y_\mathrm{l}[n]$')

# 非線形位相IIRフィルタリング
alpha = 0.8                                     # 全域通過フィルタのパラメータ
b = np.array([-alpha, 1])                       # 分子係数
a = np.array([1, -alpha])                       # 分母係数
_, Hap = signal.freqz(b, a, w)                  # 周波数応答の計算
Hnl = Hap * Hlin                                # 縦続接続の周波数応答の計算
fig5 = plt.figure()
ax5 = fig5.add_subplot(1, 1, 1)
ax5.plot(w, np.abs(Hnl))                        # 振幅特性の図示
ax5.set_xlim(0, np.pi); ax5.set_ylim(0, 1.2); ax5.grid()
ax5.set_xlabel('Frequency $\omega$ [rad]')
ax5.set_ylabel('$|H_\mathrm{nl}(e^{j\omega})|$')
fig6 = plt.figure()
ax6 = fig6.add_subplot(1, 1, 1)
ax6.plot(w, np.unwrap(np.angle(Hnl)))           # 位相特性の図示
ax6.set_xlim(0, np.pi); ax6.set_ylim(-10, 0); ax6.grid()
ax6.set_xlabel('Frequency $\omega$ [rad]')
ax6.set_ylabel('$\\theta_\mathrm{nl}(\omega)$ [rad]')
fig7 = plt.figure()
ax7 = fig7.add_subplot(1, 1, 1)
ax7.stem(n, x)                                  # 入力の図示
ax7.set_xlim(0, np.size(x) - 1); ax7.set_ylim(-2, 2); ax7.grid()
ax7.set_xlabel('Time $n$')
ax7.set_ylabel('$x[n]$')
ynonlin = signal.lfilter(b, a, ylin)            # 非線形位相フィルタリング
fig8 = plt.figure()
ax8 = fig8.add_subplot(1, 1, 1)
ax8.stem(n, ynonlin)                            # 出力の図示
ax8.set_xlim(0, np.size(x) - 1); ax8.set_ylim(-2, 2); ax8.grid()
ax8.set_xlabel('Time $n$')
ax8.set_ylabel('$y_\mathrm{nl}[n]$')
