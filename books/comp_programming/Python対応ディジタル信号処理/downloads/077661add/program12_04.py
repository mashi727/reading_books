import numpy as np
from scipy.fftpack import fft2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

N1 = 16; N2 = 16                                          # 信号のサイズ
n1, n2 = np.meshgrid(np.arange(0, N1), np.arange(0, N2))  # インデックスの範囲
a1 = 0.5; a2 = 0.75                                       # 信号のパラメータ
x = (a1**n1) * (a2**n2)                                   # 信号
X = fft2(x)                                               # 2次元離散フーリエ変換
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
ax1.plot_wireframe(n1, n2, x)                             # 信号の図示
ax1.set_xlim(0, N1 - 1); ax1.set_ylim(0, N2 - 1); ax1.set_zlim(0, 1)
ax1.set_xlabel('$n_1$')
ax1.set_ylabel('$n_2$')
ax1.set_zlabel('$x[n_1, n_2]$')
ax1.view_init(elev=30, azim=-130)
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1, projection='3d')
ax2.plot_wireframe(n1, n2, np.abs(X))                     # 振幅スペクトルの図示
ax2.set_xlim(0, N1 - 1); ax2.set_ylim(0, N2 - 1); ax2.set_zlim(0, 8)
ax2.set_xlabel('$k_1$')
ax2.set_ylabel('$k_2$')
ax2.set_zlabel('$|X[k_1, k_2]|$')
ax2.view_init(elev=30, azim=-130)
