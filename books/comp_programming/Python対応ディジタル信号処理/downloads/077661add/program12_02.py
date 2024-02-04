import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 2次元信号の生成
x = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 8  # 2次元信号
N1, N2 = np.shape(x)                                 # 信号のサイズ
L = max(N1, N2)
xzero = np.zeros([3 * L + 1, 3 * L + 1])
xzero[L:L+N1, L:L+N2] = x                            # 図示のためのゼロづめ
n = np.arange(-L, 2 * L + 1)                         # 信号の区間
n1, n2 = np.meshgrid(n, n)
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
ax1.plot_wireframe(n1, n2, xzero)                    # 信号の図示
ax1.set_xlim(-L, 2 * L); ax1.set_ylim(-L, 2 * L); ax1.set_zlim(0, 0.5)
ax1.set_xlabel('$n_1$')
ax1.set_ylabel('$n_2$')
ax1.set_zlabel('$x[n_1,n_2]$')
ax1.view_init(elev=20, azim=-130)

# 2次元離散空間フーリエ変換
M = 16                                               # 周波数の刻み数
w = np.linspace(-np.pi, np.pi, M, endpoint=False)    # 周波数の範囲と刻み
w1, w2 = np.meshgrid(w, w)
X = np.zeros([M, M], dtype=np.complex128)            # 2次元離散空間フーリエ変換の初期化
for iw1 in range(0, M):
    for iw2 in range(0, M):
        for in1 in range(0, N1):
            for in2 in range(0, N2):
                X[iw1, iw2] \
                    = X[iw1, iw2] + x[in1, in2] \
                    * np.exp(-1j * w1[iw1, iw2] * n1[in1, in2]) \
                    * np.exp(-1j * w2[iw1, iw2] * n2[in1, in2])
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1, projection='3d')
ax2.plot_wireframe(w1, w2, np.abs(X))                # 振幅特性の図示
ax2.set_xlim(-np.pi, np.pi); ax2.set_ylim(-np.pi, np.pi)
ax2.set_zlim(0, 1)
ax2.set_xlabel('Frequency $\omega_1$ [rad]')
ax2.set_ylabel('Frequency $\omega_2$ [rad]')
ax2.set_zlabel('$|X(e^{j\omega_1},e^{j\omega_2})|$')
ax2.view_init(elev=20, azim=-130)
