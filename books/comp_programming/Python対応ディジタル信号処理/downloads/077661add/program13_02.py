import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 単位インパルス応答
h = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
N1, N2 = np.shape(h)                               # フィルタのサイズ
L = max(N1, N2)
hzero = np.zeros([3 * L + 1, 3 * L + 1])
hzero[L:L+N1, L:L+N1] = h                          # 図示のためのゼロづめ
n = np.arange(-L, 2 * L + 1)                       # 図示の区間
n1, n2 = np.meshgrid(n, n)
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
ax1.plot_wireframe(n1, n2, hzero)                  # フィルタの図示
ax1.set_xlim(-L, 2 * L); ax1.set_ylim(-L, 2 * L); ax1.set_zlim(0, 0.3)
ax1.set_xlabel('$n_1$')
ax1.set_ylabel('$n_2$')
ax1.set_zlabel('$h[n_1, n_2]$')
ax1.view_init(elev=20, azim=-130)

# 2次元周波数応答
M = 16                                             # 周波数の刻み数
w = np.linspace(-np.pi, np.pi, M, endpoint=False)  # 周波数の範囲と刻み
w1, w2 = np.meshgrid(w, w)
H = np.zeros([M, M], dtype=np.complex128)          # 周波数応答の初期化
for iw1 in range(0, M):
    for iw2 in range(0, M):
        for in1 in range(0, N1):
            for in2 in range(0, N2):
                H[iw1, iw2] \
                    = H[iw1, iw2] + h[in1, in2] \
                    * np.exp(-1j * w1[iw1, iw2] * n1[in1, in2]) \
                    * np.exp(-1j * w2[iw1, iw2] * n2[in1, in2])
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1, projection='3d')
ax2.plot_wireframe(w1, w2, np.abs(H))              # 振幅特性の図示
ax2.set_xlim(-np.pi, np.pi); ax2.set_ylim(-np.pi, np.pi)
ax2.set_zlim(0, 1)
ax2.set_xlabel('Frequency $\omega_1$ [rad]')
ax2.set_ylabel('Frequency $\omega_2$ [rad]')
ax2.set_zlabel('$|H(e^{j\omega_1},e^{j\omega_2})|$')
ax2.view_init(elev=20, azim=-130)
