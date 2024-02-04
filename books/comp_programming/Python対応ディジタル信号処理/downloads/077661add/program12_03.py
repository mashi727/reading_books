import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

wc1 = np.pi / 8; wc2 = np.pi / 4                  # 周波数帯域
L = 10
n = np.arange(-L, L + 1)
n1, n2 = np.meshgrid(n, n)                        # 空間軸の範囲
x = (wc1 / np.pi * np.sinc(wc1 * n1 / np.pi)) \
    * (wc2 / np.pi * np.sinc(wc2 * n2 / np.pi))   # 信号の計算
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
ax1.plot_wireframe(n1, n2, x)                     # データの3次元プロット（ワイヤーフレーム）
ax1.set_xlim(-L, L); ax1.set_ylim(-L, L); ax1.set_zlim(-0.01, 0.035)
ax1.set_xlabel('$n_1$')
ax1.set_ylabel('$n_2$')
ax1.set_zlabel('$x[n_1,n_2]$')
ax1.view_init(elev=20, azim=-130)                 # 3次元プロットの表示角度の調整
