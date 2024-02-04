import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# 格子点を定義
t1, t2 = np.meshgrid(np.arange(-20, 21), np.arange(-20, 21))

# 2次元連続信号の生成
x = np.exp(-0.01 * (t1**2 + t2**2)) \
    + 0.5 * np.exp(-0.03 * ((t1 - 13)**2 + (t2 + 10)**2)) \
    + 0.3 * np.exp(-0.03 * ((t1 + 15)**2 + (t2)**2))

# xを立体図示
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1, projection='3d')              # 3次元プロットを設定
ax1.plot_surface(t1, t2, x, cmap=cm.gray, antialiased=False)  # データの3次元プロット（サーフェス）
ax1.set_xlim(-20, 20); ax1.set_ylim(-20, 20); ax1.set_zlim(0, 1.5)
ax1.set_xlabel('$t_1$')
ax1.set_ylabel('$t_2$')
ax1.set_zlabel('$x(t_1, t_2)$')
ax1.view_init(elev=30, azim=-130)                             # 3次元プロットの表示角度の調整

# xを平面上に濃淡図示
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1)
ax2.pcolor(t1, t2, x, cmap=cm.gray)                           # データの濃淡プロット
ax2.axis('square')
ax2.set_xlabel('$t_1$')
ax2.set_ylabel('$t_2$')
