import numpy as np
from scipy.fftpack import fft2
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 画像データの読み込みと図示
cameraman = cv2.imread('Cameraman.bmp', 0)                # 画像データの読み込み
x = cameraman[0::8, 0::8]                                 # 画像Cameramanを1/8のサイズに変更
x = np.flipud(x)                                          # 見やすさのために画像の上下を入れ替え
N1 = 32; N2 = 32                                          # 画像の水平・垂直方向のサイズ
n1, n2 = np.meshgrid(np.arange(0, N1), np.arange(0, N2))  # 画像の水平・垂直方向のインデックス
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
ax1.plot_wireframe(n1, n2, x)                             # 画像の立体的図示
ax1.set_xlim(0, N1 - 1); ax1.set_ylim(0, N2 - 1); ax1.set_zlim(0, 500)
ax1.set_xlabel('$n_1$')
ax1.set_ylabel('$n_2$')
ax1.set_zlabel('$x[n_1, n_2]$')
ax1.view_init(elev=60, azim=-110)                         # 視点の設定

# 画像の2次元離散フーリエ変換
X = fft2(x)                                               # 2次元離散フーリエ変換
absX = np.abs(X)                                          # 周波数スペクトルの絶対値
absX[0, 0] = 0                                            # 図示のため直流分を除去
fig2 = plt.figure()
ax2 = fig2.add_subplot(1, 1, 1, projection='3d')
ax2.plot_wireframe(n1, n2, absX)                          # 振幅スペクトルの立体的図示
ax2.set_xlim(0, N1 - 1); ax2.set_ylim(0, N2 - 1)
ax2.set_zlim(0, 2*10**4)
ax2.set_xlabel('$k_1$')
ax2.set_ylabel('$k_2$')
ax2.set_zlabel('$|X[k_1, k_2]|$')
ax2.view_init(elev=30, azim=-120)
