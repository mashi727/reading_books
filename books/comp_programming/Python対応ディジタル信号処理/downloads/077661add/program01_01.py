import numpy as np                      # 数値計算ライブラリを別名np として読み込み
import matplotlib.pyplot as plt         # グラフ描画ライブラリを別名plt として読み込み

RC = 1                                  # 時定数
t = np.arange(0, 5, 0.01)               # 0～5までの時刻の範囲，刻み0.01
y = 1 - np.exp(-t / RC)                 # 出力の計算
print('y = \n', y)                      # 出力信号値の表示
fig = plt.figure()                      # Figureの生成
ax = fig.add_subplot(1, 1, 1)           # Figureを1行1列に分割し，1番目に
ax.plot(t, y)                           # 出力の図示
ax.set_xlim(0, 5); ax.set_ylim(0, 1.5)  # 座標軸の範囲の指定
ax.grid()                               # 格子の図示
ax.set_xlabel('Time $t$ [sec]')         # 座標軸名の設定(x軸)
ax.set_ylabel('Output $y(t)$')          # 座標軸名の設定(y軸)
