# たたみこみの計算
# yはhとxのたたみこみ
# y[n] = h[0]x[n] + h[1]x[n-1] + ... + h[n]x[0]
# パラメータ: h = 単位インパルス応答
#          x = 入力
# 戻り値: y = 出力
import numpy as np

def myconvolve(h, x):
    hlength = len(h)                               # hの長さ
    xlength = len(x)                               # xの長さ
    hzero = np.hstack([h, np.zeros(xlength - 1)])  # hのゼロづめ
    xzero = np.hstack([x, np.zeros(hlength - 1)])  # xのゼロづめ
    ylength = hlength + xlength - 1                # 出力yの長さ
    y = np.zeros(ylength)                          # 出力yの初期化
    for n in range(0, ylength):
        for k in range(0, n + 1):
            y[n] = y[n] + hzero[k] * xzero[n - k]  # たたみこみの計算
    return y
