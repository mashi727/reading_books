# 信号hとxの循環たたみこみ
# パラメータ: h, x = 信号，二つの信号の長さは等しくなければならない
# 戻り値: yc = N点の循環たたみこみの結果
import numpy as np
import sys

def mycircconv(h, x):
    if len(h) != len(x):                           # hとxの長さが異なれば
        print('Length of h and x must be equal')   # エラーの表示
        sys.exit()                                 # 終了
    N = len(h)                                     # hの長さ
    yc = np.zeros(N)                               # ycの初期化
    for n in range(0, N):
        for k in range(0, N):
            yc[n] = yc[n] + h[k] * x[(n - k) % N]  # 循環たたみこみの実行
    return yc
