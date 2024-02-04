# 差分方程式によるディジタルフィルタリング
# 出力y は以下の差分方程式から計算される
# a[0]*y[n] = -a[1]*y[n-1] - ... - a[na]*y[n-na]
#             +b[0]*x[n] + b[1]*x[n-1] + ... + b[nb]*x[n-nb]
# パラメータ: a = フィードバック係数(a[0]≠0)
#             b = フィードフォワード係数
#             x = 入力
# 戻り値: y = 出力
import numpy as np

def mylfilter(b, a, x):
    a = a / a[0]                           # 係数aの正規化
    b = b / a[0]                           # 係数bの正規化
    order = np.max([len(a), len(b)]) - 1   # フィルタ次数
    xlength = len(x)                       # 入力信号の長さ
    x = np.hstack([np.zeros(order), x])    # 入力の前部にゼロづめ
    y = np.zeros(order + xlength)          # 出力の初期化
    # フィルタリング
    for n in range(order, order + xlength):
        for k in range(0, len(b)):
            y[n] = y[n] + b[k] * x[n - k]  # フィードフォワード部の計算
        for k in range(1, len(a)):
            y[n] = y[n] - a[k] * y[n - k]  # フィードバック部の計算
    y = y[order:]                          # 出力の調整（不要なゼロを除く）
    return y
