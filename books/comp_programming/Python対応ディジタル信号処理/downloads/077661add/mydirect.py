# ディジタルバタワース低域フィルタの直接設計
# パラメータ: wc = 遮断周波数[rad]
#             ws = 阻止域端周波数[rad]
#             As = 阻止域減衰量[dB]
# 戻り値: b, a = 設計されたフィルタの分子・分母係数
import math
import numpy as np
from scipy import signal

def mydirect(wc, ws, As):
    # 次数と零点の計算
    N = math.ceil(0.5
                  * np.log10(10**(As / 10) - 1)
                  / np.log10(np.tan(ws / 2) / np.tan(wc / 2)))  # 次数
    zr = -np.ones(N)                                            # 零点
    # q 平面の極の計算
    k = np.arange(0, 2 * N)
    if N % 2 == 0:                                              # N=偶数のとき
        qk = np.tan(wc / 2) * np.exp(1j * (2 * k + 1) * np.pi / (2 * N))
    else:                                                       # N=奇数のとき
        qk = np.tan(wc / 2) * np.exp(1j * k * np.pi / N)
    # z 平面の極の計算
    pk = ((1 + qk) / (1 - qk))                                  # 極
    pl = pk[np.abs(pk) < 1]                                     # 安定な極（絶対値1 未満）を選択
    # 伝達関数の計算
    K = np.abs(np.prod(1 - pl)) / 2**N                          # 伝達関数のゲイン
    b, a = signal.zpk2tf(zr, pl, K)                             # 零点と極，ゲインから伝達関数へ
    # 数値計算誤差（虚数部）を除去
    b = np.real(b)
    a = np.real(a)
    return b, a
