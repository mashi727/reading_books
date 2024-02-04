# JacksonとMecklenbraeukerにより訂正されたインパルス不変変換
# パラメータ: bp, ap = プロトタイプフィルタの分子・分母係数
#             T = 標本化周期
# 戻り値: bz, az = 設計されたフィルタの分子・分母係数
import numpy as np
from scipy import signal

def myimpinvar(bp, ap, T):
    Ak, sk, k = signal.residue(bp, ap)                     # プロトタイプフィルタの部分分数展開
    JM = -T * np.sum(Ak) / 2                               # JacksonとMecklenbraeukerの補正項
    bz, az = signal.invresz(T * Ak, np.exp(T * sk), [JM])  # 訂正されたインパルス不変変換
    bz = np.real(bz)                                       # 分子係数の数値計算誤差（虚数部）を除去
    az = np.real(az)                                       # 分母係数の数値計算誤差（虚数部）を除去
    return bz, az
