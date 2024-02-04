# 周波数変換
# パラメータ: b, a = プロトタイプフィルタの分子・分母係数
#             filter_type = 所望のフィルタのタイプ
#             'lp': 低域, 'hp': 高域, 'bp': 帯域, 'bs': 帯域阻止
#             thetac = プロトタイプフィルタの遮断周波数[rad]
#             wc = 所望の遮断周波数（スカラーまたはベクトル）[rad]
# 戻り値: B, A = 設計されるフィルタの分子・分母係数
# 参考文献: V. K. Ingle and J. G. Proakis, Digital Signal Processing
#             using MATLAB, Second Edition, p. 366, Prentice-Hall, 2000.
import numpy as np
from scipy import signal
import sys

def myfreqztrans(b, a, filter_type, thetac, wc):

    # 周波数変換の式の選択
    if filter_type == 'lp':                     # 低域-低域変換
        alpha = np.sin((thetac - wc) / 2) / np.sin((thetac + wc) / 2)
        NZ = np.array([-alpha, 1])
        DZ = np.array([1, -alpha])
    elif filter_type == 'hp':                   # 低域-高域変換
        alpha = -np.cos((thetac + wc) / 2) / np.cos((thetac - wc) / 2)
        NZ = -np.array([alpha, 1])
        DZ =  np.array([1, alpha])
    elif filter_type == 'bp':                   # 低域-帯域変換
        w1, w2 = wc[0], wc[1]
        alpha = np.cos((w2 + w1) / 2) / np.cos((w2 - w1) / 2)
        k = 1 / np.tan((w2 - w1) / 2) * np.tan(thetac / 2)
        NZ = -np.array([(k - 1) / (k + 1), -2 * alpha * k / (k + 1), 1])
        DZ = np.array([1, -2 * alpha * k / (k + 1), (k - 1) / (k + 1)])
    elif filter_type == 'bs':                   # 低域-帯域阻止変換
        w1, w2 = wc[0], wc[1]
        alpha = np.cos((w2 + w1) / 2) / np.cos((w2 - w1) / 2)
        k = np.tan((w2 - w1) / 2) * np.tan(thetac / 2)
        NZ = np.array([(1 - k) / (1 + k), -2 * alpha / (1 + k), 1])
        DZ = np.array([1, -2 * alpha / (1 + k), (1 - k) / (1 + k)])
    else:
        print('freqztrans: filter type error')  # エラーの表示
        sys.exit()                              # 終了

    # 伝達関数の次数
    bord = len(b) - 1                           # プロトタイプフィルタの分子多項式の次数
    aord = len(a) - 1                           # プロトタイプフィルタの分母多項式の次数
    Bord = (len(b) - 1) * (len(NZ) - 1)         # 設計されるフィルタの分子多項式の次数
    Aord = (len(a) - 1) * (len(DZ) - 1)         # 設計されるフィルタの分母多項式の次数

    # 設計されるフィルタの分子係数の計算B(Z) = b(z)|z^(-1)←N(Z)/D(Z)
    B = np.zeros(Bord + 1)
    for m in range(0, bord + 1):
        num = np.array([1.0])
        for k in range(0, m):
            num = signal.convolve(num, NZ)
        den = np.array([1.0])
        for k in range(0, bord - m):
            den = signal.convolve(den, DZ)
        B = B + b[m] * signal.convolve(num, den)

    # 設計されるフィルタの分母係数の計算A(Z) = a(z)|z^(-1)←N(Z)/D(Z)
    A = np.zeros(Aord + 1)
    for m in range(0, aord + 1):
        num = np.array([1.0])
        for k in range(0, m):
            num = signal.convolve(num, NZ)
        den = np.array([1.0])
        for k in range(0, aord - m):
            den = signal.convolve(den, DZ)
        A = A + a[m] * signal.convolve(num, den)
    B = B / A[0]                                # 分子係数の正規化
    A = A / A[0]                                # 分母係数の正規化
    return B, A
