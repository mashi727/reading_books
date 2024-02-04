import numpy as np
from scipy import signal

N = 2                                                     # 与えられた次数
Wc = 1                                                    # 与えられた遮断周波数
k = np.arange(0, 2 * N)
if N % 2 == 0:
    pk = Wc * np.exp(1j * (2 * k + 1) * np.pi / (2 * N))  # N=偶数のときの極
else:
    pk = Wc * np.exp(1j * k * np.pi / N)                  # N=奇数のときの極
pk = pk[np.real(pk) < 0]                                  # 安定な極（実数部が負の極）のみを選択
print('pk =\n', pk)
zr = np.zeros(0)                                          # 零点
b, a = signal.zpk2tf(zr, pk, Wc**N)                       # 零点と極から伝達関数の分子と分母係数を求める
b = np.real(b)                                            # 分子係数の数値計算誤差（虚数部）を除去
print('b =\n', b)
a = np.real(a)                                            # 分母係数の数値計算誤差（虚数部）を除去
print('a =\n', a)
