# 離散フーリエ変換DFT
# パラメータ: x = 信号
# 戻り値: X = x の離散フーリエ変換
import numpy as np

def mydft(x):
    N = len(x)                            # 信号の長さ
    kn = np.arange(0, N)                  # 回転因子のインデックス
    WN = np.exp(-1j * 2 * np.pi / N)
    WNkn = WN**kn                         # 回転因子
    X = np.zeros(N, dtype=complex)        # 離散フーリエ変換の初期化
    for k in range(0, N):
        for n in range(0, N):
            p = (k * n) % N               # 回転因子のべき指数の計算（Nを法とする）
            X[k] = X[k] + x[n] * WNkn[p]  # 離散フーリエ変換の計算
    return X
