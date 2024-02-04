# 基数2の時間間引き形高速フーリエ変換(FFT)
# パラメータ: x = 2のべき乗の長さの信号
# 戻り値: X = x の離散フーリエ変換
# 参考文献: L. R. Rabiner and B. Gold: Theory and Application of
#           Digital Signal Processing, p. 367, Prentice-Hall, 1975.
import numpy as np
import sys

def myfft(x):
    X = np.complex_(x)                                 # 高速フーリエ変換Xの準備（複素数）
    N = len(x)                                         # 信号の長さN
    Nbin = format(N, 'b')                              # Nの2進数表記の文字列

    # Nが2のべき乗でない場合
    if Nbin.count('1') != 1:                           # Nが2のべき乗でなければ
        print('N is not a power of 2')                 # エラーの表示
        sys.exit()                                     # 終了

    # N=1の場合
    if N == 1:                                         # Nが1であれば
        return X                                       # Xの値をそのまま出力

    # N>1の場合(一般の場合)
    # 信号xのビット逆順
    N2 = N // 2
    q = 0
    for r in range(1, N - 1):                          # ビット逆順開始
        n = N2
        while n <= q:
            q = q - n
            n = n // 2
        q = q + n                                      # qはrのビット逆順
        if r < q:                                      # r<qのとき
            X[q], X[r] = X[r], X[q]                    # X[q]とX[r]の値の入れ替え

    # 回転因子の計算
    WNk = np.exp(-1j * 2 * np.pi * np.arange(0, N2) / N)

    # バタフライによる変換
    p = len(Nbin) - 1                                  # べき指数p (N=2**p)
    for s in range(1, p + 1):                          # 第s段階
        u = 2**s
        v = 2**(s - 1)                                 # v = u // 2
        w = 2**(p - s)                                 # w = N // u
        for m in range(0, v):
            k = m * w
            for q in range(m, N, u):
                r = q + v
                WNkX = WNk[k] * X[r]
                X[q], X[r] = X[q] + WNkX, X[q] - WNkX  # バタフライ
    return X
