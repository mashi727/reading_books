import numpy as np
from myfft import myfft

x = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # 8点の信号
X = myfft(x)                            # 高速フーリエ変換（FFT）の実行
print('X = \n', X)
magX = np.abs(X)                        # 振幅スペクトル
print('magX = \n', magX)
