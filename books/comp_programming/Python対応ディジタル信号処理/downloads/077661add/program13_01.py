import numpy as np
from scipy import signal

h = np.array([[1, 2], [2, 4]])                   # 単位インパルス応答
x = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # 入力
y = signal.convolve2d(x, h)                      # たたみこみによる出力
print('y =\n', y)
