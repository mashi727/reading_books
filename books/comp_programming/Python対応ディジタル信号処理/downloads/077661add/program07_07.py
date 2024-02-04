import numpy as np
from scipy import signal

b = np.array([1, -1, -3, 1, 2])                 # 分子係数
a = np.array([1, 0.75, 0.125, -0.25, -0.0625])  # 分母係数
sos = signal.tf2sos(b, a)                       # 2次伝達関数の分子・分母
print('sos =\n', sos)
