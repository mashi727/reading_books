import numpy as np
from mycircconv import mycircconv

h = np.array([8, 4, 2, 1])
x = np.array([1, 2, 3, 4])
hp = np.hstack([h, np.zeros(len(x) - 1)])  # hにゼロづめ
print('hp = \n', hp)
xp = np.hstack([x, np.zeros(len(h) - 1)])  # xにゼロづめ
print('xp = \n', xp)
ycp = mycircconv(hp, xp)                   # 循環たたみこみ
print('ycp = \n', ycp)
