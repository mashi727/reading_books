import numpy as np

a = np.array([1, -0.9, 0.81])  # 伝達関数の分母係数
pl = np.roots(a)               # 極の計算
print('pl =\n', pl)
maxpl = np.max(np.abs(pl))     # 極の絶対値の最大値
print('maxpl =\n', maxpl)
if maxpl < 1:                  # 極の絶対値の最大値が1より小さいならば
    print('Stable')            # 安定と表示
else:
    print('Unstable')          # その他のとき不安定と表示
