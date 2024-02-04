# プログラム例 example_2.py
print('ライプニッツの公式による π の近似値')
N = 10**3
S = 0
for n in range(0, N + 1):
    p = 1 / (2 * n + 1) 
    S = S + (-1)**n * p
pi = 4 * S
print('pi= \n',pi)