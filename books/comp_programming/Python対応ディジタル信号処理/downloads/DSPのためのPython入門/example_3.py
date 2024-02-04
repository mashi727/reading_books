# プログラム例　example_3.py
print('ライプニッツの公式による π の近似値')
M = 7
for m in range(0, M+1):
    N = 10**m
    S = 0
    for n in range(0, N + 1):
        p = 1 / (2 * n + 1) 
        S = S + (-1)**n * p
    pi = 4 * S
    print('N = ', N, '    pi = ', pi)
    

