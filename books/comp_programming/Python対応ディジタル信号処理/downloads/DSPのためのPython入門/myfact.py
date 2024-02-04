# 階乗の再帰的定義 myfact.py

def myfact(n):
    if n == 0:
        return 1
    else:
        return n*myfact(n-1)