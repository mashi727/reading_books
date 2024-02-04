# ヘロンの公式の関数定義 heron.py
import numpy as np

def heron(a, b, c):
    s = (a + b + c) / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    return area
