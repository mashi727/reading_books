# -*- coding: utf-8 -*-
"""
    File name:      IirFilt.py
    Author:         Tetsuro Tatsuoka
    Date created:   08/27/2016
    Last modified:  08/27/2016
    Version:        1.0.0
    Python version: 3.5.2
"""
import numpy as np

class IirFilt:
    """　IIRフィルタクラス（1次/2次）
    """
    def __init__(self, b, a, n=2):
        """ コンストラクタ
        """
        self.set_coef(b, a)

        if n != 2:                                      # 2次以外は、1次で動作
            self.apply_filter = self.__apply_filter1    # apply_filterを置き換え

    def set_coef(self, b, a):
        """ 係数を設定しメモリスロットを初期化する
        """
        self.b = b
        self.a = a
        self.s = np.array([0.0, 0.0])

    def reset_mem_slot(self):
        """ メモリスロットのリセット
        """
        self.s = np.array([0.0, 0.0])

    def apply_filter(self, x):
        """ フィルタ処理（2次用）
        """
        y = self.b[0] * x + self.s[0]
        self.s[0] = self.b[1] * x - self.a[1] * y + self.s[1]
        self.s[1] = self.b[2] * x - self.a[2] * y
        return y

    def __apply_filter1(self, x):
        """ フィルタ処理（1次用）
        """
        y = self.b[0] * x + self.s[0]
        self.s[0] = self.b[1] * x - self.a[1] * y
        return y

    @staticmethod
    def design_notch(Fo, Fs=200, Q=2/np.sqrt(2)):
        L = np.tan(np.pi*Fo/Fs)
        D = L**2 + L/Q + 1
        b = np.array([(L**2 + 1)/D, 2*(L**2 - 1)/D, (L**2 + 1)/D])  # Numerator
        a = np.array([1, 2*(L**2 - 1)/D, 1 - 2*L/Q/D])              # Denominator
        return [b, a]
