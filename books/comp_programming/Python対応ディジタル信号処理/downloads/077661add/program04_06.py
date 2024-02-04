import numpy as np

h = np.array([8, 4, 2, 1])
x = np.array([1, 2, 3, 4])
M = len(h) + len(x) - 1                    # たたみこみの長さM
Mbin = format(M, 'b')                      # Mの2進数表記の文字列
if Mbin.count('1') != 1:                   # Mが2のべき乗でなければ
    M = 2**len(Mbin)                       # Mを2のべき乗になるよう変更
hp = np.hstack([h, np.zeros(M - len(h))])  # hにゼロづめ
print('hp = \n', hp)
xp = np.hstack([x, np.zeros(M - len(x))])  # xにゼロづめ
print('xp = \n', xp)
Hp = np.fft.fft(hp, M)                     # hpのFFT
print('Hp = \n', Hp)
Xp = np.fft.fft(xp, M)                     # xpのFFT
print('Xp = \n', Xp)
Yp = Hp * Xp                               # HpとXpの積Yp
print('Yp = \n', Yp)
yp = np.fft.ifft(Yp, M)                    # YpのIFFT yp
print('yp = \n', yp)
