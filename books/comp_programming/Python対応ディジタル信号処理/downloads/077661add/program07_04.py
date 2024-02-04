import numpy as np
import matplotlib.pyplot as plt

b = np.array([1, 0, 1])                                # 分子係数
a = np.array([1, -0.9, 0.81])                          # 分母係数
zr = np.roots(b)                                       # 零点の計算
print('zr =\n', zr)
pl = np.roots(a)                                       # 極の計算
print('pl =\n', pl)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
theta = np.linspace(-np.pi, np.pi, 1024, endpoint=False)
uc = np.exp(1j * theta)                                # 単位円の計算
ax.plot(np.real(uc), np.imag(uc), '-')                 # 単位円の図示
ax.plot(np.real(zr), np.imag(zr), 'o', label='Zeros')  # 零点の図示
ax.plot(np.real(pl), np.imag(pl), 'x', label='Poles')  # 極の図示
ax.axis('equal'); ax.grid()
ax.set_xlabel('Real part')
ax.set_ylabel('Imaginary part')
ax.legend(loc='upper left')
