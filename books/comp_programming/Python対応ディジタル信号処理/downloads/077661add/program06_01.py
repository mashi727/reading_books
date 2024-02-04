import numpy as np
import matplotlib.pyplot as plt

Nz = np.array([1, 2, 1])                               # 分子係数
Dz = np.array([1, -1, 0.75])                           # 分母係数
zr = np.roots(Nz)                                      # 零点の計算（N(z)の根）
print('zr =\n', zr)
pl = np.roots(Dz)                                      # 極の計算（D(z)の根）
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
ax.legend(loc='upper left')                            # 凡例
