import numpy as np
import matplotlib.pyplot as plt
import math

GrayCode = [0b0000, 0b0001, 0b0011, 0b0010, 0b0110, 0b0111, 0b0101, 0b0100,
            0b1100, 0b1101, 0b1111, 0b1110, 0b1010, 0b1011, 0b1001, 0b1000]

# グラフ上にテキストを併せて表示する。
# 参考: http://hanairosoft.skr.jp/programming/matplotlib-add-text.html

qpsk_v = np.array([-1.0, 1.0])
qpsk_iq = [x + y for x in qpsk_v for y in qpsk_v*1j]
qpsk_x = np.real(qpsk_iq)*math.sqrt(0.5)
qpsk_y = np.imag(qpsk_iq)*math.sqrt(0.5)
plt.figure()
plt.title('QPSK')
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.scatter(qpsk_x, qpsk_y)
plt.grid()
for i in range(len(qpsk_iq)):
    x = i%2
    y = i//2
    plt.text(qpsk_x[i], qpsk_y[i]-0.1, format(GrayCode[y], "01b")+format(GrayCode[x], "01b"),
     fontsize=8, horizontalalignment="center")

_16qam_v = np.array([-3.0, -1.0, 1.0, 3.0])*(1.0/3.0)
_16qam_iq = [x + y for x in _16qam_v for y in _16qam_v*1j]
_16qam_x = np.real(_16qam_iq)*math.sqrt(0.5)
_16qam_y = np.imag(_16qam_iq)*math.sqrt(0.5)
plt.figure()
plt.title('16QAM')
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.scatter(_16qam_x, _16qam_y)
plt.grid()
for i in range(len(_16qam_iq)):
    x = i%4
    y = i//4
    plt.text(_16qam_x[i], _16qam_y[i]-0.1, format(GrayCode[y], "02b")+format(GrayCode[x], "02b"),
     fontsize=8, horizontalalignment="center")

_64qam_v = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0])*(1.0/7.0)
_64qam_iq = [x + y for x in _64qam_v for y in _64qam_v*1j]
_64qam_x = np.real(_64qam_iq)*math.sqrt(0.5)
_64qam_y = np.imag(_64qam_iq)*math.sqrt(0.5)
plt.figure()
plt.title('64QAM')
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.scatter(_64qam_x, _64qam_y)
plt.grid()
for i in range(len(_64qam_iq)):
    x = i%8
    y = i//8
    plt.text(_64qam_x[i], _64qam_y[i]-0.1, format(GrayCode[y], "03b")+format(GrayCode[x], "03b"),
     fontsize=8, horizontalalignment="center")

_256qam_v = np.array([-15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, -1.0, 
                        1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])*(1.0/15.0)
_256qam_iq = [x + y for x in _256qam_v for y in _256qam_v*1j]
_256qam_x = np.real(_256qam_iq)*math.sqrt(0.5)
_256qam_y = np.imag(_256qam_iq)*math.sqrt(0.5)
plt.figure()
plt.title('256QAM')
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)
plt.scatter(_256qam_x, _256qam_y)
plt.grid()
for i in range(len(_256qam_iq)):
    x = i%16
    y = i//16
    plt.text(_256qam_x[i], _256qam_y[i]-0.1, format(GrayCode[y], "04b")+format(GrayCode[x], "04b"),
     fontsize=8, horizontalalignment="center")

plt.show()