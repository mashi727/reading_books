import numpy as np
import wave
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy
import commpy

bits = np.random.randint(0, 2, 1024)
rate = 9600
targetrate = rate * 15

if __name__ == "__main__":
    # BPSK変調
    bpsk_mod = bits*2 - 1.0
    # QPSK変調
    qpsk_mod = bpsk_mod[::2]+bpsk_mod[1::2]*1.0j
    qpsk_mod = qpsk_mod*(1.0/np.sqrt(2.0))
    # アップサンプリング
    upsampled = commpy.utilities.upsample(qpsk_mod, 15)
    # 時系列表示
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)
    ax1.plot(upsampled.real, drawstyle='steps-post')
    ax2.plot(upsampled.imag, drawstyle='steps-post')
    # スペクトラム表示
    yf = scipy.fft.fft(upsampled)
    yf = scipy.fft.fftshift(yf)
    N = int(len(yf))
    xf = np.linspace(-targetrate/2.0, targetrate/2.0, N)
    plt.figure()
    plt.semilogy(xf, np.abs(yf[:N]), '-b')
    # QPSKコンスタレーション表示
    plt.figure()
    plt.scatter(upsampled.real, upsampled.imag)
    plt.show()