import numpy as np
import wave
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy
import commpy

bits = np.random.randint(0, 2, 1024)
rate = 9600.0           # Samples per second
targetrate = rate * 15   # Upsampling target rate

if __name__ == "__main__":
    # BPSK変調
    bpsk_mod = bits*2 - 1.0
    # QPSK変調
    qpsk_mod = bpsk_mod[::2]+bpsk_mod[1::2]*1.0j
    qpsk_mod = qpsk_mod*(1.0/np.sqrt(2.0))
    # アップサンプリングと帯域制限フィルタ
    upsamp = sig.resample(qpsk_mod, int(len(qpsk_mod)*targetrate/rate))
    h_rrc = commpy.filters.rrcosfilter(len(upsamp), 0.8, 1/rate, targetrate)[1]
    upsampled = np.convolve(h_rrc, upsamp)
#    upsampled = upsamp
    # 時系列表示
    plt.figure()
    plt.plot(upsampled.real, drawstyle='steps-post')
    # スペクトラム表示
    yf = scipy.fft.fft(upsampled)
    yf = scipy.fft.fftshift(yf)
    N = int(len(yf))
    xf = np.linspace(-targetrate/2.0, targetrate/2.0, N)
    plt.figure()
    plt.semilogy(xf, np.abs(yf[:N]), '-b')
    # QPSKコンスタレーション表示
    plt.figure()
    plt.plot(upsampled.real, upsampled.imag, lw=1)
    plt.scatter(upsampled.real, upsampled.imag, s=10)
    plt.show()