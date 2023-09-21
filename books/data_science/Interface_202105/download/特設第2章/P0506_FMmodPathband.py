import numpy as np
import wave
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy

rate = 44100.0
targetrate = rate * 100   # アップサンプリングターゲットレート
freq_deviation = 100.0*1e3   # FM変調の周波数偏移
carrierfreq = 0.2e6
signalfreq = 1.0e3
duration = 0.1

if __name__ == "__main__":
    t = np.arange(0, duration, 1/rate)
    samp = np.sin(2.0*np.pi*t*signalfreq)
    # Remove DC offset
    samp = samp - np.mean(samp)
    # Upsample
    upsamp = sig.resample(samp, int(len(samp)*targetrate/rate))
    integrated = scipy.cumsum(upsamp)
    # FM mod
    d_theta = (integrated * 2.0*np.pi*freq_deviation/targetrate)
    basebandfmmodulated = np.exp(1j*d_theta)
    # 疑似搬送波を直交変調
    seq = np.arange(0, carrierfreq*len(basebandfmmodulated)/targetrate, carrierfreq/targetrate)
    carrier = np.exp(1j*2.0*np.pi*seq)
    fmmodulated = carrier * basebandfmmodulated
    # 時系列表示
    plt.figure()
    plt.plot(fmmodulated)
    # スペクトラム表示
    yf = scipy.fft.fft(fmmodulated)
    yf = scipy.fft.fftshift(yf)
    N = int(len(yf))
    xf = np.linspace(-targetrate/2.0, targetrate/2.0, N)
    plt.figure()
    plt.semilogy(xf, np.abs(yf[:N]), '-b')
    plt.show()