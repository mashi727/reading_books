import numpy as np
import wave
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy

rate = 44100.0 * 12         # ベースバンドサンプルレート
targetrate = 44100.0        # ダウンサンプリングターゲットレート(44.1ksps)
BBfilename = "FMmodulated.IQ"

if __name__ == "__main__":
    # I-Q File入力
    f = open(BBfilename, 'r')
    fmmodulated = np.fromfile(file=f, dtype=complex)
    f.close()
    # Downsample
    downsamp = sig.resample(fmmodulated, int(len(fmmodulated)*targetrate/rate))
    # 時系列表示
    plt.figure()
    plt.plot(downsamp)
    # スペクトラム表示
    yf = scipy.fft.fft(downsamp)
    yf = scipy.fft.fftshift(yf)
    N = int(len(yf))
    xf = np.linspace(-targetrate/2.0, targetrate/2.0, N)
    plt.figure()
    plt.semilogy(xf, np.abs(yf[:N]), '-b')
    plt.show()