import numpy as np
import wave
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy

wavfilename = "loop100513.wav"
targetrate = 44100.0 * 12   # アップサンプリングターゲットレート
freq_deviation = 50.0*1e3   # FM変調の周波数偏移
Outfilename = "FMmodulated.IQ"

def ReadWavFileAndPlot(FileName):
    wav = wave.open(FileName, "r")
    af = wav.readframes(wav.getnframes())
    wav.close()
    rate = wav.getframerate()
    samp = np.frombuffer(af, dtype="int16")
    samp = samp[::2]
    max = np.max(np.abs(samp))
    return samp, rate, max

if __name__ == "__main__":
# 音源の入手先(例)
# サイト名：otosozai.com　https://otosozai.com/
    samp, rate, max = ReadWavFileAndPlot(wavfilename)
    samp = samp / 32768.0   # 振幅の最大値を1.0に正規化
    # Remove DC offset
    samp = samp - np.mean(samp)
    # Upsample
    upsamp = sig.resample(samp, int(len(samp)*targetrate/rate))
    integrated = scipy.cumsum(upsamp)
    # FM mod
    d_theta = integrated * 2.0*np.pi*freq_deviation/targetrate
    fmmodulated = np.exp(1j*d_theta)
    # I-Q File出力
    f = open(Outfilename, 'w')
    fmmodulated.tofile(f)
    f.close()
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