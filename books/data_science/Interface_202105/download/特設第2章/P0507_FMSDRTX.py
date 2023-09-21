import numpy as np
import wave
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy
import adi

wavfilename = "loop100513.wav"
targetrate = 44100.0 * 12   # アップサンプリングターゲットレート
freq_deviation = 50.0*1e3   # FM変調の周波数偏移

def ReadWavFileAndPlot(FileName):
    wav = wave.open(FileName, "r")
    af = wav.readframes(wav.getnframes())
    wav.close()
    rate = wav.getframerate()
    samp = np.frombuffer(af, dtype="int16")
    samp = samp[::2]
    max = np.max(np.abs(samp))
    samp = np.hstack((samp, [0]*(1024-(len(samp)%1024))))
    return samp, rate, max

def SdrTxWithPluto(data):
    sdr = adi.Pluto()
    sdr.tx_rf_bandwidth = 100000
    sdr.tx_lo = 79000000
    sdr.sample_rate = int(targetrate)
    sdr.tx_hardwaregain = 0
    for idx in range(0, len(data), 1024):
        sdr.tx(data[idx:idx+1023]*1024)

if __name__ == "__main__":
# 音源の入手先(例)
# サイト名：otosozai.com　https://otosozai.com/
    samp, rate, max = ReadWavFileAndPlot(wavfilename)
    samp = samp / 32768.0   # 振幅の最大値を1.0に正規化
    # Remove DC offset
    samp = samp - np.mean(samp)
    # Upsample
    upsamp = sig.resample(samp, int(len(samp)*targetrate/rate))
    integrated = np.cumsum(upsamp)
    # FM mod
    d_theta = integrated * 2.0*np.pi*freq_deviation/targetrate
    fmmodulated = np.exp(1j*d_theta)
    # PlutoSDRで送信出力
    SdrTxWithPluto(fmmodulated)
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