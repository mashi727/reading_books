import numpy as np
import wave
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy
import math

rate = 44100.0 * 12         # ベースバンドサンプルレート
targetrate = 44100.0 * 5    # ダウンサンプリングターゲットレート(44.1ksps)
audiorate = 44100.0         # オーディオサンプリングレート
BBfilename = "FMmodulated.IQ"
AudioFilename = "rxaudio.wav"

if __name__ == "__main__":
    # I-Q File入力
    f = open(BBfilename, 'r')
    fmmodulated = np.fromfile(file=f, dtype=complex)
    f.close()
    # Downsample1
    downsamp = sig.resample(fmmodulated, int(len(fmmodulated)*targetrate/rate))
    # FM復調処理
    demod = np.angle(downsamp[1:]*np.conj(downsamp[:-1]))
    # Downsample2
    a = sig.resample(demod, int(len(demod)*audiorate/targetrate))
    a = a/math.pi * 32768.0
    audio = a.astype(np.int16)
    wav = wave.open(AudioFilename, mode="wb")
    wav.setnchannels(1)
    wav.setframerate(audiorate)
    wav.setsampwidth(2)
    wav.writeframes(audio)
    wav.close()
    # 時系列表示
    plt.figure()
    plt.plot(audio)
    # スペクトラム表示
    yf = scipy.fft.fft(audio)
    yf = scipy.fft.fftshift(yf)
    N = int(len(yf))
    xf = np.linspace(-audiorate/2.0, audiorate/2.0, N)
    plt.figure()
    plt.semilogy(xf, np.abs(yf[:N]), '-b')
    plt.show()