import numpy as np
import wave
import matplotlib.pyplot as plt
import scipy

def ReadWavFileAndPlot(FileName):
    wav = wave.open(FileName, "r")
    af = wav.readframes(wav.getnframes())
    wav.close()

    print("Channel: ", wav.getnchannels())
    print("Sample width: ", wav.getsampwidth())
    print("Frame Rate: ", wav.getframerate())
    print("Frame num: ", wav.getnframes())
    print("Params: ", wav.getparams())
    print("Total time: ", 1.0 * wav.getnframes() / wav.getframerate())

    x = np.frombuffer(af, dtype="int16")/32768.0
    x = x[::2]  # Lチャネルだけ取り出す。
    # 音声信号の時系列グラフ表示
    plt.figure()
    plt.plot(x)
    # スペクトラム表示
    yf = scipy.fft.fft(x)
    N = int(len(yf)/2)
    xf = np.linspace(0.0, 44100.0/2.0, N)
    plt.figure()
    plt.semilogy(xf, np.abs(yf[:N]), '-b')
    plt.show()

if __name__ == "__main__":
    ReadWavFileAndPlot("loop100513.wav")
# 音源の入手先(例)
# サイト名：otosozai.com　https://otosozai.com/
