import numpy as np
import wave
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy

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
    samp, rate, max = ReadWavFileAndPlot("loop100513.wav")
    # Upsample
    targetrate = 44100*5
    upsamp = sig.resample(samp, int(len(samp)*targetrate/rate))
    integrated = scipy.cumsum(upsamp)
    plt.figure()
    plt.plot(integrated)
    plt.show()