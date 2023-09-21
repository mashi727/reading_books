import numpy as np
import wave
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy
import math
import adi
import pyaudio

SampleRate = 44100.0 * 12         # ベースバンドサンプルレート
Targetrate = 44100.0 * 5    # ダウンサンプリングターゲットレート(44.1ksps)
Audiorate  = 44100.0         # オーディオサンプリングレート

CenterFreq = 84.7e6     # FMラジオ放送の周波数
TragetRate = 44.1e3
UseAGC     = False
RfAmpGain  = 50

def SdrInit():
    # SDRデバイス(PlutoSDR)の初期化
    sdr = adi.Pluto()
    sdr.rx_rf_bandwidth = 150000
    sdr.rx_lo = 80000000 #FM被変調信号(放送波)の周波数(FM横浜84.7MHz)
    sdr.sample_rate = int(SampleRate)
    sdr.rx_hardwaregain = 50
    sdr.rx_buffer_size = 32768*10
    return sdr

def SdrRxFromPluto(sdr):
    # PlutoSDRから指定されたサイズのI-Qサンプルを読み出してリターン
    rx = sdr.rx()
    return rx

if __name__ == "__main__":
    # PyAudioの初期化
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, output=True)
    # SDRデバイスの初期化
    sdr = SdrInit()
    while True:
        # SDRデバイスからI-Q信号を入力
        fmmodulated = SdrRxFromPluto(sdr)
        # Downsample1
        downsamp = sig.resample(fmmodulated, int(len(fmmodulated)*Targetrate/Samplerate))
        # FM復調処理
        demod = np.angle(downsamp[1:]*np.conj(downsamp[:-1]))
        # Downsample2
        a = sig.resample(demod, int(len(demod)*Audiorate/Targetrate))
        a = a/math.pi * 32768.0
        audio = a.astype(np.int16)
        # PyAudioで音声を再生
        stream.write(audio)
