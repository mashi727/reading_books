import numpy as np
import wave
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy
import math
import array
import rtlsdr
import pyaudio
import asyncio
import queue

Audiorate  = 24000          # オーディオサンプリングレート
SampleRate = Audiorate * 50 # ベースバンドサンプルレート
Targetrate = Audiorate * 5  # ダウンサンプリングターゲットレート
SampleLen  = 1024*55        # RTL-SDRから一度に取得するI-Qサンプル数
BufferSize = 32768*10

CenterFreq = 84.7e6         # FMラジオ放送の周波数
UseAGC     = False
RfAmpGain  = 50

def SdrInit():
    ## SDRデバイス(RTL-SDR)の初期化
    sdr = rtlsdr.RtlSdr()
    sdr.sample_rate = SampleRate
    sdr.center_freq = CenterFreq
    sdr.agc = UseAGC
    sdr.gain = RfAmpGain
    return sdr

def sdrcallback(rxsamples, rtlsdr):
    rxqueue.put(rxsamples)

def SdrRx(sdr):
    sdr.read_samples_async(sdrcallback, SampleLen)
    return True

def SignalDemod2(capture):
    # decimate 1/5 from 1.2MHz to 240kHz
    sigif = scipy.signal.decimate(capture, 5, ftype='iir')
    # convert to continuous phase angle
    phase = np.unwrap(np.angle(sigif))
    # differentiate phase brings into frequency
    pd = np.convolve(phase, [1,-1], mode='valid')
    # decimate 1/10 from 240kHz to 24kHz
    a = scipy.signal.decimate(pd, 10, ftype='iir')
    a = a/math.pi * 32767.0
    audio = a.astype(np.int16)
    return (audio)

def SignalDemod(fmmodulated):
    # Downsample1
    downsamp = sig.resample(fmmodulated, int(len(fmmodulated)*Targetrate/SampleRate))
    # FM復調処理
    demod = np.angle(downsamp[1:]*np.conj(downsamp[:-1]))
    # Downsample2
    a = sig.resample(demod, int(len(demod)*Audiorate/Targetrate))
    a = a/math.pi * 32767.0
    audio = a.astype(np.int16)
    return audio

def pacallback(in_data, frame_count, time_info, status):
    print(status, end='')
    fmsignal = rxqueue.get()
    audio = SignalDemod(fmsignal)
    # # PyAudioで音声を再生
    # buf = array.array('d', audio).tostring()
    return (audio, pyaudio.paContinue)

if __name__ == "__main__":
    # PyAudioの初期化
    p = pyaudio.PyAudio()
    rxqueue = queue.Queue(BufferSize)
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=Audiorate, output=True, stream_callback = pacallback)
    stream.start_stream()
    # SDRデバイスの初期化
    sdr = SdrInit()
    SdrRx(sdr)
    sdr.close()
    stream.close()