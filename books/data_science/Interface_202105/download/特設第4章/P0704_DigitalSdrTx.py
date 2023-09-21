import numpy as np
import wave
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy
import commpy
import adi

bits = np.random.randint(0, 2, 1024)
rate = 9600.0           # Samples per second
RF_Frequency = 1000000000
targetrate = rate * 100  # Upsampling target rate

def SdrInit():
    sdr = adi.Pluto()
    sdr.tx_rf_bandwidth = 100000        # 帯域幅100kHz
    sdr.tx_lo = RF_Frequency            # 周波数設定
    sdr.sample_rate = int(targetrate)   # サンプリングレート設定
    sdr.tx_hardwaregain = 0             # 送信アンプのゲイン=0dB
    return sdr

def SdrTxWithPluto(sdr, data):
    for idx in range(0, len(data), 1024):
        sdr.tx(data[idx:idx+1023]*1024)

if __name__ == "__main__":
    sdr = SdrInit()
    # BPSK変調
    bpsk_mod = bits*2 - 1.0
    # QPSK変調
    qpsk_mod = bpsk_mod[::2]+bpsk_mod[1::2]*1.0j
    qpsk_mod = qpsk_mod*(1.0/np.sqrt(2.0))
    # アップサンプリングと帯域制限フィルタ
    upsamp = sig.resample(qpsk_mod, int(len(qpsk_mod)*targetrate/rate))
    h_rrc = commpy.filters.rrcosfilter(len(upsamp), 0.8, 1/rate, targetrate)[1]
    upsampled = np.convolve(h_rrc, upsamp)
    while True:
        SdrTxWithPluto(sdr, upsampled)