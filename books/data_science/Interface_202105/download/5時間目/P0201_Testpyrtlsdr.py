# Test pyrtlsdr

from rtlsdr import RtlSdr   # RTL-SDRパッケージをインポート

sdr = RtlSdr()              # RTL-SDRクラスのインスタンス"sdr"を作成

# SDRデバイスのパラメータを設定
sdr.sample_rate = 2.048e6   # サンプルレート(Hz)
sdr.center_freq = 80e6      # 中心周波数(Hz)
sdr.freq_correction = 60    # 周波数補正(PPM)
sdr.gain = 'auto'           # チューナーアンプのゲイン(ここでは「自動」)

# SDRデバイス(sdr)から指定したサンプル数のI-Qサンプルを受信して表示
print(sdr.read_samples(512))