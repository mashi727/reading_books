# -*- coding: utf-8 -*-
"""
    File name:      WaveProcessing.py
    Author:         Tetsuro Tatsuoka
    Date created:   09/13/2016
    Last modified:  09/13/2016
    Version:        1.0.0
    Python version: 3.5.2
"""
import numpy as np
from scipy import signal

import IirFilt

class WaveProcessing:
    """　波形処理を担当するクラス
    """
    DEFAULT_SAMPLE_FREQ = 200
    DEFAULT_CALC_AMP_SEC = 5
    DEFAULT_CH_NUM = 2
    DEFAULT_HPF_FC = 1
    HPF_ORDER = 1
    LPF_ORDER = 2
    DEFAULT_LPF_FC = 30
    DEFAULT_NOTCH_F0 = 60

    def __init__(self, ch_num=DEFAULT_CH_NUM, fs=DEFAULT_SAMPLE_FREQ,
                 calc_amp_sec=DEFAULT_CALC_AMP_SEC):
        """ コンストラクタ
        """
        self.fs = fs

        # フィルタ有効/無効フラグ
        self.hpf_enable = []
        self.lpf_enable = []
        self.notch_enable = []
        for i in range(ch_num):
            self.hpf_enable.append(True)
            self.lpf_enable.append(True)
            self.notch_enable.append(True)
        
        # フィルタオブジェクト
        self.nyq = fs / 2
        self.hpf_1st = []
        self.lpf_2nd = []
        self.notch = []
        bh, ah = signal.butter(self.HPF_ORDER, self.DEFAULT_HPF_FC/self.nyq, 'high')
        bl, al = signal.butter(self.LPF_ORDER, self.DEFAULT_LPF_FC/self.nyq, 'low')
        bn, an = IirFilt.IirFilt.design_notch(self.DEFAULT_NOTCH_F0)
        for i in range(ch_num):
            self.hpf_1st.append(IirFilt.IirFilt(bh, ah, self.HPF_ORDER))
            self.lpf_2nd.append(IirFilt.IirFilt(bl, al, self.LPF_ORDER))
            self.notch.append(IirFilt.IirFilt(bn, an))

        # 波形バッファ
        self.buf_size = int(fs*calc_amp_sec)
        self.buf_ptr = []
        self.wav_buf = []
        for i in range(ch_num):
            self.buf_ptr.append(0)
            self.wav_buf.append(np.zeros(self.buf_size))

    def proc_each(self, dat, ch_idx):
        """ データ1ポイントずつの処理
        """
        filtered_data = dat
        if self.hpf_enable[ch_idx]:
            filtered_data = self.hpf_1st[ch_idx].apply_filter(filtered_data)
        if self.lpf_enable[ch_idx]:
            filtered_data = self.lpf_2nd[ch_idx].apply_filter(filtered_data)
        if self.notch_enable[ch_idx]:
            filtered_data = self.notch[ch_idx].apply_filter(filtered_data)
        self.wav_buf[ch_idx][self.buf_ptr[ch_idx]] = filtered_data
        self.buf_ptr[ch_idx] = (self.buf_ptr[ch_idx] + 1) % self.buf_size
        return filtered_data

    def calc_amp(self, ch_idx):
        """ 振幅を算出する
        """
        return max(self.wav_buf[ch_idx]) - min(self.wav_buf[ch_idx])

    def set_hpf_coef(self, fc, ch_idx):
        """ ハイパスフィルタの係数を設定する
        """
        if fc == 'OFF':
            self.hpf_enable[ch_idx] = False
        else:
            bh, ah = signal.butter(self.HPF_ORDER, float(fc)/self.nyq, 'high')
            self.hpf_1st[ch_idx].set_coef(bh, ah)
            self.hpf_enable[ch_idx] = True

    def set_lpf_coef(self, fc, ch_idx):
        """ ローパスフィルタの係数を設定する
        """
        if fc == 'OFF':
            self.lpf_enable[ch_idx] = False
        else:
            bl, al = signal.butter(self.LPF_ORDER, float(fc)/self.nyq, 'low')
            self.lpf_2nd[ch_idx].set_coef(bl, al)
            self.lpf_enable[ch_idx] = True

    def set_notch_coef(self, fo, ch_idx):
        """ ノッチフィルタの係数を設定する
        """
        if fo == 'OFF':
            self.notch_enable[ch_idx] = False
        else:
            bn, an = IirFilt.IirFilt.design_notch(float(fo))
            self.notch[ch_idx].set_coef(bn, an)
            self.notch_enable[ch_idx] = True

    def set_calc_amp_sec(self, calc_amp_sec):
        """ 振幅計測用の波形バッファ長を秒単位で設定する
        """
        self.buf_size = int(self.fs*calc_amp_sec)
