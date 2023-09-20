# -*- coding: utf-8 -*-
"""
    File name:      ExgBdWavData.py
    Description:    Class of EXGBD waveform data
    Author:         Tetsuro Tatsuoka
    Date created:   09/18/2016
    Last modified:  09/18/2016
    Version:        1.0.0
    Python version: 3.5.2
"""
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import datetime
import IirFilt

TEXT_FILE_EXT = '.txt'
BINARY_FILE_EXT = '.dat'
ATTR_FILE_SUFFIX = '_attr'

GAIN_200_STR = 'x200'
GAIN_200_COEF = 5

ATTR_DAT_CIDX = 1

ATTR_START_TIME_RIDX = 0
ATTR_END_TIME_RIDX = 1
ATTR_CH1_GAIN_RIDX = 2
ATTR_CH2_GAIN_RIDX = 3
ATTR_CH1_FILTER_RIDX = 4
ATTR_CH2_FILTER_RIDX = 5
ATTR_SAMPLING_RATE_RIDX = 6

HPF_ORDER = 1
LPF_ORDER = 2
DEFAULT_HPF_FC = 0.5
DEFAULT_LPF_FC = 30
DEFAULT_NOTCH_F0 = 60

class ExgBdWavData:
    """　EXGBDの波形クラス
    """

    def __init__(
                self,
                wave_filename,
                hpf_fc = DEFAULT_HPF_FC, 
                lpf_fc = DEFAULT_LPF_FC, 
                notch_f0 = DEFAULT_NOTCH_F0
                ):
        filename_without_ext = wave_filename[0:-4]
        attr_filename = filename_without_ext + ATTR_FILE_SUFFIX + TEXT_FILE_EXT

        self.__wave_data = pd.read_table(wave_filename, header=None)
        self.__attr_data = pd.read_table(attr_filename, header=None)
        
        # 波形データの長さを取得
        self.__data_len = len(self.__wave_data)
        
        # 属性データの読み取り
        start_time_str = self.__attr_data.iloc[ATTR_START_TIME_RIDX,ATTR_DAT_CIDX]
        self.__start_time = datetime.datetime.strptime(start_time_str[0:-3], '%Y%m%d_%H%M%S')
        self.__start_time = self.__start_time + \
            datetime.timedelta(microseconds=int(start_time_str[-3:]) * 1000)
        
        end_time_str = self.__attr_data.iloc[ATTR_END_TIME_RIDX,ATTR_DAT_CIDX]
        self.__end_time = datetime.datetime.strptime(end_time_str[0:-3], '%Y%m%d_%H%M%S')
        self.__end_time = self.__end_time + \
            datetime.timedelta(microseconds=int(end_time_str[-3:]) * 1000)

        self.__diff_time = self.__end_time - self.__start_time 
        
        self.__ch1_gain = self.__attr_data.iloc[ATTR_CH1_GAIN_RIDX,ATTR_DAT_CIDX]
        if self.__ch1_gain == GAIN_200_STR:
            self.__wave_data[0] = self.__wave_data[0] * GAIN_200_COEF

        self.__ch2_gain = self.__attr_data.iloc[ATTR_CH2_GAIN_RIDX,ATTR_DAT_CIDX]
        if self.__ch2_gain == GAIN_200_STR:
            self.__wave_data[1] = self.__wave_data[1] * GAIN_200_COEF
        
        self.__ch1_filter = self.__attr_data.iloc[ATTR_CH1_FILTER_RIDX,ATTR_DAT_CIDX]
        self.__ch2_filter = self.__attr_data.iloc[ATTR_CH2_FILTER_RIDX,ATTR_DAT_CIDX]
        
        sampling_rate_str = self.__attr_data.iloc[ATTR_SAMPLING_RATE_RIDX,ATTR_DAT_CIDX]
        self.__sampling_rate = int(sampling_rate_str[0:-2])   # Hzは削除
        
        # 波形データにフィルタを適用
        self.__wave_data = self.__apply_filter(
                                self.__wave_data,
                                self.__sampling_rate,
                                hpf_fc, lpf_fc, notch_f0)

    @property
    def wave_data(self):
        return self.__wave_data
        
    @property
    def wave_length(self):
        return self.__data_len

    def get_index(self, date_time):
        elapsed_time = date_time - self.start_time
        return elapsed_time.total_seconds() * self.__data_len / self.__diff_time.total_seconds()

    @property
    def start_time(self):
        return self.__start_time

    @property
    def end_time(self):
        return self.__end_time

    @property
    def ch1_gain(self):
        return self.__ch1_gain

    @property
    def ch2_gain(self):
        return self.__ch2_gain

    @property
    def ch1_filter(self):
        return self.__ch1_filter

    @property
    def ch2_filter(self):
        return self.__ch2_filter

    @property
    def sampling_rate(self):
        return self.__sampling_rate

    def __apply_filter(self, dat, smpl, hpf_fc, lpf_fc, notch_f0):
        # フィルタの設計
        nyq = smpl / 2
        bh, ah = signal.butter(HPF_ORDER, hpf_fc/nyq, 'high')
        bl, al = signal.butter(LPF_ORDER, lpf_fc/nyq, 'low')
        bn, an = IirFilt.IirFilt.design_notch(notch_f0)
        # フィルタの適用
        dat = dat.apply(lambda x: signal.filtfilt(bh, ah, x), axis=0)
        dat = dat.apply(lambda x: signal.filtfilt(bl, al, x), axis=0)
        dat = dat.apply(lambda x: signal.filtfilt(bn, an, x), axis=0)
        return dat

    def get_epoch_wave(self, start_idx, length_sec):
        points = length_sec * self.__sampling_rate
        return np.array(self.__wave_data[start_idx:start_idx+points])

    def plot_wave(self):
        plt.plot(self.__wave_data)
        plt.show()
