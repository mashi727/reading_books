# -*- coding: utf-8 -*-
"""
    File name:      CalcEegBand.py
    Description:    Class for calculating EEG band
    Author:         Tetsuro Tatsuoka
    Date created:   09/18/2016
    Last modified:  09/18/2016
    Version:        1.0.0
    Python version: 3.5.2
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

CH1_IDX = 0
CH2_IDX = 1

DELTA_IDX = 0
THETA_IDX = 1
ALPHA_IDX = 2
BETA_IDX = 3
GAMMA_IDX = 4

class CalcEegBand:
    """　脳波の周波数解析を行うクラス
    """
    
    __band_label = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    __freq_border = [0.5, 4, 8, 13, 30, 50]
    __freq_mid_alpha = 10           # Lower alpha: 8-10, Upper alpha: 10-13

    __spectrogram_vmax = 10         # スペクトログラムの最大強度

    def __init__(self, wave_array, fft_len=256, fs=200):
        
        # 各帯域の境界インデックスを算出
        self.__idx_border = []
        for s_freq in self.__freq_border:
            self.__idx_border.append(int(s_freq * fft_len / fs) + 1)

        self.__idx_mid_alpha = int(self.__freq_mid_alpha * fft_len / fs) + 1

        # データ長、チャネル数、スペクトル算出インターバル(1sec分のデータ点数）を取得
        data_len, ch_num = wave_array.shape
        fft_int = fs

        #　スペクトル密度の算出
        freq = None
        time = None
        abs_list = []
        for ch_idx in range(ch_num):
            freq, time, Sxx = signal.spectrogram(
                                wave_array[:,ch_idx],
                                fft_int,
                                window=signal.get_window('hann', fft_len),
                                nperseg=fft_len,
                                noverlap=fft_len - fft_int)
            abs_list.append(np.sqrt(Sxx))       # スペクトル密度のパワーから振幅を算出

        # 各帯域の振幅値を算出
        band_num = len(self.__freq_border) - 1  # 境界値のリストなので区間の数は-1
        fft_num = abs_list[0].shape[1]
        self.__fft_length = fft_num

        self.__band_abs = np.zeros((ch_num, fft_num, band_num))
        self.__band_ratio = np.zeros((ch_num, fft_num, band_num))
        self.__beta_alpha_ratio = np.zeros((ch_num, fft_num))
        self.__peak_alpha = np.zeros((ch_num, fft_num))

        for i in range(ch_num):
            for j in range(fft_num):
                band_total = sum(abs_list[i][:,j])

                for k in range(band_num):
                    self.__band_abs[i, j, k] = sum(
                        abs_list[i][self.__idx_border[k]:self.__idx_border[k+1], j])
                    self.__band_ratio[i, j, k] = self.__band_abs[i, j, k] / band_total
                
                self.__beta_alpha_ratio[i, j] = (
                    self.__band_abs[i, j, BETA_IDX] / self.__band_abs[i, j, ALPHA_IDX])

                idx_peak_alpha = (
                    np.argmax(
                        abs_list[i][self.__idx_border[ALPHA_IDX]:self.__idx_border[BETA_IDX], j])
                    + self.__idx_border[ALPHA_IDX])
                self.__peak_alpha[i, j] = idx_peak_alpha * fs / fft_len
                

        # Low alpha, High alphaの振幅値の算出
        self.__lo_alpha_abs = np.zeros((ch_num, fft_num))
        self.__hi_alpha_abs = np.zeros((ch_num, fft_num))

        for i in range(ch_num):
            for j in range(fft_num):
                self.__lo_alpha_abs[i, j] = sum(
                    abs_list[i][self.__idx_border[ALPHA_IDX]:self.__idx_mid_alpha, j])
                self.__hi_alpha_abs[i, j] = sum(
                    abs_list[i][self.__idx_mid_alpha:self.__idx_border[BETA_IDX], j])

    @property
    def band_label(self):
        return self.__band_label

    @property
    def freq_border(self):
        return self.__freq_border

    @property
    def band_abs(self):
        return self.__band_abs

    @property
    def band_ratio(self):
        return self.__band_abs

    @property
    def beta_alpha_ratio(self):
        return self.__beta_alpha_ratio

    @property
    def alpha_amplitude(self):
        return [self.__lo_alpha_abs, self.__hi_alpha_abs]

    @property
    def peak_alpha_frequency(self):
        return self.__peak_alpha


    def plot_band(self, title, is_save=False, file_path='./temp.png'):
        
        RIGHT_MARGIN = 0.15
        LEGEND_POS = 1.05
        
        plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
        
        plt.subplot(221)
        for abs_data in self.__band_abs[CH1_IDX].T:
            plt.plot(abs_data)
        plt.xlim([-0.5, self.__fft_length-0.5])
        plt.tick_params(labelbottom='off')
        plt.ylabel('Abs')
        plt.title('Ch1')

        plt.subplot(222)
        for abs_data in self.__band_abs[CH2_IDX].T:
            plt.plot(abs_data)
        plt.xlim([-0.5, self.__fft_length-0.5])
        plt.legend(
            self.band_label, bbox_to_anchor=(LEGEND_POS, 1), loc='upper left', borderaxespad=0)
        plt.subplots_adjust(right=1-RIGHT_MARGIN)
        plt.tick_params(labelbottom='off')
        plt.title('Ch2')

        color_map = ['b', 'g', 'r', 'c', 'm']
        
        plt.subplot(223)
        base_val = np.zeros(self.__fft_length)
        idx = 0
        for ratio in self.__band_ratio[CH1_IDX].T:
            plt.bar(
                np.arange(len(ratio)), ratio, width=1.0, bottom=base_val, color=color_map[idx])
            idx = idx + 1
            base_val = base_val + ratio
        plt.ylabel('ratio')
 
        plt.subplot(224)
        base_val = np.zeros(self.__fft_length)
        idx = 0
        for ratio in self.__band_ratio[CH2_IDX].T:
            plt.bar(
                np.arange(len(ratio)), ratio, width=1.0, bottom=base_val, color=color_map[idx])
            idx = idx + 1
            base_val = base_val + ratio
        plt.legend(
            self.band_label, bbox_to_anchor=(LEGEND_POS, 1), loc='upper left', borderaxespad=0)
        
        plt.suptitle(title, fontsize=16)
        if is_save:
            plt.savefig(file_path)

        plt.show()

    def plot_analysis_param(self, title, is_save=False, file_path='./temp.png'):
        
        RIGHT_MARGIN = 0.15
        LEGEND_POS = 1.05

        plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        
        ch_num = self.beta_alpha_ratio.shape[0]

        # Peak Alpha
        plt.subplot(311)
        legend_str = []
        for i in range(ch_num):
            plt.plot(self.__peak_alpha[i])
            legend_str.append('Ch' + str(i))
        plt.ylabel('Freq. [Hz]')
        plt.legend(
            legend_str, bbox_to_anchor=(LEGEND_POS, 1), loc='upper left', borderaxespad=0)
        plt.subplots_adjust(right=1-RIGHT_MARGIN)
        plt.tick_params(labelbottom='off')
        plt.title('Peak alpha frequency')

        # Alpha power
        plt.subplot(312)
        legend_str = []
        for i in range(ch_num):
            plt.plot(self.__lo_alpha_abs[i])
            legend_str.append('Ch' + str(i) + ' Low')
            plt.plot(self.__hi_alpha_abs[i])
            legend_str.append('Ch' + str(i) + ' High')
        plt.ylabel('A/D value')
        plt.legend(
            legend_str, bbox_to_anchor=(LEGEND_POS, 1), loc='upper left', borderaxespad=0)
        plt.subplots_adjust(right=1-RIGHT_MARGIN)
        plt.tick_params(labelbottom='off')
        plt.title('Alpha low/high amplitude')

        # Beta/Alpha
        plt.subplot(313)
        legend_str = []
        for i in range(ch_num):
            plt.plot(self.__beta_alpha_ratio[i])
            legend_str.append('Ch' + str(i))
        plt.ylabel('ratio')
        plt.legend(
            legend_str, bbox_to_anchor=(LEGEND_POS, 1), loc='upper left', borderaxespad=0)
        plt.subplots_adjust(right=1-RIGHT_MARGIN)
        plt.tick_params(labelbottom='off')
        plt.title('Beta / Alpha')
        
        plt.suptitle(title, fontsize=16)
        if is_save:
            plt.savefig(file_path)

        plt.show()
        