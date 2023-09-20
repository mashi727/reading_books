# -*- coding: utf-8 -*-
"""
    File name:      VisualStimuliTiming.py
    Description:    Class of visual stimuli timing log
    Author:         Tetsuro Tatsuoka
    Date created:   09/18/2016
    Last modified:  09/18/2016
    Version:        1.0.0
    Python version: 3.5.2
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

DISPLAY_CROSS_S = 2         # 2 sec     予告画像 提示時間[s]
DISPLAY_STIM_S = 6          # 6 sec     刺激画像 提示時間[s]

STIM_TIME_FORMAT = '%Y/%m/%d %H:%M:%S.'

TEXT_FILE_EXT = '.txt'
BINARY_FILE_EXT = '.dat'
ATTR_FILE_SUFFIX = '_attr'

HIGH_VALENCE_STR = 'HV'
LOW_VALENCE_STR = 'LV'
NEUTRAL_STR = 'N'

STIM_TYPE_STR = {
    'P':HIGH_VALENCE_STR, 'N':NEUTRAL_STR, 'A':LOW_VALENCE_STR, 'H':LOW_VALENCE_STR }

HIGH_VALENCE_VAL = 1
NEUTRAL_VAL = 0.3
BLANK_VAL = 0
PRECAUTION_VAL = -0.1
LOW_VALENCE_VAL = -1

LOG_WAVE_VAL = {
    'P':HIGH_VALENCE_VAL, 'N':NEUTRAL_VAL, 'A':LOW_VALENCE_VAL, 'H':LOW_VALENCE_VAL }

STIM_TIME_COL_IDX = 0
IMAGE_PATH_COL_IDX = 1

FILE_TYPE_IDX = -8      # GAPEDのファイル名は「XNNN.png」なので、Xは-8文字目

class VisualStimuliTiming:
    """　画像刺激提示タイミングクラス
    """

    def __init__(self, log_file_path):
        self.__log_data = pd.read_table(log_file_path, header=None)
        self.__data_len = len(self.__log_data)
        self.__log_wave = None

    @property
    def log_data(self):
        return self.__log_data
        
    @property
    def data_length(self):
        return self.__data_len

    def get_stim_timing_and_log_wave(self, exgBdWavData_obj):  #wav_file_path):
        """ 刺激呈示が開始されたインデックス（波形データの何ポイント目か）のリスト、刺激の種類、画像ファイル名、
            および、休止、予告および各刺激を値で表現した、波形データと同じ長さのnumpy.arrayオブジェクトを返す
        """
        stim_index_array = []
        stim_type = []
        pic_file_name = []
        log_wave = np.zeros(exgBdWavData_obj.wave_length)
        log_wave = log_wave + BLANK_VAL

        peek_pos = 0
        smpl_rate = exgBdWavData_obj.sampling_rate
        precaution_len = smpl_rate * DISPLAY_CROSS_S
        stim_len = smpl_rate * DISPLAY_STIM_S
        
        for i, one_line in self.__log_data.iterrows():
            # 刺激呈示時刻の取得
            stim_time_str = one_line[STIM_TIME_COL_IDX]
            stim_time = datetime.datetime.strptime(
                            stim_time_str[0:-3],    # 秒以下の3文字は削除
                            STIM_TIME_FORMAT)

            # 刺激呈示時のインデックス（波形データの何ポイント目か）
            stim_index = exgBdWavData_obj.get_index(stim_time)

            # 予告呈示区間のインデックス
            end_precaution = math.floor(stim_index)
            start_precaution = end_precaution - precaution_len                

            # 刺激呈示区間のインデックス
            start_stim = end_precaution + 1
            stim_index_array.append(start_stim)
            end_stim = start_precaution + stim_len

            # 次の休止区間のインデックス
            start_next_blank = end_stim + 1

            # 刺激画像の種類別に書き込む値を取得
            image_path = one_line[IMAGE_PATH_COL_IDX]
            file_type_letter = image_path[FILE_TYPE_IDX]
            log_wave_val = LOG_WAVE_VAL[file_type_letter]
            stim_type.append(STIM_TYPE_STR[file_type_letter])
            pic_file_name.append(image_path[FILE_TYPE_IDX:])
            
            try:
                # ログ波形ファイルの書き込み
                log_wave[peek_pos:start_precaution] = BLANK_VAL
                log_wave[start_precaution:start_stim] = PRECAUTION_VAL
                log_wave[start_stim:start_next_blank] = log_wave_val
                peek_pos = start_next_blank
            except IndexError:
                # インデックスを超えてしまったら終了
                print('IndexError')
                break
        
        stim_index_array
        self.__log_wave = log_wave
        return [stim_index_array, stim_type, pic_file_name, log_wave]

    def plot_wave(self, exgBdWavData_obj):
        if self.__log_wave is None:
            print('Execute get_stim_timing_and_log_wave() in advance.')
        else:
            plt.figure(num=None, figsize=(8, 3), dpi=80, facecolor='w', edgecolor='k')
            plt.subplot(211)
            plt.plot(self.__log_wave)
            plt.tick_params(labelbottom='off')
            plt.ylabel('log wave')
            plt.title('Visual stimuli timing')
            
            t = np.arange(len(self.__log_wave))
            y = np.arange(2)
            v = np.c_[self.__log_wave, self.__log_wave]
            v = v.T
            plt.subplot(212)
            plt.pcolormesh(t,y,v)
            plt.plot(exgBdWavData_obj.wave_data)
            plt.xlabel('Time [points]')
            plt.ylabel('color mesh')
            #plt.colorbar()
            
            plt.show()
