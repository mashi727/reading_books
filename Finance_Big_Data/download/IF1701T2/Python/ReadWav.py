# -*- coding: utf-8 -*-
""" 生体計測基板で測定したデータを読み込むスクリプト（for Python 3） """
import os
import sys
import struct as st
import numpy as np
import pandas as pd
import tkinter as tk
import tkinter.filedialog
import datetime

TEXT_FILE_EXT = '.txt'
BINARY_FILE_EXT = '.dat'
ATTR_FILE_SUFFIX = '_attr'

CH_NUM = 2
BIN_DATA_SIZE = 4

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

root = tkinter.Tk()
root.withdraw()

args = { "initialdir" : os.path.dirname(__file__),
"filetypes" : [('波形データファイル','*.txt;*.dat')],
"title" : "テスト"
}
wave_filename = tk.filedialog.askopenfilename(**args)

filename_without_ext = wave_filename[0:-4]
attr_filename = filename_without_ext + ATTR_FILE_SUFFIX + TEXT_FILE_EXT

if wave_filename[-4:] == TEXT_FILE_EXT:
    wave_data = pd.read_table(wave_filename, header=None)
    data_len = len(wave_data)
elif wave_filename[-4:] == BINARY_FILE_EXT:
    data_len = os.path.getsize(wave_filename) // CH_NUM // BIN_DATA_SIZE
    wave_data = pd.DataFrame(np.zeros([data_len,CH_NUM]))
    f = open(wave_filename, 'rb')
    for i in range(data_len * CH_NUM):
        wave_data[i%2][i//2] = st.unpack('<L', f.read(4))[0]
    f.close()
else:
    print("Invalid extention.")
    sys.exit()


# 波形データの長さを表示
print("Data length:", data_len)

# 属性データの読み取り
attr_data = pd.read_table(attr_filename, header=None)
start_time_str = attr_data.iloc[ATTR_START_TIME_RIDX,ATTR_DAT_CIDX]
start_time = datetime.datetime.strptime(start_time_str[0:-3], '%Y%m%d_%H%M%S')
start_time = start_time + \
    datetime.timedelta(microseconds=int(start_time_str[-3:]) * 1000)

end_time_str = attr_data.iloc[ATTR_END_TIME_RIDX,ATTR_DAT_CIDX]
end_time = datetime.datetime.strptime(end_time_str[0:-3], '%Y%m%d_%H%M%S')
end_time = end_time + \
    datetime.timedelta(microseconds=int(end_time_str[-3:]) * 1000)

diff_time = end_time - start_time 

ch1_gain = attr_data.iloc[ATTR_CH1_GAIN_RIDX,ATTR_DAT_CIDX]
if ch1_gain == GAIN_200_STR:
    wave_data[0] = wave_data[0] * GAIN_200_COEF

ch2_gain = attr_data.iloc[ATTR_CH2_GAIN_RIDX,ATTR_DAT_CIDX]
if ch2_gain == GAIN_200_STR:
    wave_data[1] = wave_data[1] * GAIN_200_COEF

ch1_filter = attr_data.iloc[ATTR_CH1_FILTER_RIDX,ATTR_DAT_CIDX]
ch2_filter = attr_data.iloc[ATTR_CH2_FILTER_RIDX,ATTR_DAT_CIDX]

sampling_rate_str = attr_data.iloc[ATTR_SAMPLING_RATE_RIDX,ATTR_DAT_CIDX]
sampling_rate = int(sampling_rate_str[0:-2])   # Hzは削除
        

