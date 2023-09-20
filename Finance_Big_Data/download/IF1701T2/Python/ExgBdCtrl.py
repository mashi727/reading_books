# -*- coding: utf-8 -*-
"""
    File name:      ExgBdCtrl.py
    Author:         Tetsuro Tatsuoka
    Date created:   09/10/2016
    Last modified:  09/12/2016
    Version:        1.0.0
    Python version: 3.5.2
"""
from __future__ import unicode_literals, print_function, division

import sys

import queue
import subprocess
import struct as st
import time
import threading

import numpy as np

import matplotlib
if sys.version_info[0] < 3:
    matplotlib.use('TkAgg')

import win32file

from ExgBdUtil import ExgBdUtil
import WaveProcessing

CH_NUM = 2                          # 入力チャネル数
SAMPLING_RATE = 200                 # サンプリング周波数

GAIN_ADJUST = 1.220703125           # ゲインの補正値 

NAMED_PIPE_FOR_WAV = 'wav_pipe'     # 波形受信用の名前付きパイプ
EXGBD_CTRL = 'ExgBdCtrl.exe'        # EXG BD制御モジュール

class ExgBdCtrl:
    """ EXGBDからの波形データの取得、フィルタ処理、EXGBDへのコマンド送信を行うクラス
    """
    def __init__(self):
        self.sampling_rate = SAMPLING_RATE
        self.loop_flag = True
        self.wav_fifo = queue.Queue()
        
        self.gain_adjust = []
        for i in range(CH_NUM):
            self.gain_adjust.append(ExgBdUtil.HW_GAIN_DEFALT_COEF)

        # 波形処理モジュールを生成
        self.wav_proc = WaveProcessing.WaveProcessing(CH_NUM, SAMPLING_RATE, 5)

        # EXGBD制御モジュールを起動
        self.exgbd_ctl = subprocess.Popen([EXGBD_CTRL, NAMED_PIPE_FOR_WAV],
                                          stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        time.sleep(0.5)
        
        # 波形受信用の名前付きパイプを生成
        self.pipe_handle = win32file.CreateFile('\\\\.\\pipe\\' + NAMED_PIPE_FOR_WAV,
                                                win32file.GENERIC_READ, 0, None,
                                                win32file.OPEN_EXISTING, 0, None)

        # 波形データ受信スレッドを起動
        self.acq_thread = threading.Thread(target=self.acq_wav_data)
        self.acq_thread.start()

    def release(self):
        """ リソースを開放する
        """
        req_str = ExgBdUtil.get_quit_request_str()
        self.send_command(req_str.encode())
        self.loop_flag = False
        self.acq_thread.join(0.2)
        self.exgbd_ctl.communicate()
        self.exgbd_ctl.terminate()
        self.exgbd_ctl.kill()
        win32file.CloseHandle(self.pipe_handle)

    def acq_wav_data(self):
        """ 波形データ受信スレッド
        """
        size_of_short = 2
        all_ch_one_data_bytes = CH_NUM * size_of_short
        while self.loop_flag:
            try:
                left, read_data = win32file.ReadFile(self.pipe_handle, 4096)

                for k in range(len(read_data)//all_ch_one_data_bytes):
                    wav_dat = np.zeros(CH_NUM)
                    for i in range(CH_NUM):
                        start_idx = all_ch_one_data_bytes * k + size_of_short * i
                        end_idx = start_idx + size_of_short
                        temp_dat = st.unpack('<h', read_data[start_idx:end_idx])[0]
                        temp_dat = temp_dat * self.gain_adjust[i]
                        wav_dat[i] = self.wav_proc.proc_each(temp_dat, i)

                    self.wav_fifo.put(wav_dat)
            except:
                if self.loop_flag:
                    print('Data acquisition error\n')
                time.sleep(1)

    def get_sampling_rate(self):
        """ 波形データのサンプリング周波数を取得する
        """
        return self.sampling_rate

    def get_wave_data(self):
        """ 描画用データを取得する
        """
        wav_dat_list = []
        while not self.wav_fifo.empty():
            wav_dat_list.append(self.wav_fifo.get())
        return wav_dat_list

    def send_command(self, cmd_bytes):
        """ EXGBDへコマンドを送信する
        """
        self.exgbd_ctl.stdin.write(cmd_bytes)
        self.exgbd_ctl.stdin.flush()
        print(cmd_bytes)
