# -*- coding: utf-8 -*-
"""
    File name:      ExgBdUtil.py
    Author:         Tetsuro Tatsuoka
    Date created:   09/10/2016
    Last modified:  09/11/2016
    Version:        1.0.0
    Python version: 3.5.2
"""
import copy

class ExgBdUtil:
    """ EXGBD固有の定数、コマンド文字列を扱うクラス
    """
    HW_GAIN_1000_LABEL = '1000'
    HW_GAIN_200_LABEL = '200'
    HW_GAIN_1000_VALUE = 0
    HW_GAIN_200_VALUE = 1
    HW_GAIN_1000_COEF = 1.220703125     # = 5V / 4096 / 1000 * 1000000 [uV]
    HW_GAIN_200_COEF = 6.103515625      # = 5V / 4096 / 200 * 1000000 [uV]
    HW_GAIN_DEFALT_VALUE = HW_GAIN_1000_VALUE
    HW_GAIN_DEFALT_COEF = HW_GAIN_1000_COEF

    HW_FILTER_100_LABEL = '106 Hz'
    HW_FILTER_30_LABEL = '29.5 Hz'
    HW_FILTER_100_VALUE = 0
    HW_FILTER_30_VALUE = 1
    HW_FILTER_DEFALT_VALUE = HW_FILTER_30_VALUE

    HPF_ITEMS = ['OFF', '0.5', '1', '2', '3', '5', '7', '10', '15', '20']
    LPF_ITEMS = ['OFF', '5', '10', '15', '20', '30', '40', '50', '60', '70', '100']

    SAMPLING_RATE_ITEMS = ['100', '200', '500', '1000']

    NOTCH_FILTER_OFF_LABEL = 'OFF'
    NOTCH_FILTER_50_LABEL = '50 Hz'
    NOTCH_FILTER_60_LABEL = '60 Hz'
    NOTCH_FILTER_OFF_VALUE = 'OFF'
    NOTCH_FILTER_50_VALUE = '50'
    NOTCH_FILTER_60_VALUE = '60'

    COMMAND_SET_FORMAT = 'SET XXXX YYYY ZZZZ\n'
    COMMAND_REQ_FORMAT = 'REQ XXXX\n'
    COMMAND_FILE_FORMAT = 'FILE BYTES PREFIX FORMAT\n'

    CMD_SET = {'acq_start_stop' :'ACQ_',
               'hw_gain'        :'GAIN',
               'hw_filter'      :'FILT',
               'sampling_rate'  :'SMPL',
               'led_on_off'     :'LED_',
               'save_on_off'    :'SAVE'}

    CMD_REQ = copy.deepcopy(CMD_SET)
    CMD_REQ['device_name'] = 'DEVN'
    CMD_REQ['connect_status'] = 'CONN'
    CMD_REQ['quit'] = 'QUIT'

    PARAM_GAIN = {HW_GAIN_200_VALUE: '0200', HW_GAIN_1000_VALUE: '1000'}
    PARAM_FILT = {HW_FILTER_30_VALUE: '0030', HW_FILTER_100_VALUE: '0100'}
    PARAM_SMPL = {SAMPLING_RATE_ITEMS[0]: '0100',
                  SAMPLING_RATE_ITEMS[1]: '0200',
                  SAMPLING_RATE_ITEMS[2]: '0500',
                  SAMPLING_RATE_ITEMS[3]: '1000'}

    VAL_RESERVED = '0000'
    VAL_START = '0001'
    VAL_STOP = '0000'
    VAL_ON = '0001'
    VAL_OFF = '0000'

    @classmethod
    def get_command_str(cls, xxxx, yyyy, zzzz):
        """ EXGBD制御モジュール用コマンド文字列を生成する
        """
        cmd_str = cls.COMMAND_SET_FORMAT
        cmd_str = cmd_str.replace('XXXX', xxxx)
        cmd_str = cmd_str.replace('YYYY', yyyy)
        cmd_str = cmd_str.replace('ZZZZ', zzzz)
        return cmd_str

    @classmethod
    def get_acq_command_str(cls, is_start):
        """ 測定開始/停止コマンドの文字列を生成する
        """
        xxxx = cls.CMD_SET['acq_start_stop']
        yyyy = cls.VAL_RESERVED
        zzzz = cls.VAL_START if is_start else cls.VAL_STOP
        return cls.get_command_str(xxxx, yyyy, zzzz)

    @classmethod
    def get_gain_command_str(cls, ch1, ch2):
        """ ハードウェアゲイン設定コマンドの文字列を生成する
        """
        xxxx = cls.CMD_SET['hw_gain']
        yyyy = cls.PARAM_GAIN[ch1]
        zzzz = cls.PARAM_GAIN[ch2]
        return cls.get_command_str(xxxx, yyyy, zzzz)

    @classmethod
    def get_filter_command_str(cls, ch1, ch2):
        """ ハードウェアLPF設定コマンドの文字列を生成する
        """
        xxxx = cls.CMD_SET['hw_filter']
        yyyy = cls.PARAM_FILT[ch1]
        zzzz = cls.PARAM_FILT[ch2]
        return cls.get_command_str(xxxx, yyyy, zzzz)

    @classmethod
    def get_sampling_command_str(cls, sampling):
        """ サンプリング周波数設定コマンドの文字列を生成する
        """
        xxxx = cls.CMD_SET['sampling_rate']
        yyyy = cls.VAL_RESERVED
        zzzz = cls.PARAM_SMPL[sampling]
        return cls.get_command_str(xxxx, yyyy, zzzz)

    @classmethod
    def get_led_command_str(cls, is_on):
        """ LED制御コマンドの文字列を生成する
        """
        xxxx = cls.CMD_SET['led_on_off']
        yyyy = cls.VAL_RESERVED
        zzzz = cls.VAL_ON if is_on else cls.VAL_OFF
        return cls.get_command_str(xxxx, yyyy, zzzz)

    @classmethod
    def get_save_command_str(cls, is_start):
        """ 保存開始/停止コマンドの文字列を生成する
        """
        xxxx = cls.CMD_SET['save_on_off']
        yyyy = cls.VAL_RESERVED
        zzzz = cls.VAL_START if is_start else cls.VAL_STOP
        return cls.get_command_str(xxxx, yyyy, zzzz)

    @classmethod
    def get_quit_request_str(cls):
        """ 終了リクエストの文字列を生成する
        """
        req_str = cls.COMMAND_REQ_FORMAT
        req_str = req_str.replace('XXXX', cls.CMD_REQ['quit'])
        return req_str
