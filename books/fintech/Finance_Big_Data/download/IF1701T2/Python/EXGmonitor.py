# -*- coding: utf-8 -*-
"""
    File name:      EXGmonitor.py
    Author:         Tetsuro Tatsuoka
    Date created:   08/22/2016
    Last modified:  09/09/2016
    Version:        1.0.0
    Python version: 3.5.2
"""
from __future__ import unicode_literals, print_function, division

import sys
if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk
from tkinter import ttk

import datetime

import numpy as np

import matplotlib
if sys.version_info[0] < 3:
    matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ExgBdCtrl import ExgBdCtrl
from ExgBdUtil import ExgBdUtil

# 一般
CH_NUM = 2                          # 入力チャネル数
CH1_IDX = 0                         # Ch1のインデックス
CH2_IDX = 1                         # Ch2のインデックス

# 外観
WINDOW_TITLE = 'EXG monitor'        # ウィンドウのタイトル文字列

CTRL_BUTTON_SIZE = 10               # コントロールボタンのサイズ
SPINBUTTON_SIZE = 6                 # スピンボタンのサイズ

FONT_BUTTON = u'メイリオ 12'          # ボタンのフォント
FONT_GENERAL = u'メイリオ 10'         # ボタン以外のフォント

# グラフエリアパラメータ
DEFAULT_V_SCALE = 250               # 振幅スケールの初期値 [uV]
REDRAW_INTERVAL = 200               # グラフの再描画時間間隔 [ms]
DRAW_H_RANGE = 10.0                  # グラフの横軸の表示時間 [sec]
POINTS_PER_SEC = 200                # 1秒当たりの描画点数 [points]

GRAPH_SIZE_X = 8                    # グラフ表示エリアの横サイズ
GRAPH_SIZE_Y = 6                    # グラフ表示エリアの縦サイズ

GRAPH_TITLE = ['Ch1', 'Ch2']        # グラフのタイトル
WAV_GRAPH_COLOR = ['b', 'g']        # グラフの波形色
WAV_GRAPH_BGCOLOR = 'white'         # 波形描画エリアの背景色
WAV_THICKNESS = 0.5                 # グラフの太さ

SUBPLOT_ASIGN_NUM = [211, 212]      # Subplotの指定値

# ウィジェットのキャプション
LABEL_AMPLITUDE = 'Peak to peak: '  # ピーク振幅ラベル
BUTTON_CAPTION_CONN = 'Connect'     # 接続ボタン
BUTTON_CAPTION_STOP = 'Stop'        # 停止ボタン
BUTTON_CAPTION_START = 'Start'      # 開始ボタン（停止ボタンとトグル動作）
BUTTON_CAPTION_SAVE = 'Save'        # 保存ボタン
BUTTON_CAPTION_STOP_SAVE = 'Stop storing'  # 保存ボタン

SPINBOX_CAPTION_CH1 = 'CH1:'        # Ch1振幅スケール
SPINBOX_CAPTION_CH2 = 'uV CH2:'     # Ch2振幅スケール
SPINBOX_CAPTION_TIME = 'uV Time:'   # 時間スケール
SPINBOX_CAPTION_TUNIT = 'sec'       # 時間単位

BUTTON_CAPTION_QUIT = 'Quit'        # 終了ボタン

RBUTTON_CAPTION_HW_GAIN = 'Hardware Gain'   # ハードウェアゲイン
RBUTTON_CAPTION_HW_FILTER = 'Hardware LPF'  # ハードウェアフィルタ
COMBOBOX_CAPTION_HPF = 'High pass filter'   # ハイパスフィルタ
COMBOBOX_CAPTION_LPF = 'Low pass filter'    # ローパスフィルタ
RBUTTON_CAPTION_NOTCH = 'Notch Filter'      # ノッチフィルタ

CHECHBTN_CAPTINON_LED = 'LED'       # LEDチェックボックス

class Application:
    """ アプリケーション本体
    """
    def __init__(self):
        self.root = tk.Tk()
        self.root.wm_title(WINDOW_TITLE)

        self.is_updating = True
        self.xmin = 0.0

        self.bd_ctrl = ExgBdCtrl()              # BDコントロールオブジェクトの初期化
        self.data = [[] for row in range(CH_NUM)]
        self.data_start_time = 0.0

        self.draw_y_range = []                  # グラフの縦軸スケール
        for i in range(CH_NUM):
            self.draw_y_range.append(DEFAULT_V_SCALE)
        self.draw_h_range = DRAW_H_RANGE        # グラフの横軸の表示時間
        self.points_per_sec = POINTS_PER_SEC    # 1秒当たりの描画点数

        self.fig = Figure(figsize=(GRAPH_SIZE_X, GRAPH_SIZE_Y), dpi=100)
        self.fig.patch.set_facecolor('#F0F0ED')
        self.plot_axes = [[] for row in range(CH_NUM)]

        # グラフエリア
        self.wav_plot = []
        for i in range(CH_NUM):
            self.wav_plot.append(self.fig.add_subplot(SUBPLOT_ASIGN_NUM[i]))
            self.wav_plot[i].set_axis_bgcolor(WAV_GRAPH_BGCOLOR)
            self.wav_plot[i].text(0.01, 1.02, GRAPH_TITLE[i], size=11,
                                  transform=self.wav_plot[i].transAxes)
            self.plot_axes[i] = self.wav_plot[i].plot(self.data[i],
                                                      linewidth=WAV_THICKNESS,
                                                      color=WAV_GRAPH_COLOR[i])[0]

        self.fig.subplots_adjust(left=0.05, right=0.98, bottom=0.07, top=0.95)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.show()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas._tkcanvas.grid(column=0, row=0, sticky='nsew',
                                   columnspan=1, rowspan=2)

        ### 各チャネルの設定エリア（ウィンドウ右部） ###
        self.root.option_add('*font', FONT_GENERAL)
        self.root.option_add('*Button.font', FONT_BUTTON)

        self.frame_settings = []

        self.amplitude = []

        self.frame_hw_gain = []
        self.value_hw_gain = []

        self.frame_hw_filter = []
        self.value_hw_filter = []

        self.frame_hpf = []
        self.combobox_hpf = []
        self.value_hpf = []

        self.frame_lpf = []
        self.combobox_lpf = []
        self.value_lpf = []

        self.frame_notch_filter = []
        self.value_notch_filter = []

        for i in range(CH_NUM):
            self.frame_settings.append(tk.Frame(self.root))
            self.frame_settings[i].grid(column=1, row=i, pady=15, sticky='nsew')

            self.amplitude.append(tk.Label(self.root, text=LABEL_AMPLITUDE + '---'))
            self.amplitude[i].pack(in_=self.frame_settings[i], padx=5, fill=tk.X)

            # ハードウェアゲイン
            self.value_hw_gain.append(tk.IntVar())
            self.value_hw_gain[i].set(ExgBdUtil.HW_GAIN_DEFALT_VALUE)
            self.frame_hw_gain.append(tk.LabelFrame(self.root, text=RBUTTON_CAPTION_HW_GAIN))
            tk.Radiobutton(self.frame_hw_gain[i], text=ExgBdUtil.HW_GAIN_1000_LABEL,
                           value=ExgBdUtil.HW_GAIN_1000_VALUE, variable=self.value_hw_gain[i],
                           command=self.select_hw_gain).pack(side=tk.LEFT)
            tk.Radiobutton(self.frame_hw_gain[i], text=ExgBdUtil.HW_GAIN_200_LABEL,
                           value=ExgBdUtil.HW_GAIN_200_VALUE, variable=self.value_hw_gain[i],
                           command=self.select_hw_gain).pack(side=tk.LEFT)
            self.frame_hw_gain[i].pack(in_=self.frame_settings[i], padx=5, fill=tk.X)

            # ハードウェアフィルタ
            self.value_hw_filter.append(tk.IntVar())
            self.value_hw_filter[i].set(ExgBdUtil.HW_FILTER_DEFALT_VALUE)
            self.frame_hw_filter.append(tk.LabelFrame(self.root, text=RBUTTON_CAPTION_HW_FILTER))
            tk.Radiobutton(self.frame_hw_filter[i], text=ExgBdUtil.HW_FILTER_100_LABEL,
                           value=ExgBdUtil.HW_FILTER_100_VALUE, variable=self.value_hw_filter[i],
                           command=self.select_hw_filter).pack(side=tk.LEFT)
            tk.Radiobutton(self.frame_hw_filter[i], text=ExgBdUtil.HW_FILTER_30_LABEL,
                           value=ExgBdUtil.HW_FILTER_30_VALUE, variable=self.value_hw_filter[i],
                           command=self.select_hw_filter).pack(side=tk.LEFT)
            self.frame_hw_filter[i].pack(in_=self.frame_settings[i], padx=5, fill=tk.X)

            # ハイパスフィルタ
            self.value_hpf.append(tk.StringVar())
            self.frame_hpf.append(tk.LabelFrame(self.root, text=COMBOBOX_CAPTION_HPF))
            self.combobox_hpf.append(ttk.Combobox(self.frame_hpf[i], textvariable=self.value_hpf,
                                                  values=ExgBdUtil.HPF_ITEMS))
            self.combobox_hpf[i].pack(fill=tk.X)
            self.combobox_hpf[i].bind('<<ComboboxSelected>>', self.select_hpf)
            self.combobox_hpf[i].current(1)
            self.frame_hpf[i].pack(in_=self.frame_settings[i], padx=5, fill=tk.X)

            # ローパスフィルタ
            self.value_lpf.append(tk.StringVar())
            self.frame_lpf.append(tk.LabelFrame(self.root, text=COMBOBOX_CAPTION_LPF))
            self.combobox_lpf.append(ttk.Combobox(self.frame_lpf[i], textvariable=self.value_lpf,
                                                  values=ExgBdUtil.LPF_ITEMS))
            self.combobox_lpf[i].pack(fill=tk.X)
            self.combobox_lpf[i].bind('<<ComboboxSelected>>', self.select_lpf)
            self.combobox_lpf[i].current(5)
            self.frame_lpf[i].pack(in_=self.frame_settings[i], padx=5, fill=tk.X)

            # ノッチフィルタ
            self.value_notch_filter.append(tk.StringVar())
            self.value_notch_filter[i].set(ExgBdUtil.NOTCH_FILTER_60_VALUE)   # Default 60 Hz

            self.frame_notch_filter.append(tk.LabelFrame(self.root, text=RBUTTON_CAPTION_NOTCH))
            callback_func = self.select_notch_ch1 if i == CH1_IDX else self.select_notch_ch2
            tk.Radiobutton(self.frame_notch_filter[i], text=ExgBdUtil.NOTCH_FILTER_OFF_LABEL,
                           value=ExgBdUtil.NOTCH_FILTER_OFF_VALUE, variable=self.value_notch_filter[i],
                           command=callback_func).pack(side=tk.LEFT)
            tk.Radiobutton(self.frame_notch_filter[i], text=ExgBdUtil.NOTCH_FILTER_50_LABEL,
                           value=ExgBdUtil.NOTCH_FILTER_50_VALUE, variable=self.value_notch_filter[i],
                           command=callback_func).pack(side=tk.LEFT)
            tk.Radiobutton(self.frame_notch_filter[i], text=ExgBdUtil.NOTCH_FILTER_60_LABEL,
                           value=ExgBdUtil.NOTCH_FILTER_60_VALUE, variable=self.value_notch_filter[i],
                           command=callback_func).pack(side=tk.LEFT)
            self.frame_notch_filter[i].pack(in_=self.frame_settings[i], padx=5, fill=tk.X)

        ### タイマーバーエリア（グラフエリア下部） ###
        self.disp_time = tk.IntVar()
        self.disp_time.set(0)
        self.scale_disp_time = tk.Scale(self.root, label='', orient='h', variable=self.disp_time,
                                        from_=0, to=1, state=tk.DISABLED,
                                        command=self.change_disp_time)
        self.scale_disp_time.grid(column=0, row=CH_NUM, sticky='nsew')

        ### コマンドエリア（ウィンドウ下部） ###
        self.frame_ctrl = tk.Frame(self.root)
        self.frame_ctrl.grid(column=0, row=CH_NUM+1, sticky='nsew')

        # 接続ボタン
        self.button_conn = tk.Button(master=self.root, text=BUTTON_CAPTION_CONN,
                                     width=CTRL_BUTTON_SIZE, command=self.connect)
        self.button_conn.pack(in_=self.frame_ctrl, side=tk.LEFT)

        # 停止ボタン
        self.button_stop = tk.Button(master=self.root, text=BUTTON_CAPTION_STOP,
                                     width=CTRL_BUTTON_SIZE, command=self.review)
        self.button_stop.pack(in_=self.frame_ctrl, side=tk.LEFT)

        # 保存ボタン
        self.button_save = tk.Button(master=self.root, text=BUTTON_CAPTION_SAVE,
                                     width=CTRL_BUTTON_SIZE, command=self.save)
        self.button_save.pack(in_=self.frame_ctrl, side=tk.LEFT)

        # 時間スケール
        tk.Label(self.root, text=SPINBOX_CAPTION_TUNIT).pack(in_=self.frame_ctrl, side=tk.RIGHT)
        var_time = tk.StringVar(self.root)
        var_time.set(str(DRAW_H_RANGE))
        self.spinbox_time = tk.Spinbox(self.root, from_=1, to=20, increment=1,
                                       textvariable=var_time, width=SPINBUTTON_SIZE,
                                       command=self.time_scale)
        self.spinbox_time.pack(in_=self.frame_ctrl, side=tk.RIGHT)
        tk.Label(self.root, text=SPINBOX_CAPTION_TIME).pack(in_=self.frame_ctrl, side=tk.RIGHT)

        # Ch2振幅スケール
        var_ch2 = tk.StringVar(self.root)
        var_ch2.set(str(DEFAULT_V_SCALE))
        self.spinbox_ch2 = tk.Spinbox(self.root, from_=10, to=10000, increment=10,
                                      textvariable=var_ch2, width=SPINBUTTON_SIZE,
                                      command=self.ch2_vscale)
        self.spinbox_ch2.pack(in_=self.frame_ctrl, side=tk.RIGHT)
        tk.Label(self.root, text=SPINBOX_CAPTION_CH2).pack(in_=self.frame_ctrl, side=tk.RIGHT)

        # Ch1振幅スケール
        var_ch1 = tk.StringVar(self.root)
        var_ch1.set(str(DEFAULT_V_SCALE))
        self.spinbox_ch1 = tk.Spinbox(self.root, from_=10, to=10000, increment=10,
                                      textvariable=var_ch1, width=SPINBUTTON_SIZE,
                                      command=self.ch1_vscale)
        self.spinbox_ch1.pack(in_=self.frame_ctrl, side=tk.RIGHT)
        tk.Label(self.root, text=SPINBOX_CAPTION_CH1).pack(in_=self.frame_ctrl, side=tk.RIGHT)

        # LEDチェックボタン
        self.value_led = tk.BooleanVar()
        self.value_led.set(False)
        self.check_led = tk.Checkbutton(master=self.root, text=CHECHBTN_CAPTINON_LED,
                                        variable=self.value_led, command=self.set_led)
        self.check_led.grid(column=1, row=CH_NUM, sticky='nsew')

        # 終了ボタン
        self.button_quit = tk.Button(master=self.root, text=BUTTON_CAPTION_QUIT,
                                     command=self._quit)
        self.button_quit.grid(column=1, row=CH_NUM+1, sticky='nsew')

        # グリッドの余白配分
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_rowconfigure(3, weight=1)

        # 描画の連続更新
        self.on_redraw_timer()

        # tkinterのメッセージループ
        self.root.mainloop()

    def release(self):
        """ リソースを開放する
        """
        self.bd_ctrl.release()

    def on_redraw_timer(self):
        """ 描画を一定間隔で更新する
        """
        self.time_begin = datetime.datetime.now()
        self.add_draw_data(self.bd_ctrl.get_wave_data())
        self.draw_plot()
        self.time_end = datetime.datetime.now()

        # 再描画の時間間隔から描画の時間を引いた時間を算出
        tdelta = self.time_end - self.time_begin
        tofs = datetime.timedelta(microseconds=(REDRAW_INTERVAL * 1000)) - tdelta

        if tofs.microseconds > REDRAW_INTERVAL * 1000:
            self.root.after(0, self.on_redraw_timer)
        else:
            self.root.after(tofs.microseconds // 1000, self.on_redraw_timer)

    def add_draw_data(self, data):
        """ 波形データを追加する
        """
        sampling_rate = self.bd_ctrl.get_sampling_rate()
        sampling_sec = 1. / sampling_rate

        points_per_sec = self.points_per_sec
        draw_sampling_sec = 1. / points_per_sec

        append_data = [[] for row in range(CH_NUM)]
        draw_time = 0.0
        for d in data:
            draw_time += sampling_sec
            # 描画サンプリング周期を超えた時点でデータを追加
            if draw_time >= draw_sampling_sec:
                draw_time -= draw_sampling_sec
                for i in range(CH_NUM):
                    append_data[i].append(d[i])

        # データのサンプリング周波数は描画のサンプリング周波数より十分大きくする必要あり
        for idx in range(CH_NUM):
            # データを追加
            self.data[idx] += append_data[idx]

            # 表示データ数
            remain_frame_length = int(self.draw_h_range * POINTS_PER_SEC)

            # 取得データ数が表示データ数を超えたら開始位置を移動
            self.data_start_time += max((len(self.data[idx]) - remain_frame_length),
                                        0) / float(POINTS_PER_SEC)

            self.data[idx] = self.data[idx][-remain_frame_length:]

    def draw_plot(self):
        """ 波形を描画する
        """
        if self.is_updating:
            self.xmin = self.data_start_time
            self.scale_disp_time['to'] = int(self.xmin)

        xmax = self.xmin + self.draw_h_range

        xaxis = [float(con) / self.points_per_sec + self.data_start_time
                 for con in range(len(self.data[CH1_IDX]))]

        for i in range(CH_NUM):
            ymin = -self.draw_y_range[i]
            ymax = self.draw_y_range[i]
            self.wav_plot[i].set_xbound(lower=self.xmin, upper=xmax)
            self.wav_plot[i].set_ybound(lower=ymin, upper=ymax)
            self.wav_plot[i].grid(True, color='gray')

            if self.is_updating:
                self.plot_axes[i].set_xdata(xaxis)
                self.plot_axes[i].set_ydata(np.array(self.data[i]))

            self.amplitude[i]['text'] = (LABEL_AMPLITUDE +
                                         str(round(self.bd_ctrl.wav_proc.calc_amp(i), 1)))

        self.canvas.draw()


    def select_hw_gain(self):
        """ Hardware gain値変更イベントハンドラ
        """
        cmd_str = ExgBdUtil.get_gain_command_str(self.value_hw_gain[0].get(),
                                                 self.value_hw_gain[1].get())
        self.bd_ctrl.send_command(cmd_str.encode())
        self.bd_ctrl.gain_adjust[0] = ExgBdUtil.HW_GAIN_1000_COEF \
            if self.value_hw_gain[0].get() == ExgBdUtil.HW_GAIN_1000_VALUE else \
            ExgBdUtil.HW_GAIN_200_COEF
        self.bd_ctrl.gain_adjust[1] = ExgBdUtil.HW_GAIN_1000_COEF \
            if self.value_hw_gain[1].get() == ExgBdUtil.HW_GAIN_1000_VALUE else \
            ExgBdUtil.HW_GAIN_200_COEF

    def select_hw_filter(self):
        """ Hardware filter値変更イベントハンドラ
        """
        cmd_str = ExgBdUtil.get_filter_command_str(self.value_hw_filter[0].get(),
                                                   self.value_hw_filter[1].get())
        self.bd_ctrl.send_command(cmd_str.encode())

    def select_hpf(self, evt):
        """ HPFコンボボックス値変更イベントハンドラ
        """
        if evt.widget == self.combobox_hpf[CH1_IDX]:
            self.bd_ctrl.wav_proc.set_hpf_coef(evt.widget.get(), CH1_IDX)
        else:
            self.bd_ctrl.wav_proc.set_hpf_coef(evt.widget.get(), CH2_IDX)

    def select_lpf(self, evt):
        """ LPFコンボボックス値変更イベントハンドラ
        """
        if evt.widget == self.combobox_lpf[CH1_IDX]:
            self.bd_ctrl.wav_proc.set_lpf_coef(evt.widget.get(), CH1_IDX)
        else:
            self.bd_ctrl.wav_proc.set_lpf_coef(evt.widget.get(), CH2_IDX)

    def select_notch_ch1(self):
        """ Notch値変更イベントハンドラ
        """
        fo = self.value_notch_filter[CH1_IDX].get()
        self.bd_ctrl.wav_proc.set_notch_coef(fo, CH1_IDX)

    def select_notch_ch2(self):
        """ Notch値変更イベントハンドラ
        """
        fo = self.value_notch_filter[CH2_IDX].get()
        self.bd_ctrl.wav_proc.set_notch_coef(fo, CH2_IDX)

    def connect(self):
        """ 接続ボタンクリックイベントハンドラ
        """
        print("Reserved. Currently auto start.")

    def review(self):
        """ レビューボタンクリックイベントハンドラ
        """
        if self.button_stop["text"] == BUTTON_CAPTION_STOP:
            self.scale_disp_time['state'] = tk.ACTIVE
            self.is_updating = False
            self.button_stop["text"] = BUTTON_CAPTION_START
        else:
            self.disp_time.set(0)
            self.scale_disp_time['state'] = tk.DISABLED
            self.is_updating = True
            self.button_stop["text"] = BUTTON_CAPTION_STOP

    def save(self):
        """ 保存ボタンクリックイベントハンドラ
        """
        if self.button_save["text"] == BUTTON_CAPTION_SAVE:
            cmd_str = ExgBdUtil.get_save_command_str(True)
            self.bd_ctrl.send_command(cmd_str.encode())
            self.button_save["text"] = BUTTON_CAPTION_STOP_SAVE
        else:
            cmd_str = ExgBdUtil.get_save_command_str(False)
            self.bd_ctrl.send_command(cmd_str.encode())
            self.button_save["text"] = BUTTON_CAPTION_SAVE


    def ch1_vscale(self):
        """ Ch1縦軸スケール値変更イベントハンドラ
        """
        self.draw_y_range[CH1_IDX] = int(self.spinbox_ch1.get())

    def ch2_vscale(self):
        """ Ch2縦軸スケール値変更イベントハンドラ
        """
        self.draw_y_range[CH2_IDX] = int(self.spinbox_ch2.get())

    def time_scale(self):
        """ 時間軸スケール値変更イベントハンドラ
        """
        self.draw_h_range = int(self.spinbox_time.get())

    def change_disp_time(self, tval):
        """ タイマーバー値変更イベントハンドラ
        """
        if not self.is_updating:
            self.xmin = float(tval)

    def set_led(self):
        """ LEDチェックボックス変更イベントハンドラ
        """
        cmd_str = ExgBdUtil.get_led_command_str(self.value_led.get())
        self.bd_ctrl.send_command(cmd_str.encode())

    def _quit(self):
        """ 終了ボタンクリックイベントハンドラ
        """
        self.root.quit()     # stops mainloop
        self.root.destroy()  # this is necessary on Windows to prevent

if __name__ == '__main__':
    app = Application()
    app.release()
    app = None
    