# -*- coding: utf-8 -*-
"""
    File name:      DisplayVisualStimuli.py
    Description:    Display pictures specified in ImagedDef.py randomly.
                    Pressing any key, the picture can be skipped.
    Author:         Tetsuro Tatsuoka
    Date created:   09/16/2016
    Last modified:  09/16/2016
    Version:        1.0.0
    Python version: 3.5.2
"""
import random
import copy
from datetime import datetime
import os
import sys

if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk

import cv2          # OpenCV のインストールが必要（コマンドプロンプトから下記を実行）
                    # conda install -c https://conda.binstar.org/menpo opencv3

import ImageDef     # 刺激画像を ImageDef.py に書いておく（順番は後でランダムになる）


PICTURE_FOLDER = './GAPED/'     # 刺激画像のフォルダ（この下に「A」「H」「N」「P」などのフォルダがある）

DISPLAY_CROSS_MS = 2000         # 2 sec     予告画像 提示時間[ms]
DISPLAY_STIM_MS = 6000          # 6 sec     刺激画像 提示時間[ms]
DISPLAY_BLANK_MS = 15000        # 15 sec    休止画像 提示時間[ms]
DISPLAY_END_MS = 2000           # 2 sec     終了画像 提示時間[ms]

CROSS_IMAGE = 'cross.png'       # 予告画像
BLANK_IMAGE = 'blank.png'       # 休止画像
END_IMAGE = 'end.png'           # 終了画像

STIM_TIME_FORMAT = '%Y/%m/%d %H:%M:%S.'

WINDOW_NAME = 'fullscreen'
PROJECT_PREFIX = 'visual_stimuli_'
SAVE_FILE_FOLDER = './' + PROJECT_PREFIX + 'log/'

# 刺激タイプの選択
select_type  = 'a'
BUTTON_WIDTH = 20
BUTTON_HEIGHT = 5
root = tk.Tk()
root.wm_title('Select stimulation type')
root.option_add('*font', ('FixedSys, 14'))
def quit_gui():
    root.quit()
    root.destroy()
def sel_hv():
    global select_type
    select_type= 'h'
    quit_gui();
def sel_nt():
    global select_type
    select_type= 'n'
    quit_gui();
def sel_lv():
    global select_type
    select_type= 'l'
    quit_gui();
b_hv = tk.Button(master=root, text='High valence',
                 width=BUTTON_WIDTH, height=BUTTON_HEIGHT, command=sel_hv)
b_nt = tk.Button(master=root, text='Neutral',
                 width=BUTTON_WIDTH, height=BUTTON_HEIGHT, command=sel_nt)
b_lv = tk.Button(master=root, text='Low valence',
                 width=BUTTON_WIDTH, height=BUTTON_HEIGHT, command=sel_lv)
b_hv.pack(in_=root, side=tk.LEFT)
b_nt.pack(in_=root, side=tk.LEFT)
b_lv.pack(in_=root, side=tk.LEFT)
root.mainloop()

# 刺激画像の読み込み（A/A001.bmpの形式でセットされる）
if select_type == 'h':
    IMAGES = [f[0] + '/' + f for f in ImageDef.IMAGES_HV]
elif select_type == 'n':
    IMAGES = [f[0] + '/' + f for f in ImageDef.IMAGES_NT]
elif select_type == 'l':
    IMAGES = [f[0] + '/' + f for f in ImageDef.IMAGES_LV]
else:
    IMAGES = [f[0] + '/' + f for f in ImageDef.IMAGES_HV]
    IMAGES = IMAGES + [f[0] + '/' + f for f in ImageDef.IMAGES_NT]
    IMAGES = IMAGES + [f[0] + '/' + f for f in ImageDef.IMAGES_LV]

# 画像提示
cv2.namedWindow(WINDOW_NAME, 0)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

image_paths = [PICTURE_FOLDER + p for p in IMAGES]
num_images = len(image_paths)
disp_counter = 0

random_image = copy.copy(image_paths)
random.shuffle(random_image)    # ランダムに並び替え

if not os.path.exists(SAVE_FILE_FOLDER):
    os.mkdir(SAVE_FILE_FOLDER)

save_file_name = PROJECT_PREFIX + datetime.now().strftime('%Y%m%d%H%M%S') + '.txt'
save_file = open(SAVE_FILE_FOLDER + save_file_name, "w")

os.system('rundll32 user32.dll,SetCursorPos')   # マウスカーソルを右上隅に移動

try:
    for fname in random_image:
        disp_counter = disp_counter + 1

        # 予告画像の表示
        img = cv2.imread(CROSS_IMAGE)
        cv2.imshow(WINDOW_NAME, img)
        cv2.waitKey(DISPLAY_CROSS_MS)

        # 刺激画像の表示
        img = cv2.imread(fname)
        disp_time = datetime.now()      # 画像提示時刻を取得
        cv2.imshow(WINDOW_NAME, img)
        cv2.waitKey(DISPLAY_STIM_MS)
        
        # 画像提示時刻の書き出し
        write_str = (
            disp_time.strftime(STIM_TIME_FORMAT) +
            '%03d'%(disp_time.microsecond//1000) + '\t' + fname + '\n'
            )  # if change the above '%03d', revise LENGTH_UNDER_SEC value
        save_file.write(write_str)

        # 休止、終了画像の表示
        if disp_counter < num_images:
            img = cv2.imread(BLANK_IMAGE)
            cv2.imshow(WINDOW_NAME, img)
            cv2.waitKey(DISPLAY_BLANK_MS)
        else:
            img = cv2.imread(END_IMAGE)
            cv2.imshow(WINDOW_NAME, img)
            cv2.waitKey(DISPLAY_END_MS)

except Exception as ex:
    print('Error message:' + str(sys.exc_info()[0]))
    print(type(ex))
    print(ex.args)

finally:
    save_file.close()
    cv2.destroyAllWindows()
