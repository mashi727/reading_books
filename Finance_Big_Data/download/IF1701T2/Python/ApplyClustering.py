# -*- coding: utf-8 -*-
""" k-meansを用いたクラスタリングを適用するスクリプト（第２章） """
import numpy as np
from scipy import signal
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, dendrogram, ward
import pandas as pd                                                    
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import manifold
from sklearn.cluster import KMeans

import ExgBdWavData

sns.set_style("whitegrid")

""" 特徴量データの準備 """
hpf_fc = 5
lpf_fc = 30 
notch_f0 = 60
wav_obj = ExgBdWavData.ExgBdWavData('./wave_data/rec_20161001_194534452.txt',
                                    hpf_fc, lpf_fc, notch_f0)     # じゃんけんデータ

# 全波整流
wav = np.abs(wav_obj.wave_data)

# 1次ローパス（両方向フィルタ）
smpl = 200
nyq = smpl / 2
LPF_ORDER = 1
lpf_fc2 = 2
bl, al = signal.butter(LPF_ORDER, lpf_fc2/nyq, 'low')
wav = wav.apply(lambda x: signal.filtfilt(bl, al, x), axis=0)

# 波形の切り出し、振幅計算
def get_epoch_data(wav):
    idx = []
    amp = []
    amp_3part = []

    th_val_high = 10
    th_val_low = 5
    x_pre = 50
    x_post = 250
    th_high = []
    th_low = []
    
    detect_ch = 0   # # Ch1で切り出しタイミングを抽出
    
    # 上側の閾値を超える点を抽出
    th_high.append(wav[detect_ch] > th_val_high)
    th_high[detect_ch][1:][list(th_high[detect_ch][:-1] & th_high[detect_ch][1:])] = False
    
    # 下側の閾値を下回る点を抽出
    th_low.append(wav[detect_ch] < th_val_low)
    th_low[detect_ch][1:][list(th_low[detect_ch][:-1] & th_low[detect_ch][1:])] = False
    
    # 上側閾値のうち、前都の間隔がx_postより間隔の短い点、下側閾値を下回る前に現れた点は除去
    peak_idx = np.where(th_high[detect_ch])[0]
    idx_flag = [True] * len(peak_idx)
    for i in range(1,len(peak_idx)):
        if ((peak_idx[i] - peak_idx[i-1] < x_post) or
            (sum(th_low[detect_ch][peak_idx[i-1]:peak_idx[i]]) < 1)):
            idx_flag[i] = False
    peak_idx = peak_idx[np.array(idx_flag)]

    # 各チャネルの波形を切り出して、前半、中盤、後半の3つの振幅を抽出        
    for ch in range(wav.shape[1]):
        amp_list = []
        amp_3part_list = []
        wav_len = wav[ch].shape[0]
        for i in peak_idx:
            start_peak_idx = (i-x_pre) if (i-x_pre) > 0 else 0
            end_peak_idx = (i+x_post) if (i+x_post < wav_len) else wav_len
            ep = np.array(wav[ch][start_peak_idx:end_peak_idx]) # 波形確認用
            start_valid_idx = x_pre
            end_valid_idx = np.max(np.where(ep>th_val_low))
            duration = end_valid_idx - start_valid_idx
            early_range = range(start_valid_idx, start_valid_idx+duration//3)
            middle_range = range(start_valid_idx+duration//3, start_valid_idx+duration*2//3)
            late_range = range(start_valid_idx+duration*2//3, end_valid_idx)
            amp_list.append(np.mean(ep[range(start_valid_idx, end_valid_idx)]))
            amp_3part_list.append(np.array([
                                   np.mean(ep[early_range]), 
                                   np.mean(ep[middle_range]),
                                   np.mean(ep[late_range])]))
        idx.append(peak_idx)
        amp.append(amp_list)
        amp_3part.append(amp_3part_list)
    return [idx, amp, amp_3part]

idx_dat, amp_dat, amp_3part_dat = get_epoch_data(wav)  # 振幅を取得

amp = np.c_[np.array(amp_dat[0]), np.array(amp_dat[1])]
amp_3part = np.c_[np.array(amp_3part_dat[0]), np.array(amp_3part_dat[1])]    # Ch1, Ch2 を結合

""" 散布図の描画（図３） """
CLASS_LEN = 15
MARKER_SIZE = 80
plt.figure()
par_idx = range(0, CLASS_LEN)
goo_idx = range(CLASS_LEN, CLASS_LEN*2)
choki_idx = range(CLASS_LEN*2, CLASS_LEN*3)
plt.scatter(amp[par_idx, 0], amp[par_idx, 1], marker='o', s=MARKER_SIZE, c='b', label='Par')
plt.scatter(amp[goo_idx, 0], amp[goo_idx, 1], marker='o', s=MARKER_SIZE, c='g', label='Goo')
plt.scatter(amp[choki_idx, 0], amp[choki_idx, 1], marker='o', s=MARKER_SIZE, c='r', label='Choki')

""" 相関分析（2変数の散布図）（図８） """
CLASS_LEN = 15
amp_3part_df = pd.DataFrame(amp_3part)
amp_3part_df.columns = ['Ch1 early','Ch1 mid','Ch1 late','Ch2 early','Ch2 mid','Ch2 late']
amp_3part_df['label'] = ['M1']*CLASS_LEN + ['M2']*CLASS_LEN + ['M3']*CLASS_LEN
sns.pairplot(amp_3part_df, hue='label')

""" MDSの描画（図９） """
mds = manifold.MDS(n_components=2,
                   dissimilarity="precomputed",
                   random_state=1)
similarities = squareform(pdist(amp_3part,'seuclidean'))
pos = mds.fit_transform(similarities)
CLASS_LEN = 15
MARKER_SIZE = 80
plt.figure()
par_idx = range(0, CLASS_LEN)
goo_idx = range(CLASS_LEN, CLASS_LEN*2)
choki_idx = range(CLASS_LEN*2, CLASS_LEN*3)
plt.scatter(pos[par_idx, 0], pos[par_idx, 1], marker='o', s=MARKER_SIZE, c='b', label='Par')
plt.scatter(pos[goo_idx, 0], pos[goo_idx, 1], marker='o', s=MARKER_SIZE, c='g', label='Goo')
plt.scatter(pos[choki_idx, 0], pos[choki_idx, 1], marker='o', s=MARKER_SIZE, c='r', label='Choki')
plt.show()

""" クラスタ数kの検討（図１０） """
MAX_CLUSTERS = 10
plt.figure()
distortions = []
y_km = []
for i in range(1, MAX_CLUSTERS):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    y_km.append(km.fit_predict(amp_3part))
    distortions.append(km.inertia_)

plt.plot(range(1,MAX_CLUSTERS), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

""" k-means++の実行（特徴量１） """
N=3
km = KMeans(n_clusters=N,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0)

y_km = km.fit_predict(amp)

""" k-means++の実行（特徴量２） """
N=3
km = KMeans(n_clusters=N,
            init='k-means++',   
            n_init=10,
            max_iter=300,
            random_state=0)

y_km = km.fit_predict(amp_3part)


""" 階層的クラスタリング（コラム２） """
def plot_dendrogram(mat, labels, title):
    """ デンドログラムの描画 """
    plt.figure()
    dendrogram(mat, labels= labels)
    plt.title(title)
    plt.show()

def plot_dendrogram_heatmap(df, mat, title):
    """ デンドログラムとヒートマップの描画 """
    g = sns.clustermap(df,
                       col_cluster=False,
                       cmap='rainbow',
                       figsize=(8,8),
                       row_linkage=mat)
    #g.cax.set_visible(False)  # Uncomment if removing colorbar
    cb = g.cax.get_position()
    g.cax.set_position([cb.x0,
                        cb.y0+cb.height*.2,
                        cb.width,
                        cb.height*0.8])
    g.ax_col_dendrogram.set_visible(False)
    hm = g.ax_heatmap.get_position()
    plt.setp(g.ax_heatmap.yaxis.
             get_majorticklabels(),
             fontsize=10, rotation=0)
    g.ax_heatmap.set_position([hm.x0+hm.width*0.75,
                               hm.y0, hm.width*0.25,
                               hm.height*1.2])
    row = g.ax_row_dendrogram.get_position()
    g.ax_row_dendrogram.set_position([
             row.x0,
             row.y0,
             row.width+hm.width*0.75,
             row.height*1.2])
    plt.suptitle(title)

# データの準備
labels = ['Par']*15 + ['Goo']*15 + ['Choki']*15
amp_3part_df = pd.DataFrame(amp_3part)
# 階層的クラスタリング（ユークリッド距離による最長距離法）
mat = linkage(amp_3part_df.values,
                    method='complete',
                    metric='euclidean')
title = 'Euclidean distance'
# デンドログラムの描画（コラム２図C左）
plot_dendrogram(mat, labels, title)
# デンドログラムとヒートマップの描画（コラム２図D左）
plot_dendrogram_heatmap(amp_3part_df, mat, title)

# 階層的クラスタリング（ウォード法）
mat = ward(amp_3part)
title = 'Ward\'s method'
# デンドログラムの描画（コラム２図C右）
plot_dendrogram(mat, labels, title)
# デンドログラムとヒートマップの描画（コラム２図D右）
plot_dendrogram_heatmap(amp_3part_df, mat, title)

