# -*- coding: utf-8 -*-
""" k-NNを用いた変化点検知を適用するスクリプト（第３章） """
import sys
import time
import numpy as np
import pandas as pd                                                    
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import signal
from scipy.cluster.hierarchy import ward, cut_tree

from sklearn.neighbors import NearestNeighbors

import ExgBdWavData

sns.reset_orig()

""" 特徴量データの準備 """
hpf_fc = 0.5
lpf_fc = 60
notch_f0 = 60
wav_obj = ExgBdWavData.ExgBdWavData('./wave_data/rec_20161005_102809097.txt',
                                    hpf_fc, lpf_fc, notch_f0)

ecg_wav = wav_obj.wave_data.iloc[:,1]

smpl = 200
nyq = smpl / 2
HPF_ORDER = 2
hpf_fc2 = 10
bh, ah = signal.butter(HPF_ORDER, hpf_fc2/nyq, 'high')
x = signal.filtfilt(bh, ah, ecg_wav)

# 部分時系列の生成
w = 40              # ウィンドウサイズ

d = np.zeros((len(x)-w+1, w))
for i in range(w):
    start_idx = i
    end_idx = len(x)-w+1+i
    d[:,i] = x[start_idx:end_idx]

train_len = 6000    # 学習データ数
train_start = 0
train_end = train_start + train_len

# 学習データのプロット
plt.figure()
plt.plot(x[train_start:train_end])
plt.suptitle('Training data (' + str(train_start) + ':' + str(train_end) + ')')
plt.show()

""" 変化点の検出 """
# k近傍クラスのオブジェクトを生成
nk = 1              # 近傍点数
nn = NearestNeighbors(n_neighbors=nk,
                      p=2,
                      metric='minkowski')   # minkowski の p=2 はユークリッド距離

# テスト範囲
test_start = 0
test_end = len(x)   # 全範囲

# スコア（距離）を算出
proc_start = time.time()
nn.fit(d[train_start:train_end,:])                      # 学習
dist, idx = nn.kneighbors(d[test_start:test_end,:])     # テストデータを適用
dist = np.array(dist)
dm = np.mean(dist, axis=1)
elapsed_per = (time.time() - proc_start)
print('Elapsed time:', elapsed_per)

# スコア（距離）をプロット
plt.figure()
plt.plot(dm)
plt.suptitle('Score data (' + 'Train: ' + str(train_start) + ':' + str(train_end) +
            ', Test: ' + str(test_start) + ':' + str(test_end) +
            ', Window size: ' + str(w) +
            ', k size: ' + str(nk) + ')')
plt.show()

# テストデータをプロット
plt.figure()
plt.plot(x[test_start:test_end])
plt.suptitle('Test data (' + str(test_start) + ':' + str(test_end) + ')')
plt.show()

# 閾値を超えた箇所を抽出
ANOMALY_TH = 200    # スコアの閾値
MASK_INT = 1500     # マスク期間
an_idx = (np.where(dm>ANOMALY_TH))[0]
an_idx = an_idx[np.r_[True, (an_idx[1:]-an_idx[:-1] > MASK_INT)]]

""" 波形とスコアをプロット（図３） """
PRE_LEN = 500       # 左側の余白
# for idx in an_idx: # 全て描画する場合はこちら
for idx in [216282, 676424, 916375]:
    s_idx = idx - PRE_LEN
    e_idx = idx + MASK_INT
    plt.figure()
    plt.subplot(211)
    plt.plot(x[s_idx:e_idx])
    plt.subplot(212)
    plt.plot(dm[s_idx:e_idx])
    plt.suptitle('Sampling range: ' + str(s_idx) + ':' + str(e_idx))

sys.exit() # 以降は時間がかかるため、ここで終了しています（実行する場合はコメントアウト）

""" スコア算出時間の短縮（クラスタリングにより学習データの数を低減） """
sns.set_style("whitegrid")
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

# ウォード法による階層的クラスタリング
mat = ward(d[train_start:train_end,:])

# デンドログラムの描画
title = 'Ward\'s method'
plot_dendrogram_heatmap(pd.DataFrame(d[train_start:train_end,:]), mat, title)

# クラスタ数を変更（デンドログラムの高さで指定）し、スコア算出時間と変化点の検出数を調べる
th_range = np.r_[np.arange(10,100,10),
                 np.arange(100,500,50),
                 np.arange(500,1000,100)]

for height_th in th_range:
    print('\nThreshold:', height_th)
    # 閾値でデンドログラムをカット
    cutree = cut_tree(mat, height=height_th)
    branches = len(np.unique(cutree))
    print('Branches:', branches)
    cutree_df = pd.DataFrame(np.c_[
                             d[train_start:train_end,:],
                             cutree])
    # 各クラスタの平均データを算出
    grouped = cutree_df.groupby(cutree_df.shape[1]-1)
    cutree_mean = grouped.mean()
    
    proc_start = time.time()    # 処理時間計測開始
    # 学習データをセット
    nn.fit(cutree_mean)
    # テストデータの距離（スコア（異常値））を算出
    dist, idx = nn.kneighbors(d[test_start:test_end,:])
    dist = np.array(dist)
    dm = np.mean(dist, axis=1)
    elapsed_per = (time.time() - proc_start)  # 計測終了
    print('Elapsed time:', elapsed_per)

    # スコアが閾値を超える変化点を抽出    
    ANOMALY_TH = 200    # 閾値
    MASK_INT = 1500     # 1度検出したら連続
    an_idx = (np.where(dm>ANOMALY_TH))[0]
    an_idx = an_idx[np.r_[True,
                    (an_idx[1:]-an_idx[:-1] > MASK_INT)]]
    print('Number of detection:', an_idx.shape[0])
    
    # スコアを画像として保存
    plt.plot(dm)
    plt.suptitle('Threshold: ' + str(height_th) +
                ', Branches: ' + str(branches) +
                ', Window size: ' + str(w) +
                ', k size: ' + str(nk))
    plt.savefig('height_' + str(height_th) + '.png')
    print('Saved height_' + str(height_th) + '.png')

