# -*- coding: utf-8 -*-
""" SVMを用いたクラス分類を適用するスクリプト（第１章） """
import numpy as np
from scipy.spatial.distance import squareform,pdist                                                              
import matplotlib.pyplot as plt

from sklearn import manifold
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

import ExgBdWavData
import VisualStimuliTiming
import CalcEegBand
import ClfUtils

""" 脳波データ、刺激提示タイミングデータの読み込み """
w_hv = ExgBdWavData.ExgBdWavData('./wave_data/rec_20160920_194429052.txt')
t_hv = VisualStimuliTiming.VisualStimuliTiming(
                    './visual_stimuli_log/visual_stimuli_20160920194009.txt')
time_hv, label_hv, pic_hv, log_wav = t_hv.get_stim_timing_and_log_wave(w_hv)

w_nt = ExgBdWavData.ExgBdWavData('./wave_data/rec_20160920_194936848.txt')
t_nt = VisualStimuliTiming.VisualStimuliTiming(
                    './visual_stimuli_log/visual_stimuli_20160920194534.txt')
time_nt, label_nt, pic_nt, log_wav = t_nt.get_stim_timing_and_log_wave(w_nt)

w_lv = ExgBdWavData.ExgBdWavData('./wave_data/rec_20160920_195426648.txt')
t_lv = VisualStimuliTiming.VisualStimuliTiming(
                    './visual_stimuli_log/visual_stimuli_20160920195023.txt')
time_lv, label_lv, pic_lv, log_wav = t_lv.get_stim_timing_and_log_wave(w_lv)

""" ラベルデータの準備 """
label_str_and_float = {
    VisualStimuliTiming.HIGH_VALENCE_STR: 1.0,
    VisualStimuliTiming.NEUTRAL_STR: 0.0,
    VisualStimuliTiming.LOW_VALENCE_STR: -1.0
}

cl_hv = np.array([label_str_and_float[x] for x in label_hv])
cl_nt = np.array([label_str_and_float[x] for x in label_nt])
cl_lv = np.array([label_str_and_float[x] for x in label_lv])

y = np.r_[cl_hv, cl_nt, cl_lv]      # ラベルデータ

# FFTパラメータ、刺激提示情報の設定
FFT_N = 256                         # FFTデータ点数
FFT_INT = w_hv.sampling_rate        # FFTインターバル

STIM_S = 5                          # 刺激呈示時間[s]
RANGE_X = STIM_S * FFT_INT          # 波形グラフのデータ点数

""" 特徴量の抽出 """
def get_band_abs(data_len, w_obj, time_list):
    """ 帯域の絶対値を取得 """
    ret_data = []
    for i in range(data_len):
        wave_dat = w_obj.get_epoch_wave(time_list[i], STIM_S+1)
        band_obj = CalcEegBand.CalcEegBand(wave_dat, FFT_N, FFT_INT)
        ret_data.append(np.array(band_obj.band_abs).flatten())
    return ret_data

def get_alpha_peak(data_len, w_obj, time_list):
    """ αピーク周波数を取得 """
    ret_data = []
    for i in range(data_len):
        wave_dat = w_obj.get_epoch_wave(time_list[i], STIM_S+1)
        band_obj = CalcEegBand.CalcEegBand(wave_dat, FFT_N, FFT_INT)
        ret_data.append(band_obj.peak_alpha_frequency.flatten())
    return ret_data

def get_alpha_amp(data_len, w_obj, time_list):
    """ α振幅を取得 """
    ret_data = []
    for i in range(data_len):
        wave_dat = w_obj.get_epoch_wave(time_list[i], STIM_S+1)
        band_obj = CalcEegBand.CalcEegBand(wave_dat, FFT_N, FFT_INT)
        ret_data.append(np.array(band_obj.alpha_amplitude).flatten())
    return ret_data

def get_all_features(data_len, w_obj, time_list):
    """ 帯域の絶対値、αピーク、α振幅を取得 """
    ret_data = []
    for i in range(data_len):
        wave_dat = w_obj.get_epoch_wave(time_list[i], STIM_S+1)
        band_obj = CalcEegBand.CalcEegBand(wave_dat, FFT_N, FFT_INT)
        band_alpha_peak_amp = np.r_[np.array(band_obj.band_abs).flatten(),
                                    band_obj.peak_alpha_frequency.flatten(),
                                    np.array(band_obj.alpha_amplitude).flatten()]
        ret_data.append(np.array(band_alpha_peak_amp).flatten())
    return ret_data

""" ランダムフォレストによる特徴量の重要度確認（図７） """
if False:    # 時間がかかるのでスキップしています（出力するときはTrueにする）
    X_hv = np.array(get_all_features(len(cl_hv), w_hv, time_hv))
    X_nt = np.array(get_all_features(len(cl_nt), w_nt, time_nt))
    X_lv = np.array(get_all_features(len(cl_lv), w_lv, time_lv))
    X = np.r_[X_hv, X_nt, X_lv]         # 特徴量データ
    
    # トレーニングデータとテストデータの分割
    X_train, X_test, y_train, y_test = train_test_split(
                                       X, y, test_size=0.25, random_state=0, stratify=y)
    # 特徴量ラベルの生成
    ch_str = ['ch1 ', 'ch2 ']
    band_str = ['Delta ', 'Theta ' ,'Alpha ', 'Beta ', 'Gamma ']
    a_f_str = 'AlphaFreq '
    a_p_str = ['Low alpha ', 'High alpha ']
    feat_labels = []
    N = 2*5*5
    for i in range(N):
        feat_labels = feat_labels + [ch_str[i//(N//2)] + band_str[(i//5)%5] + str(i%5)]
    N = 2*5
    for i in range(N):
        feat_labels = feat_labels + [ch_str[i//(N//2)] + a_f_str + str(i%5)]
    N = 2*2*5
    for i in range(N):
        feat_labels = feat_labels + [ch_str[i//(N//2)] + a_p_str[(i//5)%2] + str(i%5)]
    
    # 決定木：10,000本、全コアで並列演算
    forest = RandomForestClassifier(n_estimators=10000,
                                    random_state=0,
                                    n_jobs=-1)
    # 訓練用の特徴量データとラベルデータ与えて実行
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    # 重要度を抽出
    indices = np.argsort(importances)[::-1]
    
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, 
                                feat_labels[indices[f]], 
                                importances[indices[f]]))
    
    plt.title('Feature Importances')
    plt.bar(range(X_train.shape[1]), 
            importances[indices],
            color='lightblue', 
            align='center')
    
    plt.xticks(range(X_train.shape[1]), 
               np.array(feat_labels)[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    #plt.savefig('./random_forest.png', dpi=300)
    plt.show()

""" MDSによる可視化 """
def draw_MDS(X_train, X_test, y_train, y_test, title):
    """ MDSの描画 """
    test_flag = []
    for x1 in X:
        flag = False
        for x2 in X_test:
            if np.all(x1 == x2):
                flag = True
        test_flag.append(flag)
    
    mds = manifold.MDS(n_components=2,
                       dissimilarity="precomputed",
                       random_state=1)
    similarities = squareform(pdist(X,'seuclidean'))
    pos = mds.fit_transform(similarities)
    hv_flag = np.array([label_str_and_float[VisualStimuliTiming.HIGH_VALENCE_STR] == x for x in y])
    nt_flag = np.array([label_str_and_float[VisualStimuliTiming.NEUTRAL_STR] == x for x in y])
    lv_flag = np.array([label_str_and_float[VisualStimuliTiming.LOW_VALENCE_STR] == x for x in y])
    hv_train_flag = np.array([x and (not y) for (x, y) in zip(hv_flag, test_flag)])
    nt_train_flag = np.array([x and (not y) for (x, y) in zip(nt_flag, test_flag)])
    lv_train_flag = np.array([x and (not y) for (x, y) in zip(lv_flag, test_flag)])
    hv_test_flag = np.array([x and y for (x, y) in zip(hv_flag, test_flag)])
    nt_test_flag = np.array([x and y for (x, y) in zip(nt_flag, test_flag)])
    lv_test_flag = np.array([x and y for (x, y) in zip(lv_flag, test_flag)])
    M_SIZE = 40
    plt.figure()
    plt.scatter(pos[hv_train_flag, 0], pos[hv_train_flag, 1],
                marker='o', s=M_SIZE, c='r', label='High valence')
    plt.scatter(pos[hv_test_flag, 0], pos[hv_test_flag, 1],
                marker='d', s=M_SIZE, c='r', label='Test data')
    plt.scatter(pos[nt_train_flag, 0], pos[nt_train_flag, 1],
                marker='o', s=M_SIZE, c='g', label='Neutral')
    plt.scatter(pos[nt_test_flag, 0], pos[nt_test_flag, 1],
                marker='d', s=M_SIZE, c='g', label='Test data')
    plt.scatter(pos[lv_train_flag, 0], pos[lv_train_flag, 1],
                marker='o', s=M_SIZE, c='b', label='Low valence')
    plt.scatter(pos[lv_test_flag, 0], pos[lv_test_flag, 1],
                marker='d', s=M_SIZE, c='b', label='Test data')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., 1.02),
               loc=3, ncol=3, mode="expand", borderaxespad=0.)
    plt.suptitle('MDS plot (' + title + ')', y=0.99, fontsize=20)
    plt.subplots_adjust(top=0.82)
    plt.show()

# alphaピーク
X_hv = np.array(get_alpha_peak(len(cl_hv), w_hv, time_hv))
X_nt = np.array(get_alpha_peak(len(cl_nt), w_nt, time_nt))
X_lv = np.array(get_alpha_peak(len(cl_lv), w_lv, time_lv))
X = np.r_[X_hv, X_nt, X_lv]         # 特徴量データ

# トレーニングデータとテストデータの分割
X_train, X_test, y_train, y_test = train_test_split(
                                   X, y, test_size=0.25, random_state=0, stratify=y)

draw_MDS(X_train, X_test, y_train, y_test, 'Alpha peak')    # MDSの描画（図１１）

# 帯域の絶対値
X_hv = np.array(get_band_abs(len(cl_hv), w_hv, time_hv))
X_nt = np.array(get_band_abs(len(cl_nt), w_nt, time_nt))
X_lv = np.array(get_band_abs(len(cl_lv), w_lv, time_lv))
X = np.r_[X_hv, X_nt, X_lv]         # 特徴量データ

# トレーニングデータとテストデータの分割
X_train, X_test, y_train, y_test = train_test_split(
                                   X, y, test_size=0.25, random_state=0, stratify=y)

draw_MDS(X_train, X_test, y_train, y_test, 'Band abs.')     # MDSの描画（図１０）


""" SVMによる判別 """
print('*'*10 + ' SVM ' + '*'*10)

# クロスバリデーション用データの生成
#（各クラスターが同数となるよう分割）
kf = StratifiedKFold(y_train, n_folds=9)

# ハイパーパラメータのグリッド探索
param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
              'gamma': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
              
# カーネル毎に実行
kernels = ['linear', 'poly', 'rbf']

for kernel in kernels:
    print('\n' + '*'*3 + ' ' + kernel + ' ' + '*'*3)

    # 平均0、分散1に標準化
    scl = StandardScaler()
    X_train_std = scl.fit_transform(X_train)

    # グリッドサーチ
    gs = GridSearchCV(SVC(kernel=kernel, random_state=1),
                          param_grid, cv=kf)
    gs.fit(X_train_std, y_train)

    # 結果出力
    print('Best score: %.3f' % gs.best_score_ +
          ' for ' + str(gs.best_params_))
    print('\nMean score for each parameters:')
    for params, mean_score, all_scores in gs.grid_scores_:
        print('%.3f (+/- %.3f) for %s' %
              (mean_score, all_scores.std() / 2, params))

    # テストデータの結果出力
    X_test_std = scl.transform(X_test)
    clf = gs.best_estimator_
    print('\nTest accuracy: %.3f' %
          clf.score(X_test_std, y_test))    
    
    y_true, y_pred = y_test, clf.predict(X_test_std)
    print('\nClassification report')    
    print(classification_report(y_true, y_pred))

""" LDAによる判別 """
def draw_regions(X, y, clf):
    """ 分類エリアの図示 """
    DECISION_REGION_SCALE = [[-16, 16], [-10, 10]]
    plt.figure()
    ClfUtils.plot_decision_regions(X, y, classifier=clf, scale=DECISION_REGION_SCALE)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.show()

print('*'*10 + ' LDA ' + '*'*10)
# スケーリング（標準化）、LDAによる判別
pipe_lda = Pipeline([('scl', StandardScaler()),
                    ('lda', LinearDiscriminantAnalysis(n_components=2))])
X_train_lda = pipe_lda.fit_transform(X_train, y_train)
score = pipe_lda.score(X_test, y_test)
# テストデータの結果出力
print('\nTest accuracy: %.3f' % score)    
y_true, y_pred = y_test, pipe_lda.predict(X_test)
print('\nClassification report')    
print(classification_report(y_true, y_pred))

# 学習データ
lda_lda = LinearDiscriminantAnalysis(n_components=2)
lda_lda.fit(X_train_lda, y_train)
draw_regions(X_train_lda, y_train, lda_lda)   # コラム３図F(a)左

# テストデータ
X_test_lda = pipe_lda.transform(X_test)
draw_regions(X_test_lda, y_test, lda_lda)   # コラム３図F(a)右

""" LDA + ロジスティック回帰による判別 """
print('*'*10 + ' Logistic Regression ' + '*'*10)
# スケーリング（標準化）、LDAによる次元削減、ロジスティック回帰
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('lda', LinearDiscriminantAnalysis(n_components=2)),
                    ('clf', LogisticRegression(random_state=1))])

## k分割交差検証法を使用する場合
#NUM_OF_KFOLD = 9
#kfold = StratifiedKFold(y=y_train, n_folds=NUM_OF_KFOLD, random_state=1)
#scores = []
#for k, (train, test) in enumerate(kfold):
#    pipe_lr.fit(X_train[train], y_train[train])
#    score = pipe_lr.score(X_train[test], y_train[test])
#    scores.append(score)
#    print('Fold: %s, Class dist.: %s, Acc: %.3f' %
#          (k+1, np.bincount((y_train[train]).astype(int)+1), score))
#print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
#
#scores = cross_val_score(estimator=pipe_lr,
#                         X=X_train,
#                         y=y_train,
#                         cv=NUM_OF_KFOLD,
#                         n_jobs=1)
#print('CV accuracy scores: %s' % scores)
#print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

pipe_lr.fit(X_train, y_train)
score = pipe_lr.score(X_test, y_test)
# テストデータの結果出力
print('\nTest accuracy: %.3f' % score)    
y_true, y_pred = y_test, pipe_lr.predict(X_test)
print('\nClassification report')    
print(classification_report(y_true, y_pred))

# 学習データ
X_train_lda = pipe_lda.fit_transform(X_train, y_train)
lr = LogisticRegression(random_state=1)
lr = lr.fit(X_train_lda, y_train)
draw_regions(X_train_lda, y_train, lr)   # コラム３図F(b)左

# テストデータ
X_test_lda = pipe_lda.transform(X_test)
draw_regions(X_test_lda, y_test, lr)   # コラム３図F(b)右

""" LDA + SVMによる判別 """

for kernel in kernels:
    print('\n' + '*'*3 + ' ' + kernel + ' ' + '*'*3)
    pipe_lda = Pipeline([('scl', StandardScaler()),
                        ('lda', LinearDiscriminantAnalysis(n_components=2))])
    pipe_lda.fit(X_train, y_train)
    X_train_lda = pipe_lda.transform(X_train)
    X_test_lda = pipe_lda.transform(X_test)
    X_lda = pipe_lda.transform(X)
    gs = GridSearchCV(SVC(kernel=kernel, random_state=1), param_grid, cv=kf)
    gs = gs.fit(X_train_lda, y_train)

    print('\nBest score:')
    print(gs.best_score_)
    print('\nBest parameter:')
    print(gs.best_params_)
    print('\nResult of test data:')
    clf = gs.best_estimator_
    print('Test accuracy: %.3f' % clf.score(X_test_lda, y_test))
    y_true, y_pred = y_test, clf.predict(X_test_lda)
    print('Classification report')    
    print(classification_report(y_true, y_pred))

    # 学習データ
    X_train_svc = clf.fit(X_train_lda, y_train)
    draw_regions(X_train_lda, y_train, X_train_svc)   # コラム３図F(c),(d),(e)左
    
    # テストデータ
    draw_regions(X_test_lda, y_test, X_train_svc)   # コラム３図F(c),(d),(e)右

