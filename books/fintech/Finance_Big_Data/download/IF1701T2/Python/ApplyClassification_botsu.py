# -*- coding: utf-8 -*-
""" SVMを用いたクラス分類を適用するスクリプト """
import sys
import numpy as np
from scipy.spatial.distance import squareform,pdist                                                              
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import manifold
from sklearn.cross_validation import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.learning_curve import learning_curve

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import ExgBdWavData
import VisualStimuliTiming
import CalcEegBand
import ClfUtils

sns.set_style("whitegrid")

# 脳波データ、刺激提示タイミングデータの読み込み
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

# ラベルデータの準備
label_str_and_float = {
    VisualStimuliTiming.HIGH_VALENCE_STR: 1.0,
    VisualStimuliTiming.NEUTRAL_STR: 0.0,
    VisualStimuliTiming.LOW_VALENCE_STR: -1.0
}

cl_hv = np.array([label_str_and_float[x] for x in label_hv])
cl_nt = np.array([label_str_and_float[x] for x in label_nt])
cl_lv = np.array([label_str_and_float[x] for x in label_lv])

y = np.r_[cl_hv, cl_nt, cl_lv]                  # ラベルデータ

# FFT、刺激提示情報の取得
FFT_N = 256                                     # FFTデータ点数
FFT_INT = w_hv.sampling_rate                    # FFTインターバル

STIM_S = 5                                      # 刺激呈示時間[s]
RANGE_X = STIM_S * FFT_INT                      # 波形グラフのデータ点数


# 特徴量の抽出
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

def get_alpha_power(data_len, w_obj, time_list):
    """ αパワーを取得 """
    ret_data = []
    for i in range(data_len):
        wave_dat = w_obj.get_epoch_wave(time_list[i], STIM_S+1)
        band_obj = CalcEegBand.CalcEegBand(wave_dat, FFT_N, FFT_INT)
        ret_data.append(np.array(band_obj.alpha_amplitude).flatten())
    return ret_data

X_hv = np.array(get_band_abs(len(cl_hv), w_hv, time_hv))
X_nt = np.array(get_band_abs(len(cl_nt), w_nt, time_nt))
X_lv = np.array(get_band_abs(len(cl_lv), w_lv, time_lv))

X = np.r_[X_hv, X_nt, X_lv]                     # 特徴量データ

# MDSによる特徴量の可視化
plt.figure()
mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=1)
similarities = squareform(pdist(X,'seuclidean'))
pos = mds.fit_transform(similarities)
hv_flag = np.array([label_str_and_float[VisualStimuliTiming.HIGH_VALENCE_STR] == x for x in y])
nt_flag = np.array([label_str_and_float[VisualStimuliTiming.NEUTRAL_STR] == x for x in y])
lv_flag = np.array([label_str_and_float[VisualStimuliTiming.LOW_VALENCE_STR] == x for x in y])
plt.scatter(pos[hv_flag, 0], pos[hv_flag, 1], marker = 'o', c='r', label='High valence')
plt.scatter(pos[nt_flag, 0], pos[nt_flag, 1], marker = 'o', c='g', label='Neutral')
plt.scatter(pos[lv_flag, 0], pos[lv_flag, 1], marker = 'o', c='b', label='Low valence')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
plt.suptitle('MDS plot', y=1.05)
plt.show()

# 代表波形の抽出
REP_WAVE_SCALE = 40

#mean_hv = np.mean(X_hv, axis=0)
#mean_nt = np.mean(X_nt, axis=0)
#mean_lv = np.mean(X_lv, axis=0)
#
#norm_hv = np.apply_along_axis(lambda x: np.linalg.norm(x - mean_hv), 1, X_hv)
#norm_nt = np.apply_along_axis(lambda x: np.linalg.norm(x - mean_nt), 1, X_nt)
#norm_lv = np.apply_along_axis(lambda x: np.linalg.norm(x - mean_lv), 1, X_lv)
#
#represent_hv_idx = np.argmin(norm_hv)
#represent_nt_idx = np.argmin(norm_nt)
#represent_lv_idx = np.argmin(norm_lv)
#
#rep_wave_hv = w_hv.get_epoch_wave(time_hv[represent_hv_idx], STIM_S)
#rep_wave_nt = w_nt.get_epoch_wave(time_nt[represent_nt_idx], STIM_S)
#rep_wave_lv = w_lv.get_epoch_wave(time_lv[represent_lv_idx], STIM_S)
#
#col = ['r', 'r', 'g', 'g', 'b', 'b']
#wav = [rep_wave_hv, rep_wave_hv, rep_wave_nt, rep_wave_nt, rep_wave_lv, rep_wave_lv]
#plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
#for i in range(6):
#    pos = 320+(i+1)
#    plt.subplot(pos)
#    plt.plot(wav[i][:,(i%2)], c=col[i])
#    plt.ylim([-REP_WAVE_SCALE, REP_WAVE_SCALE])
#    if i < 2:
#        plt.title('Ch' + str(i+1))
#plt.show()
#
##sys.exit()

# テストデータとトレーニングデータの分類
NUM_OF_KFOLD = 9

# トレーニングデータとテストデータの分割
X_train, X_test, y_train, y_test = train_test_split(
                                            X, y, test_size=0.25, random_state=0, stratify=y)

test_flag = []
for x1 in X:
    flag = False
    for x2 in X_test:
        if np.all(x1 == x2):
            flag = True
    test_flag.append(flag)

plt.figure()
mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=1)
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
plt.scatter(pos[hv_train_flag, 0], pos[hv_train_flag, 1], marker='o', c='r', label='High valence')
plt.scatter(pos[hv_test_flag, 0], pos[hv_test_flag, 1], marker='x', c='r', label='Test data')
plt.scatter(pos[nt_train_flag, 0], pos[nt_train_flag, 1], marker='o', c='g', label='Neutral')
plt.scatter(pos[nt_test_flag, 0], pos[nt_test_flag, 1], marker='x', c='g', label='Test data')
plt.scatter(pos[lv_train_flag, 0], pos[lv_train_flag, 1], marker='o', c='b', label='Low valence')
plt.scatter(pos[lv_test_flag, 0], pos[lv_test_flag, 1], marker='x', c='b', label='Test data')
plt.legend(bbox_to_anchor=(0., 1.02, 1., 1.02), loc=3, ncol=3, mode="expand", borderaxespad=0.)
plt.suptitle('MDS plot', y=0.99, fontsize=20)
plt.subplots_adjust(top=0.82)
plt.show()

sys.exit()

DECISION_REGION_SCALE = [[-16, 16], [-10, 10]]

# SVMによる判別
print('SVM')

# クロスバリデーション用データの生成
kf = StratifiedKFold(y_train, n_folds=NUM_OF_KFOLD)                        # 各クラスターが同数となるよう分割
#kf = StratifiedKFold(n=len(X_train), n_folds=NUM_OF_KFOLD, shuffle=True)  # ランダムに分割する場合はこちら

# ハイパーパラメータのグリッド探索
param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
              'gamma': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}

kernels = ['linear', 'rbf', 'poly']

for kernel in kernels:
    print('\n' + '*'*3 + ' ' + kernel + ' ' + '*'*3)

    pipe_lda = Pipeline([('scl', StandardScaler()),
                        ('lda', LinearDiscriminantAnalysis(n_components=2))])

    pipe_lda.fit(X_train, y_train)
    X_train_lda = pipe_lda.transform(X_train)
    X_test_lda = pipe_lda.transform(X_test)
    X_lda = pipe_lda.transform(X)

#    pipe_svc = Pipeline([('scl', StandardScaler()),
#                        ('lda', LinearDiscriminantAnalysis(n_components=2)),
#                        ('clf', SVC(kernel=kernel, random_state=1))])

    gs = GridSearchCV(SVC(kernel=kernel, random_state=1), param_grid, cv=kf)
#    gs = GridSearchCV(estimator=pipe_svc,
#                      param_grid=param_grid,
#                      scoring='accuracy',
#                      cv=kf,
#                      n_jobs=1)
    gs = gs.fit(X_train_lda, y_train)

    print('\nBest score:')
    print(gs.best_score_)
    print('\nBest parameter:')
    print(gs.best_params_)

    if False:  # Debug
        print('\nMean score by cross validation with training data:')
        for params, mean_score, all_scores in gs.grid_scores_:
            print('{:.3f} (+/- {:.3f}) for {}'.format(mean_score, all_scores.std() / 2, params))

    print('\nResult of test data:')
    clf = gs.best_estimator_
    
    print('Test accuracy: %.3f' % clf.score(X_test_lda, y_test))    
    
    y_true, y_pred = y_test, clf.predict(X_test_lda)
    print('Classification report')    
    print(classification_report(y_true, y_pred))

    plt.figure()
    X_train_svc = clf.fit(X_train_lda, y_train)
    ClfUtils.plot_decision_regions(X_train_lda, y_train, classifier= X_train_svc, scale=DECISION_REGION_SCALE)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.show()

    plt.figure()
    ClfUtils.plot_decision_regions(X_test_lda, y_test, classifier= X_train_svc, scale=DECISION_REGION_SCALE)
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.show()
    
    # 学習曲線の描画
    plt.figure()
    training_sizes, train_scores, test_scores = learning_curve(SVC(
                        kernel=kernel, C=gs.best_params_['C'], gamma=gs.best_params_['gamma'],),
                        X_train, y_train, cv=NUM_OF_KFOLD, scoring='mean_squared_error',
                        train_sizes=np.arange(0.1,1.0,0.1))
    plt.plot(training_sizes, train_scores.mean(axis=1), label='training scores')
    plt.plot(training_sizes, test_scores.mean(axis=1), label='test scores')
    plt.legend(loc='best')
    plt.suptitle('Training curve')
    plt.show()



#kfold = StratifiedKFold(y=y_train, n_folds=NUM_OF_KFOLD, random_state=1)
#scores = []
#
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




# LDAによる判別
print('Pipeline')
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('lda', LinearDiscriminantAnalysis(n_components=2)),
                    ('clf', LogisticRegression(random_state=1))])

kfold = StratifiedKFold(y=y_train, n_folds=NUM_OF_KFOLD, random_state=1)
scores = []

for k, (train, test) in enumerate(kfold):
    pipe_lr.fit(X_train[train], y_train[train])
    score = pipe_lr.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %s, Class dist.: %s, Acc: %.3f' %
          (k+1, np.bincount((y_train[train]).astype(int)+1), score))
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

scores = cross_val_score(estimator=pipe_lr,
                         X=X_train,
                         y=y_train,
                         cv=NUM_OF_KFOLD,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

pipe_lr.fit(X_train, y_train)
score = pipe_lr.score(X_test, y_test)

print('Test accuracy: %.3f' % score)    

y_true, y_pred = y_test, pipe_lr.predict(X_test)
print('Classification report')    
print(classification_report(y_true, y_pred))




print('Train')
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
lr = LogisticRegression(random_state=1)
lr = lr.fit(X_train_lda, y_train)
plt.figure()
ClfUtils.plot_decision_regions(X_train_lda, y_train, classifier= lr, scale=DECISION_REGION_SCALE)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.show()

print('Test')
X_test_lda = lda.transform(X_test_std)
plt.figure()
ClfUtils.plot_decision_regions(X_test_lda, y_test, classifier= lr, scale=DECISION_REGION_SCALE)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
#plt.legend(loc='lower left')
plt.show()

plt.figure()
lda = LinearDiscriminantAnalysis(n_components=None, solver='eigen', shrinkage='auto')
lda.fit(X_train_std, y_train)
plt.bar(range(X.shape[1]), lda.explained_variance_ratio_)

