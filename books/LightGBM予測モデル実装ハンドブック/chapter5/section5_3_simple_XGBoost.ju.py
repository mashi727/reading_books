# %% [markdown]
"""
# XGBoostの可視化
"""

# %%
# # Colabでバージョンを変更するとき、コメントアウトして実行してください
# !pip install numpy==1.22.4
# !pip install matplotlib==3.7.1
# !pip install graphviz==0.20.1
# !pip install scikit-learn==1.2.2
# !pip install xgboost==1.7.5

# %%
# ライブラリのインポート
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# %%
# バージョンの確認
import matplotlib
import sklearn
print(np.__version__)
print(matplotlib.__version__)
print(sklearn.__version__)
print(xgb.__version__)

# %%
# 特徴量と目的変数の設定
X_train = np.array([[10], [20], [30], [40], [50], [60], [70], [80]])
y_train = np.array([6, 5, 7, 1, 2, 1, 6, 4])

# %%
# ハイパーパラメータの設定

xgb_train = xgb.DMatrix(X_train, label=y_train)

params = {
    'objective': 'reg:squarederror', # 損失関数
    'eval_metric': 'rmse', # 評価指標
    'max_depth': 2, # 深さの最大値
    'learning_rate': 0.8, # 学習率
    'base_score': 4, # 予測の初期値
    'min_split_loss': 0, # 枝刈り
    'reg_lambda': 0, # L2正則化
    'reg_alpha': 0, # L1正則化
    'seed': 0, # 乱数
}

# %%
# XGBoostの学習
model = xgb.train(params,
                  xgb_train,
                  evals=[(xgb_train, 'train')],
                  num_boost_round=2)

# %%
# XGBoostの予測
model.predict(xgb.DMatrix(X_train))

# %%
# 1本目の木の可視化
xgb.to_graphviz(model, num_trees=0)

# %%
# 2本目の木の可視化
xgb.to_graphviz(model, num_trees=1)

# %%


# %%


# %% [markdown]
"""
# 予測値の検証
"""

# %%
# 二乗誤差の重み
def weight(res, lam=0):
    if len(res)==0:
        return 0
    return sum(res)/(len(res)+lam)

# %%
# 二乗誤差の類似度
def similarity(res, lam=0):
    if len(res)==0:
        return 0
    return  sum(res)**2/(len(res)+lam)

# %%
# 分割点ごとの左葉類似度＋右葉類似度の計算
def split(X_train, residual):
  # プロット用のリスト
  index_plt = []
  similarity_plt = []
  # L2正則化
  lam = 0
  # 2次元配列を1次元配列
  X_train = X_train.flatten()
  # 分割点ごとの重みと類似度を計算
  for i in range(1, len(X_train)):
      X_left = np.array(X_train[:i])
      X_right = np.array(X_train[i:])
      res_left = np.array(residual[:i])
      res_right = np.array(residual[i:])
      # 分割点のインデックス      
      print('*****')
      print('index', i)
      index_plt.append(i)
      # 分割後の配列
      print('X_left:', X_left)
      print('X_right:', X_right)
      print('res_left:', res_left)
      print('res_right:', res_right)
      # 重み
      print('res_weight_left:', weight(res_left, lam))
      print('res_weight_right:', weight(res_right, lam))
      # 類似度
      print('similarity_left:', similarity(res_left, lam))
      print('similarity_right:', similarity(res_right, lam))      
      # 左葉類似度＋右葉類似度の合計
      print('similarity_total:', similarity(res_left, lam) + similarity(res_right, lam))
      similarity_plt.append(similarity(res_left, lam) + similarity(res_right, lam))     
      print('')

  # 1次元配列→2次元配列
  index_plt = np.array(index_plt)
  X_plt = index_plt[:, np.newaxis]
  # 分割点ごとの類似度を可視化
  plt.figure(figsize=(10, 4)) #プロットのサイズ指定  
  plt.plot(X_plt, similarity_plt)
  plt.xlabel('index')
  plt.ylabel('Similarity Score')
  plt.title('Similarity Score vs Split Point index')
  plt.grid()
  plt.show()

# %%
# 初期値の計算
pred0 = np.mean(y_train)
pred0

# %%
# 残差1=正解値-初期値
residual1 = y_train - pred0
residual1

# %%
# 残差1の深さ1の分割点
X_train = np.array([[10], [20], [30], [40], [50], [60], [70], [80]])
split(X_train, residual1)

# %%
# 左葉：残差1の深さ2の分割点
X_train1_L = np.array([[10], [20], [30]])
residual1_L = np.array([2, 1, 3])
split(X_train1_L, residual1_L)

# %%
# 右葉：残差1の深さ2の分割点
X_train1_R = np.array([[40], [50], [60], [70], [80]])
residual1_R = np.array([-3, -2, -3,  2,  0])
split(X_train1_R, residual1_R)

# %%
# 学習率×1回ブースティングした重み
weight1 = np.array([1.5, 1.5, 3, -2.6666666666666665, -2.6666666666666665, -2.6666666666666665, 1, 1])
0.8 * weight1

# %%
# 予測値1=初期値+学習率×重み1
pred1 = pred0 + 0.8 * weight1
pred1

# %%
# 予測値1のRMSE
print('RMSE train: %.5f' % (mean_squared_error(y_train, pred1) ** 0.5))

# %%
# 残差2=正解値-予測値1
residual2 = y_train - pred1
residual2

# %%
# 残差2の深さ1の分割点
X_train = np.array([[10], [20], [30], [40], [50], [60], [70], [80]])
split(X_train, residual2)

# %%
# 左葉：残差2の深さ2の分割点
X_train2_L = np.array([[10], [20], [30]])
residual2_L = np.array([ 0.8, -0.2, 0.6])
split(X_train2_L, residual2_L)

# %%
# 右葉：残差2の深さ2の分割点
X_train2_R = np.array([[40], [50], [60], [70], [80]])
residual2_R  = np.array([-0.86666667, 0.13333333, -0.86666667, 1.2, -0.8])
split(X_train2_R, residual2_R)

# %%
# 学習率×2回ブースティングした重み
weight2 = np.array([0.8, 0.2, 0.2, -0.5333333366666667, -0.5333333366666667, -0.5333333366666667, 0.19999999999999996, 0.19999999999999996])
0.8 * weight2

# %%
# 予測値2=予測値1+学習率×重み2
pred2 = pred1 + 0.8 * weight2
pred2

# %%
# 予測値2のRMSE
print('RMSE train: %.5f' % (mean_squared_error(y_train, pred2) ** 0.5))

# %%


# %%


# %% [markdown]
"""
# 枝刈り
"""

# %%
# ハイパーパラメータの設定
params2 = {
    'objective': 'reg:squarederror', # 損失関数
    'eval_metric': 'rmse', # 評価指標
    'max_depth': 2, # 深さの最大値
    'learning_rate': 0.8, # 学習率
    'base_score': 4, # 初期値
    'min_split_loss': 1.51, # 枝刈り
    'reg_lambda': 0, # L2正則化
    'reg_alpha': 0, # L1正則化
    'seed': 0
    }

# %%
# XGBoostの学習
model2 = xgb.train(params2,
                  xgb_train,
                  evals=[(xgb_train, 'train')],
                  num_boost_round=1)

# %%
# XGBoostの予測
model2.predict(xgb.DMatrix(X_train))

# %%
# 1本目の木の可視化
xgb.to_graphviz(model2, num_trees=0)

# %%


