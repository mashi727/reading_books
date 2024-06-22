# %% [markdown]
"""
# DecisionTreeRegressorの可視化
"""

# %%
# # Colabでバージョンを変更するとき、コメントアウトして実行してください
# !pip install numpy==1.22.4
# !pip install matplotlib==3.7.1
# !pip install graphviz==0.20.1
# !pip install scikit-learn==1.2.2

# %%
# ライブラリのインポート
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

# %%
# バージョンの確認
import matplotlib
import sklearn
print(np.__version__)
print(matplotlib.__version__)
print(graphviz.__version__)
print(sklearn.__version__)

# %%
# 特徴量と目的変数の設定
X_train = np.array([[10], [20], [30], [40], [50], [60], [70], [80]])
y_train = np.array([6, 5, 7, 1, 2, 1, 6, 4])

# %%
# 回帰木の学習と予測
model = DecisionTreeRegressor(criterion='squared_error', max_depth=2, min_samples_leaf=1, ccp_alpha=0, random_state=0)
model.fit(X_train, y_train)
model.predict(X_train)

# %%
# ハイパーパラメータの表示
model.get_params()

# %%
# データと予測値の可視化
plt.figure(figsize=(8, 4)) #プロットのサイズ指定

# 学習データの最小値から最大値まで0.01刻みのX_pltを作成し、予測
X_plt = np.arange(X_train.min(), X_train.max(), 0.01)[:, np.newaxis]
y_pred = model.predict(X_plt)

# 学習データの散布図と予測値のプロット
plt.scatter(X_train, y_train, color='blue', label='data')
plt.plot(X_plt, y_pred, color='red', label='Decision tree')
plt.ylabel('y')
plt.xlabel('X')
plt.title('simple data')
plt.legend(loc='upper right')
plt.show()

# %%
# 木の可視化
dot_data = tree.export_graphviz(model, out_file=None, rounded=True, feature_names=['X'], filled=True)
graphviz.Source(dot_data, format='png')

# %%


# %% [markdown]
"""
# 予測値の検証
"""

# %%
# 分割点の計算
def loss(X_train, y_train):
  index =[]
  loss =[]
  # 表示のため、2次元配列のX_trainを1次元配列に変換
  X_train = X_train.flatten()
  # 分割点ごとの予測値とSSE,MSEを計算  
  for i in range(1, len(X_train)):
      X_left = np.array(X_train[:i])
      X_right = np.array(X_train[i:])
      y_left = np.array(y_train[:i])
      y_right = np.array(y_train[i:])
      # 分割点のインデックス
      print('*****')
      print('index', i)
      index.append(i)
      # 左右の分割
      print('X_left:', X_left)
      print('X_right:', X_right)
      print('y_left:', y_left)
      print('y_right:', y_right)
      # 予測値の計算
      print('y_pred_left:', np.mean(y_left))
      print('y_pred_right:', np.mean(y_right))
      # SSEの計算
      y_error_left = y_left - np.mean(y_left)
      y_error_right = y_right - np.mean(y_right)
      SSE = np.sum(y_error_left * y_error_left) + np.sum(y_error_right * y_error_right)
      print('SSE:', SSE)
      loss.append(SSE)
      # MSEの計算
      MSE_left = 1/len(y_left) * np.sum(y_error_left * y_error_left)
      MSE_right = 1/len(y_right) * np.sum(y_error_right * y_error_right)
      print('MSE_left:', MSE_left)
      print('MSE_right:', MSE_right)
      print('')

  # プロットのため、1次元配列のX_trainを2次元配列に変換
  index = np.array(index)
  X_plt = index[:, np.newaxis]
  # 分割点ごとのSSEを可視化
  plt.figure(figsize=(10, 4)) #プロットのサイズ指定
  plt.plot(X_plt, loss)
  plt.xlabel('index')
  plt.ylabel('SSE')
  plt.title('SSE vs Split Point index')
  plt.grid()
  plt.show()

# %%
# 全レコードの深さ1の分割点
X_train = np.array([[10], [20], [30], [40], [50], [60], [70], [80]])
y_train = np.array([6, 5, 7, 1, 2, 1, 6, 4])
loss(X_train, y_train)

# %%
# 左葉レコードの深さ2の分割点
X_train_L = np.array([[10], [20], [30]])
y_train_L = np.array([6, 5, 7])
loss(X_train_L, y_train_L)

# %%
# 右葉レコードの深さ2の分割点
X_train_R = np.array([[40], [50], [60], [70], [80]])
y_train_R = np.array([1, 2, 1, 6, 4])
loss(X_train_R, y_train_R)

# %%


# %%


