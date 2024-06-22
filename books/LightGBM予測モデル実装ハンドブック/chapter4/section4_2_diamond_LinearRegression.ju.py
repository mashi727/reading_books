# %% [markdown]
"""
# 線形回帰
"""

# %%
# # Colabでバージョンを変更するとき、コメントアウトして実行してください
# !pip install pandas==1.5.3
# !pip install numpy==1.22.4
# !pip install matplotlib==3.7.1
# !pip install seaborn==0.12.2
# !pip install scikit-learn==1.2.2

# %%
# ライブラリのインポート
# %matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# %%
# バージョンの確認
import matplotlib
import sklearn
print(pd.__version__) 
print(np.__version__)
print(matplotlib.__version__)
print(sns.__version__)
print(sklearn.__version__) 

# %%
# データ読み込み、外れ値の除外

# データセットの読み込み
df = sns.load_dataset('diamonds')

# 外れ値除外の前処理
df = df.drop(df[(df['x'] == 0) | (df['y'] == 0)| (df['z'] == 0)].index, axis=0)
df = df.drop(df[(df['x'] >= 10) | (df['y'] >= 10) | (df['z'] >= 10)].index, axis=0)
df.reset_index(inplace=True, drop=True)
print(df.shape)
df.head()

# %%
# 特徴量と目的変数の設定
X = df.drop(['price'], axis=1)
y = df['price']

# %%
# one-hot encoding
X = pd.concat([X, pd.get_dummies(X['cut'], prefix='cut', drop_first=True)], axis=1)
X = pd.concat([X, pd.get_dummies(X['color'], prefix='color', drop_first=True)], axis=1)
X = pd.concat([X, pd.get_dummies(X['clarity'], prefix='clarity', drop_first=True)], axis=1)
X = X.drop(['cut', 'color', 'clarity'], axis=1)
print(X.shape)
X.head()

# %%
# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
print('X_trainの形状：', X_train.shape, ' y_trainの形状：', y_train.shape, ' X_testの形状：', X_test.shape, ' y_testの形状：', y_test.shape)

# %%
# 数値の特徴量
X.columns[0:6]

# %%
# 特徴量の標準化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() # 変換器の作成
num_cols =  X.columns[0:6] # 数値型の特徴量を取得
scaler.fit(X_train[num_cols]) # 学習データでの標準化パラメータの計算
X_train[num_cols] = scaler.transform(X_train[num_cols]) # 学習データの変換
X_test[num_cols] = scaler.transform(X_test[num_cols]) # テストデータの変換

display(X_train.iloc[:2]) # 標準化された学習データの特徴量

# %%
# 学習データの一部を検証データに分割
X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=0)
print('X_trの形状：', X_tr.shape, ' y_trの形状：', y_tr.shape, ' X_vaの形状：', X_va.shape, ' y_vaの形状：', y_va.shape)

# %%
# モデルの学習
from sklearn.linear_model import LinearRegression

model = LinearRegression() # 線形回帰モデル
model.fit(X_tr, y_tr)
model.get_params()

# %%
# 検証データの予測と評価
y_va_pred = model.predict(X_va) 
print('MAE valid: %.2f' % (mean_absolute_error(y_va, y_va_pred)))

# %%
# テストデータの予測と評価
y_test_pred = model.predict(X_test) 
print('MAE test: %.2f' % (mean_absolute_error(y_test, y_test_pred)))

# %%
# テストデータの価格の統計情報
y_test.describe()

# %%
# テストデータの正解値と予測値の比較
print('正解値：', y_test[:5].values)
print('予測値：', y_test_pred[:5])
print('残差=正解値-予測値：', y_test[:5].values - y_test_pred[:5])

# %%
# 残差のプロット

# 残差の計算
residuals = y_test - y_test_pred
# 残差と予測値の散布図
plt.figure(figsize=(8, 4)) #プロットのサイズ指定
plt.scatter(y_test_pred, residuals, s=3)
plt.xlabel('Predicted values', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.title('Residuals vs Predicted values', fontsize=16)
plt.grid()
plt.show()

# %%


# %%
# パラメータ
print('回帰係数 w = [w1, w2, … , w23]:', model.coef_)
print('')
print('定数項 w0:', model.intercept_) 

# %%
# 特徴量の列テキスト表示
X.columns

# %%
# 回帰係数の可視化
importances = model.coef_ # 回帰係数
indices = np.argsort(importances)[::-1] # 回帰係数を降順にソート

plt.figure(figsize=(8, 4)) #プロットのサイズ指定
plt.title('Regression coefficient') # プロットのタイトルを作成
plt.bar(range(X.shape[1]), importances[indices]) # 棒グラフを追加
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90) # X軸に特徴量の名前を追加
plt.show() # プロットを表示

# %%


# %% [markdown]
"""
# Lasso回帰
"""

# %%
# ハイパーパラメータalphaとMAEの可視化

# ハイパーパラメータalphaとMAEの計算
from sklearn.linear_model import Lasso

params = np.arange(1, 10)
mae_metrics = []

for param in params:
  model_l1 = Lasso(alpha = param)
  model_l1.fit(X_tr, y_tr)
  y_va_pred = model_l1.predict(X_va) 
  mae_metric = mean_absolute_error(y_va, y_va_pred)
  mae_metrics.append(mae_metric)

# ハイパーパラメータalphaとMAEのプロット
plt.figure(figsize=(8, 4)) #プロットのサイズ指定
plt.plot(params, mae_metrics)
plt.xlabel('alpha', fontsize=14)
plt.ylabel('MAE', fontsize=14)
plt.title('MAE vs alpha', fontsize=16)
plt.grid()
plt.show()  

# %%


# %%
# モデルの学習
from sklearn.linear_model import Lasso

model_l1 = Lasso(alpha=6.0) # Lasso回帰
model_l1.fit(X_tr, y_tr)
model_l1.get_params()

# %%
# 検証データの予測と評価
y_va_pred = model_l1.predict(X_va) 
print('MAE valid: %.2f' % (mean_absolute_error(y_va, y_va_pred)))

# %%
# テストデータの予測と評価
y_test_pred = model_l1.predict(X_test) 
print('MAE test: %.2f' % (mean_absolute_error(y_test, y_test_pred)))

# %%
# パラメータ
print('回帰係数 w = [w1, w2 , … , w23]:', model_l1.coef_)
print('')
print('定数項 w0:', model_l1.intercept_) 

# %%
# 回帰係数の可視化
importances = model_l1.coef_ # 回帰係数
indices = np.argsort(importances)[::-1] # 回帰係数を降順にソート

plt.figure(figsize=(8, 4)) #プロットのサイズ指定
plt.title('Regression coefficient') # プロットのタイトルを作成
plt.bar(range(X.shape[1]), importances[indices]) # 棒グラフを追加
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90) # X軸に特徴量の名前を追加
plt.show() # プロットを表示

# %%


