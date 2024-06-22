# %% [markdown]
"""
# 重回帰（特徴量：ALL）の学習→予測→評価

"""

# %%
# # Colabでバージョンを変更するとき、コメントアウトして実行してください
# !pip install pandas==1.5.3
# !pip install numpy==1.22.4
# !pip install matplotlib==3.7.1
# !pip install scikit-learn==1.2.2

# %%
# ライブラリのインポート
# %matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# %%
# バージョンの確認
import matplotlib
import sklearn
print(pd.__version__) 
print(np.__version__)
print(matplotlib.__version__)
print(sklearn.__version__) 

# %%
# データセットの読み込み
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
df.columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

# %%
# 特徴量と目的変数の設定
X = df.drop(['MEDV'], axis=1)
y = df['MEDV']
X.head()

# %%
# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
print('X_trainの形状：', X_train.shape, ' y_trainの形状：', y_train.shape, ' X_testの形状：', X_test.shape, ' y_testの形状：', y_test.shape)

# %%
# 特徴量の標準化
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() # 変換器の作成
num_cols =  X.columns[0:13] # 全て数値型の特徴量なので全て取得
scaler.fit(X_train[num_cols]) # 学習データでの標準化パラメータの計算
X_train[num_cols] = scaler.transform(X_train[num_cols]) # 学習データの変換
X_test[num_cols] = scaler.transform(X_test[num_cols]) # テストデータの変換

display(X_train.iloc[:2]) # 標準化された学習データの特徴量

# %%
# モデルの学習
from sklearn.linear_model import LinearRegression

model = LinearRegression() # 線形回帰モデル
model.fit(X_train, y_train)
model.get_params()

# %%
# テストデータの予測と評価
y_test_pred = model.predict(X_test)
print('RMSE test: %.2f' % (mean_squared_error(y_test, y_test_pred) ** 0.5))

# %%
# テストデータの目的関数の統計情報
y_test.describe()

# %%


# %% [markdown]
"""
# 予測値の解釈
"""

# %%
# パラメータ
print('回帰係数 w = [w1, w2, … , w13]:', model.coef_)
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
# 予測値のリスト
y_test_pred

# %%
# 15件目の予測値
y_test_pred[14]

# %%
# 15件目の特徴量
print('15件目の特徴量 X = [x1, x2, … , x13]:', X_test.values[14]) # pandasをnumpyに変換

# %%
# 15件目予測値の検証
# y = w * X + w0
np.sum(model.coef_ * X_test.values[14]) + model.intercept_

# %%
  

# %%


