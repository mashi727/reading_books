# %% [markdown]
"""
# LightGBM（特徴量：RM）の予測値の可視化
"""

# %%
# # Colabでバージョンを変更するとき、コメントアウトして実行してください
# !pip install pandas==1.5.3
# !pip install numpy==1.22.4
# !pip install matplotlib==3.7.1
# !pip install scikit-learn==1.2.2
# !pip install lightgbm==3.3.5

# %%
# ライブラリのインポート
# %matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# %%
# バージョンの確認
import matplotlib
import sklearn
import lightgbm as lgb
print(pd.__version__) 
print(np.__version__)
print(matplotlib.__version__)
print(sklearn.__version__) 
print(lgb.__version__)

# %%
# データセットの読み込み
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
df.columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
#df.head()

# %%
# 特徴量と目的変数の設定
X_train = df.loc[:99, ['RM']] # 特徴量に100件のRM（平均部屋数）を設定
y_train = df.loc[:99, 'MEDV'] # 正解値に100件のMEDV（住宅価格）を設定
print('X_train:', X_train[:3])
print('y_train:', y_train[:3])

# %%
# ハイパーパラメータの設定
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train)

params = {
    'objective': 'mse',
    'metric': 'mse',
    'learning_rate': 0.8,
    'max_depth': 1,
    'min_data_in_leaf': 1,
    'min_data_in_bin': 1,
    'max_bin': 100,
    'seed': 0,
    'verbose': -1,
}

# %%
# モデルの学習
model = lgb.train(params,
                  lgb_train,
                  num_boost_round=1,
                  valid_sets=[lgb_train],
                  valid_names=['train'])

# %%
# 学習データの予測と評価
y_train_pred = model.predict(X_train)
print('MSE train: %.2f' % (mean_squared_error(y_train, y_train_pred)))

# %%
# 予測値
model.predict(X_train)

# %%
# 木の可視化
lgb.plot_tree(model, tree_index=0, figsize=(10, 10))

# %%
# データと予測値の可視化
plt.figure(figsize=(8, 4)) #プロットのサイズ指定
X = X_train.values.flatten() # numpy配列に変換し、1次元配列に変換
y = y_train.values # numpy配列に変換

# Xの最小値から最大値まで0.01刻みのX_pltを作成し、2次元配列に変換
X_plt = np.arange(X.min(), X.max(), 0.01)[:, np.newaxis]
y_pred = model.predict(X_plt) # 住宅価格を予測

# 学習データ(平均部屋数と住宅価格)の散布図と予測値のプロット
plt.scatter(X, y, color='blue', label='data')
plt.plot(X_plt, y_pred, color='red', label='LightGBM')
plt.ylabel('Price in $1000s [MEDV]')
plt.xlabel('average number of rooms [RM]')
plt.title('Boston house-prices')
plt.legend(loc='upper right')
plt.show()

# %%


# %% [markdown]
"""
# 分割点と予測値の検証
"""

# %%
# 初期値
print('samples:', len(y)) # レコード数
pred0 = sum(y)/len(y) # 予測値（平均）
print('pred0:', pred0)

# %%
# 左葉のレコード
threshold = 6.793 # 左右に分割する分割点
X_left = X[X<=threshold] # 左葉の特徴量
y_left = y[X<=threshold] # 左葉の正解値
print('X_left:', X_left)
print('')
print('y_left:', y_left)

# %%
# 左葉の予測値
print('samples_left:', len(y_left)) # 左葉のレコード数
residual_left = y_left - pred0 # 残差
weight_left = sum(residual_left)/len(y_left) # 重み
print('weight_left:', weight_left)
y_pred_left = pred0 + 0.8 * weight_left # 左葉の予測値
print('y_pred_left:', y_pred_left)

# %%
# 右葉のレコード
X_right = X[threshold<X] # 右葉の特徴量
y_right = y[threshold<X] # 右葉の正解値
print('X_right:', X_right)
print('y_right:', y_right)

# %%
# 右葉の予測値
print('samples_right:', len(y_right)) # 右葉のレコード数
residual_right = y_right - pred0 # 残差
weight_right = sum(residual_right)/len(y_right) # 重み
print('weight_right:', weight_right)
y_pred_right = pred0 + 0.8 * weight_right # 右葉の予測値
print('y_pred_right:', y_pred_right)

# %%


# %%


# %%


# %% [markdown]
"""
# 特徴量RMのヒストグラム(5.4節のLightGBMの説明で掲載)
"""

# %%
# max_bin=20
X_train = df.loc[:99, ['RM']] # 特徴量に100件のRM（平均部屋数）を設定
X_train.hist(bins=20) # 100件レコードに対してbinが20のヒストグラム

# %%
# max_bin=10
X_train.hist(bins=10) # 100件レコードに対してbinが10のヒストグラム

# %%


