# %% [markdown]
"""
# データ概要
"""

# %%
# # Colabでバージョンを変更するとき、コメントアウトして実行してください
# !pip install pandas==1.5.3
# !pip install numpy==1.22.4
# !pip install matplotlib==3.7.1
# !pip install seaborn==0.12.2

# %%
# ライブラリのインポート
# %matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# バージョンの確認
import matplotlib
import sklearn
print(pd.__version__) 
print(np.__version__)
print(matplotlib.__version__)
print(sns.__version__)

# %%
# データセットの読み込み
df = sns.load_dataset('diamonds')
df.head()

# %%
# # ローカルファイルアップロード
# from google.colab import files
# uploaded = files.upload()

# %%
# # データセットの読み込み
# df = pd.read_csv('diamonds.csv', index_col=0)
# df.head()

# %%
# データ形状
df.shape

# %%
# 欠損値の有無
df.isnull().sum()

# %%
# データ型
df.info()

# %%
# 数値の統計情報
df.describe().T

# %%


# %% [markdown]
"""
# 1数値変数EDA
"""

# %%
# 数値のヒストグラム
plt.rcParams['figure.figsize'] = (10, 6)
df.hist(bins=20)
plt.tight_layout() 
plt.show()

# %%
# ダイヤモンド価格の統計情報
df['price'].describe()

# %%
# ダイヤモンド価格のヒストグラム
plt.figure(figsize=(6, 4))
df['price'].hist(bins=20)

# %% [markdown]
"""
#2数値変数EDA
"""

# %%
# 相関係数
plt.figure(figsize=(8, 6))
df_corr = df.corr()
sns.heatmap(df_corr, vmax=1, vmin=-1, center=0, annot=True, cmap = 'Blues')

# %%
# 数値×数値の散布図
num_cols = ['carat', 'depth', 'table', 'x', 'y', 'z', 'price']
sns.pairplot(df[num_cols], size=2.5)

# %%


# %% [markdown]
"""
# カテゴリ変数EDA
"""

# %%
# カテゴリ変数の統計情報
df.describe(exclude='number').T

# %%
# カテゴリ変数のリスト
cat_cols = ['cut', 'color', 'clarity']

for col in cat_cols:
    print('%s: %s' % (col, list(df[col].unique())))

# %%
# カテゴリ変数のラベル内訳
plt.rcParams['figure.figsize'] = (10, 6)

for i, name in enumerate(cat_cols):
  ax = plt.subplot(2, 2, i+1)
  df[name].value_counts().plot(kind='bar', ax=ax)

plt.tight_layout() 
plt.show()

# %%


# %%


# %% [markdown]
"""
# 前処理
"""

# %%
# x、y、zが0mmの外れ値
df[(df['x'] == 0) | (df['y'] == 0)| (df['z'] == 0)].shape

# %%
# x、y、zが0mmの外れ値のインデックス
df[(df['x'] == 0) | (df['y'] == 0)| (df['z'] == 0)].index

# %%
# x、y、zが0mmの外れ値の除外
df = df.drop(df[(df['x'] == 0) | (df['y'] == 0)| (df['z'] == 0)].index, axis=0)
df.shape

# %%
# x、y、zが10mm以上の外れ値
df[(df['x'] >= 10) | (df['y'] >= 10) | (df['z'] >= 10)]

# %%
# x、y、zが10mm以上の外れ値の除外
df = df.drop(df[(df['x'] >= 10) | (df['y'] >= 10) | (df['z'] >= 10)].index, axis=0)
df.reset_index(inplace=True, drop=True)
df.shape

# %%
# 外れ値除外後の統計値
df.describe().T

# %%
# ダイヤモンド価格の統計情報（外れ値を除外したあと）
df['price'].describe()

# %%
# ダイヤモンド価格のヒストグラム（外れ値除外後）
plt.figure(figsize=(6, 4))
df['price'].hist(bins=20)

# %%


# %%


