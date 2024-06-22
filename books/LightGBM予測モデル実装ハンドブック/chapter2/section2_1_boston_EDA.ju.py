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
print(pd.__version__) 
print(np.__version__)
print(matplotlib.__version__)
print(sns.__version__)

# %%
# データセットの読み込み
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
df.columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

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
# 1変数EDA
"""

# %%
# 住宅価格の統計情報
df['MEDV'].describe()

# %%
# 住宅価格のヒストグラム
df['MEDV'].hist(bins=30)

# %%


# %% [markdown]
"""
# 2変数EDA
"""

# %%
# 相関係数
plt.figure(figsize=(12, 10))
df_corr = df.corr()
sns.heatmap(df_corr, vmax=1, vmin=-1, center=0, annot=True, cmap = 'Blues')

# %%
# 散布図
num_cols = ['LSTAT', 'RM', 'MEDV']
sns.pairplot(df[num_cols], size=2.5)

# %%


