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
# !pip install scikit-learn==1.2.2

# %%
# ライブラリのインポート
# %matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

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
# データセットの読み込み
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None)
df.columns =['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
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

# %% [markdown]
"""
# 数値変数EDA
"""

# %%
# 数値のヒストグラム
plt.rcParams['figure.figsize'] = (10, 6)
df.hist(bins=20)
plt.tight_layout() 
plt.show()

# %% [markdown]
"""
# カテゴリ変数EDA
"""

# %%
# カテゴリ変数の統計情報
df.describe(exclude='number').T

# %%
# カテゴリ変数のリスト表示
cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']

for col in cat_cols:
    print('%s: %s' % (col, list(df[col].unique())))

# %%
# カテゴリ変数の棒グラフ
cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']

plt.rcParams['figure.figsize'] = (20, 20)

for i, name in enumerate(cat_cols):
  ax = plt.subplot(5, 2, i+1)
  df[name].value_counts().plot(kind='bar', ax=ax)

plt.tight_layout() 
plt.show()

# %%


# %% [markdown]
"""
# 前処理
"""

# %%
# 半角スペースの削除
cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']

for s in cat_cols:
  df[s] =df[s].str.replace(' ', '')

# %%
# カテゴリ変数のリスト表示
for col in cat_cols:
    print('%s: %s' % (col, list(df[col].unique())))

# %%
# レコードの絞り込み
df = df[df['native-country'].isin(['United-States'])]
df = df.drop(['native-country'], axis=1)
df.reset_index(inplace=True, drop=True)
df.shape

# %%
# 前処理後のincome件数内訳
df['income'].value_counts()

# %%
# 前処理後のincome件数可視化
plt.figure(figsize=(6, 3))
sns.countplot(x='income', data=df)

# %%
# 正解ラベルの作成
df['income'] = df['income'].replace('<=50K', 0)
df['income'] = df['income'].replace('>50K', 1)
# df.head()

# %%
# 前処理後のデータ
print(df.shape)
df.head()

# %%


# %% [markdown]
"""
# 混同行列と正解率の検証
"""

# %%
# 特徴量と目的変数の設定
X = df.drop(['income'], axis=1)
y = df['income']

# %%
# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=0)
print('X_trainの形状：', X_train.shape, ' y_trainの形状：', y_train.shape, ' X_testの形状：', X_test.shape, ' y_testの形状：', y_test.shape)

# %%
# 学習データとテストデータのラベル件数内訳
print(y_train.value_counts())
print(y_test.value_counts())

# %%
# 正解ラベル
y_test.values # numpy化

# %%
# 予測ラベル0の作成
y_test_zeros = np.zeros(5834) # テストデータレコード数の0を作成
y_test_zeros

# %%
# 予測ラベル0の混同行列
cm = confusion_matrix(y_test, y_test_zeros)
plt.figure(figsize = (6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('pred')
plt.ylabel('label')

# %%
# 予測ラベル0の評価指標
ac_score = accuracy_score(y_test, y_test_zeros)
pr_score = precision_score(y_test, y_test_zeros)
rc_score = recall_score(y_test, y_test_zeros)
f1 = f1_score(y_test, y_test_zeros)

print('accuracy = %.2f' % (ac_score))
print('precision = %.2f' % (pr_score))
print('recall = %.2f' % (rc_score))
print('F1-score = %.2f' % (f1))

# %%
# 予測ラベル1の作成
y_test_ones = np.ones(5834) # テストデータレコード数の1を作成
y_test_ones

# %%
# 予測ラベル1の混同行列
cm = confusion_matrix(y_test, y_test_ones)
plt.figure(figsize = (6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('pred')
plt.ylabel('label')

# %%
# 予測ラベル1の評価指標
ac_score = accuracy_score(y_test, y_test_ones)
pr_score = precision_score(y_test, y_test_ones)
rc_score = recall_score(y_test, y_test_ones)
f1 = f1_score(y_test, y_test_ones)

print('accuracy = %.2f' % (ac_score))
print('precision = %.2f' % (pr_score))
print('recall = %.2f' % (rc_score))
print('F1-score = %.2f' % (f1))

# %%


