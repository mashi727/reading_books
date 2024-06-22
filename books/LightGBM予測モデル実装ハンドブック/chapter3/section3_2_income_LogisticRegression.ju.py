# %% [markdown]
"""
# ロジスティック回帰の学習→予測→評価
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
# 前処理

# 文字列の半角スペース削除
cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
for s in cat_cols:
  df[s] =df[s].str.replace(' ', '')

# United-Statesのレコードに絞り特徴量native-countryを削除
df = df[df['native-country'].isin(['United-States'])]
df = df.drop(['native-country'], axis=1)
df.reset_index(inplace=True, drop=True)

# 正解ラベルの数値への置換
df['income'] = df['income'].replace('<=50K', 0)
df['income'] = df['income'].replace('>50K', 1)

print(df.shape)
df.head()

# %%
# 特徴量と目的変数の設定
X = df.drop(['income'], axis=1)
y = df['income']

# %%
# カテゴリ変数
X.describe(exclude='number').T

# %%
# one-hot encoding
X = pd.concat([X, pd.get_dummies(X['workclass'], prefix='workclass', drop_first=True)], axis=1)
X = pd.concat([X, pd.get_dummies(X['education'], prefix='education', drop_first=True)], axis=1)
X = pd.concat([X, pd.get_dummies(X['marital-status'], prefix='marital-status', drop_first=True)], axis=1)
X = pd.concat([X, pd.get_dummies(X['occupation'], prefix='occupation', drop_first=True)], axis=1)
X = pd.concat([X, pd.get_dummies(X['relationship'], prefix='relationship', drop_first=True)], axis=1)
X = pd.concat([X, pd.get_dummies(X['race'], prefix='race', drop_first=True)], axis=1)
X = pd.concat([X, pd.get_dummies(X['gender'], prefix='gender', drop_first=True)], axis=1)
X = X.drop(['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender'], axis=1)
print(X.shape)
#X.head()

# %%
# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=0)
print('X_trainの形状：', X_train.shape, ' y_trainの形状：', y_train.shape, ' X_testの形状：', X_test.shape, ' y_testの形状：', y_test.shape)

# %%
# 学習データとテストデータのラベル件数内訳
print(y_train.value_counts())
print(y_test.value_counts())

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
# モデルの学習
from sklearn.linear_model import LogisticRegression

# ロジスティック回帰モデル
model = LogisticRegression(max_iter=100, multi_class = 'ovr', solver='liblinear', C=0.1, penalty='l1', random_state=0)
model.fit(X_train, y_train)
model.get_params()

# %%
# 予測の確率のリスト
model.predict_proba(X_test)

# %%
# 予測のラベルのリスト
model.predict(X_test)

# %%
# 正解ラベルのリスト
y_test.values # pandasをnumpyに変換

# %%
# テストデータの予測と評価
y_test_pred = model.predict(X_test)
ac_score = accuracy_score(y_test, y_test_pred)
print('accuracy = %.2f' % (ac_score))

f1 = f1_score(y_test, y_test_pred)
print('F1-score = %.2f' % (f1))

# %%
# 混同行列
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize = (6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('pred')
plt.ylabel('label')

# %%


# %% [markdown]
"""
# 予測値の解釈
"""

# %%
# パラメータ
print('回帰係数 w = [w1, w2, … , w59]:', model.coef_[0])
print('')
print('定数項 w0:', model.intercept_)

# %%
# 特徴量の列テキスト表示
X.columns

# %%
# 回帰係数の可視化
importances = model.coef_[0] # 回帰係数
indices = np.argsort(importances)[::-1] # 回帰係数を降順にソート

plt.figure(figsize=(32, 8)) #プロットのサイズ指定
plt.title('Regression coefficient') # プロットのタイトルを作成
plt.bar(range(len(indices)), importances[indices]) # 棒グラフを追加
plt.xticks(range(len(indices)), X.columns[indices], rotation=90) # X軸に特徴量の名前を追加

plt.show() # プロットを表示

# %%
# 回帰係数（上位30件）の可視化
importances = model.coef_[0] # 回帰係数
indices = np.argsort(importances)[::-1][:30] # 回帰係数を降順にソート

plt.figure(figsize=(10, 4)) #プロットのサイズ指定
plt.title('Regression coefficient') # プロットのタイトルを作成
plt.bar(range(len(indices)), importances[indices]) # 棒グラフを追加
plt.xticks(range(len(indices)), X.columns[indices], rotation=90) # X軸に特徴量の名前を追加

plt.show() # プロットを表示

# %%
# 回帰係数（下位30件）の可視化
importances = model.coef_[0] # 回帰係数
indices = np.argsort(importances)[::-1][-30:] # 回帰係数を降順にソート

plt.figure(figsize=(10, 4)) #プロットのサイズ指定
plt.title('Regression coefficient') # プロットのタイトルを作成
plt.bar(range(len(indices)), importances[indices]) # 棒グラフを追加
plt.xticks(range(len(indices)), X.columns[indices], rotation=90) # X軸に特徴量の名前を追加

plt.show() # プロットを表示

# %%
# 最後から3件目のクラス0とクラス1の確率
model.predict_proba(X_test)[-3]

# %%
# 最後から3件目の特徴量
print('最後から3件目の特徴量 X = [x1, x2, … , x59]:', X_test.values[-3]) # pandasをnumpyに変換

# %%
# 最後から3件目 logit = w × X + w0 
logit = sum(np.multiply(model.coef_[0] , X_test.values[-3])) + model.intercept_
logit

# %%
# シグモイド関数でlogitから確率に変換
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

sigmoid(logit)

# %%


