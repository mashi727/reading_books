# %% [markdown]
"""
# 前処理
"""

# %%
# # Colabでバージョンを変更するとき、コメントアウトして実行してください
# !pip install pandas==1.5.3
# !pip install numpy==1.22.4
# !pip install matplotlib==3.7.1
# !pip install seaborn==0.12.2
# !pip install scikit-learn==1.2.2
# !pip install xgboost==1.7.5
# !pip install lightgbm==3.3.5

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
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb

# %%
# バージョンの確認
import matplotlib
import sklearn
print(pd.__version__) 
print(np.__version__)
print(matplotlib.__version__)
print(sns.__version__)
print(sklearn.__version__)
print(xgb.__version__) 
print(lgb.__version__)

# %%
# Google Driveにマウント
from google.colab import drive
drive.mount('/content/drive')

# %%
# ディレクトリ移動
# %cd '/content/drive/MyDrive/Colab Notebooks/lightgbm_sample/chapter5'

# %%
!ls

# %%
# データセットの読み込み
df = pd.read_csv('exoTrain.csv')
df.head()

# %%
# データ形状
df.shape

# %%
# データ型
df.info()

# %%
df.describe()

# %%
# 欠損値の有無
df.isnull().sum().sum()

# %%
# 正解ラベルの件数内訳
df['LABEL'].value_counts()

# %%
# 前処理

# 正解ラベルの置換
df['LABEL'] = df['LABEL'].replace(1, 0)
df['LABEL'] = df['LABEL'].replace(2, 1)

# 置換後の正解ラベルの件数内訳
df['LABEL'].value_counts()

# %%
# 特徴量と目的変数の設定
X_train = df.drop(['LABEL'], axis=1)
y_train = df['LABEL']

# %%


# %% [markdown]
"""
# 学習時間比較
"""

# %%
# 実行時間の表示
import time
start = time.time()

df.info()

end = time.time()
elapsed = end - start

print('\nRun Time: ' + str(elapsed) + ' seconds.')

# %%
# DecisionTreeClassifierの実行時間

# DecisionTreeClassifierの学習
start = time.time()

model_tree = DecisionTreeClassifier(max_depth=2, random_state=0)
model_tree.fit(X_train, y_train)
y_train_pred = model_tree.predict(X_train)

end = time.time()

elapsed = end - start
print('Run Time: ' + str(elapsed) + ' seconds')

# 混同行列
cm = confusion_matrix(y_train, y_train_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('pred')
plt.ylabel('label')

# %%
# GradientBoostingClassifierの実行時間

# GradientBoostingClassifierの学習
start = time.time()

model_gbdt = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2, criterion='squared_error', random_state=0)
model_gbdt.fit(X_train, y_train)
y_train_pred = model_gbdt.predict(X_train)

end = time.time()

elapsed = end - start
print('Run Time: ' + str(elapsed) + ' seconds')

# 混同行列
cm = confusion_matrix(y_train, y_train_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('pred')
plt.ylabel('label')

# %%
# XGBoostハイパーパラメータの設定

xgb_train = xgb.DMatrix(X_train, label=y_train)

params_xgb = {
    'objective': 'reg:logistic', # 損失関数
    'max_depth': 2, # 深さの最大値
    'learning_rate': 0.1, # 学習率
    'base_score': 0.5, # 初期値
    'min_child_weight': 1e-3, # 葉の2階微分の最小値
    'min_split_loss': 0, # 枝刈り
    'reg_alpha': 0, # L1正則化
    'reg_lambda': 0, # L2正則化
    'tree_method': 'auto', # 計算方法
    'nthread': 1, # スレッド数
    'seed': 0, # 乱数
}

# %%
# XGBoostの実行時間

# XGBoostの学習
start = time.time()

model_xgb = xgb.train(params_xgb,
                      xgb_train,
                      num_boost_round=100)

y_train_pred_proba= model_xgb.predict(xgb.DMatrix(X_train))
y_train_pred = np.round(y_train_pred_proba)

end = time.time()

elapsed = end - start
print('Run Time: ' + str(elapsed) + ' seconds')

# 混同行列
cm = confusion_matrix(y_train, y_train_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('pred')
plt.ylabel('label')

# %%
# LightGBMハイパーパラメータの設定
lgb_train = lgb.Dataset(X_train, y_train)

params_lgb = {
    'objective': 'binary', # 損失関数
    'max_depth': 2, # 深さの最大値
    'learning_rate': 0.1, # 学習率
    'min_data_in_leaf': 20, # 葉の最小のレコード数
    'min_sum_hessian_in_leaf': 1e-3, # 葉の2階微分の最小値
    'max_bin': 255, # ヒストグラムの最大のbin数 
    'min_data_in_bin': 3, # binの最小のレコード数
    'min_gain_to_split': 0, # 枝刈り
    'lambda_l1': 0, # L1正則化
    'lambda_l2': 0, # L2正則化
    'num_threads': 1, # スレッド数
    'seed': 0, # 乱数
    'verbose': -1, # ログ表示
}

# %%
# LightGBMの実行時間

# LightGBMの学習
start = time.time()

model_lgb = lgb.train(params_lgb,
                      lgb_train,
                      num_boost_round=100)

y_train_pred_proba = model_lgb.predict(X_train)
y_train_pred = np.round(y_train_pred_proba)

end = time.time()

elapsed = end - start
print('Run Time: ' + str(elapsed) + ' seconds')

# 混同行列
cm = confusion_matrix(y_train, y_train_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('pred')
plt.ylabel('label')

# %%


# %% [markdown]
"""
# scikit-learn API（本には記載なし）
"""

# %%
# XGBoostの実行時間（scikit-learn API）
from xgboost import XGBClassifier

# XGBoost分類の学習
start = time.time()

model_xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=2, reg_lambda=0, min_child_weight=1e-3, nthread=1, random_state=0)
model_xgb.fit(X_train, y_train)
y_train_pred = model_xgb.predict(X_train)

end = time.time()

elapsed = end - start
print('Run Time: ' + str(elapsed) + ' seconds')

# 混同行列
cm = confusion_matrix(y_train, y_train_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('pred')
plt.ylabel('label')

# %%
# LightGBMの実行時間（scikit-learn API）
from lightgbm import LGBMClassifier

# LigitGBM分類の学習
start = time.time()

model_lgb = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=2, num_threads=1, random_state=0)
model_lgb.fit(X_train, y_train)
y_train_pred = model_lgb.predict(X_train)

end = time.time()

elapsed = end - start
print('Run Time: ' + str(elapsed) + ' seconds')

# 混同行列
cm = confusion_matrix(y_train, y_train_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('pred')
plt.ylabel('label')

# %%


