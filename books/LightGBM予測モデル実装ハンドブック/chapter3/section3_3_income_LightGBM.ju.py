# %% [markdown]
"""
# LightGBMの学習→予測→評価
"""

# %%
# # Colabでバージョンを変更するとき、コメントアウトして実行してください
# !pip install pandas==1.5.3
# !pip install numpy==1.22.4
# !pip install matplotlib==3.7.1
# !pip install seaborn==0.12.2
# !pip install scikit-learn==1.2.2
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

# %%
# バージョンの確認
import matplotlib
import sklearn
import lightgbm as lgb
print(pd.__version__) 
print(np.__version__)
print(matplotlib.__version__)
print(sns.__version__)
print(sklearn.__version__) 
print(lgb.__version__)

# %%
# データセットの読み込み
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None)
df.columns =['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
#df.head()

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

#print(df.shape)
#df.head()

# %%
# 特徴量と目的変数の設定
X = df.drop(['income'], axis=1)
y = df['income']

# %%
# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=0)
print('X_trainの形状：', X_train.shape, ' y_trainの形状：', y_train.shape, ' X_testの形状：', X_test.shape, ' y_testの形状：', y_test.shape)

# %%
# カテゴリ変数のlabel encoding
from sklearn.preprocessing import LabelEncoder

cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender']

for c in cat_cols:
    le = LabelEncoder()
    le.fit(X_train[c])
    X_train[c] = le.transform(X_train[c])
    X_test[c] = le.transform(X_test[c])

#X_train.info()

# %%
# カテゴリ変数のデータ型をcategory型に変換
for c in cat_cols:
    X_train[c] = X_train[c].astype('category')
    X_test[c] = X_test[c].astype('category')

X_train.info()

# %%
# ハイパーパラメータの設定
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train)

params = {
    'objective': 'binary',
    'num_leaves': 5,
    'seed': 0,
    'verbose': -1,
}

# %%
# モデルの学習
model = lgb.train(params,
                  lgb_train,
                  num_boost_round=500,
                  valid_sets=[lgb_train],
                  valid_names=['train'],
                  callbacks=[lgb.log_evaluation(100)])

# %%
# テストデータの予測と評価
y_test_pred_proba = model.predict(X_test) # ラベル1の確率
print('ラベル1の確率：', y_test_pred_proba)
y_test_pred = np.round(y_test_pred_proba) # 確率をラベル0 or 1に変換
print('予測ラベル値：', y_test_pred)

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
# 特徴量重要度の可視化
importances = model.feature_importance(importance_type='gain') # 特徴量重要度
indices = np.argsort(importances)[::-1] # 特徴量重要度を降順にソート

plt.figure(figsize=(8, 4)) #プロットのサイズ指定
plt.title("Feature Importance") # プロットのタイトルを作成
plt.bar(range(len(indices)), importances[indices]) # 棒グラフを追加
plt.xticks(range(len(indices)), X.columns[indices], rotation=90) # X軸に特徴量の名前を追加
plt.show() # プロットを表示

# %%
# 1本目の木の可視化
lgb.plot_tree(model, tree_index=0, figsize=(20, 20))

# %%


# %%


# %% [markdown]
"""
# SHAP
"""

# %%
# 最後から3件目の予測値
y_test_pred_proba[-3]

# %%
# ライブラリshapのインストール
! pip install shap

# %%
import shap
shap.__version__

# %%
# explainerの作成
import shap

explainer = shap.TreeExplainer(
    model = model,
    feature_pertubation = 'tree_path_dependent')

# %%
# SHAP値の計算
shap_values = explainer(X_test)

# %%
# 全件レコードの期待値
explainer.expected_value

# %%
# 最後から3件目のSHAP値
shap_values[-3]

# %%
# shap_valuesのラベル1のへの絞り込み
shap_values.values = shap_values.values[:, :, 1] # 貢献度
shap_values.base_values = explainer.expected_value[1] #期待値

# %%
# 最後から3件目のSHAP値（ラベル1）
shap_values[-3]

# %%
# 最後から3件目の貢献度
shap_values.values[-3]

# %%
# 最後から3件目の貢献度合計
shap_values.values[-3].sum()

# %%
# 期待値＋最後から3件目の貢献度合計
shap_values[-3].base_values + shap_values.values[-3].sum()

# %%
# 最後から3件目のラベル1の確率

# SHAP値合計をlogitに設定
logit = shap_values[-3].base_values + shap_values.values[-3].sum()

# シグモイド関数でlogitから確率に変換
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

sigmoid(logit)

# %%
# 最後から3件目の予測値
y_test_pred_proba[-3]

# %%
# 最後から3件目のSHAP値の可視化
shap.plots.waterfall(shap_values[-3])

# %%
# 重要度の可視化
shap.plots.bar(shap_values=shap_values)

# %%


