# %% [markdown]
"""
# LightGBM（特徴量：ALL）の学習→予測→評価
"""

# %%
# # Colabでバージョンを変更するとき、コメントアウトして実行してください
# !pip install pandas==1.5.3
# !pip install numpy==1.22.4
# !pip install matplotlib==3.7.1
# !pip install scikit-learn==1.2.2
# !pip install lightgbm==3.3.5
# !pip install shap==0.41.0

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
X = df.drop(['MEDV'], axis=1)
y = df['MEDV']

# %%
# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
print('X_trainの形状：', X_train.shape, ' y_trainの形状：', y_train.shape, ' X_testの形状：', X_test.shape, ' y_testの形状：', y_test.shape)

# %%
# ハイパーパラメータの設定
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train)

params = {'objective': 'mse',
          'num_leaves': 5,
          'seed': 0,
          'verbose': -1,
}

# %%
# モデルの学習
model = lgb.train(params,
                  lgb_train,
                  num_boost_round=50,
                  valid_sets=[lgb_train],
                  valid_names=['train'],
                  callbacks=[lgb.log_evaluation(10)])

# %%
# 学習データの予測と評価
y_train_pred = model.predict(X_train) 
print('MSE train: %.2f' % (mean_squared_error(y_train, y_train_pred)))
print('RMSE train: %.2f' % (mean_squared_error(y_train, y_train_pred) ** 0.5))

# %%
# テストデータの予測と評価
y_test_pred = model.predict(X_test)
print('RMSE test: %.2f' % (mean_squared_error(y_test, y_test_pred) ** 0.5))

# %%
# 特徴量の重要度の可視化
importances = model.feature_importance(importance_type='gain') # 特徴量の重要度
indices = np.argsort(importances)[::-1] # 特徴量の重要度を降順にソート

plt.figure(figsize=(8, 4)) #プロットのサイズ指定
plt.title('Feature Importance') # プロットのタイトルを作成
plt.bar(range(len(indices)), importances[indices]) # 棒グラフを追加
plt.xticks(range(len(indices)), X.columns[indices], rotation=90) # X軸に特徴量の名前を追加
plt.show() # プロットを表示

# %%
# 1本目の木の可視化
lgb.plot_tree(model, tree_index=0, figsize=(20, 20))

# %%
# 50本目の木の可視化
lgb.plot_tree(model, tree_index=-1, figsize=(20, 20))

# %%


# %%


# %% [markdown]
"""
# SHAP
"""

# %%
# ライブラリshapのインストール
!pip install shap

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
# # explainerの作成（interventional）
# import shap

# explainer = shap.TreeExplainer(
#     model = model,
#     data = X_test,
#     feature_pertubation = 'interventional')

# %%
# SHAP値の計算
shap_values = explainer(X_test)

# %%
# 全件レコードの期待値
explainer.expected_value

# %%
# 予測値のリスト
y_test_pred

# %%
# 15件目のSHAP値
shap_values[14]

# %%
# 15件目の貢献度
shap_values.values[14]

# %%
# 15件目の貢献度合計
shap_values.values[14].sum()

# %%
# 期待値＋15件目の貢献度合計
shap_values[14].base_values + shap_values.values[14].sum()

# %%
# 15件目の予測値
y_test_pred[14]

# %%
# 15件目のSHAP値の可視化
shap.plots.waterfall(shap_values[14])

# %%
# 11件目のSHAP値の可視化
shap.plots.waterfall(shap_values[10])

# %%
# 特徴量重要度の可視化
shap.plots.bar(shap_values=shap_values)

# %%


