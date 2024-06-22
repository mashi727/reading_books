# %% [markdown]
"""
# GradientBoostingRegressorの可視化
"""

# %%
# # Colabでバージョンを変更するとき、コメントアウトして実行してください
# !pip install numpy==1.22.4
# !pip install matplotlib==3.7.1
# !pip install graphviz==0.20.1
# !pip install scikit-learn==1.2.2

# %%
# ライブラリのインポート
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import tree

# %%
# バージョンの確認
import matplotlib
import sklearn
print(np.__version__)
print(matplotlib.__version__)
print(graphviz.__version__)
print(sklearn.__version__)

# %%
# 特徴量と目的変数の設定
X_train = np.array([[10], [20], [30], [40], [50], [60], [70], [80]])
y_train = np.array([6, 5, 7, 1, 2, 1, 6, 4])

# %%
# 勾配ブースティング回帰の学習と予測
model = GradientBoostingRegressor(n_estimators=2, learning_rate=0.8, criterion='squared_error', loss ='squared_error', max_depth=2, min_samples_leaf=1, ccp_alpha=0, random_state=0)
model.fit(X_train, y_train)
model.predict(X_train)

# %%
# ハイパーパラメータの表示
model.get_params()

# %%
# データと予測値の可視化
plt.figure(figsize=(8, 4)) #プロットのサイズ指定

# 学習データの最小値から最大値まで0.01刻みのX_pltを作成し、予測
X_plt = np.arange(X_train.min(), X_train.max(), 0.01)[:, np.newaxis]
y_pred = model.predict(X_plt)

# 学習データの散布図と予測値のプロット
plt.scatter(X_train, y_train, color='blue', label='data')
plt.plot(X_plt, y_pred, color='red', label='GradientBoostingRegressor')
plt.ylabel('y')
plt.xlabel('X')
plt.title('simple data')
plt.legend(loc='upper right')
plt.show()

# %%
# ブースティング1回目の木の可視化
dot_data = tree.export_graphviz(model.estimators_[0, 0], out_file=None, rounded=True, feature_names=['X'], filled=True)
graphviz.Source(dot_data, format='png')

# %%
# ブースティング2回目の木の可視化
dot_data = tree.export_graphviz(model.estimators_[1, 0], out_file=None, rounded=True, feature_names=['X'], filled=True)
graphviz.Source(dot_data, format='png')

# %%


# %%


# %% [markdown]
"""
# 予測値の検証
"""

# %%
# 初期値の計算
pred0 = np.mean(y_train)
print(pred0)

# %%
# 残差1=正解値-初期値
residual1 = y_train - pred0
print(residual1)

# %%
# 重み1の計算

# 特徴量と残差1で回帰木1の学習
model_tree = DecisionTreeRegressor(criterion='squared_error', max_depth=2, min_samples_leaf=1, ccp_alpha=0, random_state=0)
model_tree.fit(X_train, residual1)

# 重み1の予測値
weight1 = model_tree.predict(X_train)
print(weight1)

# %%
# 1本目の回帰木の可視化
dot_data = tree.export_graphviz(model_tree, out_file=None, rounded=True, feature_names=['X'], filled=True)
graphviz.Source(dot_data, format='png')

# %%
# 予測値1=初期値+学習率×重み1
pred1 = pred0 + 0.8 * weight1
print(pred1)

# %%
# 残差2=正解値-予測値1
residual2 = y_train - pred1
print(residual2)

# %%
# 重み2の計算

# 特徴量と残差2で回帰木2の学習
model_tree.fit(X_train, residual2)

# 重み2の予測値
weight2 = model_tree.predict(X_train)
print(weight2)

# %%
# 2本目の回帰木の可視化
dot_data = tree.export_graphviz(model_tree, out_file=None, rounded=True, feature_names=['X'], filled=True)
graphviz.Source(dot_data, format='png')

# %%
# 予測値2=予測値1+学習率×重み2
pred2 = pred1 + 0.8 * weight2
print(pred2)

# %%


# %%


