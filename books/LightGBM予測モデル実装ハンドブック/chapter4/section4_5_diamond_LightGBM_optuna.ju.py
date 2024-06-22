# %% [markdown]
"""
# 前処理＋特徴量エンジアリング
"""

# %%
# # Colabでバージョンを変更するとき、コメントアウトして実行してください
# !pip install pandas==1.5.3
# !pip install numpy==1.22.4
# !pip install matplotlib==3.7.1
# !pip install seaborn==0.12.2
# !pip install scikit-learn==1.2.2
# !pip install lightgbm==3.3.5
# !pip install optuna==3.1.1
# !pip install plotly==5.13.1

# %%
# ライブラリのインポート
# %matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

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
# データ読み込み、外れ値の除外

# データセットの読み込み
df = sns.load_dataset('diamonds')

# 外れ値除外の前処理
df = df.drop(df[(df['x'] == 0) | (df['y'] == 0)| (df['z'] == 0)].index, axis=0)
df = df.drop(df[(df['x'] >= 10) | (df['y'] >= 10) | (df['z'] >= 10)].index, axis=0)
df.reset_index(inplace=True, drop=True)
print(df.shape)
df.head()

# %%
# 特徴量と目的変数の設定
X = df.drop(['price'], axis=1)
y = df['price']

# %%
# 数値×数値の特徴量エンジニアリング

# 密度（重さ/体積）
X['density'] = X['carat'] / (X['x'] * X['y'] * X['z'])

# 差分
X['x-y'] = (X['x'] - X['y']).abs()
X['y-z'] = (X['y'] - X['z']).abs()
X['z-x'] = (X['x'] - X['y']).abs()

# 比率
X['x/y'] = X['x'] / X['y']
X['y/z'] = X['y'] / X['z']
X['z/x'] = X['z'] / X['x']

# 中央値との差分
X['x-median_x'] = (X['x'] - X['x'].median()).abs()
X['y-median_y'] = (X['y'] - X['y'].median()).abs()
X['z-median_z'] = (X['z'] - X['z'].median()).abs()

print('追加した特徴量')
display(X.head())

# %%
# カテゴリ変数×数値の特徴量エンジニアリング

# カテゴリ変数cutごとにcarat中央値を集計
X_carat_by_cut = X.groupby('cut')['carat'].agg('median').reset_index()
X_carat_by_cut.columns = ['cut', 'median_carat_by_cut']
print('cutごとにcarat中央値を集計')
display(X_carat_by_cut)
X = pd.merge(X, X_carat_by_cut, on='cut', how = 'left')
X['carat-median_carat_by_cut'] = (X['carat'] - X['median_carat_by_cut'])
X['carat/median_carat_by_cut'] = (X['carat'] / X['median_carat_by_cut'])

# カテゴリ変数colorごとにcarat中央値を集計
X_carat_by_color = X.groupby('color')['carat'].agg('median').reset_index()
X_carat_by_color.columns = ['color', 'median_carat_by_color']
print('colorごとにcarat中央値を集計')
display(X_carat_by_color)
X = pd.merge(X, X_carat_by_color, on='color', how = 'left')
X['carat-median_carat_by_color'] = (X['carat'] - X['median_carat_by_color'])
X['carat/median_carat_by_color'] = (X['carat'] / X['median_carat_by_color'])

# カテゴリ変数clarityごとにcarat中央値を集計
X_carat_by_clarity = X.groupby('clarity')['carat'].agg('median').reset_index()
X_carat_by_clarity.columns = ['clarity', 'median_carat_by_clarity']
print('clarityごとにcarat中央値を集計')
display(X_carat_by_clarity)
X = pd.merge(X, X_carat_by_clarity, on='clarity', how = 'left')
X['carat-median_carat_by_clarity'] = (X['carat'] - X['median_carat_by_clarity'])
X['carat/median_carat_by_clarity'] = (X['carat'] / X ['median_carat_by_clarity'])

display(X.head())

# %%
# カテゴリ変数×カテゴリ変数の特徴量エンジニアリング

# cut*colorの出現割合
X_tbl = pd.crosstab(X['cut'], X['color'], normalize='index')
X_tbl = X_tbl.reset_index()
print('cut*colorの出現割合')
display(X_tbl)
X_tbl = pd.melt(X_tbl, id_vars='cut', value_name='rate_cut*color')
X = pd.merge(X, X_tbl, on=['cut', 'color'], how='left' )

# color*clarityの出現割合
X_tbl = pd.crosstab(X['color'], X['clarity'], normalize='index')
X_tbl = X_tbl.reset_index()
print('color*clarityの出現割合')
display(X_tbl)
X_tbl = pd.melt(X_tbl, id_vars='color', value_name='rate_color*clarity')
X = pd.merge(X, X_tbl, on=['color', 'clarity'], how='left' )

# clarity*cutの出現割合
X_tbl = pd.crosstab(X['clarity'], X['cut'], normalize='index')
X_tbl = X_tbl.reset_index()
print('clarity*cutの出現割合')
display(X_tbl)
X_tbl = pd.melt(X_tbl, id_vars='clarity', value_name='rate_clarity*cut')
X = pd.merge(X, X_tbl, on=['clarity', 'cut'], how='left' )

display(X.head())

# %%
# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
print('X_trainの形状：', X_train.shape, ' y_trainの形状：', y_train.shape, ' X_testの形状：', X_test.shape, ' y_testの形状：', y_test.shape)

# %%
# カテゴリ変数のlabel encoding
from sklearn.preprocessing import LabelEncoder

cat_cols = ['cut', 'color', 'clarity']

for c in cat_cols:
    le = LabelEncoder()
    le.fit(X_train[c])
    X_train[c] = le.transform(X_train[c])
    X_test[c] = le.transform(X_test[c])

# %%
# カテゴリ変数のデータ型をcategoryに変換
cat_cols = ['cut', 'color', 'clarity']

for c in cat_cols:
    X_train[c] = X_train[c].astype('category')
    X_test[c] = X_test[c].astype('category')

X_train.info()

# %%
# 学習データの一部を検証データに分割
X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=1)
print('X_trの形状：', X_tr.shape, ' y_trの形状：', y_tr.shape, ' X_vaの形状：', X_va.shape, ' y_vaの形状：', y_va.shape)

# %% [markdown]
"""
# ハイパーパラメータ最適化
"""

# %%
# ライブラリoptunaのインストール
!pip install optuna

# %%
# optunaのバージョン確認
import optuna
optuna.__version__

# %%
# 固定値のハイパーパラメータ
params_base = {
    'objective': 'mae',
    'random_seed': 1234,
    'learning_rate': 0.02,
    'min_data_in_bin': 3,
    'bagging_freq': 1,
    'bagging_seed': 0,
    'verbose': -1,
}

# %%
# ハイパーパラメータ最適化

# ハイパーパラメータの探索範囲
def objective(trial):
  params_tuning = {
      'num_leaves': trial.suggest_int('num_leaves', 50, 200),
      'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2, 30),
      'max_bin': trial.suggest_int('max_bin', 200, 400),
      'bagging_fraction': trial.suggest_float('bagging_fraction', 0.8, 0.95),
      'feature_fraction': trial.suggest_float('feature_fraction', 0.35, 0.65),
      'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.01, 1, log=True),
      'lambda_l1': trial.suggest_float('lambda_l1', 0.01, 1, log=True),
      'lambda_l2': trial.suggest_float('lambda_l2', 0.01, 1, log=True),
  }
  
  # 探索用ハイパーパラメータの設定
  params_tuning.update(params_base)

  lgb_train = lgb.Dataset(X_tr, y_tr)
  lgb_eval = lgb.Dataset(X_va, y_va)
  
  # 探索用ハイパーパラメータで学習
  model = lgb.train(params_tuning,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=[lgb_train, lgb_eval],
                    valid_names=['train', 'valid'],
                    callbacks=[lgb.early_stopping(100),
                               lgb.log_evaluation(500)])
  
  y_va_pred = model.predict(X_va, num_iteration=model.best_iteration)
  score =  mean_absolute_error(y_va, y_va_pred)
  print('')

  return score

# %%
# ハイパーパラメータ最適化の実行
import optuna
study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0), direction='minimize')
study.optimize(objective, n_trials=200)

# %%
# 最適化の結果確認
trial = study.best_trial
print(f'trial {trial.number}')
print('MAE bset: %.2f'% trial.value)
display(trial.params)

# %%
# 最適化ハイパーパラメータの設定
params_best = trial.params
params_best.update(params_base)
display(params_best)

# %%
optuna.visualization.plot_param_importances(study).show()

# %%
optuna.visualization.plot_slice(study, params=['min_data_in_leaf', 'num_leaves']).show()

# %%
fig = optuna.visualization.plot_slice(study, params=['max_bin', 'min_gain_to_split'])
fig.show()

# %%
fig = optuna.visualization.plot_slice(study, params=['lambda_l1', 'lambda_l2'])
fig.show()

# %%
fig = optuna.visualization.plot_slice(study, params=['bagging_fraction', 'feature_fraction'])
fig.show()

# %% [markdown]
"""
# LightGBM（特徴量エンジニアリングあり×ハイパーパラメータ最適値）
"""

# %%
# 学習データの一部を検証データに分割
X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=0)
print('X_trの形状：', X_tr.shape, ' y_trの形状：', y_tr.shape, ' X_vaの形状：', X_va.shape, ' y_vaの形状：', y_va.shape)

# %%
# 最適化ハイパーパラメータを用いた学習
lgb_train = lgb.Dataset(X_tr, y_tr)
lgb_eval = lgb.Dataset(X_va, y_va, reference=lgb_train)

# 最適化ハイパーパラメータを読み込み
model = lgb.train(params_best,
                  lgb_train,
                  num_boost_round=10000,
                  valid_sets=[lgb_train, lgb_eval],
                  valid_names=['train', 'valid'],
                  callbacks=[lgb.early_stopping(100),
                             lgb.log_evaluation(500)])

y_va_pred = model.predict(X_va, num_iteration=model.best_iteration)
score = mean_absolute_error(y_va, y_va_pred)
print(f'MAE valid: {score:.2f}')

# %%
# 検証データの予測と評価
y_va_pred = model.predict(X_va, num_iteration=model.best_iteration) 
print('MAE valid: %.2f' % (mean_absolute_error(y_va, y_va_pred)))

# %%
# テストデータの予測と評価
y_test_pred = model.predict(X_test, num_iteration=model.best_iteration) 
print('MAE test: %.2f' % (mean_absolute_error(y_test, y_test_pred)))

# %%
# 特徴量重要度の可視化
importances = model.feature_importance(importance_type='gain') # 特徴量重要度
indices = np.argsort(importances)[::-1] # 特徴量重要度を降順にソート

plt.figure(figsize=(8, 4)) #プロットのサイズ指定
plt.title('Feature Importance') # プロットのタイトルを作成
plt.bar(range(len(indices)), importances[indices]) # 棒グラフを追加
plt.xticks(range(len(indices)), X.columns[indices], rotation=90) # X軸に特徴量の名前を追加
plt.show() # プロットを表示

# %%


# %% [markdown]
"""
# クロスバリデーション
"""

# %%
# 最適化ハイパーパラメータを用いたクロスバリデーション
from sklearn.model_selection import KFold

# 格納用データの作成
valid_scores = []
models = []
oof = np.zeros(len(X_train))

# KFoldを用いて学習データを5分割してモデルを作成
kf = KFold(n_splits=5, shuffle=True, random_state=0)
for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
    X_tr = X_train.iloc[tr_idx]
    X_va = X_train.iloc[va_idx]
    y_tr = y_train.iloc[tr_idx]
    y_va = y_train.iloc[va_idx]
    
    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_eval = lgb.Dataset(X_va, y_va, reference=lgb_train)

    # 最適化ハイパーパラメータを読み込み
    model = lgb.train(params_best,
                      lgb_train,
                      num_boost_round=10000,
                      valid_sets=[lgb_train, lgb_eval],
                      valid_names=['train', 'valid'],
                      callbacks=[lgb.early_stopping(100),
                      lgb.log_evaluation(500)])

    y_va_pred = model.predict(X_va, num_iteration=model.best_iteration)
    score = mean_absolute_error(y_va, y_va_pred)
    print(f'fold {fold+1} MAE valid: {score:.2f}')
    print('')

    # スコア、モデル、予測値の格納
    valid_scores.append(score)
    models.append(model)
    oof[va_idx] = y_va_pred

# クロスバリデーションの平均スコア
cv_score = np.mean(valid_scores)
print(f'CV score: {cv_score:.2f}')

# %%
# 検証データの誤差
valid_scores

# %%
# 検証データの誤差平均
cv_score

# %%
# 検証データの誤差平均
print('MAE CV: %.2f' % (
      mean_absolute_error(y_train, oof)))

# %%


