# %% [markdown]
"""
# LightGBM（特徴量エンジニアリングあり×ハイパーパラメータ初期値）
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
# データ型
df.info()

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
display(X[['density', 'x/y', 'y/z', 'z/x', 'x-y', 'y-z', 'z-x', 'x-median_x', 'y-median_y', 'z-median_z']].head())

# %%
# カテゴリ変数cutで集計したcarat中央値の特徴量追加

# カテゴリ変数cutごとにcarat中央値を集計
X_carat_by_cut = X.groupby('cut')['carat'].agg('median').reset_index()
X_carat_by_cut.columns = ['cut', 'median_carat_by_cut']
print('cutごとのcarat中央値')
display(X_carat_by_cut)

# 集計した特徴量の追加
X = pd.merge(X, X_carat_by_cut, on='cut', how = 'left')

# caratとcarat中央値の差分
X['carat-median_carat_by_cut'] = (X['carat'] - X['median_carat_by_cut'])
# caratとcarat中央値の比率
X['carat/median_carat_by_cut'] = (X['carat'] / X['median_carat_by_cut'])

print('カテゴリ変数＋追加した特徴量')
display(X[['cut', 'carat', 'median_carat_by_cut', 'carat-median_carat_by_cut', 'carat/median_carat_by_cut']].head())

# %%
# カテゴリ変数colorで集計したcarat中央値の特徴量追加

# カテゴリ変数colorごとにcarat中央値を集計
X_carat_by_color = X.groupby('color')['carat'].agg('median').reset_index()
X_carat_by_color.columns = ['color', 'median_carat_by_color']
print('colorごとのcarat中央値')
display(X_carat_by_color)

# 集計した特徴量の追加
X = pd.merge(X, X_carat_by_color, on='color', how = 'left')

# caratとcarat中央値の差分
X['carat-median_carat_by_color'] = (X['carat'] - X['median_carat_by_color'])
# caratとcarat中央値の比率
X['carat/median_carat_by_color'] = (X['carat'] / X['median_carat_by_color'])

print('カテゴリ変数＋追加した特徴量')
display(X[['color', 'carat', 'median_carat_by_color', 'carat-median_carat_by_color', 'carat/median_carat_by_color']].head())

# %%
# カテゴリ変数clarityで集計したcarat中央値の特徴量追加

# カテゴリ変数clarityごとにcarat中央値を集計
X_carat_by_clarity = X.groupby('clarity')['carat'].agg('median').reset_index()
X_carat_by_clarity.columns = ['clarity', 'median_carat_by_clarity']
print('clarityごとのcarat中央値')
display(X_carat_by_clarity)

# 集計した特徴量の追加
X = pd.merge(X, X_carat_by_clarity, on='clarity', how = 'left')

# caratとcarat中央値の差分
X['carat-median_carat_by_clarity'] = (X['carat'] - X['median_carat_by_clarity'])
# caratとcarat中央値の比率
X['carat/median_carat_by_clarity'] = (X['carat'] / X['median_carat_by_clarity'])


print('カテゴリ変数＋追加した特徴量')
display(X[['clarity', 'carat', 'median_carat_by_clarity', 'carat-median_carat_by_clarity', 'carat/median_carat_by_clarity']].head())

# %%
# カテゴリ変数cut×colorで集計した出現割合の特徴量追加

# クロス集計表の出現割合
X_cross = pd.crosstab(X['cut'], X['color'], normalize='index')
X_cross = X_cross.reset_index()
print('cut*colorのクロス集計表')
display(X_cross)

# クロス集計表のテーブルへの変換
X_tbl = pd.melt(X_cross, id_vars='cut', value_name='rate_cut*color')
print('cut*colorのテーブル')
display(X_tbl)

# 出現割合の特徴量追加
X = pd.merge(X, X_tbl, on=['cut', 'color'], how='left' )
print('カテゴリ変数＋追加した特徴量')
display(X[['cut', 'color', 'clarity', 'rate_cut*color']].head())

# %%
# カテゴリ変数color×clarityで集計した出現割合の特徴量追加

# クロス集計表の出現割合
X_cross = pd.crosstab(X['color'], X['clarity'], normalize='index')
X_cross = X_cross.reset_index()
print('color*clarityのクロス集計表')
display(X_cross)

# クロス集計表のテーブルへの変換
X_tbl = pd.melt(X_cross, id_vars='color', value_name='rate_color*clarity')
print('color*clarityのテーブル')
display(X_tbl)

# 出現割合の特徴量追加
X = pd.merge(X, X_tbl, on=['color', 'clarity'], how='left' )
print('カテゴリ変数＋追加した特徴量')
display(X[['cut', 'color', 'clarity', 'rate_color*clarity']].head())

# %%
# カテゴリ変数clarity×cutで集計した出現割合の特徴量追加

# クロス集計表の出現割合
X_cross = pd.crosstab(X['clarity'], X['cut'], normalize='index')
X_cross = X_cross.reset_index()
print('clarity*cutのクロス集計表')
display(X_cross)

# クロス集計表のテーブルへの変換
X_tbl = pd.melt(X_cross, id_vars='clarity', value_name='rate_clarity*cut')
print('clarity*cutのテーブル')
display(X_tbl)

# 出現割合の特徴量追加
X = pd.merge(X, X_tbl, on=['clarity', 'cut'], how='left' )
print('カテゴリ変数＋追加した特徴量')
display(X[['cut', 'color', 'clarity', 'rate_clarity*cut']].head())

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
# カテゴリ変数のデータ型をcategory型に変換
cat_cols = ['cut', 'color', 'clarity']

for c in cat_cols:
    X_train[c] = X_train[c].astype('category')
    X_test[c] = X_test[c].astype('category')

# %%
X_train.info()

# %%
# 学習データの一部を検証データに分割
X_tr, X_va, y_tr, y_va = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=0)
print('X_trの形状：', X_tr.shape, ' y_trの形状：', y_tr.shape, ' X_vaの形状：', X_va.shape, ' y_vaの形状：', y_va.shape)

# %%
# ハイパーパラメータの設定
import lightgbm as lgb

lgb_train = lgb.Dataset(X_tr, y_tr)
lgb_eval = lgb.Dataset(X_va, y_va, reference=lgb_train)

params = {
    'objective': 'mae',
    'seed': 0,
    'verbose': -1,
}

# %%
# モデルの学習
model = lgb.train(params,
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

# %% [markdown]
"""
# クロスバリデーション
"""

# %%
# クロスバリデーション
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

    model = lgb.train(params,
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
# foldごとの検証データの誤差
valid_scores

# %%


# %%


