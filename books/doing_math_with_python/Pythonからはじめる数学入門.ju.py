# %% [markdown]
"""
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#数を扱う" data-toc-modified-id="数を扱う-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>数を扱う</a></span><ul class="toc-item"><li><span><a href="#基本数学演算" data-toc-modified-id="基本数学演算-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>基本数学演算</a></span></li><li><span><a href="#ラベル：名前に数を割り当てる" data-toc-modified-id="ラベル：名前に数を割り当てる-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>ラベル：名前に数を割り当てる</a></span></li><li><span><a href="#さまざまな種類の数" data-toc-modified-id="さまざまな種類の数-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>さまざまな種類の数</a></span><ul class="toc-item"><li><span><a href="#分数を扱う" data-toc-modified-id="分数を扱う-1.3.1"><span class="toc-item-num">1.3.1&nbsp;&nbsp;</span>分数を扱う</a></span></li><li><span><a href="#複素数" data-toc-modified-id="複素数-1.3.2"><span class="toc-item-num">1.3.2&nbsp;&nbsp;</span>複素数</a></span></li></ul></li><li><span><a href="#ユーザー入力を受け取る" data-toc-modified-id="ユーザー入力を受け取る-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>ユーザー入力を受け取る</a></span></li></ul></li></ul></div>
"""

# %% [markdown]
"""
# 数を扱う

とりあえず、Pythonの使い方に慣れるお。

まずは、基本数学演算からはじめて、数を扱い理解する簡単なプログラムを書くお。


## 基本数学演算
"""

# %%
1+2

# %%
1+3.5

# %% [markdown]
"""
$%$は、剰余だお。
"""

# %%
9%2

# %%
8**(1/3)

# %% [markdown]
"""
## ラベル：名前に数を割り当てる

複雑な数式を扱うことができるように、名前に数を割り当てて代数計算ができるようにするお。
"""

# %%
a = 3
a+1

# %%
a=5
a+1

# %% [markdown]
"""
変数と数字の間には、スペースを入れた方が見やすいお。
"""

# %%
a = 3
a + 3

# %% [markdown]
"""
## さまざまな種類の数

これまでの数学演算では、2種類の数を扱ってきました。整数と浮動小数点数です。

人間は、数がどのように書かれようと、整数、浮動小数点数、分数、ローマ字表記でも、その数を正しく認識して問題なく処理することができますが、コンピューターはそういうわけにはいきません。

Pythonは、整数と浮動小数点数を異なる型（type）だと考えます。type()関数を使用すると、入力した数の種類がわかります。
"""

# %%
type(3)

# %%
type(3.)

# %% [markdown]
"""
### 分数を扱う

Pythonで分数を扱うためには、fractionsモジュールを使用する必要があります。モジュール(module)は、自分のプログラムの中で使うことができる第三者が書いたプログラムだお。

モジュールには

- クラス
- 関数
- ラベル（変数）定義

が含まれているお。

Pythonの標準ライブラリに入っていることもあれば、第三者が配布されることもあるお。後者の場合、使う前にモジュールをインストールする必要があるお。
"""

# %%
from fractions import Fraction
f = Fraction(3, 4)
f

# %%
Fraction(3,4) + 1 + 1.5

# %%
Fraction(3, 4) + 1 + Fraction(1/4)

# %%
Fraction(3, 4) + 1 + Fraction(1,4)

# %% [markdown]
"""
### 複素数

Pythonでは、複素数も扱うことができるお。
"""

# %%
a = 2 + 3j
type(a)

# %% [markdown]
"""
`complex()`関数を使って、複素数を定義することもできるお。
複素数の実部と虚部をcomplex()関数の引数として渡すと、複素数を返すお。
"""

# %%
complex(3, 4)

# %%
print(a)

# %% [markdown]
"""
## ユーザー入力を受け取る

プログラムを書き始めると、input()関数でユーザーの入力を巻単位受け取ることができるようになるお。
"""

# %%
a = input()

# %%
print(a)

# %%
a

# %% [markdown]
"""
**この違いを確認する必要があるなと。**

思いましたが、解説がありました。

**input関数は、入力を文字列で返す関数。**


一重引用符（シングルクオート）は、
"""

# %%
1+3

# %%
a = 1
a + 1

# %%
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
# データの作成
x = np.arange(0, 6, 0.1) # 0 から6 まで0.1 刻みで生成
y1 = np.sin(x)
y2 = np.cos(x)
# グラフの描画
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle = "--", label="cos") # 破線で描画
plt.xlabel("x") # x 軸のラベル
plt.ylabel("y") # y 軸のラベル
plt.title('sin & cos') # タイトル
plt.legend()
plt.show()

# %%


