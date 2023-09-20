# -*- coding: utf-8 -*-
""" 判別領域を図示するスクリプト
    書籍「Pyrhon機械学習プログラミング 達人データサイエンティストによる理論と実践」（インプレス社）
    より一部改変（第1版p132参照） """
import numpy as np
import matplotlib.pyplot as plt

import warnings
from matplotlib.colors import ListedColormap

def versiontuple(v):
    return tuple(map(int, (v.split("."))))

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02, scale=None):

    # setup marker generator and color map
    marker_col_dic = {1: 2, -1: 0, 0: 1}
    markers = ('s', '^', 'o', 'v', 'x')
    colors = ('blue', 'green', 'red', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    if scale==None:
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    else:
        x1_min, x1_max = scale[0]
        x2_min, x2_max = scale[1]
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        marker_color_idx = marker_col_dic[cl]
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    c=cmap(marker_color_idx),
                    alpha=0.8,
                    marker=markers[marker_color_idx],
                    s=80, label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')
