# -*- coding: utf-8 -*-
""" Apply SVM to convolution data """
from matplotlib.colors import ListedColormap
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

sns.set_style("whitegrid")

N = 200
t = np.arange(0, N*2, 2)
x1d = 6.5
x2d = 6.5
r_x = 1
r_y = 1
ax_scale = 50

# Draw original plot
np.random.seed(0)
sin1_dat = np.c_[np.sin(2*np.pi*t/N), np.cos(2*np.pi*t/N)]
sin2_dat = np.c_[np.sin(2*np.pi*t/N+np.pi), np.cos(2*np.pi*t/N+np.pi)]  # Shift 180 deg
t_dat = np.linspace(0, 2*np.pi, N)
rand_dat = np.random.randn(N, 2)
X1 = x1d * np.c_[t_dat, t_dat] * sin1_dat + r_x * rand_dat
X2 = x2d * np.c_[t_dat, t_dat] * sin2_dat + r_y * rand_dat
X = np.r_[X1, X2]
y = np.r_[np.ones(N), -1*np.ones(N)]
plt.scatter(X[y==1,0], X[y==1,1], c='r', marker='o')
plt.scatter(X[y==-1,0], X[y==-1,1], c='b', marker='s')

plt.xlim([-ax_scale, ax_scale])
plt.ylim([-ax_scale, ax_scale])
plt.axes().set_aspect('equal')
plt.show()

# Draw with SVM edge
clf = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
clf.fit(X, y)
resolution = 0.02

markers = ('s', 'o')
colors = ('blue', 'red')
cmap = ListedColormap(colors)

# plot the decision surface
xx1, xx2 = np.meshgrid(np.arange(-ax_scale, ax_scale, resolution),
                       np.arange(-ax_scale, ax_scale, resolution))
Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)

plt.contour(xx1, xx2, Z, cmap=plt.cm.get_cmap('Blues'))
plt.xlim([-ax_scale, ax_scale])
plt.ylim([-ax_scale, ax_scale])
    
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                alpha=0.8, c=cmap(idx),
                marker=markers[idx])

plt.axes().set_aspect('equal')
plt.show()

# Draw Kernel plot
resolution = 1

# plot the decision surface
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                       np.arange(x2_min, x2_max, resolution))
Z = clf.decision_function(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)
ZZ = np.zeros_like(Z)
    
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(xx1, xx2, Z, alpha=0.2, color='gray')
ax.plot_wireframe(xx1, xx2, ZZ, alpha=0.1, color='gray')
    
for idx, cl in enumerate(np.unique(y)):
    ZZZ = clf.decision_function(X)
    ax.scatter3D(np.ravel(X[y == cl, 0]),
                 np.ravel(X[y == cl, 1]),
                 np.ravel(ZZZ[y == cl]),
                 c=cmap(idx),
                 marker=markers[idx])

for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()

plt.show()
    