# 複数のグラフを重ねて描く example10.py  
import numpy as np  
import matplotlib.pyplot as plt

t = np.arange(0, 2*np.pi, 0.1)
ys = np.sin(t)
yc = np.cos(t)
fig = plt.figure()  
ax = fig.add_subplot(1, 1, 1)
ax.plot(t, ys)
ax.plot(t, yc)
ax.set_xlim(0, 2*np.pi); ax.set_ylim(-2, 2) 
ax.grid()
ax.set_xlabel('Time $t$'); ax.set_ylabel('Output $ys(t), yc(t)$')
plt.show()
