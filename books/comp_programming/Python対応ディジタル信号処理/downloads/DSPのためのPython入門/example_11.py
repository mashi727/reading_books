# 複数のグラフを別々に描く example11.py
import numpy as np
import matplotlib.pyplot as plt    

t = np.arange(0, 1, 0.01)
ys = np.sin(2 * np.pi * t) 
n = np.arange(0, 32)
yc = np.cos((2 * np.pi / 32) * n)
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)   
ax1.plot(t, ys)
ax1.grid()
ax1.set_xlim(0, 1); ax1.set_ylim(-2, 2) 
ax1.set_xlabel('Time $t$'); ax1.set_ylabel('Output $y_s(t)$')  
ax2 = fig.add_subplot(2, 1, 2)   
ax2.stem(n, yc)
ax2.grid()
ax2.set_xlabel('$n$'); ax2.set_ylabel('$y_c[n]$')
ax2.set_xlim(0, 32); ax2.set_ylim(-2, 2) 
plt.show()