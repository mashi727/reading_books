# Pyplotインターフェースを用いたグラフの描画 example8.py
import numpy as np
import matplotlib.pyplot as plt 

t = np.arange(0, 2*np.pi, 0.1)
y = np.sin(t)
plt.plot(t, y)
plt.grid(True)
plt.xlabel('Time $t$'); plt.ylabel('Output $y(t)$') 
plt.show()