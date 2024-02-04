# プログラム例　example_6.py
import numpy as np
x = -2 
print('x =', x)
if  x >= 0:
    root_x = sqrt(x)
else:
    root_x = np.sqrt(-x) * 1j
print('root_x =', root_x)