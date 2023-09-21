import numpy as np
import scipy
import matplotlib.pyplot as plt

SPEED_OF_LIGHT = 299792458.0            # 光速(m/sec)

def CalcPathloss(freq, d):
    _lambda = SPEED_OF_LIGHT / freq     # 光速と周波数から波長を計算
    loss = (4.0*np.pi*d/_lambda)**2     # 距離の2乗に比例、波長の2乗に反比例
    return loss

def PlotPathloss(freq, dist, loss):
    plt.figure()
    ax = plt.gca()
    title = "Distance attenuation at %dHz"%(freq)
    plt.title(title)
    ax.axes.xaxis.set_visible(True)
    ax.axes.yaxis.set_visible(True)

    plt.grid(True)
    plt.xlabel('distance(m)')
    plt.ylabel('pathloss(dB)')
    plt.grid = True
    plt.plot(dist, loss, drawstyle='steps-post')
    plt.show()

if __name__ == "__main__":
    print("Frequency:", end='')
    freq = float(input())
    print("Max Distance(m):", end='')
    dist = float(input())
    d = np.arange(dist/1000.0,dist,dist/1000.0)
    loss = CalcPathloss(freq, d)
    loss_db = -10.0*np.log10(loss)
    PlotPathloss(freq, d, loss_db)