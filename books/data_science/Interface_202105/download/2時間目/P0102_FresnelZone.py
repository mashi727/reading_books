import math

SPEED_OF_LIGHT = 299792458.0            # 光速(m/sec)

print("周波数=")
s = input()
freq = float(s)
_lambda = SPEED_OF_LIGHT / freq
print("d1=")
s = input()
d1 = float(s)
print("d2=")
s = input()
d2 = float(s)
print("n(フレネルゾーンの次数)=")
s = input()
n = int(s)
print("障害物の高さ(m)=")
s = input()
H2 = int(s)
dh = (d1+d2)/2.0
Rn1 = math.sqrt(n*dh*dh*_lambda/(d1+d2))
Rn2 = math.sqrt(n*d1*d2*_lambda/(d1+d2))
print("周波数%0.1f(MHz)、伝搬距離%0.1f(m)の時のフレネルゾーン最大半径=%0.1f(m)\n"%(freq/1e6, d1+d2, Rn1))
print("送信アンテナから%0.1f(m)地点、受信アンテナから%0.1f(m)地点のフレネル半径=%0.1f(m)\n"%(d1, d2, Rn2))
print("障害物の干渉を避けるのに必要な送受信アンテナの高さ=%0.1f(m)\n"%(Rn2+H2))