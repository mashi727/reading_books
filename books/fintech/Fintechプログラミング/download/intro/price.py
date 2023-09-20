import requests

# bitFlyerからビットコインの価格を取得する
r = requests.get('https://api.bitflyer.jp/v1/ticker?product_code=BTC_JPY')
json = r.json()
print('bitFlyer  btc_jpy: ' + str(json["ltp"]))

# Zaifからビットコインの価格を取得する
r = requests.get('https://api.zaif.jp/api/1/ticker/btc_jpy')
json = r.json()
print('Zaif      btc_jpy: ' + str(json["last"]))

# coincheckからビットコインの価格を取得する
r = requests.get('https://coincheck.com/api/ticker')
json = r.json()
print('coincheck btc_jpy: ' + str(json["last"]))