from bflib import BfApi
import time
    
API_KEY=""
API_SECRET="" 
#取引所のパラメータ
order_min_size=0.001   #BTC数量最小値
order_digit   =8       #BTC数量の桁数　ex. 3=0.001
fee_rate      =0.15    #取引手数料のレート(%)
#取引パラメータ
buy_unit      =0.001  #購入単位
profit        =200        #価格差

api=BfApi(API_KEY,API_SECRET)

while True:    
    #板情報を取得
    ob=api.orderbook()    
    #購入予定価格を決定(bidの先頭)
    buy_price=ob["bids"][0][0]
    #購入数量を計算。 購入数量 = 数量*(1+fee*2) - BTC残高
    balance=api.balance()
    buy_amount=round(buy_unit*(1+0.01*fee_rate*2) - balance["btc"],order_digit)
    if buy_amount > 0:
        #BTC残高が不十分なら注文の最小値を考慮して追加購入。
        buy_amount=max(order_min_size,buy_amount)
        #単位の整形
        if buy_amount==int(buy_amount):
            buy_amount=int(buy_amount)
        #JPY残高の確認
        if balance["jpy"]<buy_amount*buy_price:
            print("Log : Insufficient JPY balance")
            break
        #注文 BTCの場合はpriceを整数に強制する。
        print("Log : Buy order {0} x {1}".format(int(buy_price),buy_amount))
        oid=api.buy(int(buy_price),buy_amount)
        print("Log : Buy oid={0}".format(oid))
        #注文がサーバーで処理されるまで少し待つ
        time.sleep(10)
        #さらに最大30秒間、注文が約定するのを待つ
        for i in range(0,10):
            if api.is_active_order(oid)==False:
                oid=None
                break
            print("Log : Buy Wait")
            time.sleep(5)
        #注文が残っていたらキャンセルする
        if oid!=None:
            api.cancel(oid)
            print("Log : Buy canceled! oid={0}".format(oid))
            time.sleep(5)
        else:
            print("Log : Buy completed! oid={0}".format(oid))
    else:
        #売却するBTCがすでにあるなら何もしない
        print("Log : Sufficient BTC balance")
    #BTC残高を調べる
    balance=api.balance()
    #売却数量は,BTC残高*(1-fee)
    sell_amount=round(balance["btc"]*(1-0.01*fee_rate),order_digit)
    if sell_amount<order_min_size:
        #部分的な約定などで最小売却単位に届かないならもう一度購入に戻る
        print("Log : Insufficient BTC balance")
        continue
    else:
        #単位の整形
        if sell_amount==int(sell_amount):
            sell_amount=int(sell_amount)       
        print("Log : Sell order {0} x {1}".format(int(buy_price+profit),sell_amount))
        #利益をのせて注文　BTCの場合はpriceを整数に強制する。
        oid=api.sell(int(buy_price+profit),sell_amount)
        print("Log : Sell oid={0}".format(oid))   
        #注文がサーバーで処理されるまで少し待つ
        time.sleep(10)
        #注文が成立するまで永遠に待つ
        while api.is_active_order(oid):
            print("Log : Sell Wait")
            time.sleep(3)
        print("Log : Sell completed! oid={0}".format(oid))

