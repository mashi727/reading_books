
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from datetime import date
from dateutil.relativedelta import relativedelta
from dateutil import parser
import numpy as np
import math
from alpha_vantage.techindicators import TechIndicators



from alpha_vantage.timeseries import TimeSeries
from pprint import pprint



symbol = "AAPL"


# Sampleコードは、AVを使用だっかけど、データが取得できれば何でも良いのでyfinanceで取得した。

from dataclasses import dataclass
@dataclass
class Data():
    span_av :str = '1min'
    # yahooFin [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
    span_yh :str = '1d'

import yfinance
def fetch_from_yahoo(ticker_symbol, span):
    df = yfinance.download(
        tickers=ticker_symbol, # ナスダック100指数
        period="max",
        interval=span,
    )
    df.index.name = 'date'
    return df


#data = fetch_from_yahoo(symbol, Data.span_yh)





class DrawGraph():
    def __init__(self, *args, **kwargs):
        super(DrawGraph, self).__init__(*args, **kwargs)

    def draw_graph(self):
        area = DockArea()
        self.layout.addWidget(area)  

        #d1 = Dock("pyqtgraph example: Basic Plots", size=(1, 1))
        #area.addDock(d1) ## place d1

        win = pg.GraphicsLayoutWidget()
        win.setWindowTitle('Awesome Graph')
        #d1.addWidget(win)


        #dt, data = DrawGraph.make_dataframe(FilePath.data_filename)
        filename = FilePath.data_filename
        data = pd.read_csv(filename,index_col='date', parse_dates=True)
        
        '''
        データを取得して、いくつかの指標を追加する。
        '''


        #dataframeのりネーム
        data = data.rename(columns={'Open': '1. open', 'Close': '4. close', 'High': '2. high', 'Low': '3. low'})
        print(data)

        # 前日より値段が上がった場合(1),下がった場合(0)
        data['Price_Up_From_PreviousDay'] = np.where(data['4. close'] > data['4. close'].shift(-1), 1, 0) 
        # その日の終値の値段が上がった場合(1)、下がった場合(0)
        data['Today_Price_Up'] = np.where(data['4. close'] > data['1. open'], 1, 0)
        # 翌日に値段が上がった場合(1)、下がった場合(0)
        data['Tomorrow_Price_Up'] = np.where(data['4. close'].shift(1) > data['4. close'], 1, 0)
        # 前日比での上下率
        data['Percentage'] = (data['4. close'] - data['4. close'].shift(-1)) / data['4. close'].shift(-1)
        # ある一定期間後の終値（今回は10日）
        data['SomeDaysAfterClosingPrice'] = data['4. close'].shift(10) # X days later price
        # 任意の期間でデータを可視化できるRegionの威力は絶大だな！



        #%%
        '''
        以上をベースにテクニカル指標を見ていく

        移動平均からゴールデンクロスへ！

        '''

        API_KEY = '0A5DAC7F3S5UWRJZ'

        ti = TechIndicators(key=API_KEY, output_format='pandas')
        ts = TimeSeries(key=API_KEY, output_format='pandas')


        # Moving Average 5 and 20
        dataSMA_short, meta_data_sma_short = ti.get_sma(symbol=symbol, time_period=5)
        dataSMA_short.columns = ['SMA_short']
        data = data.merge(dataSMA_short, left_index=True, right_index=True)

        dataSMA_long, meta_data_sma_long = ti.get_sma(symbol=symbol, time_period=20)
        dataSMA_long.columns = ['SMA_long']
        data = data.merge(dataSMA_long, left_index=True, right_index=True)

        data['SMATrend'] = np.where(data['SMA_short'] > data['SMA_long'], 1, 0) # Up trend is 1,, Down Trend is 0
        data['GoldenCrossHappened'] = np.where(data['SMATrend'] > data['SMATrend'].shift(-1), 1, 0)
        #pd.set_option('display.max_rows', None)




        # theory 1
        # we should see price up after seeing golden cross between SMA 5 and SMA20.
        # this is giving as some days later price as example
        turningPointChange = 0
        priceActuallyUp = 0
        for index, row in data.iterrows():
            if row['GoldenCrossHappened'] == 1:
                turningPointChange += 1
            if row['4. close'] < row['SomeDaysAfterClosingPrice']:
                priceActuallyUp += 1

        print("Golden Cross Point:  " + str(turningPointChange))
        print("Actual Price Up:  " + str(priceActuallyUp))
        print("Percentage:  " + str((priceActuallyUp/turningPointChange)*100) + "%")






        # Loop through each stock (while ignoring time columns with index 0)
        #df = data
        #for i in df.columns[:]:
        #    if i == "4. close" or i == "SMA_short" or i == "SMA_long":
        #        p1.plot(x = dt, y = df[i], name = i) # add a new Scatter trace

        #p1 = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})


    