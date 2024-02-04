'''
finance plot Dummy data
'''
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea

# setpropは、それぞれのモデル側から読み込む（docstring的にオイシイから）
from my_modules.prop_func import *
from my_modules.finance_lib import *

import pandas as pd

from dataclasses import dataclass
# tiker symbol
# 'S&P500', 'TSLA',

FinData.symbol = 'AAPL'
# 1min, 5min, 15min, 30min, 60min, over 60min

if '1min' in FilePath.data_filename:
    FinData.duration = '1min'
elif '5min' in FilePath.data_filename:
    FinData.duration = '5min'
elif '15min' in FilePath.data_filename:
    FinData.duration = '15min'
elif '30min' in FilePath.data_filename:
    FinData.duration = '30min'
elif '60min' in FilePath.data_filename:
    FinData.duration = '60min'
else:
    FinData.duration = '60min_over'
FinData.region_disp = 0.2


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

        #dt, df_calculated = DrawGraph.make_dataframe(FilePath.data_filename)
        filename = FilePath.data_filename
        df = pd.read_csv(filename,index_col='date', parse_dates=True)
        
        p1 = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        #p1 = graphics_lw2.addPlot(axisItems={'bottom': TimeAxisItem(orientation='bottom')},tickLength=500)
        p1.addItem(CandlestickItem(df_calculated[['Open', 'Close', 'Low', 'High']]))

        setprop_stock_func(p1)
        fontCssLegend = '<style type="text/css"> p {font-family: Helvetica, HackGen35 Console NFJ; font-size: 15pt; color: "#ffffff"} </style>'
        p1.addLegend()
        pen = pg.mkPen(color = '#f0f8', style = QtCore.Qt.SolidLine, width=3)
        pen1 = pg.mkPen(color = '#fffa', style = QtCore.Qt.SolidLine, width=5)
        pen2 = pg.mkPen(color = '#0ff5', style = QtCore.Qt.SolidLine, width=3)
        pen3 = pg.mkPen(color = '#ff06', style = QtCore.Qt.SolidLine, width=1)
        pen4 = pg.mkPen(color = '#ff06', style = QtCore.Qt.SolidLine, width=5)

        p1.plot(x=dt,
                y=df_calculated['sma_short'],
                name= fontCssLegend +'<p>SMA short</p>',
                pen=pen,
                )
        p1.plot(x=dt,
                y=df_calculated['sma_med'],
                name= fontCssLegend +'<p>SMA med</p>',
                pen=pen1,
                )
        p1.plot(x=dt,
                y=df_calculated['sma_long'],
                name= fontCssLegend +'<p>SMA long</p>',
                pen=pen2,
                )
        brushes = [0.5, (255,255,0, 40), 0.5]
        #df_calculated[lower]=df_calculated['BB_up']
        #df_calculated[upper]=df_calculated['BB_down']
        upper = pg.PlotDataItem(dt, df_calculated['BB_up'], pen='g', symbol=None)
        lower = pg.PlotDataItem(dt, df_calculated['BB_down'], pen='g', symbol=None)
        fills = pg.FillBetweenItem(lower, upper, brushes[1])
        fills.setZValue(-100)
        p1.plot(x=dt,
                y=df_calculated['BB_up'],
                name= fontCssLegend +'<p>BB upper</p>',
                pen=pen3,
                )
        p1.plot(x=dt,
                y=df_calculated['BB_middle'],
                name= fontCssLegend +'<p>BB middle</p>',
                pen=pen4,
                )
        p1.plot(x=dt,
                y=df_calculated['BB_down'],
                name= fontCssLegend +'<p>BB down</p>',
                pen=pen3,
                )
        p1.addItem(fills)




        # p1.enableAutoRange(axis='y', enable=True) # うまく動かないので、別に作成
        p1.getAxis('bottom').setHeight(0)
        p1.getAxis('right').setLabel(**fontCss)
        p1.setLabels(right='ローソク足')
        d1 = Dock('Main Chart', widget=p1, size=(1, 10))
        area.addDock(d1)

        #graphics_lw3 = pg.GraphicsLayoutWidget(show=True)

        # MACDの作成
        plt_macd = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        setprop_stock(plt_macd)
        # 以下は、MACDのヒストグラムの幅
        if FinData.duration == '1min':
            width = 50
        elif FinData.duration == '5min':
            width = 250
        elif FinData.duration == '15min':
            width = 750
        elif FinData.duration == '30min':
            width = 1500
        elif FinData.duration == '60min':
            width = 3000
        else:
            width = 3000
        plt_macd.addItem(pg.BarGraphItem(x=dt, height=df_calculated['macd_hist'], width=width, pen=None ,alpha=1, brush='#0D0'))
        plt_macd.addItem(pg.PlotDataItem(dt, df_calculated['macd'], pen='w'))
        plt_macd.addItem(pg.PlotDataItem(dt, df_calculated['macd_sig'], pen='r'))
        plt_macd.getAxis('right').setLabel(**fontCss)
        plt_macd.setLabels(right='MACD')
        plt_macd.setXLink(p1)
        area.addDock(Dock('MACD', widget=plt_macd, size=(1, 4))) ## place d1 at left edge of dock area (it will fill 

        # regionの作成
        plt_region = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        plt_region.plot(dt, df_calculated['Open'], pen='#fff6')
        # Regionの初期設定
        # デフォルトのRegionは、後ろの10％のデータ
        num_of_df = len(dt)
        #self.region.setRegion([dt[num_of_df-math.floor(num_of_df/10)], dt[num_of_df-1]])
        region = pg.LinearRegionItem([dt[num_of_df-math.floor(num_of_df*FinData.region_disp)], dt[num_of_df-1]]) # http://www.pyqtgraph.org/documentation/graphicsItems/linearregionitem.html
        region.setZValue(10)
        plt_region.addItem(region, ignoreBounds=True)
        area.addDock(Dock('Region', widget=plt_region, size=(1, 0.2))) ## place d1 at left edge of dock area (it will fill 

        def updatePlot():
            p1.setXRange(*region.getRegion(), padding=0)
            minx, maxx = region.getRegion()
            # ローソク足のチャートのy軸最適表示  
            df = df_calculated.dropna(how='any')
            dt2 = df.index.view(np.int64)//10**9
            sma_long = df['sma_short'].values
            sma_med = df['sma_med'].values
            sma_short = df['sma_long'].values
            # y軸の最適表示のためのindexを作成
            idx2 = (dt2>=minx) & (dt2<=maxx)
            if idx2.sum()<2:
                return
            miny = min([sma_short[idx2].min(),sma_med[idx2].min(),sma_long[idx2].min()])
            maxy = max([sma_short[idx2].max(),sma_med[idx2].max(),sma_long[idx2].max()])
            p1.setYRange(miny*(1-0.001), maxy*(1+0.001))
            plt_region.setYRange(miny*(1-0.01), maxy*(1+0.01))


        def updateRegion():
            region.setRegion(p1.getViewBox().viewRange()[0])

        region.sigRegionChanged.connect(updatePlot)
        p1.sigXRangeChanged.connect(updateRegion)
        updatePlot()



    # 4valuesからもろもろの値を算出
    def make_dataframe(filename):
        @dataclass
        class finData:
            sma_short : float = 5
            sma_med : float = 25
            sma_long : float = 60
        # https://note.nkmk.me/python-pandas-time-series-datetimeindex/
        # を参考にして、csvの読み込み時にインデックスの設定をしたよ。
        try:
            df = pd.read_csv(filename,index_col='date', parse_dates=True)
        except:
            df = pd.read_csv(filename,index_col='Datetime', parse_dates=True)
            df.index.name = 'date'


        # 移動平均
        df['sma_short'] = df['Close'].rolling(finData.sma_short).mean() # sma_shortの移動平均。sma_short分の窓をずらすので最初のsma_short-1 はNaNになる。
        df['sma_med'] = df['Close'].rolling(finData.sma_med).mean()
        df['sma_long'] = df['Close'].rolling(finData.sma_long).mean()
            
        #MACD
        df['macd'] = df['Close'].rolling(12).mean()-df['Close'].rolling(26).mean() # sma版
        df['macd_sig'] = df['macd'].rolling(9).mean()
        df['macd_hist'] = df['macd']-df['macd_sig']
        print(df)
        dt = df.index.view(np.int64)//10**9

        # BB
        import talib
        bb_period = 10
        sigma = 2
        matype = 0

        df['BB_up'], df['BB_middle'], df['BB_down'] = talib.BBANDS(np.array(df['Close']), bb_period, sigma, sigma, matype)
        return dt, df
