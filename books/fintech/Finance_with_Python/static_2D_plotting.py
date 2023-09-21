import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtGui import QFont
from PyQt5.QtCore import *


import numpy as np
import sys

import inspect
def print_method(obj):
    for x in inspect.getmembers(obj, inspect.ismethod):
        print(x[0])
    print(type(obj))


setprop = lambda x: (x.showGrid(x=True, y=True, alpha=0.5),
                    #x.getGrid(color='#000'),
                    x.setAutoVisible(y=True),
                    x.enableAutoRange('y'),
                    x.getAxis('left').setTickSpacing(major = 5, minor =10),
                    x.getAxis('bottom').setTickSpacing(major = 5, minor =10),

                    x.getAxis('right').setWidth(60),
                    x.getAxis('right').setLabel(**fontCss),
                    x.getAxis('right').setStyle(tickFont = QFont("Roman times",10)),
                    x.getAxis('right').setPen(color='#790', width=2),
                    x.showAxis('right'),
                    x.setLabels(right='右側のラベル'),


                    x.getAxis('top').setHeight(50),
                    x.getAxis('top').setPen(color='#666', width=2),
                    x.getAxis('top').setLabel(**fontCss),
                    x.getAxis('top').setStyle(tickFont = QFont("Roman times",10)),
                    x.showAxis('top'),
                    x.setLabels(top='上側のラベル'),

                    x.getAxis('bottom').setHeight(50),
                    x.getAxis('bottom').setLabel(**fontCss),
                    x.getAxis('bottom').setPen(color='#660', width=2),
                    x.getAxis('bottom').setStyle(tickFont = QFont("Roman times",10)),
                    x.showAxis('bottom'),
                    x.setLabels(bottom='下側のラベル'),

                    x.getAxis('left').setPen(color='#066', width=2),
                    x.getAxis('left').setWidth(60),
                    x.getAxis('left').setStyle(tickFont = QFont("Roman times",10)),
                    x.getAxis('left').setLabel(**fontCss),
                    x.showAxis('left'),
                    x.setLabel('left','Label with <span style="color: #f0f">color</span>'),
                    )



class PlotGraph:
    def __init__(self):
        # UIを設定
        # アンチエイリアスを指定するとプロットがより綺麗になる
        pg.setConfigOptions(antialias=True)
        pg.setConfigOption('background', (255, 255, 255)) # 背景＝薄水色
        pg.setConfigOption('foreground', (0, 0, 0))       # 前景＝黒

        self.win = pg.GraphicsWindow()
        self.win.setWindowTitle('Static 2D Plotting')


        self.plt = self.win.addPlot()
        viewbox = self.plt.getViewBox()
        viewbox.setBackgroundColor((234, 234, 242))

        import numpy as np
        np.random.seed(1000)
        y = np.random.standard_normal(20)
        x = np.arange(len(y))
        '''
        self.win.setStyleSheet("""
                        border-top: 5px solid '#099';
                        border-style: outset;
                        border-width: 5px;
                        border-color: '#70f';
                        padding: 5px;
                        """)
        '''

        fontCss = {'font-family': "Arial, Noto Sans Mono Regular", 'font-size': '16pt'}
        #fontCss["color"] = '#000'
        penGrid = pg.mkPen(color = '#0003', style = Qt.CustomDashLine, width=5)
        penGrid.setDashPattern([10, 40, 50, 40])
        grid = pg.GridItem(pen=penGrid)
        print_method(grid)   
        #self.plt.addItem(grid)

        setprop(self.plt)


        pen = pg.mkPen(color = '#09f9', style = Qt.SolidLine, width=2)
        self.plt.plot(x,y,pen=(0,0,200), symbolBrush=(0,0,200), symbolPen='w', symbol='o', symbolSize=14, name="symbol='o'")

        legend = self.plt.addLegend(offset=(10,10))
        fontCssLegend = '<style type="text/css"> p {font-family: Arial, Noto Sans Mono Regular; font-size: 11pt; color: "#000F"} </style>'
        legend.addItem(pg.PlotCurveItem(pen=pen , antialias = True), name = fontCssLegend + '<p>plot1（ぷろっと1）</p>')




if __name__ == "__main__":
    graphWin = PlotGraph()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()