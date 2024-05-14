import sys

from PySide6 import *
from PySide6 import QtCore
from PySide6.QtGui import QFont, QPainter, QPen
import pyqtgraph as pg  # import PyQtGraph after Qt


def setprop_func(plt):
    fontCss = {'font-family': "Arial, Meiryo", 'font-size': '16pt'}
    plt.showGrid(x=True, y=True, alpha=0.5),
    plt.setAutoVisible(y=True),
    plt.getAxis('top').setLabel(**fontCss),
    plt.setLabels(top='上側のラベル'),
    plt.showAxis('right'),
    plt.getAxis('left').setWidth(60),
    plt.getAxis('bottom').setHeight(50),
    plt.showAxis('top'),
    plt.getAxis('top').setHeight(20),
    plt.getAxis('right').setWidth(60),
    plt.enableAutoRange('y'),
    plt.addLegend(offset=(20,20)) # legendには、nameが必要


class Draw_interface(QtWidgets.QMainWindow):
    def __init__(self):
        super(Draw_interface, self).__init__()
        # グラフの表示位置と画面サイズ
        self.setGeometry(10, 10, 1280, 768)
        pg.setConfigOptions(antialias=True)
        pg.setConfigOption('background', (255, 255, 255)) # 背景＝白
        pg.setConfigOption('foreground', (0, 0, 0))       # 前景＝黒

        # Add two charts
        self.plt= pg.GraphicsLayoutWidget(show=True)
        self.plt.setWindowTitle('Static 2D Plotting')
        self.setCentralWidget(self.plt)
        
        self.p1 = self.plt.addPlot(row=0, col=0)
        # 背景色の設定
        viewbox_p1 = self.p1.getViewBox()
        viewbox_p1.setBackgroundColor('#dff4')
        # Add Title
        self.p1.setTitle("Your p1 Title Here", color="b", size="20pt")

        self.p2 = self.plt.addPlot(row=1, col=0)
        viewbox_p2 = self.p2.getViewBox()
        viewbox_p2.setBackgroundColor((234, 234, 242))
        self.p2.setTitle("Your p2 Title Here", color="b", size="20pt")

        '''
        self.plt.setStyleSheet("""
                        border-top: 5px solid '#099';
                        border-style: outset;
                        border-width: 5px;
                        border-color: '#70f';
                        padding: 5px;
                        """)
        '''

        # Draw charts

        import numpy as np
        np.random.seed(1000)
        y = np.random.standard_normal(20)
        x = np.arange(len(y))

        pen = pg.mkPen(color = '#09f9', style = QtCore.Qt.SolidLine, width=3)
        # Add Axis Labels
        styles = {"color": "#32f", "font-size": "20px"}
        self.p1.setLabel("left", "Temperature (deg)", **styles)
        self.p1.setLabel("bottom", "Hour (H)", **styles)
        # Add legend
        setprop_func(self.p1)
        fontCssLegend = '<style type="text/css"> p {font-family: Arial, Meiryo; font-size: 14pt; color: "#55F"} </style>'
        self.p1.plot(x,
                    y,
                    name= fontCssLegend +'<p>Sensor 1</p>',
                    pen=pen,
                    symbol="o",
                    symbolSize=15,
                    symbolBrush=("#f7f"),
                    )


        styles2 = {"color": "#F0F", "font-size": "20px"}
        self.p2.setLabel("left", "Temperature (deg)", **styles2)
        self.p2.setLabel("bottom", "Hour (H)", **styles2)

        setprop_func(self.p2)
        self.p2.plot(
                    x,
                    y,
                    name="Sensor 2",
                    pen=pen,
                    symbol="o",
                    symbolSize=15,
                    symbolBrush=("b"),
                    )
    

def main():
    app= QtWidgets.QApplication(sys.argv)
    main= Draw_interface()
    main.show()
    sys.exit(app.exec())
    
if __name__== '__main__':
    main()
