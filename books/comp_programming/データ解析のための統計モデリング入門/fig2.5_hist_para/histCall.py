import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from PyQt5 import uic
from pyqtgraph.Point import Point

#QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
#QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('plotUi3.ui', self)
        self.win = self.graphicsView
        
        graph_obj = ['p1']

        num = len(graph_obj)+1
        for case in range(1,num):
            graph_id = graph_obj[case-1]
            # 凡例のフォントの大きさと、表示するテキストの指定
            self.graph_id = self.set_graph_ui(graph_id)
            self.plot_xy(graph_id)
            self.vb = self.graph_id.vb
        self.doubleSpinBox.valueChanged.connect(self.plot_xy)

    def set_graph_ui(self,graph_id):
        setprop = lambda x: (x.showGrid(x=True, y=True, alpha = 1), x.addLegend(offset=(0,5.5)), x.showGrid(x=True, y=True))
        styles = {'color':'white','font-size':'30px', 'font-style':'bold'}       
        pg.setConfigOptions(antialias=True)
        p1 = self.win.addPlot()
        p1.setTitle('<font size=\'12\' color=\'#FFFFFF\'>'+ 'Histogram' +'</font>')
        p1.setLabel('left', text='frequency', units='', **styles)
        p1.setLabel('bottom', text='data', units='', **styles)
        setprop(p1)
        self.p1 = p1
        return p1

    def plot_xy(self,graph_id):
        p = self.doubleSpinBox.value()
        self.graph_id.clear()
         # Histogram
        # RData形式のデータを読み込み
        import pyreadr
        result = pyreadr.read_r('./data/data.RData')
        data = result["data"].to_numpy()
        data_range = int(data.max()-data.min())
        y,x = np.histogram(data, bins=np.linspace(int(data.min()), data_range + 1, data_range + 2 ), density = False)
        ## Using stepMode="center" causes the plot to draw two lines for each sample.
        ## notice that len(x) == len(y)+1
        ## 頻度ヒストグラム表示を一致させるため、グラフのx軸を-0.5ほどオフセットします。
        self.graph_id.plot(x-0.5, y, stepMode=1, fillLevel=0, fillOutline=True, brush=(255,255,0,100))
        from scipy.stats import poisson
        s = poisson.pmf(x, p)*len(data)
        log_L = self.logLikelihoodFunction(p,data)
        legend = '<font size=\'5\' color=\'#FFFFFF\'>'+ 'p = '+format(p, '.2f')+ '<br>'+ 'log_L = '+ format(log_L, '.2f') +'</font>'
        self.graph_id.plot(x, s, name = legend, pen='#0F0' ,alpha=1, symbolBrush='#0F0', symbolSize=10)

    def logLikelihoodFunction(self,p,data):
        import math
        #print(data[49][0])
        log_L = 0
        for i in range(len(data)):
            log_L += data[i][0] * math.log(p) - p  - math.log(math.factorial(int(data[i][0])))
        return log_L


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()