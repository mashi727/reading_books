import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, uic

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('simplePlotUi.ui', self)
        self.win = self.graphicsView

        graph_obj = ['p1','p2','p3','p4','p5','p6','p7','p8','p9']
        average = [2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.4, 4.8, 5.2]
        num_of_fold = 3
        
        num = len(graph_obj)+1
        for case in range(1,num):
            graph_id = graph_obj[case-1]
            average_num = average[case-1]
            # 凡例のフォントの大きさと、表示するテキストの指定
            #legend = '<font size=\'5\' color=\'#FFFFFF\'>'+ 'λ='+str(average_num) +'</font>'
            self.graph_id = self.set_graph_ui(graph_id, case, num_of_fold)
            self.plot_xy(graph_id,average_num)
            self.vb = self.graph_id.vb

    def set_graph_ui(self,graph_id, num, num_of_fold):
        # いつもは、こちら。
        # setprop = lambda x: (x.showGrid(x=True, y=True, alpha = 1),x.setAutoVisible(y=True), x.addLegend(), x.showGrid(x=True, y=True))
        # setAutoVisible(y=True)を有効にすると、IndexError: boolean index did not match indexed array along dimension 0;
        # dimension is 39 but corresponding boolean dimension is 40 がでます。
        setprop = lambda x: (x.showGrid(x=True, y=True, alpha = 1), x.addLegend(offset=(0,5.5)), x.showGrid(x=True, y=True))
        styles = {'color':'white','font-size':'20px', 'font-style':'bold'}       
        graph_id = self.win.addPlot()
        graph_id.setTitle('<font size=\'5\' color=\'#FFFFFF\'>'+ 'Histogram' +'</font>')
        graph_id.setLabel('left', text='Frequency', units='', **styles)
        graph_id.setLabel('bottom', text='data', units='', **styles)
        setprop(graph_id)
        self.graph_id = graph_id
        if num % num_of_fold == 0:
            self.win.nextRow()
        return graph_id



    def plot_xy(self,graph_id,average):
        # Histogram
        # RData形式のデータを読み込み
        import pyreadr
        result = pyreadr.read_r('./data/data.RData')
        data = result["data"].to_numpy()
        # compute parameter
        data_range = int(data.max()-data.min())
        y,x = np.histogram(data, bins=np.linspace(int(data.min()), data_range + 1, data_range + 2 ), density = False)
        ## Using stepMode="center" causes the plot to draw two lines for each sample.
        ## notice that len(x) == len(y)+1
        ## 頻度ヒストグラム表示を一致させるため、グラフのx軸を-0.5ほどオフセットします。
        self.graph_id.plot(x-0.5, y, stepMode=1, fillLevel=0, fillOutline=True, brush=(255,255,0,100))
        from scipy.stats import poisson
        s = poisson.pmf(x, average)*len(data)
        log_L = self.logLikelihoodFunction(average,data)
        legend = '<font size=\'5\' color=\'#FFFFFF\'>'+ 'p = '+format(average, '.2f')+ '<br>'+ 'log_L = '+ format(log_L, '.2f') +'</font>'
        self.graph_id.plot(x, s, name=legend, pen='#0F0' ,alpha=1, symbolBrush='#0F0', symbolSize=10)

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