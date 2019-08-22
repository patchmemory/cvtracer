import sys
import numpy as np
import pyqtgraph as pg
#from pyqtgraph.Qt import QtCore, QtGui
from PyQt5 import QtCore, QtWidgets, QtGui


class DataStream(QtWidgets.QWidget):

  def __init__(self, data_source, parent=None):
    super().__init__(parent)
    self.data = np.load(data_source)

    self.frame        =  0
    self.frame_buffer = 30
    self.fps          = 30

    self.t = np.arange(0,len(self.data)/self.fps,1./self.fps)

    self.win = pg.GraphicsWindow()
    self.win.setWindowTitle('Streaming Data')

    self.p1 = self.win.addPlot()
    self.win.nextRow()
    self.p2 = self.win.addPlot()
    self.win.nextRow()
    self.p3 = self.win.addPlot()

    self.setDownsampling()
    self.plot_curves()

    self.slowdown = 2 # how much to slow down playback
    self.timer = pg.QtCore.QTimer()
    self.timer.timeout.connect(self.play)

  def setClipToView(self):
    self.p1.setClipToView(True)
    self.p2.setClipToView(True)
    self.p3.setClipToView(True)

  def setDownsampling(self):
    self.p1.setDownsampling(mode='peak')
    self.p2.setDownsampling(mode='peak')
    self.p3.setDownsampling(mode='peak')

  def set_trange(self):
    tmin = ( self.frame - self.frame_buffer ) / self.fps 
    tmax = ( self.frame + self.frame_buffer ) / self.fps 
    self.p1.setRange(xRange=[tmin,tmax])
    self.p2.setRange(xRange=[tmin,tmax])
    self.p3.setRange(xRange=[tmin,tmax])

  def plot_curves(self):
    self.set_trange()
    self.curve1 = self.p1.plot(self.t,self.data[:,0],pen=pg.mkPen('c',width=1))
    self.curve2 = self.p2.plot(self.t,self.data[:,1],pen=pg.mkPen('c',width=1))
    self.curve3 = self.p3.plot(self.t,self.data[:,2],pen=pg.mkPen('c',width=1))
    self.p2.setXLink(self.p1)
    self.p3.setXLink(self.p1)
    self.p1.showGrid(x=True,y=True)
    self.p2.showGrid(x=True,y=True)
    self.p3.showGrid(x=True,y=True)
    self.p1.setLabel('left','distance from wall','cm')
    self.p2.setLabel('left','speed','cm/s')
    self.p3.setLabel('left','angular velocity','rad/s')
    self.p3.setLabel('bottom', 'time', 's')

  def frame_fwd(self):
    self.timer.stop()
    self.frame += 1
    self.set_trange()
  
  def frame_bwd(self):
    self.timer.stop()
    self.frame -= 1
    self.set_trange()

  def play(self):
    self.frame_fwd()
    self.timer.start(self.slowdown / self.fps * 1e3) # ms 

  def pause(self):
    self.timer.stop()


class MainWidget(QtWidgets.QWidget):
    def __init__(self, data_source, parent=None):

        super().__init__(parent)
        self.data_stream = DataStream(data_source)

        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.data_stream)

        self.tslider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        layout.addWidget(self.tslider)

        buttons = QtWidgets.QHBoxLayout()

        self.bwd_button = QtWidgets.QPushButton('Bwd')
        self.bwd_button.clicked.connect(self.data_stream.frame_bwd)
        buttons.addWidget(self.bwd_button)
        self.bwd_key = QtWidgets.QShortcut(QtGui.QKeySequence("Left"), self)
        self.bwd_key.activated.connect(self.data_stream.frame_bwd)

        self.play_button = QtWidgets.QPushButton('Play')
        self.play_button.clicked.connect(self.data_stream.play)
        buttons.addWidget(self.play_button)
        self.play_key = QtWidgets.QShortcut(QtGui.QKeySequence("Up"), self)
        self.play_key.activated.connect(self.data_stream.play)

        self.pause_button = QtWidgets.QPushButton('Pause')
        self.pause_button.clicked.connect(self.data_stream.pause)
        buttons.addWidget(self.pause_button)
        self.pause_key = QtWidgets.QShortcut(QtGui.QKeySequence("Down"), self)
        self.pause_key.activated.connect(self.data_stream.pause)

        self.fwd_button = QtWidgets.QPushButton('Fwd')
        self.fwd_button.clicked.connect(self.data_stream.frame_fwd)
        buttons.addWidget(self.fwd_button)
        self.fwd_key = QtWidgets.QShortcut(QtGui.QKeySequence("Right"), self)
        self.fwd_key.activated.connect(self.data_stream.frame_fwd)

        self.quit_button = QtWidgets.QPushButton('Quit')
        self.quit_button.clicked.connect(quit)
        buttons.addWidget(self.quit_button)
        self.quit_key = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+q"), self)
        self.quit_key.activated.connect(quit)

        layout.addLayout(buttons)

        self.setLayout(layout)



def main(data_source):
    app = QtWidgets.QApplication(sys.argv)

    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget(data_source)
    main_window.setCentralWidget(main_widget)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
  video_source="/data1/cavefish/social/video/20190504_SF_SF_n5_t1_3051_track_mask.mp4"
  #video_source="/data1/cavefish/social/video/20190503_SF_SF_n1_t1_3001_track_mask.mp4"
  subdir = video_source.split('/')[-1].split('.')[0] + "_kinematics/kinematic_scatter.npy"
  data_source='/'.join(video_source.split('.')[0].split('/')[:-2]) + '/data/' + subdir 
    

main(data_source)
