import argparse, sys, os, time 
import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
#from PyQt5 import QtCore
from PyQt5 import QtWidgets
#from PyQt5 import QtGui

class CVPlayerControl(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, video_src, parent=None):
        super().__init__(parent)
        self.video = cv2.VideoCapture(video_src)
        self.im_factor = 1.5 
        self.timer = QtCore.QBasicTimer()
        self.slowdown = 2 
        self.fps = 30
        self.fnum = -1

    def load_video(self):
        self.get_frame()

    def get_frame(self):
        if self.video.isOpened():
            ret, self.frame = self.video.read()
            width, height, chan = self.frame.shape
            self.frame = cv2.resize(self.frame, ( int(width/self.im_factor), int(height/self.im_factor) ) , interpolation = cv2.INTER_LINEAR)
            self.fnum += 1
            if ret:
                self.image_data.emit(self.frame)
                cv2.waitKey(0)
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def frame_fwd(self):
        self.get_frame()

    def frame_bwd(self):
        self.fnum -= 2
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.fnum)
        self.get_frame()

    def set_frame(self,fnum):
        self.fnum = fnum-1
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.fnum)

    def mouseMoveEvent (self, eventQMouseEvent):
        self.x, self.y = eventQMouseEvent.x(), eventQMouseEvent.y()
        self.circle = cv2.circle(self.frame,(self.x,self.y),3,thickness=-1)

    def mouseClickEvent (self, eventQMouseEvent):
        self.x, self.y = eventQMouseEvent.x(), eventQMouseEvent.y()
        self.circle = cv2.circle(self.frame,(self.x,self.y),3,thickness=-1)


class CVPlayerWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QtGui.QImage()

    def image_data_slot(self, image_data):
        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())
        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()


class DataStream(QtWidgets.QWidget):

    def __init__(self, data_src, parent=None):
        super().__init__(parent)
        self.data = np.load(data_src)

        self.frame        =  0
        self.frame_buffer = 30
        self.fps          = 30
        self.nframes      = len(self.data)

        self.t = np.arange(0,len(self.data)/self.fps,1./self.fps)

        self.win = pg.GraphicsLayoutWidget()
        self.win.setWindowTitle('Streaming Data')

        self.p1 = self.win.addPlot()
        self.win.nextRow()
        self.p2 = self.win.addPlot()
        self.win.nextRow()
        self.p3 = self.win.addPlot()

        self.setDownsampling()
        self.plot_curves()

        self.slowdown = 2 # how much to slow down playback

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
        self.frame += 1
        self.set_trange()
    
    def frame_bwd(self):
        self.frame -= 1
        self.set_trange()

    def set_frame(self,fnum):
        self.frame = fnum
        self.set_trange()

    def get_fnum(self):
        return self.frame


class MainWidget(QtWidgets.QWidget):
    def __init__(self, video_src, data_src, parent=None):
        super().__init__(parent)

        self.timer = QtCore.QBasicTimer()
        self.fps = 30
        self.slowdown = 2

        self.data_stream = DataStream(data_src)

        self.cvp_widget = CVPlayerWidget() 
        self.cvp_control = CVPlayerControl(video_src)
        image_data_slot = self.cvp_widget.image_data_slot
        self.cvp_control.image_data.connect(image_data_slot)

        buttons = QtWidgets.QHBoxLayout()

        self.play_button = QtWidgets.QPushButton('Play')
        self.play_button.clicked.connect(self.play)
        buttons.addWidget(self.play_button)
        self.play_key = QtWidgets.QShortcut(QtGui.QKeySequence("Up"), self)
        self.play_key.activated.connect(self.play)

        self.pause_button = QtWidgets.QPushButton('Pause')
        self.pause_button.clicked.connect(self.pause)
        buttons.addWidget(self.pause_button)
        self.pause_key = QtWidgets.QShortcut(QtGui.QKeySequence("Down"), self)
        self.pause_key.activated.connect(self.pause)

        self.bwd_button = QtWidgets.QPushButton('Bwd')
        self.bwd_button.clicked.connect(self.frame_bwd)
        buttons.addWidget(self.bwd_button)
        self.bwd_key = QtWidgets.QShortcut(QtGui.QKeySequence("Left"), self)
        self.bwd_key.activated.connect(self.frame_bwd)


        self.fwd_button = QtWidgets.QPushButton('Fwd')
        self.fwd_button.clicked.connect(self.frame_fwd)
        buttons.addWidget(self.fwd_button)
        self.fwd_key = QtWidgets.QShortcut(QtGui.QKeySequence("Right"), self)
        self.fwd_key.activated.connect(self.frame_fwd)

        self.quit_button = QtWidgets.QPushButton('Quit')
        self.quit_button.clicked.connect(quit)
        buttons.addWidget(self.quit_button)
        self.quit_key = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+q"), self)
        self.quit_key.activated.connect(quit)

        cvp_w_buttons = QtWidgets.QVBoxLayout()
        cvp_w_buttons.addWidget(self.cvp_widget)
        cvp_w_buttons.addLayout(buttons)

        self.tslider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.tslider.setMinimum(0)
        self.tslider.setMaximum(self.data_stream.nframes)
        self.tslider.valueChanged[int].connect(self.slide)

        self.time_box = QtWidgets.QLineEdit(self)
        self.time_box.setText(self.t_hhmmsscs())
        self.time_box.setAlignment(QtCore.Qt.AlignRight)
        self.time_box.setFixedWidth(83)

        self.tslider_w_t = QtWidgets.QHBoxLayout()
        self.tslider_w_t.addWidget(self.time_box)
        self.tslider_w_t.addWidget(self.tslider)

        data_w_tslider = QtWidgets.QVBoxLayout()
        data_w_tslider.addWidget(self.data_stream.win)
        data_w_tslider.addLayout(self.tslider_w_t)

        cvp_and_data = QtWidgets.QHBoxLayout()
        cvp_and_data.addLayout(cvp_w_buttons)
        cvp_and_data.addLayout(data_w_tslider)

        self.setLayout(cvp_and_data)

        self.cvp_control.load_video()
        #self.setFocusPolicy(QtCore.Qt.StrongFocus)


    def t_curr_cs(self):
        return 1e2*float(self.data_stream.get_fnum()) / self.fps

    def t_hhmmsscs(self):
        t_ds = self.t_curr_cs()
        ds = t_ds % 1e2 
        ss = ( t_ds / 1e2 ) % 60 
        mm = ( t_ds / ( 1e2 * 60) ) % 60 
        hh = ( t_ds / ( 1e2 * 60 * 60 ) ) % 24 
        return "%02i:%02i:%02i.%02i" % (hh,mm,ss,ds)

    def step_fwd(self):
        if self.data_stream.frame < self.data_stream.nframes - 1:
          self.cvp_control.frame_fwd()
          self.data_stream.frame_fwd()
          self.tslider.setValue(self.data_stream.frame)
          self.time_box.setText(self.t_hhmmsscs())

    def step_bwd(self):
        if self.data_stream.frame > 0:
          self.cvp_control.frame_bwd()
          self.data_stream.frame_bwd()
          self.tslider.setValue(self.data_stream.frame)
          self.time_box.setText(self.t_hhmmsscs())

    def frame_fwd(self):
        self.timer.stop()
        self.step_fwd()
    
    def frame_bwd(self):
        self.timer.stop()
        self.step_bwd()
 
    def play(self):
        self.timer.start(self.slowdown / self.fps * 1e3,self) # ms 

    def pause(self):
        self.timer.stop()

    def slide(self, value):
        self.data_stream.set_frame(value)
        self.data_stream.update()
        self.cvp_control.set_frame(value)
        self.cvp_control.get_frame()
        self.time_box.setText(self.t_hhmmsscs())

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return
        self.step_fwd()
        self.time_box.setText(self.t_hhmmsscs())



def main():
    parser = argparse.ArgumentParser(description="Open-CV and Tracking Data Player")
    parser.add_argument("videofile", type=str, help="video file-path")
    parser.add_argument("-ds", "--datafile", type=str, help="data file-path, if not in standard location") 
    args = parser.parse_args()

    cwd = os.path.realpath(os.getcwd())
    vpath = os.path.realpath(args.videofile)
    video_src = os.path.join(cwd,vpath)
    print(video_src)
    if args.datafile:
        data_src = os.path.join(cwd,os.path.realpath(args.datafile)) 
    else:
        subdir = video_src.split('/')[-1].split('.')[0] + "_kinematics/kinematic_scatter.npy"
        data_src='/'.join(video_src.split('.')[0].split('/')[:-2]) + '/data/' + subdir 

    app = QtWidgets.QApplication(sys.argv)
    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget(video_src, data_src)
    main_window.setFocusPolicy(QtCore.Qt.StrongFocus)
    main_window.setCentralWidget(main_widget)
    main_window.show()
    sys.exit(app.exec_())


main()
