import sys
from os import path
import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui

video_source="/data1/cavefish/social/video/20190504_SF_SF_n5_t1_3051_track_mask.mp4"

class CVPlayerControl(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, video_source=video_source, parent=None):
        super().__init__(parent)
        self.video = cv2.VideoCapture(video_source)
        self.timer = QtCore.QBasicTimer()
        self.slowdown = 1 
        self.fps = 30
        self.fnum = -1
        self.width = 400
        self.height = 400

    def load_video(self):
        self.timer.stop()
        self.get_frame()

    def get_frame(self):
        if self.video.isOpened():
            ret, frame = self.video.read()
            resize = cv2.resize(frame, ( self.width, self.height ) , interpolation = cv2.INTER_LINEAR)
            self.fnum += 1
            if ret:
                self.image_data.emit(frame)
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(resize, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def get_frame_rev(self):
        self.fnum -= 2
        self.video.set(cv2.CAP_PROP_POS_FRAMES, self.fnum)
        return self.get_frame()

    def frame_fwd(self):
        self.timer.stop()
        self.get_frame()
    
    def frame_bwd(self):
        self.timer.stop()
        self.get_frame_rev()
 
    def play(self):
        self.timer.start(self.slowdown / self.fps * 1e3,self) # ms 

    def pause(self):
        self.timer.stop()

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return
        self.get_frame()


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



class MainWidget(QtWidgets.QWidget):
    def __init__(self, video_source, parent=None):
        super().__init__(parent)

        self.cvp_widget = CVPlayerWidget() 
        self.cvp_control = CVPlayerControl(video_source)

        image_data_slot = self.cvp_widget.image_data_slot
        self.cvp_control.image_data.connect(image_data_slot)

        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(self.cvp_widget)

        buttons = QtWidgets.QHBoxLayout()

        self.bwd_button = QtWidgets.QPushButton('Bwd')
        self.bwd_button.clicked.connect(self.cvp_control.frame_bwd)
        buttons.addWidget(self.bwd_button)
        self.bwd_key = QtWidgets.QShortcut(QtGui.QKeySequence("Left"), self)
        self.bwd_key.activated.connect(self.cvp_control.frame_bwd)

        self.play_button = QtWidgets.QPushButton('Play')
        self.play_button.clicked.connect(self.cvp_control.play)
        buttons.addWidget(self.play_button)
        self.play_key = QtWidgets.QShortcut(QtGui.QKeySequence("Up"), self)
        self.play_key.activated.connect(self.cvp_control.play)

        self.pause_button = QtWidgets.QPushButton('Pause')
        self.pause_button.clicked.connect(self.cvp_control.pause)
        buttons.addWidget(self.pause_button)
        self.pause_key = QtWidgets.QShortcut(QtGui.QKeySequence("Down"), self)
        self.pause_key.activated.connect(self.cvp_control.pause)

        self.fwd_button = QtWidgets.QPushButton('Fwd')
        self.fwd_button.clicked.connect(self.cvp_control.frame_fwd)
        buttons.addWidget(self.fwd_button)
        self.fwd_key = QtWidgets.QShortcut(QtGui.QKeySequence("Right"), self)
        self.fwd_key.activated.connect(self.cvp_control.frame_fwd)

        self.quit_button = QtWidgets.QPushButton('Quit')
        self.quit_button.clicked.connect(quit)
        buttons.addWidget(self.quit_button)
        self.quit_key = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+q"), self)
        self.quit_key.activated.connect(quit)

        layout.addLayout(buttons)


        self.setLayout(layout)

        self.cvp_control.load_video()
        #self.setFocusPolicy(QtCore.Qt.StrongFocus)


def main():
    app = QtWidgets.QApplication(sys.argv)

    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget(video_source)
    main_window.setCentralWidget(main_widget)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    video_source = path.abspath(video_source)

main()
