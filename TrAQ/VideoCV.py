import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

class VideoCV:

    def __init__(self, video_source, view_scale = 1, RGB = False):        
        self.fvideo_in      = video_source
        self.fvideo_ext     = "mp4"
        self.fvideo_out     = "cv.mp4"
        
        # initialize video playback details
        self.fps            = 30
        self.width          = 400
        self.height         = 400
        self.view_scale     = view_scale
        self.RGB            = RGB
        self.codec          = 'mp4v'
        if ( self.fvideo_ext == ".avi" ): self.codec = 'DIVX' 
        self.online_viewer  = False
        self.GPU            = False

        # initialize openCV video capture
        self.cap            = None
        self.frame          = None
        self.frame_num      = -1
        self.frame_start    = 0
        self.frame_end      = -1
        self.init_video_capture()
        
        # initialize contour working variables
        self.contours       = []
        self.contour_list   = []
        self.n_pix_avg      = 3
        self.block_size     = 15
        self.offset         = 13
        self.min_area       = 20
        self.max_area       = 1000
        self.thresh         = []
        
        # initialize lists for current and previous coordinates
        self.coord_now      = []
        self.coord_pre      = []


    ############################
    # cv2.VideoCapture functions
    ############################

    def init_video_capture(self):
        self.cap = cv2.VideoCapture(self.fvideo)
        if self.cap.isOpened() == False:
            sys.exit('Video file cannot be read! Please check input_vidpath ' 
                     + 'to ensure it is correctly pointing to the video file.')
        # Video writer class to output video with contour and centroid of tracked
        # object(s) make sure the frame size matches size of array 'final'
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        output_framesize = ( int(self.cap.read()[1].shape[1]*self.view_scale),
                             int(self.cap.read()[1].shape[0]*self.view_scale)  )

        self.out = cv2.VideoWriter( filename = self.fvideo_out, fourcc = fourcc, 
                                    fps = self.fps, frameSize = output_framesize, 
                                    isColor = self.RGB )
        
        if self.frame_end < 0:
            self.frame_end = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_start)


    def get_frame(self):
        if self.cap.isOpened():
            ret, self.frame = self.cap.read()
            if ret == True:
                self.frame_num += 1
                if self.online_viewer:
                    cv2.resize(self.frame, ( self.width, self.height ), 
                               interpolation = cv2.INTER_LINEAR)
                if self.GPU:
                    self.frame = cv2.UMat(self.frame)
                return True
        return False

    def write_frame(self):
        self.out.write(self.frame)
        if ( self.online_viewer ):
            cv2.imshow('frame', self.frame) 
            return_key = 13
            esc_key    = 27 
            space_key  = 32
            if cv2.waitKey(33) == esc_key:
                return 0
            elif cv2.waitKey(33) == space_key:
                while(True):
                    k = cv2.waitKey(33)
                    if k == return_key:
                        break
                    elif k == -1:
                        continue
        return 1


    ############################
    # Contour functions
    ############################

    def contour_detect(self):
        self.threshold_detect()
        # find all contours 
        img, contours, hierarchy = cv2.findContours( self.thresh, 
                                                     cv2.RETR_TREE, 
                                                     cv2.CHAIN_APPROX_SIMPLE )
        # test found contours against area constraints
        i = 0
        while i < len(self.contours):
            area = cv2.contourArea(self.contours[i])
            if area < self.min_area or area > self.max_area:
                del self.contours[i]
            else:
                i += 1

    def threshold_detect(self, hist = False):
        # blur and current image for smoother contours
        blur = cv2.GaussianBlur(self.frame, (self.n_pix_avg, self.n_pix_avg), 0)
        #blur = cv2.blur(frame, (n_pix_avg,n_pix_avg))
       
        # convert to grayscale
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    
        if hist:
            plt.title("Raw image grayscale histogram")
            plt.hist(self.frame.ravel()[self.frame.ravel() > 0],256)
            plt.show()
        
        
            plt.title("Blurred grayscale histogram")
            plt.hist(blur.ravel()[blur.ravel() > 0],256)
            plt.show()
        
            plt.title("Grayscale histogram")
            plt.hist(gray.ravel()[gray.ravel() > 0],256)
            plt.show()
    
        # calculate thresholds, more info:
        #   https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
        self.thresh = cv2.adaptiveThreshold( gray, 
                                             160, 
                                             cv2.ADAPTIVE_THRESH_MEAN_C, 
                                             cv2.THRESH_BINARY_INV,
                                             self.block_size, 
                                             self.offset )
        # NOTE: adaptiveThreshold(..., 160, ...) is ad-hoc following histograms
    
    def analyze_contours(self):
        self.coord_pre = self.coord_now.copy()
        del self.coord_now[:]
        self.coord_now = []
        for i in range(len(self.contours)):
            M = cv2.moments(self.contours[i])
    
            if M['m00'] != 0:
                cx = M['m10']/M['m00']
                cy = M['m01']/M['m00']
                mu20 = M['m20']/M['m00'] - pow(cx,2)
                mu02 = M['m02']/M['m00'] - pow(cy,2)
                mu11 = M['m11']/M['m00'] - cx*cy
            else:
            	cx = 0
            	cy = 0
            ry = 2*mu11
            rx = mu20-mu02
            theta = 0.5*np.arctan2(ry,rx)
            self.coord_now.append([cx,cy,theta])
        # then run the theta-correcting function to properly align    
        self.correct_theta()

    def correct_theta(self):
        for i in range(len(self.coord_now)):
            cx = self.coord_now[i][0]
            cy = self.coord_now[i][1]
            theta = self.coord_now[i][2]
            
            cx_p = self.coord_pre[i][0]
            cy_p = self.coord_pre[i][1]
            theta_p = self.coord_now[i][2]
            
            vx = cx - cx_p / 2
            vy = cy - cy_p / 2
            speed = np.sqrt(np.pow(vx,2) + np.pow(vy,2))
          
            # note speed here is in pixels, so this is somewhat arbitrary until
            # we know the drift speed in pixels/frame, but basically I will 
            pix_speed_min = 3
            if ( speed > pix_speed_min):
                dot_prod = vx * np.cos(theta) + vy * np.sin(theta) 
                if dot_prod < 0:
                    theta = np.mod(theta+2*np.pi,2*np.pi) - np.pi
            else:
                dot_prod = np.cos(theta_p) * np.cos(theta) + np.sin(theta_p) * np.sin(theta) 
                if dot_prod < 0:
                    theta = np.mod(theta+2*np.pi,2*np.pi) - np.pi
            
            self.coord_now[i][2] = theta 

    
    ############################
    # Masking functions
    ############################
    
    def tank_mask(self, tank):
        x_cm = int(tank.x_cm) + 1
        y_cm = int(tank.y_cm) + 1
        R = int(tank.R) + 1
        tank_mask = np.full_like(self.frame,0)
        tank_only = np.full_like(self.frame,0)
        cv2.circle(tank_mask,(x_cm,y_cm),R,(1,1,1),thickness=-1)
        tank_only[tank_mask==1] = self.frame[tank_mask==1] 
        return tank_only

    def contour_mask(self):
        self.contour_masks = []
        # add contour areas to mask
        for i in range(len(self.contours)):
            mask = np.zeros_like(self.frame)
            cv2.drawContours(mask, self.contours, i, 1, -1)
            self.contour_masks.append(mask)


    ############################
    # Drawing functions
    ############################
    
    def draw_all(self):
        self.draw_tstamp()
        self.draw_contours()
        self.draw_points()
        self.draw_directors()

    def draw_tstamp(self):
        if self.RGB:
            color = (0,0,0)
        else:
            color = 0
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        t_str = "%02i:%02i:%02i.%02i" % ( 
                int(self.frame_num / self.fps / 3600),
                int(self.frame_num / self.fps / 60 % 60),
                int(self.frame_num / self.fps % 60),
                int(self.frame_num % self.fps * 100 / self.fps) )
        
        cv2.putText(self.frame, t_str, (5,30), font, 1, color, 2)
    
    def draw_tank(self, tank):
        if RGB:
            cv2.circle(self.frame, (int(tank.x_cm), int(tank.y_cm)), int(tank.R),
                       (0,0,0), thickness=7)
        else:
            cv2.circle(self.frame, (int(tank.x_cm), int(tank.y_cm)), int(tank.R),
                       0, thickness=7)    

    def draw_contours(self):
        if self.RGB:
            color = (0,0,0)
        else:
            color = 0
        # draw contours in list 
        for i in self.contour_list: 
            cv2.drawContours(self.frame, self.contours, i, color, 2)

    def draw_points(self):
        self.coord_now = np.array(self.coord_now)
        for i in range(len(self.coord_now)):
          if self.RGB:
            cv2.circle(self.frame, tuple([int(x) for x in self.coord_now[i,(0,1)]]), 
                       3, self.colors[i%len(self.colors)], -1, cv2.LINE_AA)
          else:
            cv2.circle(self.frame, (int(self.coord_now[i][0]), int(self.coord_now[i][1])), 
                       3, 0, -1)
        self.coord_now = list(self.coord_now)

    def draw_directors(self):
        self.coord_now = np.array(self.coord_now)
        self.coord_pre = np.array(self.coord_pre)
        vsize = 7
        for i in range(len(self.coord_now)):
            cx = self.coord_now[i][0]
            cy = self.coord_now[i][1]
            theta = self.coord_now[i][2]

            # coordinates for director line segement
            x0, y0 = int(cx - vsize*np.cos(theta)), int(cy - vsize*np.sin(theta))
            x1, y1 = int(cx + vsize*np.cos(theta)), int(cy + vsize*np.sin(theta))    

            if self.RGB:
                cv2.line(self.frame, (x0, y0), (x1, y1), self.colors[i%len(self.colors)], 2)
                cv2.circle(self.frame, (x1, y1), 3, self.colors[i%len(self.colours)], -1)
            else:
                
                cv2.line(self.frame, (x0, y0), (x1, y1), 0, 2)
                cv2.circle(self.frame, (x1, y1), 3, 255, -1)

