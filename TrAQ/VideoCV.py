import sys
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from TrAQ.Trial import Trial
from Analysis.Math import angle_diff


class VideoCV:

    def __init__(self, trial, frame_start = 0, frame_end = -1, 
                 n_pixel_blur = 3, block_size = 15, threshold_offset = 13, 
                 min_area = 20, max_area = 400, len_trail = 3,
                 RGB = False, online = False, view_scale = 1, GPU = False ):
        
        self.fvideo_in      = trial.fvideo_raw
        self.fvideo_ext     = "mp4"
        self.fvideo_out     = os.path.abspath("%s/traced.mp4" % trial.fdir)
        trial.fvideo_out    = self.fvideo_out
        
        # initialize video playback details
        self.view_scale     = view_scale
        self.RGB            = RGB
        self.codec          = 'mp4v'
        if ( self.fvideo_ext == ".avi" ): self.codec = 'DIVX' 
        self.online_viewer  = online
        self.GPU            = GPU

        # initialize openCV video capture
        self.cap            = None
        self.frame          = None
        self.frame_num      = -1
        self.frame_start    = frame_start
        self.frame_end      = frame_end
        self.fps            = trial.fps
        self.init_video_capture()
        
        # initialize contour working variables
        self.contours       = []
        self.contour_list   = []
        self.contour_repeat = []
        self.n_pix_avg      = n_pixel_blur
        self.block_size     = block_size
        self.offset         = threshold_offset
        self.min_area       = min_area
        self.max_area       = max_area
        self.thresh         = []

        # initialize lists for current and previous coordinates
        self.n_ind          = trial.n
        self.coord_now      = []
        self.coord_pre      = []
        self.ind_pre_now    = []
        self.trail          = []
        self.len_trail      = 3
        
        # choose whether to randomize or preset color list,
        #     (not necessary if running grayscale)
        self.preset_color_list()
        
        
        
    def preset_color_list(self):
        self.colors             = [ (   0,   0, 255),
                                    (   0, 255, 255),
                                    ( 255,   0, 255),
                                    ( 255, 255, 255),
                                    ( 255, 255,   0),
                                    ( 255,   0,   0),
                                    (   0, 255,   0),
                                    (   0,   0,   0) ]

    def random_color_list(self):
        self.colors = []
        low = 0
        high = 255
        print("    Randomizing colors for VideoCV object...")
        for i in range(self.n_ind):
            color = []
            for i in range(3):
                color.append(np.uint8(np.random.random_integers(low,high)))
            self.colors.append((color[0],color[1],color[2]))
        print("    Color list set as follows, ")
        print(self.colors)


    ############################
    # cv2.VideoCapture functions
    ############################

    def init_video_capture(self):
        self.cap = cv2.VideoCapture(self.fvideo_in)
        if self.cap.isOpened() == False:
            sys.exit('Video file cannot be read! Please check input_vidpath ' 
                     + 'to ensure it is correctly pointing to the video file.')
            
        # Video writer class to output video with contour and centroid of tracked
        # object(s) make sure the frame size matches size of array 'final'
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        output_framesize = ( int(self.cap.read()[1].shape[1]*self.view_scale),
                             int(self.cap.read()[1].shape[0]*self.view_scale)  )
        self.width = output_framesize[0]
        self.height = output_framesize[1]

        self.out = cv2.VideoWriter( filename = self.fvideo_out, fourcc = fourcc, 
                                    fps = self.fps, frameSize = output_framesize, 
                                    isColor = self.RGB )
        
        self.frame_num = self.frame_start
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)
        if self.frame_end < 0:
            self.frame_end = self.n_frames()

    def release(self):
        # release the capture
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        sys.stdout.write("\n")
        sys.stdout.write("       Video capture released.\n")
        sys.stdout.flush()

    def tstamp(self):
        return float(self.frame_num)/self.fps

    def n_frames(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    
    def tracked_frames(self):
        return self.frame_num - self.frame_start

    def set_frame(self, i):
        self.frame_num = i
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)

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
        if self.GPU:
            self.frame = cv2.UMat.get(self.frame)
        
        self.out.write(self.frame)
        
    def show_frame(self):
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

    def print_current_frame(self):
        t_csc = int(self.frame_num/self.fps * 100)
        t_sec = int(t_csc/100) 
        t_min = t_sec/60
        t_hor = t_min/60
        sys.stdout.write("       Current tracking time: %02i:%02i:%02i:%02i \r" 
                         % (t_hor, t_min % 60, t_sec % 60, t_csc % 100) )
        sys.stdout.flush()
        

    ############################
    # Contour functions
    ############################

    def detect_contours(self):
        self.threshold_detect()
        # find all contours 
        self.contours, hierarchy = cv2.findContours( self.thresh, 
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
        self.coord_now = []
        for contour in self.contours:
            M = cv2.moments(contour)
    
            if M['m00'] != 0:
                cx   = M['m10'] / M['m00']
                cy   = M['m01'] / M['m00']
                mu20 = M['m20'] / M['m00'] - pow(cx,2)
                mu02 = M['m02'] / M['m00'] - pow(cy,2)
                mu11 = M['m11'] / M['m00'] - cx*cy
            else:
            	cx = 0
            	cy = 0
            ry = 2 * mu11
            rx = mu20 - mu02
            theta = 0.5 * np.arctan2(ry, rx)
            self.coord_now.append([cx, cy, theta])
        
    def correct_theta(self):
        if len(self.trail) < 1:
            return

        for i in range(len(self.coord_now)):
            cx    = self.coord_now[i][0]
            cy    = self.coord_now[i][1]
            theta = self.coord_now[i][2]
            
            cx_p    = self.coord_pre[i][0]
            cy_p    = self.coord_pre[i][1]
            theta_p = self.coord_pre[i][2]
            
            vx = ( cx - cx_p ) / 2
            vy = ( cy - cy_p ) / 2
            speed = np.sqrt(math.pow(vx,2) + math.pow(vy,2))
          
            # note speed here is in pixels, so this is somewhat arbitrary until
            # we know the drift speed in pixels/frame, but basically I will 
            pix_speed_min = 1
            if ( speed > pix_speed_min):
                dot_prod = vx * np.cos(theta) + vy * np.sin(theta) 
                if dot_prod < 0:
                    theta = np.mod( theta + np.pi, 2*np.pi )
            else:
                dot_prod = np.cos(theta_p) * np.cos(theta) + np.sin(theta_p) * np.sin(theta) 
                if dot_prod < 0:
                    theta = np.mod( theta + np.pi, 2*np.pi)
            self.coord_now[i][2] = theta 

    # kmeans_contours uses contour traces and runs clustering algorithm on all 
    # of them to best locate different individuals, works OK for small-n_ind
    def kmeans_contours(self):
        # convert contour points to arrays
        clust_points = np.vstack(self.contours)
        clust_points = clust_points.reshape(clust_points.shape[0], clust_points.shape[2])
        # run KMeans clustering
        kmeans_init = 50
        kmeans = KMeans( n_clusters = self.n_ind, 
                         random_state = 0, 
                         n_init = kmeans_init ).fit(clust_points)
        theta = -13
        del self.coord_now[:]
        for cc in kmeans.cluster_centers_:
            x = int(tuple(cc)[0])
            y = int(tuple(cc)[1])
            self.coord_now.append([x,y,theta])
    
    def trail_update(self):
        self.trail.append(self.coord_now)
        if len(self.trail) > self.len_trail:
            self.trail.pop(0)
    
    
    # calculate predicted trajectory based on previous three points 
    def predict_next(self):
        prediction = self.coord_pre.copy()
        for i in range(len(self.coord_pre)):
            prediction[i][0] = ( self.trail[-1][i][0] 
                                    + ( 3*self.trail[-1][i][0] 
                                      - 2*self.trail[-2][i][0] 
                                      -   self.trail[-3][i][0] ) / 4. )
            prediction[i][1] = ( self.trail[-1][i][1] 
                                    + ( 3*self.trail[-1][i][1] 
                                      - 2*self.trail[-2][i][1] 
                                      -   self.trail[-3][i][1] ) / 4. )
        return prediction
    
    # in the absence of contours, if there is a trail, make a guess based on
    # last few frames in trail,
    def guess(self):
        self.coord_now = self.predict_next()
    
    def contour_connect(self):
        self.coord_pre = self.predict_next()
        xy_pre = np.array(self.coord_pre)[:,[0,1]]
        xy_now = np.array(self.coord_now)[:,[0,1]]
        # calculate the distance matrix and then look  
        dist_arr = cdist(xy_pre,xy_now)
        self.ind_pre_now = [ 0 for i in range(len(self.coord_pre)) ]
        for i in range(len(self.coord_pre)):
            index = np.argmin(dist_arr[i]) # i'th array has distances to predictions from pre
            self.ind_pre_now[i] = index # for i, what is corresponding index of pre (j)
    
        # identify any unclaimed contours
        contourID_unclaimed = []
        for i in range(len(self.coord_now)):
            for j in range(len(self.coord_pre)):
                if i == self.ind_pre_now[j]:
                    break # exits current for loop, contour is located
                if j == len(self.coord_pre) - 1:
                    contourID_unclaimed.append(i)
    
        # choose the closest contour to guess even if two choose the same
        for i in range(len(contourID_unclaimed)):
            index = np.argmin(dist_arr[:,contourID_unclaimed[i]])
            self.ind_pre_now[index] = contourID_unclaimed[i]
    
        # identify contours containing multiple points
        self.contour_repeat = []
        contour_deficit = len(self.coord_pre) - len(self.coord_now)
        sorted_ind_pre_now = sorted(self.ind_pre_now)
        i = 0  
        while len(self.contour_repeat) < contour_deficit:
            i += 1
            if sorted_ind_pre_now[i] == sorted_ind_pre_now[i-1]:
                self.contour_repeat.append(sorted_ind_pre_now[i])
        # then run the theta-correcting function to properly align



    def reorder_hungarian(self):
        # try this out... do hungarian algorithm on prediction of next step
        # ... just remove if it is making a mess...
        #self.coord_pre = self.predict_next()
        # reorder contours based on results of the hungarian algorithm 
        # (originally in tracktor, but i've reorganized)
        row_ind, col_ind = self.hungarian_algorithm()
        equal = np.array_equal(col_ind, list(range(len(col_ind))))
        if equal == False:
            current_ids = col_ind.copy()
            reordered = [i[0] for i in sorted(enumerate(current_ids), key=lambda x:x[1])]
            self.coord_now = [x for (y,x) in sorted(zip(reordered,self.coord_now))]
    
    def hungarian_algorithm(self):
        xy_pre = np.array(self.coord_pre)[:,[0,1]]
        xy_now  = np.array(self.coord_now)[:,[0,1]]    
        xy_pre = list(xy_pre)
        xy_now  = list(xy_now)
        cost = cdist(xy_pre, xy_now)
        row_ind, col_ind = linear_sum_assignment(cost)
        return row_ind, col_ind


    def reorder_connected(self):
        meas_now_tmp = self.coord_now
        self.coord_now = self.coord_pre
        for i in range(len(self.coord_now)):
            self.coord_now[i] = meas_now_tmp[self.ind_pre_now[i]]
        


    ############################
    # Masking functions
    ############################
    
    def mask_tank(self, tank):
        row_c = int(tank.row_c) + 1
        col_c = int(tank.col_c) + 1
        R = int(tank.r) + 1
        if self.RGB:
            tank_mask = np.zeros_like(self.frame)
        else:
            tank_mask = cv2.UMat(np.zeros_like(self.frame))
        
        cv2.circle(tank_mask, (row_c,col_c), R, (255, 255, 255), thickness=-1)
        self.frame = cv2.bitwise_and(self.frame, tank_mask)

    def mask_contours(self):
        self.contour_masks = []
        # add contour areas to mask
        for i in range(len(self.contours)):
            mask = np.zeros_like(self.frame, dtype=bool)
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
        if self.RGB:
            cv2.circle(self.frame, (int(tank.row_c), int(tank.col_c)), int(tank.r),
                       (0,0,0), thickness=7)
        else:
            cv2.circle(self.frame, (int(tank.row_c), int(tank.col_c)), int(tank.r),
                       0, thickness=7)    

    def draw_contour_repeat(self):
        if self.RGB:
            color = (0,0,0)
        else:
            color = 0
        # draw contours in list 
        for i in self.contour_list: 
            cv2.drawContours(self.frame, self.contour_repeat, i, color, 2)

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
                cv2.circle(self.frame, (x1, y1), 3, self.colors[i%len(self.colors)], -1)
            else:
                
                cv2.line(self.frame, (x0, y0), (x1, y1), 0, 2)
                cv2.circle(self.frame, (x1, y1), 3, 255, -1)

