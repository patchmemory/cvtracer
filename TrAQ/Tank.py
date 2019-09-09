import sys
import os
import argparse
import numpy as np
import cv2
import TrAQ.Trace as Trace

class Tank:

  def __init__(self):
    self.points = np.zeros((3,2))    
    self.n_point = 0
    self.row_c = 0
    self.col_c = 0
    self.r = 0
    self.found = False

  def add_point(self,x,y):
    if self.n_point > 2:
      if self.n_point == 3: 
        print("    Note: Only green points are used for calculation.")
      x_tmp, y_tmp = self.points[self.n_point%3][0], self.points[self.n_point%3][1]
      cv2.circle(frame, (int(x_tmp), int(y_tmp)), 4, (0, 0, 255), -1)
      cv2.imshow('image',frame)
      cv2.waitKey(cv2.EVENT_LBUTTONUP)
    self.points[self.n_point%3] = [x,y]
    self.n_point += 1
    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
    cv2.imshow('image',frame)
    cv2.waitKey(cv2.EVENT_LBUTTONUP)
    if self.n_point > 2:
      self.calculate_circle()
      print("    Locating tank edges... ")

  def calculate_circle(self):
    midpoint = []
    m = []
    b = []
    for i in range(2):
      midpoint.append([(self.points[i+1][0]+self.points[i][0])/2,
                       (self.points[i+1][1]+self.points[i][1])/2])
      slope = ((self.points[i+1][1]-self.points[i][1])/
                   (self.points[i+1][0]-self.points[i][0]))
      m.append(-1./slope)
      b.append(midpoint[i][1]-m[i]*midpoint[i][0])
 
    self.row_c = (b[1]-b[0])/(m[0]-m[1])
    self.col_c = m[0]*self.row_c + b[0]
    self.r = np.sqrt(pow(self.row_c-self.points[0][0],2) + pow(self.col_c-self.points[0][1],2))
    self.found = True
  
def add_circle_point(event,x,y,flags,param):
  if event == cv2.EVENT_LBUTTONDOWN:
    tank.add_point(x,y)

def select_circle(event,x,y,flags,param):
  if event == cv2.EVENT_LBUTTONDOWN:
    px,py = x,y
    if np.sqrt(pow(px-tank.row_c,2)+pow(py-tank.col_c,2)) > tank.r:
      tank.found = False
      cv2.circle(frame, (int(tank.row_c), int(tank.col_c)), int(tank.r), (0, 0, 255), -1)
      cv2.imshow('image',frame)
      cv2.waitKey(0)
    else:
      cv2.circle(frame, (int(tank.row_c), int(tank.col_c)), int(tank.r), (0, 255, 0), -1)
      cv2.imshow('image',frame)
      cv2.waitKey(0)


def locate_tank():
    
    args = Trace.arg_parse()
    
    fps = args.frames_per_second
    tank_radius_cm = args.tank_diameter/2.
    frame_start = int(args.t_start * fps)
    frame_end   = int(args.t_end   * fps)
    
    home_path = os.path.realpath(args.work_dir)
    input_loc = os.path.realpath(args.raw_video)
    data_output_dir = "data/"
    video_output_dir = "video/"
    output_str = "cv"
    input_vidpath, output_vidpath, output_filepath, output_text, codec = Trace.organize_filenames(
                                    home_path,input_loc,video_output_dir,data_output_dir,output_str)
    tank_info_file = "%s_tank.dat" % (output_filepath.split('.')[0])
    
    tank = Tank()
    
    try:
        f = open(tank_info_file,'r')
        f.readline()
        row_col_rad = f.readline().split()
        tank.row_c = float(row_col_rad[0])
        tank.col_c = float(row_col_rad[1])
        tank.r     = float(row_col_rad[2])
    except FileNotFoundError:
        cap = cv2.VideoCapture(input_vidpath)
        if cap.isOpened() == False:
            sys.exit('Video file cannot be read! Please check input_vidpath to ensure it is correctly pointing to the video file.')
      
        if frame_end < 0:
            frame_end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
      
        while True:
            # open frame
            i_frame = int(np.random.uniform(frame_start,frame_end))
            cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame-1)
            ret, frame = cap.read()
            if ret == True:
                # put frame in grayscale Transparent API
                frame = cv2.UMat(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.UMat(gray)
                # try locating the tank
                cv2.namedWindow('image')
                cv2.setMouseCallback('image',add_circle_point)
                cv2.imshow('image', frame)
                cv2.waitKey(0)
                # show results and allow user to choose if it looks right
                cv2.circle(frame, (int(tank.row_c), int(tank.col_c)), int(tank.r), (0, 255, 0), 4)
                cv2.circle(frame, (int(tank.row_c), int(tank.col_c)), 5, (0, 255, 0), -1)
                cv2.setMouseCallback('image',select_circle)
                cv2.imshow('image',frame)
                cv2.waitKey(0)
                # if the user decides the tank location is good, then exit loop
                if tank.found:
                    break
                else:
                    continue
      
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
      
        f = open(tank_info_file,'w')
        f.write("#row col radius\n")
        f.write("%f %f %f\n" % (tank.row_c, tank.col_c, tank.r) )
        f.close()
      
        sys.stdout.write("\n")
        sys.stdout.write("       Tank detection complete.\n")
        sys.stdout.flush()

    if calculate_kinematics:
        q = np.load(output_filepath)
        sys.stdout.write("\n")
        sys.stdout.write("       Converting pixels to (x,y) space in (cm,cm).\n")
        sys.stdout.flush()
        for i in range(len(q)):
            q[i].convert_pixels(tank.row_c,tank.col_c,tank.r,tank_radius_cm)
            q[i].tstamp_reformat(fps)
        sys.stdout.write("\n")
        sys.stdout.write("       %s converted according to tank size and location.\n" % output_filepath)
        sys.stdout.flush()
        
        sys.stdout.write("\n")
        sys.stdout.write("       Calculating kinematics...\n")
        sys.stdout.flush()
        for i in range(len(q)):
            print("         Fish %2i" % (i+1)) 
            q[i].calculate_dwall(tank_radius_cm)
            q[i].calculate_velocity(fps)
            q[i].calculate_acceleration(fps)
            q[i].calculate_director(fps)
            q[i].calculate_angular_velocity(fps)
            q[i].calculate_angular_acceleration(fps)
            q[i].calculate_local_acceleration(fps)
    
        np.save(output_filepath,q)
      
        sys.stdout.write("\n")
        sys.stdout.write("       %s kinematic quantities have been calculated.\n" % output_filepath)
        sys.stdout.flush()



def arg_parse():
    parser = argparse.ArgumentParser(description="OpenCV2 Tank Detector")
    parser.add_argument("raw_video", type=str, help="path to raw video")
    #parser.add_argument("-c","--convert", help="convert pixels", action='store_true') 
    parser.add_argument("-ck","--calculate_kinematics", help="calculate kinematics after conversion", action='store_true') 
    parser.add_argument("-td","--tank_diameter", type=float, help="tank diameter", default=111.)
    parser.add_argument("-ts","--t_start", type=float, help="start time, in seconds", default=0)
    parser.add_argument("-te","--t_end", type=float, help="end time, in seconds", default=-1)
    parser.add_argument("-fps","--frames_per_second", type=float, help="frames per second in raw video",default=30)
    parser.add_argument("-wd","--work_dir", type=str, help="root working directory if not current",default=os.getcwd())
    parser.add_argument("-vs","--view_scale", type=float, help="factor to scale viewer to fit window", default=1)
    return parser.parse_args()

# locates edges of a circular tank using radius guess and contours
def tank_detect(frame, tank_R_guess = 465, min_area = 1000, max_area = 1e7):
    # return a list 'tank_measure' with entries (radius, x_cm, y_cm) in pixels
    tank_measure = [] 
    # first apply threshold (input parameters tuned adhoc)
    n_pix = 3
    block_size = 151
    offset = 1
    thresh = threshold_detect(frame, n_pix, block_size, offset) 
    # next find contours
    img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # then choose the contour/s that best fit with tank_R_guess
    i = 0
    while i < len(contours):
        area = cv2.contourArea(contours[i])
        if area < min_area or area > max_area:
            del contours[i]
        else:
            (tank_xcm, tank_ycm), tank_R = cv2.minEnclosingCircle(contours[i])   
            if abs(tank_R - tank_R_guess) < 0.1 * tank_R_guess:
                tank_measure.append(np.array([tank_R,tank_xcm,tank_ycm]))
            i += 1
  # return the list, ideally of length 1
  return tank_measure


def tank_detect_avg(tank_info, n_samples):
  tank_info      = np.array(tank_info)
  tank_R_avg     = np.mean(tank_info[:,0])
  tank_R_err     = np.std(tank_info[:,0])/np.sqrt(len(tank_info)-1)
  tank_x_com_avg = np.mean(tank_info[:,1])
  tank_x_com_err = np.std(tank_info[:,1])/np.sqrt(len(tank_info)-1)
  tank_y_com_avg = np.mean(tank_info[:,2])
  tank_y_com_err = np.std(tank_info[:,2])/np.sqrt(len(tank_info)-1)
  print("       Tank center and radius estimated from %i random samples.\n" % (n_samples) )
  print("         R = %0.2e +/- %0.2e pixels" % (tank_R_avg, tank_R_err) )
  print("         x = %0.2e +/- %0.2e pixels" % (tank_x_com_avg, tank_x_com_err) )
  print("         y = %0.2e +/- %0.2e pixels" % (tank_y_com_avg, tank_y_com_err) )
  print("\n")
  return tank_x_com_avg, tank_y_com_avg, tank_R_avg
