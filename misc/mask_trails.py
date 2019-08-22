##################################################################################
#
#  this is a hacked version of tracktor, first composed by Vivekh Sridhar of
#  the Couzin lab, the base be found here: github.com/vivekhsridhar/tracktor 
#  
#  using Sridhar's tracktor as a starting point, i have tuned parameters and
#  modifed several features adhoc to study collective behavior of adult and
#  larval astyanax mexicanus ---adam patch, fau jupiter, jan 2019
#
##################################################################################
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/data1/cavefish/social/python/tracktor')
#import tracktor as tr
import tracktor_revised as tr
import cv2
import scipy.signal
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from dataclasses import dataclass
import matplotlib.pyplot as plt

def pause():
  programPause = input("<return> to continue...")

if ( len(sys.argv) != 10 ):
  print("  Usage: %s <home_path> <raw_video> <block> <offset> <viewer [0/1]> <nsamples> <n_ind> <t_start> <t_end>" % sys.argv[0])
  print("\n    suggested values: \n")
  print("          block size = 15")
  print("              offset = 13\n")
  exit()

RGB = False 
if ( RGB ):
  color = tr.colours

n_ind = int(sys.argv[7])
#frame_start = 0
fps=30
tank_D_cm = 111.
tank_R_cm = tank_D_cm/2.
t_start = float(sys.argv[8])
t_end = float(sys.argv[9])
frame_start = int(t_start * fps)
frame_end = int(t_end * fps)

@dataclass
class Kinematic:
  x: float
  y: float
  vx: float
  vy: float
  ax: float
  ay: float
  theta: float
  omega: float
  alpha: float


# the scaling parameter can be used to speed up tracking if video resolution is
# too high (use value 0-1)
scaling = 1.0

# this is the block_size and offset used for adaptive thresholding (block_size
# should always be odd) these values are critical for tracking performance
block_size = int(sys.argv[3])
offset     = int(sys.argv[4])

# the scaling parameter can be used to speed up tracking if video resolution is
# too high (use value 0-1)
scaling = 1.0

# name of source video and paths
home_path = sys.argv[1]
input_loc = sys.argv[2]
data_output_dir = "data/"
video_output_dir = "video/"
output_str = "masked_tracked"
input_vidpath, output_vidpath, output_filepath, output_text, codec = tr.organize_filenames(
                                      home_path,input_loc,video_output_dir,data_output_dir,output_str)

tr.print_title(n_ind,input_vidpath,output_vidpath,output_filepath,output_text)

cap = cv2.VideoCapture(input_vidpath)
if cap.isOpened() == False:
  sys.exit('Video file cannot be read! Please check input_vidpath to ensure it is correctly pointing to the video file.')

# Video writer class to output video with contour and centroid of tracked
# object(s) make sure the frame size matches size of array 'final'
fourcc = cv2.VideoWriter_fourcc(*codec)
output_framesize = ( int(cap.read()[1].shape[1]*scaling),
                     int(cap.read()[1].shape[0]*scaling)  )

if ( RGB ):
  out = cv2.VideoWriter( filename = output_vidpath, 
                           fourcc = fourcc, 
                              fps = fps, 
                        frameSize = output_framesize, 
                          isColor = True )
else:
  out = cv2.VideoWriter( filename = output_vidpath, 
                           fourcc = fourcc, 
                              fps = fps, 
                        frameSize = output_framesize, 
                          isColor = False )

# online viewer code block
view_online = bool(int(sys.argv[5]))


# this first loop provides a preliminary look at the tank by
#   1. randomly selecting a set a frames to fit circle to tank edges
#   2. determining the size of   
trail_count = []
tank_info = []
n_tank_meas = int(sys.argv[6])
while True:
  # exit loop when enough measurements have been made
  if ( len(tank_info) > n_tank_meas): break 

  # capture random frame within range of time 
  i_frame = int(np.random.uniform(frame_start,frame_end))
  cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame-1)
  ret, frame = cap.read()
  if ret == True:

    # values found from visual test, would be better to point and click in a GUI
    tank_R_guess = 465
    min_area_tank = 1e3
    max_area_tank = 1e7
    tank_info_tmp = tr.tank_detect(frame, tank_R_guess, min_area_tank, max_area_tank)

    if ( len(tank_info_tmp) > 0 ):

      for tank_measure in tank_info_tmp:
        tank_info.append(tank_measure)
        if ( view_online ):
          cv2.circle( frame, (int(tank_measure[1]),int(tank_measure[2])),
                              int(tank_measure[0]), (0,255,0), 5 )
 
      if ( view_online ):
        cv2.imshow('frame', frame) # requires cv2.waitKey() to follow 
        space_key=32
        return_key=13
        esc_key=27 # code for escape key to break (only works in view-finder)
        if cv2.waitKey(33) == esc_key:
          break
        elif cv2.waitKey(33) == space_key:
          while(True):
            k = cv2.waitKey(33)
            if k == return_key:
              break
            elif k == -1:
              continue

# quick analysis of tank measurement
tank_info = np.array(tank_info)
tank_R_avg = np.mean(tank_info[:,0])
tank_R_err = np.std(tank_info[:,0])/np.sqrt(len(tank_info)-1)
tank_xcm_avg = np.mean(tank_info[:,1])
tank_xcm_err = np.std(tank_info[:,1])/np.sqrt(len(tank_info)-1)
tank_ycm_avg = np.mean(tank_info[:,2])
tank_ycm_err = np.std(tank_info[:,2])/np.sqrt(len(tank_info)-1)

print("%i %e %e %e %e %e %e" % (n_tank_meas, tank_R_avg, tank_R_err, tank_xcm_avg, tank_xcm_err, tank_ycm_avg, tank_ycm_err ) )

##################################
#                                #
#  tank detection now complete!  #
#                                #
##################################


# next step is to locate the fish using contours
# here is a way to keep a trail of the previous moments
trail_count = np.zeros(output_framesize)
#trail_count = cv2.cvtColor(trail_count, cv2.COLOR_RGB2GRAY)
#print(trail_count.shape)

q = []
contour_count = []
directors = False
tracking = True

if directors:
  meas_last = list(np.zeros((n_ind,3)))
  meas_now = list(np.zeros((n_ind,3)))
else:
  meas_last = list(np.zeros((n_ind,2)))
  meas_now = list(np.zeros((n_ind,2)))

len_trail = int(.2 * fps) # 1/2 second tail
mask_list = []
contour_list = []
contourID_repeat = []
contourID_unclaimed = []
for i_frame in range(frame_start,frame_end):
  frame_count = i_frame - frame_start
  cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame-1)
  ret, frame = cap.read()
  if ret == True:
    if int(scaling) != 1:
      frame = cv2.resize( frame, None,
                          fx = scaling,
                          fy = scaling,
                          interpolation = cv2.INTER_LINEAR )

    # remove everything outside of water region 
    tank_only  = tr.tank_mask(frame, tank_xcm_avg, tank_ycm_avg, tank_R_avg)

    # find contours (note: following parameters have been tuned adhoc)
    min_area = 20   # min and max area should be changed according to 
    max_area = 1000 # fish size and the camera resolution
    #block_size = 15 # block size and offset should be changed
    #offset     = 13 # depending on the fish and quality of recording
    n_pix = 5
    contours = tr.contour_detect(tank_only,min_area,max_area,block_size,offset,n_pix)
    n_contour = len(contours)
    contour_count.append(n_contour)
    #tr.threshold_detect_hist(tank_only,n_pix,block_size,offset)
    

    meas_last, meas_now = tr.points_detect(contours, meas_last, meas_now)
    
    if (tracking):
      if n_contour == n_ind:
        # do a simple reordering based on hugarian min-dist algorithm
        meas_last, meas_now = tr.reorder_hungarian(meas_last,meas_now)
      else:
        print("n_contours = %i" % n_contour)
        if n_contour == 0:
          meas_last, meas_now = tr.temporary_guess(q,meas_last,meas_now)
        elif len(q) < 3 or n_ind < 3:   # for initial frames, make "educated guesses"
          meas_now = tr.kmeans_contours(contours, n_ind, meas_now)
          meas_last, meas_now = tr.reorder_hungarian(meas_last,meas_now)
        else:
          # if have previous info, do something smarter to connect overlapping
          # fish that may be in one combined contour
          ind_last_now, contourID_repeat, contourID_unclaimed = \
                tr.contour_connect(q, n_ind, meas_now, meas_last, contours)
          # then reorder according to connections
          meas_last, meas_now = tr.reorder_connected(meas_last, meas_now, ind_last_now) 
    

    #####
    # trail mask maintenance
    # find current mask and add to list of masks
    mask = tr.contour_mask_binary(tank_only,contours)
    mask_list.append(mask)
    # add current mask to the trail count
    trail_count[mask == 1] += 1
    # add current contours to the contour list (for each frame)
    contour_list.append(contours)
    # update the lists by removing outdated frame (trail and contours)
    if frame_count > len_trail:
      trail_count[mask_list.pop(0) == 1] -= 1
      contour_list.pop(0)

    #comp_mask = cv2.cvtColor(tank_only, cv2.COLOR_BGR2GRAY)
    #comp_mask[mask != 1] = 0
    #plt.title("Masked image grayscale histogram")
    #plt.hist(comp_mask.ravel()[comp_mask.ravel() > 0],256)
    #plt.show()

    #####
    # make final frame with white background
    trails = np.full_like(mask,255)
    trails[trail_count > 0] = 221 # draw trails in light-grey 

    ##### 
    # choose focus frame (e.g. use current point or middle point) 
    focus_index = len(mask_list) - 1
    #focus_index = int(len(mask_list)/2) # middle point shows past/future
    trails[mask_list[focus_index] == 1] = 131 # draw focus mask darker 

    # add black circle to tank (note: 0 rather than (0,0,0) for grayscale)
    trails = tr.tank_draw_gray(trails,tank_xcm_avg,tank_ycm_avg,tank_R_avg)
    if (tracking):
      if ( n_contour != n_ind ):
        trails = tr.contour_draw_gray(trails, contours, contourID_repeat)
    
    if ( RGB ):
      trails = cv2.cvtColor(trails, cv2.COLOR_GRAY2BGR)
      trails = tr.points_draw_RGB(trails, meas_now, color)
      trails = tr.frame_number_label_RGB(trails,frame_count)
    else: 
      trails = tr.points_draw_gray(trails, meas_now)
      trails = tr.frame_number_label_gray(trails,frame_count)

    if (tracking):
      q_tmp = [ Kinematic(0.,0.,0.,0.,0.,0.,0.,0.,0.) for i in range(len(meas_now)) ]
      for i in range(len(meas_now)):
        q_tmp[i].x = meas_now[i][0]
        q_tmp[i].y = meas_now[i][1]
        if ( directors ):
          q_tmp[i].theta = meas_now[i][2]
      q.append(q_tmp)

    out.write(trails)
    if ( view_online ):
      cv2.imshow('frame', trails)
      return_key = 13
      esc_key    = 27 
      space_key  = 32
      if cv2.waitKey(33) == esc_key:
        break
      elif cv2.waitKey(33) == space_key:
        while(True):
          k = cv2.waitKey(33)
          if k == return_key:
            break
          elif k == -1:
            continue
          
  # output time of current frame.
  tr.print_current_frame(i_frame,fps)
# <home_path> <raw_video> <block> <offset> <viewer [0/1]> <nsamples> <n_ind> <t_start> <t_end>

# release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

sys.stdout.write("\n")
sys.stdout.write("       Tracking complete.\n")
sys.stdout.flush()

q = tr.recenter_positions(q,tank_xcm_avg,tank_ycm_avg)
q = tr.scale_length(q,tank_R_avg,tank_R_cm)
q = tr.analyze_velocities(q,fps)
q = tr.analyze_accelerations(q,fps)
#q = tr.reorient_directors(q)
q = tr.replace_directors(q) # uses velocity for director
q = tr.analyze_angular_velocities(q,fps)
q = tr.analyze_angular_accelerations(q,fps)

q = np.array(q)
np.save(output_filepath,q)

#tr.write_kinematics(q,output_text,n_ind,fps)
#tr.write_kinematics_CM_frame(q,output_text,n_ind,fps)
