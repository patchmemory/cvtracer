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
import sys, os
import numpy as np
import cv2
import pandas as pd
import scipy.signal
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from KinematicDataFrame import KinematicDataFrame
import Tracker as tr

args = tr.arg_parse()

n_ind       = args.n_individual
fps         = args.frames_per_second
tank_R_cm   = args.tank_diameter/2.
frame_start = int(args.t_start * fps)
frame_end   = int(args.t_end   * fps)
block_size  = args.block_size 
offset      = args.thresh_offset
gpu_on      = args.gpu_on

home_path = os.path.realpath(args.work_dir)
input_loc = os.path.realpath(args.raw_video)
data_output_dir = "data/"
video_output_dir = "video/"
output_str = "track_video"
input_vidpath, output_vidpath, output_filepath, output_text, codec = tr.organize_filenames(
                                home_path,input_loc,video_output_dir,data_output_dir,output_str)

tr.print_title(n_ind,input_vidpath,output_vidpath,output_filepath,output_text)

cap = cv2.VideoCapture(input_vidpath)
if cap.isOpened() == False:
  sys.exit('Video file cannot be read! Please check input_vidpath to ensure it is correctly pointing to the video file.')

# Video writer class to output video with contour and centroid of tracked
# object(s) make sure the frame size matches size of array 'final'
fourcc = cv2.VideoWriter_fourcc(*codec)
output_framesize = ( int(cap.read()[1].shape[1]*args.view_scale),
                     int(cap.read()[1].shape[0]*args.view_scale)  )
if ( args.RGB ):
  color = tr.colours
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

if frame_end < 0:
  frame_end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1


detect_tank = False
if detect_tank:
  # this first loop provides a preliminary look at the tank by
  #   1. randomly selecting a set a frames to fit circle to tank edges
  #   2. determining the size of   
  tank_info = []
  while True:
    # exit loop when enough measurements have been made
    if ( len(tank_info) > args.sample_frames): break 
    # capture random frame within range of time 
    i_frame = int(np.random.uniform(frame_start,frame_end))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame-1)
    ret, frame = cap.read()
    if ret == True:
      if gpu_on:
        frame = cv2.UMat(frame)
      # values found from visual test, would be better to point and click in a GUI
      tank_R_guess = 465
      min_area_tank = 5e2
      max_area_tank = 1e7
      tank_info_tmp = tr.tank_detect(frame, tank_R_guess, min_area_tank, max_area_tank)
  
      if ( len(tank_info_tmp) > 0 ):
        for tank_measure in tank_info_tmp:
          tank_info.append(tank_measure)
          if ( args.online_viewer):
            cv2.circle( frame, (int(tank_measure[1]),int(tank_measure[2])),
                                int(tank_measure[0]), (0,255,0), 5 )
   
        if ( args.online_viewer ):
          cv2.imshow('frame', frame) # requires cv2.waitKey() to follow 
          space_key=32
          return_key=13
          esc_key=27 # code for escape key to break (only works in view-finder)
          if cv2.waitKey(3) == esc_key:
            break
          elif cv2.waitKey(3) == space_key:
            while(True):
              k = cv2.waitKey(33)
              if k == return_key:
                break
              elif k == -1:
                continue

  # quick analysis of tank measurement
  tank_x_com_avg, tank_y_com_avg, tank_R_avg = tr.tank_detect_avg(tank_info,args.sample_frames)


##################################
#                                #
#  tank detection now complete!  #
#                                #
##################################


# next step is to locate the fish using contours

q = [ KinematicDataFrame() for i in range(n_ind) ]
contour_count = []
directors = args.directors

if directors:
  meas_last = list(np.zeros((n_ind,3)))
  meas_now = list(np.zeros((n_ind,3)))
else:
  meas_last = list(np.zeros((n_ind,2)))
  meas_now = list(np.zeros((n_ind,2)))

draw_mask = False

contour_list = []
contourID_repeat = []
contourID_unclaimed = []
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
for i_frame in range(frame_start,frame_end+1):
  frame_count = i_frame - frame_start
  current_time = i_frame * fps 
  ret, frame = cap.read()
  if ret == True:

    if gpu_on:
      frame = cv2.UMat(frame)

    if int(args.view_scale) != 1:
      frame = cv2.resize( frame, None,
                          fx = args.view_scale,
                          fy = args.view_scale,
                          interpolation = cv2.INTER_LINEAR )

    # find contours (note: following parameters have been tuned adhoc)
    min_area = 20   # min and max area should be changed according to 
    max_area = 1000 # fish size and the camera resolution
    n_pix = 5

    # remove everything outside of water region 
    if draw_mask:
      if detect_tank: 
        frame  = tr.tank_mask(frame, tank_x_com_avg, tank_y_com_avg, tank_R_avg)
      contours = tr.contour_detect(frame,min_area,max_area,block_size,offset,n_pix)
    #  tr.threshold_detect_hist(frame,n_pix,block_size,offset)
    else:
      contours = tr.contour_detect(frame,min_area,max_area,block_size,offset,n_pix)

    n_contour = len(contours)
    contour_count.append(n_contour)
    meas_last, meas_now = tr.points_detect(contours, meas_last, meas_now)
    
    if n_contour == n_ind:
      # do a simple reordering based on hugarian min-dist algorithm
      meas_last, meas_now = tr.reorder_hungarian(meas_last,meas_now)
    else:
      print("n_contours = %i" % n_contour)
      if n_contour == 0:
        meas_last, meas_now = tr.temporary_guess(q,meas_last,meas_now,frame_count)
      elif q[0].n_entries() < 3 or n_ind < 3:   # for initial frames, make "educated guesses"
        meas_now = tr.kmeans_contours(contours, n_ind, meas_now)
        meas_last, meas_now = tr.reorder_hungarian(meas_last,meas_now)
      else:
        # if have previous info, do something smarter to connect overlapping
        # fish that may be in one combined contour
        ind_last_now, contourID_repeat, contourID_unclaimed = \
              tr.contour_connect(q, n_ind, meas_now, meas_last, contours, frame_count)
        # then reorder according to connections
        meas_last, meas_now = tr.reorder_connected(meas_last, meas_now, ind_last_now) 
    

    if draw_mask:
      mask = tr.contour_mask_binary(frame,contours)
      #comp_mask = cv2.cvtColor(tank_only, cv2.COLOR_BGR2GRAY)
      #comp_mask[mask != 1] = 0
      #plt.title("Masked image grayscale histogram")
      #plt.hist(comp_mask.ravel()[comp_mask.ravel() > 0],256)
      #plt.show()
      #####
      # make final frame with white background
      final = np.full_like(mask,255)
      final[mask == 1] = 131 # draw focus mask darker 
    else:
      final = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # add black circle to tank (note: 0 rather than (0,0,0) for grayscale)
    if detect_tank:
      final = tr.tank_draw_gray(final,tank_x_com_avg,tank_y_com_avg,tank_R_avg)

    if ( n_contour != n_ind ):
      final = tr.contour_draw_gray(final, contours, contourID_repeat)
    
    if ( args.RGB ):
      final = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
      final = tr.points_draw_RGB(final, meas_now, color)
      final = tr.frame_number_label_RGB(final,frame_count)
    else: 
      final = tr.points_draw_gray(final, meas_now)
      final = tr.time_frame_label_gray(final,i_frame)

    if directors:
      for i in range(len(meas_now)):
        q[i].add_entry(current_time,meas_now[i][0], meas_now[i][1], meas_now[i][2])
    else:
      for i in range(len(meas_now)):
        q[i].add_entry(current_time,meas_now[i][0], meas_now[i][1], 0.)

    if gpu_on:
      final = cv2.UMat.get(final)
 
    out.write(final)
    if ( args.online_viewer ):
      cv2.imshow('frame', final) 
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

# release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)

sys.stdout.write("\n")
sys.stdout.write("       Tracking complete.\n")
sys.stdout.flush()

#q = tr.reorient_directors(q)
#for i in range(len(q)):
#  if detect_tank:
#    q[i].convert_pixels(tank_x_com_avg,tank_y_com_avg,tank_R_avg,tank_R_cm)
#  q[i].calculate_velocity(fps)
#  q[i].calculate_acceleration(fps)
#  q[i].calculate_director(fps,theta_replace=True)
#  q[i].calculate_angular_velocity(fps)
#  q[i].calculate_angular_acceleration(fps)

q = np.array(q)
np.save(output_filepath,q)
