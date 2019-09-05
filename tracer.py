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
import Tracker as tr
from KinematicDataFrame import KinematicDataFrame

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
output_str = "cv"
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


q = [ KinematicDataFrame() for i in range(n_ind) ]
contour_count = []
directors = args.directors

meas_last = list(np.zeros((n_ind,3)))
meas_now = list(np.zeros((n_ind,3)))

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

    contours = tr.contour_detect(frame,min_area,max_area,block_size,offset,n_pix)

    n_contour = len(contours)
    contour_count.append(n_contour)
    meas_last, meas_now = tr.points_directors_detect(contours, meas_last, meas_now)
    
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
    


    if args.RGB:
      final = frame
      if ( n_contour != n_ind ):
        final = tr.contour_draw(final, contours, contourID_repeat, RGB=True)
      final = tr.points_draw(final, meas_now, RGB=True)
      final = tr.directors_draw(final, meas_last, meas_now, RGB=True)
      final = tr.time_frame_label(final,i_frame, RGB=True)
    else:
      final = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      if ( n_contour != n_ind ):
        final = tr.contour_draw(final, contours, contourID_repeat)
      final = tr.points_draw(final, meas_now)
      final = tr.directors_draw(final, meas_last, meas_now)
      final = tr.time_frame_label(final,i_frame)

    for i in range(len(meas_now)):
      q[i].add_entry(current_time,meas_now[i][0], meas_now[i][1], meas_now[i][2])

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

q = np.array(q)
np.save(output_filepath,q)
