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
sys.path.insert(0, '/home/apatch/code-tracking/tracktor')
import tracktor as tr
sys.path.insert(0, '/data1/cavefish/experiment')
import tracktor_revised as tr_r
import cv2
import scipy.signal
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from dataclasses import dataclass

def pause():
  programPause = input("<return> to continue...")

@dataclass
class Kinematic:
  x: float
  y: float
  vx: float
  vy: float
  theta: float

ellipses = False
directors = True

if ( len(sys.argv) != 10 ):
  print("  Usage: %s <home_path> <raw_video> <n_individuals> <fps> <block> <offset> <viewer [0/1]> <t_start> <t_end>" % sys.argv[0])
  print("\n    suggested values: \n")
  print("          block size = 15")
  print("              offset = 13\n")
  exit()

fps = float(sys.argv[4])
t_start = int(sys.argv[8])
t_end = int(sys.argv[9])
frame_start = int(t_start * fps)
frame_end = int(t_end * fps)

# color is a vector of BGR values which are used to identify individuals in
# the video since we only have one individual, the program will only use the
# first element from this array i.e. (0,0,255) - red number of elements in
# color should be greater than n_ind (THIS IS NECESSARY FOR VISUALISATION
# ONLY)
n_ind = int(sys.argv[3])
#color = tr_r.random_color_list(n_ind)
color = tr_r.colours[1]

# this is the block_size and offset used for adaptive thresholding (block_size
# should always be odd) these values are critical for tracking performance
#block_size = 15
#offset = 13
block_size = int(sys.argv[5])
offset     = int(sys.argv[6])

# the scaling parameter can be used to speed up tracking if video resolution is
# too high (use value 0-1)
scaling = 1.0

# minimum area and maximum area occupied by the animal in number of pixels this
# parameter is used to get rid of other objects in view that might be hard to
# threshold out but are differently sized
min_area = 20
max_area = 1000

# mot determines whether the tracker is being used in noisy conditions to track
# a single object or for multi-object using this will enable k-means clustering
# to force n_ind number of animals
mot = True 
kmeans_all = False

# name of source video and paths
home_path = sys.argv[1]
input_loc = sys.argv[2]
output_dir = "output/"
output_str = "tracked"
input_vidpath, output_vidpath, output_filepath, codec = tr_r.organize_filenames(
                                      home_path,input_loc,output_dir,output_str)

tr_r.print_title(n_ind,input_vidpath,output_vidpath,output_filepath)

# open video
cap = cv2.VideoCapture(input_vidpath)
if cap.isOpened() == False:
  sys.exit('Video file cannot be read! Is input_vidpath linked to video file?')

# Video writer class to output video with contour and centroid of tracked
# object(s) make sure the frame size matches size of array 'final'
fourcc = cv2.VideoWriter_fourcc(*codec)
output_framesize = (int(cap.read()[1].shape[1]*scaling),
                    int(cap.read()[1].shape[0]*scaling))
out = cv2.VideoWriter(filename = output_vidpath, 
                        fourcc = fourcc, 
                           fps = fps, 
                     frameSize = output_framesize, 
                       isColor = True)

# online viewer code block
view_online = bool(int(sys.argv[7]))
## set view_online = true to show window during tracking
#if ( view_online ):
#  online_W=int(800)
#  online_H=int(500)
#  cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
#  cv2.resizeWindow('frame', (online_W,online_H))

# Individual location(s) measured in the last and current step
if ( directors ):
  meas_last = list(np.zeros((n_ind,3)))
  meas_now = list(np.zeros((n_ind,3)))
else:
  meas_last = list(np.zeros((n_ind,3)))
  meas_now = list(np.zeros((n_ind,3)))

q = []
last = 0
df = []
frame_num = 0
contour_count = []

for i_frame in range(frame_start,frame_end):
  # Capture frame-by-frame
  cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame-1)
  # Capture frame-by-frame
  ret, frame = cap.read()
  this = cap.get(1)
  if ret == True:

    if int(scaling) != 1:
      frame = cv2.resize(frame, None,
                        fx = scaling,
                        fy = scaling,
             interpolation = cv2.INTER_LINEAR)

    thresh = tr.colour_to_thresh(frame, block_size, offset)

    final, contours, meas_last, meas_now = tr.detect_and_draw_contours(
                    frame, thresh, meas_last, meas_now, min_area, max_area, ellipses, directors)

    contour_count.append(len(meas_now))

    # use kmeans if specified or if less than n_ind found
    if kmeans_all or ( mot and len(meas_now) != n_ind): 
      if len(q) < 3 or n_ind < 3:
        contours,meas_now,final = tr.apply_k_means(contours, n_ind, meas_now,final, directors)
        row_ind, col_ind = tr.hungarian_algorithm(meas_last, meas_now)
        final, meas_now, df = tr.reorder_and_draw(
                      final, color, n_ind, col_ind, meas_now, df, mot, this, frame_start)
      else:

        unclaimed_contour_indices, ind_last_now = find_overlapped_contours(
                                                    q, n_ind, meas_now, meas_last, contours)
        final, meas_now, df = tr.reorder_and_draw_new(
             final, color, n_ind, ind_last_now, meas_last, meas_now, df, mot, this, frame_start)

    else:
      row_ind, col_ind = tr.hungarian_algorithm(meas_last, meas_now)
      final, meas_now, df = tr.reorder_and_draw(
                    final, color, n_ind, col_ind, meas_now, df, mot, this, frame_start)
    
    q_tmp = [ Kinematic(0.,0.,0.,0.,0.) for i in range(len(meas_now)) ]
    for i in range(len(meas_now)):
      q_tmp[i].x = meas_now[i][0]
      q_tmp[i].y = meas_now[i][1]
      if ( directors ):
        q_tmp[i].theta = meas_now[i][2]
    q.append(q_tmp)
    
    out.write(final)
    if ( view_online ):
      # Display the resulting frame
      cv2.imshow('frame', final)

    # break options  
    esc_key=27 # code for escape key to break (only works in view-finder)
    gutter=20
    if cv2.waitKey(1) == esc_key \
    or meas_now[0][0] < gutter or meas_now[0][0] > cap.get(3) - gutter \
    or meas_now[0][1] < gutter or meas_now[0][1] > cap.get(4) - gutter:
      break

    space_key=32
    return_key=13
    if cv2.waitKey(33) == space_key:
      while(True):
        k = cv2.waitKey(33)
        if k == return_key:
          break
        elif k == -1:
          continue
          
  # output time of current frame.
  tr_r.print_current_frame(frame_num,fps)
  frame_num += 1

  if last >= this:
    break
  last = this

sys.stdout.write("\n")
sys.stdout.write("       Raw tracking complete.\n")
sys.stdout.flush()


sys.stdout.write("\n\n")
sys.stdout.write("       Calculating velocities and reorienting directors...")
sys.stdout.flush()
# calculate velocities and rotate directors (only nematic on pi until this)
for i in range(1,len(q)-1):
  for j in range(len(q[i])):
    q[i][j].vx = ( q[i+1][j].x - q[i-1][j].x ) / ( 2. / fps ) 
    q[i][j].vy = ( q[i+1][j].y - q[i-1][j].y ) / ( 2. / fps )

    if ( q[i][j].theta == -13 ):
      q[i][j].theta = ( np.arctan2(q[i][j].vy,q[i][j].vx) + q[i-1][j].theta ) / 2
      
    edotv = q[i][j].vx*np.cos(q[i][j].theta) + q[i][j].vy*np.sin(q[i][j].theta)
    speed = np.sqrt(pow(q[i][j].vx,2) + pow(q[i][j].vy,2))

    if ( speed < 0.1 ):
      psi = 1
    else:
      cos_psi = edotv / speed

    if ( cos_psi < 0 ): 
      q[i][j].theta = np.fmod(q[i][j].theta + np.pi,2*np.pi) 
  

sys.stdout.write("\n\n")
sys.stdout.write("       Writing tracks...")
sys.stdout.flush()
f = open(output_filepath,'w')
header = "#  frame  time  n_contours n_kmeans"
for i in range(n_ind):
  header += "  x" + str(i) + " y" + str(i) + " theta" + str(i) + " vx" + str(i) + " vy" + str(i)
f.write(header + "\n")
for i in range(len(q)):
  line = " %7i %10.5f %i %i" % (i,float(i)/fps,contour_count[i],len(q[i]))
  for j in range(n_ind):
    if ( j > len(q[i])-1 ):
      line += "    -     - "
    else:
      line += "  %10.5f %10.5f %7.5f %10.5f %10.5f" % (q[i][j].x,q[i][j].y,q[i][j].theta,q[i][j].vx,q[i][j].vy)
  line += "\n"
  f.write(line)

sys.stdout.write(" done.\n\n\n")
sys.stdout.flush()

## When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
