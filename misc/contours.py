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
import tracktor as tr
#import tracktor_revised as tr
import cv2
import sys
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
  print("  Usage: %s <home_path> <raw_video> <n_individuals> <fps> <block> <offset> <viewer [0/1]> <t_start (s)> <t_end (s)>" % sys.argv[0])
  print("\n    suggested values: \n")
  print("          block size = 16")
  print("              offset = 13\n")
  exit()

fps = float(sys.argv[4])
t_start = int(sys.argv[8])
t_end = int(sys.argv[9])
frame_start = int(t_start * fps)
frame_end = int(t_end * fps)

# colours is a vector of BGR values which are used to identify individuals in
# the video since we only have one individual, the program will only use the
# first element from this array i.e. (0,0,255) - red number of elements in
# colours should be greater than n_inds (THIS IS NECESSARY FOR VISUALISATION
# ONLY)
n_inds = int(sys.argv[3])
colours = [ (   0,   0, 255),
            (   0, 255,   0),
            ( 255,   0,   0),
            (   0, 255, 255),
            ( 255,   0, 255),
            ( 255, 255,   0),
            ( 255, 255, 255),
            (   0,   0, 122),
            (   0, 122,   0),
            ( 122,   0,   0),
            (   0, 122, 122),
            ( 122,   0, 122),
            ( 122, 122,   0),
            ( 122, 122, 122),
            (   0,   0,   0)]


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
# to force n_inds number of animals
mot = True 
kmeans_all = False

# name of source video and paths
home_path = sys.argv[1]
if home_path[-1] != '/': home_path += '/'
input_loc = sys.argv[2]
video = input_loc.split('/')[-1]
print(video)
video_str = '.'.join(video.split('.')[:-1])
video_ext = '.' + video.split('.')[-1]
input_vidpath = home_path + input_loc
t_str = "_%04i_%04i" % (t_start,t_end)
output_viddir = 'video/'
output_datadir = 'data/'
output_vidpath = home_path + output_viddir + video_str + t_str + '_tracked' + video_ext 
output_filepath = home_path + output_datadir + video_str + t_str + '_tracked.dat'
if ( video_ext == ".avi" ):
  codec = 'DIVX' 
else:
  codec = 'mp4v'

# try other codecs if the default doesn't work (e.g. 'DIVX', 'avc1', 'XVID') 

sys.stdout.write("\n\n")
sys.stdout.write("   ###################################################\n")
sys.stdout.write("   #                                                 #\n")
sys.stdout.write("   #   tracktor modified (open source tracking)      #\n")
sys.stdout.write("   #                                                 #\n")
sys.stdout.write("   #           v1.0, adam patch 2019                 #\n")
sys.stdout.write("   #                   (fau jupiter)                 #\n")
sys.stdout.write("   #                                                 #\n")
sys.stdout.write("   #                          press <esc> to exit    #\n")
sys.stdout.write("   #                                                 #\n")
sys.stdout.write("   ###################################################\n")
sys.stdout.write(" \n\n")
sys.stdout.write("       Tracking %i fish in \n" % n_inds)
sys.stdout.write("\n")
sys.stdout.write("         %s \n" % (input_vidpath))
sys.stdout.write(" \n\n")
sys.stdout.write("       Writing output to \n")
sys.stdout.write("\n")
sys.stdout.write("         %s \n" % (output_vidpath))
sys.stdout.write("         %s \n" % (output_filepath))
sys.stdout.write(" \n\n")
sys.stdout.flush()

# open video
cap = cv2.VideoCapture(input_vidpath)
if cap.isOpened() == False:
  sys.exit('Video file cannot be read! Please check input_vidpath to ensure it is correctly pointing to the video file.')

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
  meas_last = list(np.zeros((n_inds,3)))
  meas_now = list(np.zeros((n_inds,3)))
else:
  meas_last = list(np.zeros((n_inds,3)))
  meas_now = list(np.zeros((n_inds,3)))

q = []
last = 0
df = []
contour_count = []

#while(True):
for i_frame in range(frame_start,frame_end):
  # Capture frame-by-frame
  cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame-1)
  ret, frame = cap.read()
  this = cap.get(1)
  frame_num = this 
  
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

    # use kmeans if specified or if less than n_inds found
    if kmeans_all or ( mot and len(meas_now) != n_inds): 
      if len(q) < 3 or n_inds < 3:
        contours,meas_now,final = tr.apply_k_means(contours, n_inds, meas_now,final, directors)
        row_ind, col_ind = tr.hungarian_algorithm(meas_last, meas_now)
        final, meas_now, df = tr.reorder_and_draw(
                      final, colours, n_inds, col_ind, meas_now, df, mot, this, frame_start)
      else:
        ind_last_now, contourID_repeat, contourID_unclaimed = tr.contour_connect(
                                    q, n_inds, meas_now, meas_last, contours)

        final = tr.contour_draw_RGB(final, contours, contourID_repeat)

        final, meas_now, df = tr.reorder_and_draw_new(
              final, colours, n_inds, ind_last_now, meas_last, meas_now, df, mot, this, frame_start)

    else:
      row_ind, col_ind = tr.hungarian_algorithm(meas_last, meas_now)
      final, meas_now, df = tr.reorder_and_draw(
                    final, colours, n_inds, col_ind, meas_now, df, mot, this, frame_start)
    
   # # Create output line (dataframe)
   # outputlist = []
   # outputlist.append(len(meas_now))
   # for i in range(len(meas_now)):
   #   outputlist.append(meas_now[i][0])
   #   outputlist.append(meas_now[i][1])
   # df.append(outputlist)

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
  t_csc = int(frame_num/fps*100)
  t_sec = int(t_csc/100) 
  t_min = t_sec/60
  t_hor = t_min/60
  sys.stdout.write("       Current tracking time: %02i:%02i:%02i:%02i \r" % (t_hor,t_min%60,t_sec%60,t_csc%100))
  sys.stdout.flush()

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
for i in range(n_inds):
  header += "  x" + str(i) + " y" + str(i) + " theta" + str(i) + " vx" + str(i) + " vy" + str(i)
f.write(header + "\n")

# replaced using Kinematic class
for i in range(len(q)):
  line = " %7i %10.5f %i %i" % (i,float(i)/fps,contour_count[i],len(q[i]))
  for j in range(n_inds):
    if ( j > len(q[i])-1 ):
      line += "    -     - "
    else:
      line += "  %10.5f %10.5f %7.5f %10.5f %10.5f" % (q[i][j].x,q[i][j].y,q[i][j].theta,q[i][j].vx,q[i][j].vy)
  line += "\n"
  f.write(line)

#for i in range(len(df)):
#  line = " %7i %10.5f %i" % (i,float(i)/fps,df[i][0])
#  for j in range(n_inds):
#    if ( j >= (len(df[i])-1)/2 ):
#      line += "    -     - "
#    else:
#      line += " %10.5f %10.5f" % (df[i][2*j+1],df[i][2*j+2])
#  line += "\n"
#  f.write(line)

sys.stdout.write(" done.\n\n\n")
sys.stdout.flush()

## When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
