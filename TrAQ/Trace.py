import sys, os
import argparse
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def random_color_list(n_ind):
  colors = []
  low = 0
  high = 255
  for i in range(n_ind):
    color = []
    for i in range(3):
      color.append(np.uint8(np.random.random_integers(low,high)))
    colors.append((color[0],color[1],color[2]))
  print(colors)
  return colors

# default color list
colours = [ (   0,   0, 255),
            (   0, 255, 255),
            ( 255,   0, 255),
            ( 255, 255, 255),
            ( 255, 255,   0),
            ( 255,   0,   0),
            (   0, 255,   0),
            (   0,   0,   0)]

def distance(pos1,pos2):
  return np.sqrt(pow(pos1[0]-pos2[0],2)+pow(pos1[1]-pos2[1],2))

def angle_diff(q2,q1):
  return min(q2-q1,q2-q1,q2-q1+2*np.pi,q2-q1-2*np.pi, key=abs)

# rotate a point xp,yp about x0,y0 by angle
def transform_point(x0,y0,xp,yp,angle):
  cos_theta, sin_theta = np.cos(angle), np.sin(angle)
  x, y = xp - x0, yp - y0
  xf =   x * cos_theta + y * sin_theta 
  yf = - x * sin_theta + y * cos_theta 
  return xf,yf

def arg_parse():
  parser = argparse.ArgumentParser(description="OpenCV2 Fish Tracker")
  parser.add_argument("raw_video", type=str, help="path to raw video")
  parser.add_argument("n_individual", type=int, help="number of individuals in group")
  parser.add_argument("-ts","--t_start", type=float, help="start time, in seconds", default=0)
  parser.add_argument("-te","--t_end", type=float, help="end time, in seconds", default=-1)
  parser.add_argument("-td","--tank_diameter", type=float, help="tank diameter", default=111.)
  parser.add_argument("-ds","--datafile", type=str, help="data file-path, if not standard location") 
  parser.add_argument("-bs","--block_size", type=int, help="contour block size",default=15) 
  parser.add_argument("-th","--thresh_offset", type=int, help="thresholding offset",default=13) 
  parser.add_argument("-sf","--sample_frames", type=int, help="number of frames to locate tank center and radius", default=50)
  parser.add_argument("-fps","--frames_per_second", type=float, help="frames per second in raw video",default=30)
  parser.add_argument("-wd","--work_dir", type=str, help="root working directory if not current",default=os.getcwd())
  parser.add_argument("-vs","--view_scale", type=float, help="factor to scale viewer to fit window", default=1)
  parser.add_argument("-RGB", help="generate movie in RGB (default is greyscale)", action='store_true')
  parser.add_argument("-dir","--directors", help="calculate directors", action='store_true')
  parser.add_argument("-on","--online_viewer", help="view tracking results in real time", action='store_true') 
  parser.add_argument("-gpu","--gpu_on", help="use UMat file handling for OpenCV Transparent API", action='store_true') 
  return parser.parse_args()


def path_slash(path):
  if path[-1] != '/': 
    path += '/'
  return path


def organize_filenames(home_path, input_loc, video_output_dir, data_output_dir, output_str):
  home_path = path_slash(home_path)
  video_output_dir = path_slash(video_output_dir)
  data_output_dir = path_slash(data_output_dir)
  if home_path in input_loc:
    input_loc = input_loc.replace(home_path, '')
    
  video = input_loc.split('/')[-1]
  video_dir = input_loc.split('/')[-2]
  video_str = '.'.join(video.split('.')[:-1])
  video_ext = '.' + video.split('.')[-1]
  input_vidpath   = home_path + input_loc
  output_vidpath  = home_path + video_output_dir + video_dir + "_" + video_str + '_' + output_str + video_ext 
  output_filepath = home_path + data_output_dir + video_dir + "_" + video_str + '_' + output_str + "_kinematics.npy" 
  output_text     = home_path + data_output_dir + video_dir + "_" + video_str + '_' + output_str + ".dat" 
  if ( video_ext == ".avi" ):
    codec = 'DIVX' 
  else:
    codec = 'mp4v'
  # try other codecs if the default doesn't work (e.g. 'DIVX', 'avc1', 'XVID') 
  return input_vidpath, output_vidpath, output_filepath, output_text, codec


def print_title(n_ind,input_vidpath,output_vidpath,output_filepath, output_text):
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
  sys.stdout.write("       Tracking %i fish in \n" % n_ind)
  sys.stdout.write("\n")
  sys.stdout.write("         %s \n" % (input_vidpath))
  sys.stdout.write(" \n\n")
  sys.stdout.write("       Writing output to \n")
  sys.stdout.write("\n")
  sys.stdout.write("         %s \n" % (output_vidpath))
  sys.stdout.write("         %s \n" % (output_filepath))
  sys.stdout.write("         %s \n" % (output_text))
  sys.stdout.write(" \n\n")
  sys.stdout.flush()
  return


def print_current_frame(frame_num,fps):
  t_csc = int(frame_num/fps*100)
  t_sec = int(t_csc/100) 
  t_min = t_sec/60
  t_hor = t_min/60
  sys.stdout.write("       Current tracking time: %02i:%02i:%02i:%02i \r" % (t_hor,t_min%60,t_sec%60,t_csc%100))
  sys.stdout.flush()
  return


def threshold_detect_hist(frame, n_pix_avg = 3, block_size = 15, offset = 13):

  plt.title("Raw image grayscale histogram")
  plt.hist(frame.ravel()[frame.ravel() > 0],256)
  plt.show()

  # blur and current image for smoother contours
  blur = cv2.GaussianBlur(frame, (n_pix_avg,n_pix_avg),0)
  blur2 = cv2.blur(frame, (n_pix_avg,n_pix_avg))

  plt.title("Blurred grayscale histogram")
  plt.hist(blur.ravel()[blur.ravel() > 0],256)
  plt.show()

  # convert to grayscale
  gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

  plt.title("Grayscale histogram")
  plt.hist(gray.ravel()[gray.ravel() > 0],256)
  plt.show()

  # calculate thresholds, more info:
  #   https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
  thresh = cv2.adaptiveThreshold( gray, 
                                  255, 
                                  cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY_INV,
                                  block_size, 
                                  offset )
  return thresh


# previously colour_to_thresh(...)
def threshold_detect(frame, n_pix_avg = 3, block_size = 15, offset = 13):
  # blur and current image for smoother contours
  blur = cv2.GaussianBlur(frame, (n_pix_avg,n_pix_avg),0)
  #blur = cv2.blur(frame, (n_pix_avg,n_pix_avg))

  # convert to grayscale
  gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
  # calculate thresholds, more info:
  #   https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
  thresh = cv2.adaptiveThreshold( gray, 
                                  160,
                                  cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY_INV,
                                  block_size, 
                                  offset )
  return thresh


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
      if abs(tank_R - tank_R_guess) < 0.1*tank_R_guess:
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


def tank_mask(frame, x_cm, y_cm, R):
  x_cm = int(x_cm)
  y_cm = int(y_cm)
  R = int(R)
  tank_mask = np.full_like(frame,0)
  tank_only = np.full_like(frame,0)
  cv2.circle(tank_mask,(x_cm,y_cm),R,(1,1,1),thickness=-1)
  tank_only[tank_mask==1] = frame[tank_mask==1] 
  return tank_only


def tank_draw(frame, x_cm, y_cm, R, RGB = False):
    if RGB:
        cv2.circle(frame,(int(x_cm),int(y_cm)),int(R),(0,0,0),thickness=7)
    else:
        cv2.circle(frame,(int(x_cm),int(y_cm)),int(R),0,thickness=7)
  return frame


def contour_detect(frame, min_area = 20, max_area = 500, block_size = 15, offset = 13, n_pix_avg = 3):
  thresh = threshold_detect(frame,n_pix_avg, block_size, offset)
  # find all contours 
  img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  # test found contours against area constraints
  i = 0
  while i < len(contours):
    area = cv2.contourArea(contours[i])
    if area < min_area or area > max_area:
      del contours[i]
    else:
      i += 1
  return contours


def contour_mask_binary(frame, contours):
  # create mask layer
  mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  mask = np.zeros_like(mask)
  # add contour areas to mask
  for i in range(len(contours)):
    cv2.drawContours(mask, contours, i, 1, -1)
  return mask


def contour_mask(frame, contours):
  # create mask layer
  mask = np.zeros_like(frame)
  for i in range(len(contours)):
    cv2.drawContours(mask, contours, i, (1,1,1), -1)
  # only include unmasked parts of frame (hopefully fish) 
  frame_masked = np.full_like(frame,255)
  frame_masked[mask == (1,1,1)] = frame[mask == (1,1,1)]
  return frame_masked


def points_detect(contours, meas_last, meas_now):
  meas_last = meas_now.copy()
  del meas_now[:]
  meas_now = []
  for i in range(len(contours)):
    M = cv2.moments(contours[i])
    if M['m00'] != 0:
      cx = M['m10']/M['m00']
      cy = M['m01']/M['m00']
    else:
    	cx = 0
    	cy = 0

    meas_now.append([cx,cy])

  return meas_last, meas_now


def points_directors_detect(contours, meas_last, meas_now):
  meas_last = meas_now.copy()
  del meas_now[:]
  meas_now = []
  
  for i in range(len(contours)):
    M = cv2.moments(contours[i])

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
    #if rx == 0:
    #  atan = 0.5*np.pi
    #  if ry < 0: atan *= -1 
    #  director = np.fmod(0.5*atan,2*np.pi) + np.pi
    #else:
    #  director = np.fmod(0.5*np.arctan(ry/rx),2*np.pi) + np.pi
    #if (rx < 0):
    #  director += np.pi/2.

    #vsize = 10
    #cv2.line(final,
    #  (int(cx - vsize*np.cos(theta)), int(cy - vsize*np.sin(theta))),
    #  (int(cx + vsize*np.cos(theta)), int(cy + vsize*np.sin(theta))), 
    #  (255,0,0),2)
    meas_now.append([cx,cy,theta])

  return meas_last, meas_now



def detect_and_draw_contours(frame, thresh, meas_last, meas_now, min_area = 0, max_area = 10000, ellipses = False, directors = False):
    # Detect contours and draw them based on specified area thresholds
    img, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    final = frame.copy()

    i = 0
    meas_last = meas_now.copy()
    del meas_now[:]
    director = 0. 
    rx = ry = 0.
    cx = cy = 0.

    fname_min_enc_C = "min_enc_C.dat"
    f_min_enc_C = open(fname_min_enc_C,'a+')
    R_min_enc_C = x_min_enc_C = y_min_enc_C = 0.
    
    while i < len(contours):
        area = cv2.contourArea(contours[i])
        if area < min_area or area > max_area:
            del contours[i]
        else:

            cv2.drawContours(final, contours, i, (0,0,255), 1)
            # add ellipse here
            if ( ellipses ):
              ellipse = cv2.fitEllipse(contours[i])
              cv2.ellipse(final,ellipse,(0,255,0),2)
            M = cv2.moments(contours[i])

            # here is the ouput showing minEnclosingCircle, which should
            # basically give a long-axis measurement of any given ellipse
            (x_min_enc_C, y_min_enc_C), R_min_enc_C = cv2.minEnclosingCircle(contours[i]) 
            f_min_enc_C.write("%e %e %e\n" %(x_min_enc_C,y_min_enc_C,R_min_enc_C))

            if M['m00'] != 0:
              cx = M['m10']/M['m00']
              cy = M['m01']/M['m00']
              if ( directors ):
                mu20 = M['m20']/M['m00'] - pow(cx,2)
                mu02 = M['m02']/M['m00'] - pow(cy,2)
                mu11 = M['m11']/M['m00'] - cx*cy
            else:
            	cx = 0
            	cy = 0

            if ( directors ):
              ry = 2*mu11
              rx = mu20-mu02
              if rx == 0:
                atan = 0.5*np.pi
                if ry < 0: atan *= -1 
                director = np.fmod(0.5*atan,2*np.pi) + np.pi
              else:
                director = np.fmod(0.5*np.arctan(ry/rx),2*np.pi) + np.pi
              if (rx < 0):
                director += np.pi/2.

              vsize = 10
              cv2.line(final,
                (int(cx - vsize*np.cos(director)), int(cy - vsize*np.sin(director))),
                (int(cx + vsize*np.cos(director)), int(cy + vsize*np.sin(director))), 
                (255,0,0),2)
              meas_now.append([cx,cy,director])
            else: 
              meas_now.append([cx,cy])

            i += 1

    f_min_enc_C.close()

    fname_ndist = "ndist.dat"
    f_ndist = open(fname_ndist,'a+')
    meas_now = np.array(meas_now)
    for i in range(len(meas_now)):
      for j in range(i+1,len(meas_now)):
        f_ndist.write("%e \n" % distance(meas_now[i,:-1],meas_now[j,:-1]))
    f_ndist.close()
    meas_now = list(meas_now)
         
    return final, contours, meas_last, meas_now

def kmeans_contours(contours, n_ind, meas_now):
    # convert contour points to arrays
    clust_points = np.vstack(contours)
    clust_points = clust_points.reshape(clust_points.shape[0], clust_points.shape[2])
    # run KMeans clustering
    kmeans_init = 50
    kmeans = KMeans(n_clusters=n_ind, random_state=0, n_init = kmeans_init).fit(clust_points)
    l = len(kmeans.cluster_centers_)
    meas_now_tmp = meas_now
    theta=-13
    del meas_now[:]
    for i in range(l):
        x = int(tuple(kmeans.cluster_centers_[i])[0])
        y = int(tuple(kmeans.cluster_centers_[i])[1])
        meas_now.append([x,y,theta])
    return meas_now


def apply_k_means(contours, n_ind, meas_now, final, directors):
    """
    This function applies the k-means clustering algorithm to separate merged
    contours. The algorithm is applied when detected contours are fewer than
    expected objects(number of animals) in the scene.
    
    Parameters
    ----------
    contours: list
        a list of all detected contours that pass the area based threhold criterion
    n_ind: int
        total number of individuals being tracked
    meas_now: array_like, dtype=float
        individual's location on current frame
        
    Returns
    -------
    contours: list
        a list of all detected contours that pass the area based threhold criterion
    meas_now: array_like, dtype=float
        individual's location on current frame
    """

    # Clustering contours to separate individuals
    myarray = np.vstack(contours)
    myarray = myarray.reshape(myarray.shape[0], myarray.shape[2])

    kmeans_init = 50
    kmeans = KMeans(n_clusters=n_ind, random_state=0, n_init = kmeans_init).fit(myarray)
    #kmeans = KMeans(n_clusters=n_ind, init='k-means++', n_init = kmeans_init, n_jobs = 8, algorithm="full").fit(myarray)
    #km_label = kmeans.labels_
    #for i in range(len(myarray)):
    #  cv2.circle(final, list(myarray[i]), 1, (0,0,255), thickness=1)
    l = len(kmeans.cluster_centers_)

    if (directors):
      meas_now_tmp = meas_now
    del meas_now[:]

    theta=-13

    for i in range(l):
        x = int(tuple(kmeans.cluster_centers_[i])[0])
        y = int(tuple(kmeans.cluster_centers_[i])[1])
        if ( directors ):
          meas_now.append([x,y,theta])
        else:
          meas_now.append([x,y])

    return contours, meas_now, final

def hungarian_algorithm(meas_last, meas_now):
    """
    The hungarian algorithm is a combinatorial optimisation algorithm used
    to solve assignment problems. Here, we use the algorithm to reduce noise
    due to ripples and to maintain individual identity. This is accomplished
    by minimising a cost function; in this case, euclidean distances between 
    points measured in previous and current step. The algorithm here is written
    to be flexible as the number of contours detected between successive frames
    changes. However, an error will be returned if zero contours are detected.
   
    Parameters
    ----------
    meas_last: array_like, dtype=float
        individual's location on previous frame
    meas_now: array_like, dtype=float
        individual's location on current frame
        
    Returns
    -------
    row_ind: array, dtype=int64
        individual identites arranged according to input ``meas_last``
    col_ind: array, dtype=int64
        individual identities rearranged based on matching locations from 
        ``meas_last`` to ``meas_now`` by minimising the cost function
    """
    xy_last = np.array(meas_last)[:,[0,1]]
    xy_now = np.array(meas_now)[:,[0,1]]
    #if meas_now.shape != meas_last.shape:
    #    if meas_now.shape[0] < meas_last.shape[0]:
    #        while meas_now.shape[0] != meas_last.shape[0]:
    #            meas_last = np.delete(meas_last, meas_last.shape[0]-1, 0)
    #    else:
    #        result = np.zeros(meas_now.shape)
    #        result[:meas_last.shape[0],:meas_last.shape[1]] = meas_last
    #        meas_last = result

    xy_last = list(xy_last)
    xy_now = list(xy_now)
    cost = cdist(xy_last, xy_now)
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind

def contour_draw(final, contours, contour_list, RGB=False):
  if RGB:
    color = (0,0,0)
  else:
    color = 0
  # draw contours in list 
  for contourID in contour_list: 
    cv2.drawContours(final, contours, contourID, color, 2)
  return final


def contour_draw_gray(final, contours, contour_list, val_gray = 0):
  # draw contours in list 
  for contourID in contour_list: 
    cv2.drawContours(final, contours, contourID, val_gray, 2)
  return final


def contour_draw_RGB(final, contours, contour_list, val_RGB = (0,0,255)):
  # draw contours in list 
  for contourID in contour_list: 
    cv2.drawContours(final, contours, contourID, val_RGB, 2)
  return final

# in the absence of contours, assuming there was something previously, make a
# guess should only be done for a limited number of frames, hence "temporary"
def temporary_guess(q,meas_last, meas_now, i_frame):
  prev = i_frame - 1 
  # use meas_last to calculate predicted trajectory based on previous three points
  #   (meas_now contains contours, meas_last contains guess) 
  meas_now = meas_last.copy()
  for i in range(len(meas_last)):
    meas_now[i][0] = q[i].loc(prev,'x') + ( 3*q[i].loc(prev,'x') - 2*q[i].loc(prev-1,'x') - q[i].loc(prev-2,'x') ) / 4.
    meas_now[i][1] = q[i].loc(prev,'y') + ( 3*q[i].loc(prev,'y') - 2*q[i].loc(prev-1,'y') - q[i].loc(prev-2,'y') ) / 4.
  return meas_last, meas_now


def contour_connect(q, n_ind, meas_now, meas_last, contours, i_frame):
  prev = i_frame - 1
  # use meas_last to calculate predicted trajectory based on previous three points
  #   (meas_now contains contours, meas_last contains guess) 
  for i in range(len(meas_last)):
    meas_last[i][0] = q[i].loc(prev,'x') + ( 3*q[i].loc(prev,'x') - 2*q[i].loc(prev-1,'x') - q[i].loc(prev-2,'x') ) / 4.
    meas_last[i][1] = q[i].loc(prev,'y') + ( 3*q[i].loc(prev,'y') - 2*q[i].loc(prev-1,'y') - q[i].loc(prev-2,'y') ) / 4.

  # calculate the distance matrix and then look  
  dist_arr = cdist(meas_last,meas_now)
  ind_last_now = [ 0 for i in range(len(meas_last)) ]
  for i in range(len(meas_last)):
    index = np.argmin(dist_arr[i]) # i'th array has distances to predictions from last
    ind_last_now[i] = index # for i, what is corresponding index of last (j)

  # identify any unclaimed contours
  contourID_unclaimed = []
  for i in range(len(meas_now)):
    for j in range(len(meas_last)):
      if i == ind_last_now[j]:
        break # exits current for loop, contour is located
      if j == len(meas_last) - 1:
        contourID_unclaimed.append(i)

  # choose the closest contour to guess even if two choose the same
  for i in range(len(contourID_unclaimed)):
    index = np.argmin(dist_arr[:,contourID_unclaimed[i]])
    ind_last_now[index] = contourID_unclaimed[i]

  # identify contours containing multiple points
  contourID_repeat = []
  contour_deficit = len(meas_last) - len(meas_now)
  sorted_ind_last_now = sorted(ind_last_now)
  i = 0  
  while len(contourID_repeat) < contour_deficit:
    i += 1
    if sorted_ind_last_now[i] == sorted_ind_last_now[i-1]:
      contourID_repeat.append(sorted_ind_last_now[i])

  return ind_last_now, contourID_repeat, contourID_unclaimed

def time_frame_label_gray(final, frame_number, fps=30, color_gray = 0):
  font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
  t_str = "%02i:%02i:%02i.%02i" % (int(frame_number/fps/3600),int(frame_number/fps/60 % 60),int(frame_number/fps%60),int(frame_number%fps*100/fps))
  cv2.putText(final, t_str, (5,30), font, 1, color_gray, 2)
  return final

def time_frame_label(final, frame_number, fps=30, RGB=False):
  if RGB:
    color = (0,0,0)
  else:
    color = 0
  font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
  t_str = "%02i:%02i:%02i.%02i" % (int(frame_number/fps/3600),int(frame_number/fps/60 % 60),int(frame_number/fps%60),int(frame_number%fps*100/fps))
  cv2.putText(final, t_str, (5,30), font, 1, color, 2)
  return final

def frame_number_label_gray(final, frame_number, color_gray = 0):
  font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
  cv2.putText(final, str(int(frame_number)), (5,30), font, 1, color_gray, 2)
  return final

def frame_number_label_RGB(final, frame_number, color_RGB = (255,255,255)):
  font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
  cv2.putText(final, str(int(frame_number)), (5,30), font, 1, color_RGB, 2)
  return final

def reorder_hungarian(meas_last, meas_now):
  # reorder contours based on results of the hungarian algorithm 
  # (originally in tracktor, but i've reorganized)
  row_ind, col_ind = hungarian_algorithm(meas_last,meas_now)
  equal = np.array_equal(col_ind, list(range(len(col_ind))))
  if equal == False:
      current_ids = col_ind.copy()
      reordered = [i[0] for i in sorted(enumerate(current_ids), key=lambda x:x[1])]
      meas_now = [x for (y,x) in sorted(zip(reordered,meas_now))]
  return meas_last, meas_now

def reorder_connected(meas_last, meas_now, ind_last_now):
  meas_now_tmp = meas_now
  meas_now = meas_last
  for i in range(len(meas_now)):
    meas_now[i] = meas_now_tmp[ind_last_now[i]]
  return meas_last, meas_now

def points_draw(final, meas_now, RGB=False):
    meas_now = np.array(meas_now)
    for i in range(len(meas_now)):
      if RGB:
        cv2.circle(final, tuple([int(x) for x in meas_now[i,(0,1)]]), 3, colours[i%len(colours)], -1, cv2.LINE_AA)
      else:
        cv2.circle(final, (int(meas_now[i][0]),int(meas_now[i][1])),3, 0, -1)

    meas_now = list(meas_now)
    return final

def points_draw_gray(final, meas_now):
    meas_now = np.array(meas_now)
    for i in range(len(meas_now)):
      cv2.circle(final, (int(meas_now[i][0]),int(meas_now[i][1])),3, 0, -1)
    meas_now = list(meas_now)
    return final

def points_draw_RGB(final, meas_now, colors):
    meas_now = np.array(meas_now)
    for i in range(len(meas_now)):
      cv2.circle(final, tuple([int(x) for x in meas_now[i,(0,1)]]), 3, colours[i%len(colors)], -1, cv2.LINE_AA)
    meas_now = list(meas_now)
    return final

def directors_draw_gray(final, meas_last, meas_now):
    meas_now = np.array(meas_now)
    meas_last = np.array(meas_last)
    vsize = 7
    for i in range(len(meas_now)):
      cx = meas_now[i][0]
      cy = meas_now[i][1]
      theta = meas_now[i][2]

      cx_p = meas_last[i][0]
      cy_p = meas_last[i][1]
      theta_p = meas_now[i][2]
      vx = cx - cx_p
      vy = cy - cy_p

      dot_prod = vx*np.cos(theta) + vy*np.sin(theta) 
      if dot_prod < 0:
        theta = np.mod(theta+2*np.pi,2*np.pi) - np.pi

      #dot_prod = np.cos(theta_p)*np.cos(theta) + np.sin(theta_p)*np.sin(theta) 
      #if dot_prod < 0:
      #  theta = np.mod(theta+2*np.pi,2*np.pi) - np.pi


      cv2.line(final, (int(cx - vsize*np.cos(theta)), int(cy - vsize*np.sin(theta))),
                      (int(cx + vsize*np.cos(theta)), int(cy + vsize*np.sin(theta))), 
                      (0,0,0),2)
      cv2.circle(final,(int(cx + vsize*np.cos(theta)), int(cy + vsize*np.sin(theta))),3,255,-1)
    return final


def directors_draw(final, meas_last, meas_now, RGB=False):
    meas_now = np.array(meas_now)
    meas_last = np.array(meas_last)
    vsize = 7
    for i in range(len(meas_now)):
      cx = meas_now[i][0]
      cy = meas_now[i][1]
      theta = meas_now[i][2]

      cx_p = meas_last[i][0]
      cy_p = meas_last[i][1]
      theta_p = meas_now[i][2]
      vx = cx - cx_p
      vy = cy - cy_p

      dot_prod = vx*np.cos(theta) + vy*np.sin(theta) 
      if dot_prod < 0:
        theta = np.mod(theta+2*np.pi,2*np.pi) - np.pi

      #dot_prod = np.cos(theta_p)*np.cos(theta) + np.sin(theta_p)*np.sin(theta) 
      #if dot_prod < 0:
      #  theta = np.mod(theta+2*np.pi,2*np.pi) - np.pi

      if RGB:
        cv2.line(final, (int(cx - vsize*np.cos(theta)), int(cy - vsize*np.sin(theta))),
                        (int(cx + vsize*np.cos(theta)), int(cy + vsize*np.sin(theta))), 
                        colours[i%len(colours)], 2)
        cv2.circle(final,(int(cx + vsize*np.cos(theta)), int(cy + vsize*np.sin(theta))),
                        3, colours[i%len(colours)], -1)
      else:
        cv2.line(final, (int(cx - vsize*np.cos(theta)), int(cy - vsize*np.sin(theta))),
                        (int(cx + vsize*np.cos(theta)), int(cy + vsize*np.sin(theta))), 
                        0,2)
        cv2.circle(final,(int(cx + vsize*np.cos(theta)), int(cy + vsize*np.sin(theta))),
                        3, 255, -1)

    return final


def reorder_and_draw_new(final, colours, n_ind, ind_last_now, meas_last, meas_now, df, mot, fr_no, frame_start):
    """
    This function reorders the measurements in the current frame to match
    identity from previous frame. This is done by using the results of the
    hungarian algorithm from the array col_inds.
    
    Parameters
    ----------
    final: ndarray, shape(n_rows, n_cols, 3)
        final output image composed of the input frame with object contours 
        and centroids overlaid on it
    colours: list, tuple
        list of tuples that represent colours used to assign individual identities
    n_ind: int
        total number of individuals being tracked
    col_ind: array, dtype=int64
        individual identities rearranged based on matching locations from 
        ``meas_last`` to ``meas_now`` by minimising the cost function
    meas_now: array_like, dtype=float
        individual's location on current frame
    df: pandas.core.frame.DataFrame
        this dataframe holds tracked coordinates i.e. the tracking results
    mot: bool
        this boolean determines if we apply the alogrithm to a multi-object
        tracking problem
        
    Returns
    -------
    final: ndarray, shape(n_rows, n_cols, 3)
        final output image composed of the input frame with object contours 
        and centroids overlaid on it
    meas_now: array_like, dtype=float
        individual's location on current frame
    df: pandas.DataFrame
        this dataframe holds tracked coordinates i.e. the tracking results
    """
    # Reorder contours based on results of the hungarian algorithm
    equal = np.array_equal(ind_last_now, list(range(len(ind_last_now))))
    meas_now = np.array(meas_now)
    meas_now_tmp = meas_now
    meas_now = np.array(meas_last)
    for i in range(len(meas_now)):
        meas_now[i] = meas_now_tmp[ind_last_now[i]]
      
    # apatch: commented the following for loop because its range is not
    # appropriate for systems where the number of individuals may change
    # frame to frame... for example, due to occlusions when fish cross over
    #for i in range(n_ind):
    meas_now = np.array(meas_now)
    for i in range(len(meas_now)):
        cv2.circle(final, tuple([int(x) for x in meas_now[i,(0,1)]]), 5, colours[i%len(colours)%n_ind], -1, cv2.LINE_AA)
    
    meas_now = list(meas_now)
    # add frame number
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    fr_no_print = fr_no - frame_start
    cv2.putText(final, str(int(fr_no_print)), (5,30), font, 1, (255,255,255), 2)
        
    return final, meas_now, df


def reject_outliers(data, m):
    """
    This function removes any outliers from presented data.
    
    Parameters
    ----------
    data: pandas.Series
        a column from a pandas dataframe that needs smoothing
    m: float
        standard deviation cutoff beyond which, datapoint is considered as an outlier
        
    Returns
    -------
    index: ndarray
        an array of indices of points that are not outliers
    """
    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d/mdev if mdev else 0.
    return np.where(s < m)


def analyze_accelerations(q,fps):
  sys.stdout.write("\n\n")
  sys.stdout.write("       Calculating accelerations from velocities...")
  sys.stdout.flush()
  # calculate velocities by differentiating position 
  frame_gap = 2
  for i in range(frame_gap,len(q)-frame_gap):
    for j in range(len(q[i])):
      q[i][j].ax = ( q[i+1][j].vx - q[i-1][j].vx ) / ( 2. / fps ) 
      q[i][j].ay = ( q[i+1][j].vy - q[i-1][j].vy ) / ( 2. / fps )
  sys.stdout.write("         ... done calculating accelerations.")
  sys.stdout.write("\n\n")
  return q


def reorient_directors(q):
  sys.stdout.write("\n\n")
  sys.stdout.write("       Reorienting directors with velocities...")
  sys.stdout.flush()
  # rotate directors (only nematic on pi until this point)
  frame_gap = 1 
  for i in range(frame_gap,len(q)-frame_gap):
    for j in range(len(q[i])):
      speed_cos_psi = q[i][j].vx*np.cos(q[i][j].theta) + q[i][j].vy*np.sin(q[i][j].theta)
      if ( cos_psi < 0 ): 
        q[i][j].theta = np.fmod(q[i][j].theta + np.pi,2*np.pi) 
  sys.stdout.write("         ... done reorienting directors.")
  sys.stdout.write("\n\n")
  return q


def replace_directors(q):
  sys.stdout.write("\n\n")
  sys.stdout.write("       Replacing directors with velocity...")
  sys.stdout.flush()
  # rotate directors (only nematic on pi until this point)
  for i in range(1,len(q)-1):
    for j in range(len(q[i])):
      q[i][j].theta = np.arctan2(q[i][j].vy,q[i][j].vx)
  sys.stdout.write("         ... done replacing directors.")
  sys.stdout.write("\n\n")
  return q


def analyze_angular_velocities(q,fps):
  sys.stdout.write("\n\n")
  sys.stdout.write("       Calculating angular velocities...")
  sys.stdout.flush()
  # calculate velocities by differentiating position 
  frame_gap = 1
  for i in range(frame_gap,len(q)-frame_gap):
    for j in range(len(q[i])):
      q[i][j].omega = angle_diff(q[i+1][j].theta,q[i-1][j].theta) / ( 2. / fps ) 
  sys.stdout.write("         ... done calculating angular velocities.")
  sys.stdout.write("\n\n")
  return q


def analyze_angular_accelerations(q,fps):
  sys.stdout.write("\n\n")
  sys.stdout.write("       Calculating angular accelerations...")
  sys.stdout.flush()
  # calculate accelerations by differentiating position 
  frame_gap = 2 
  for i in range(frame_gap,len(q)-frame_gap):
    for j in range(len(q[i])):
      q[i][j].alpha = angle_diff(q[i+1][j].omega,q[i-1][j].omega) / ( 2. / fps ) 
  sys.stdout.write("         ... done calculating angular accelerations.")
  sys.stdout.write("\n\n")
  return q


def write_kinematics(q,output_fname,n_ind,fps):

  if (len(output_fname.split('.')) != 2 ):
    print("ERROR: Multiple '.' in filename creates error in tr.write_kinematics(...).")
    exit()

  ofname_str1 = output_fname.split('.')[0]
  ofname_str2 = output_fname.split('.')[1]

  for j in range(n_ind):
    ofname_tmp = "%s_%02i.%s" % (ofname_str1,j,ofname_str2)
    f = open(ofname_tmp,'w')
    header = "#t x y theta vx vy omega ax ay alpha"
    f.write(header + "\n")
    for i in range(len(q)):
      line  = "%10.5f  " % (float(i)/fps)
      line += "%10.5f  %10.5f  %10.5f  " % (q[i][j].x , q[i][j].y , q[i][j].theta)
      line += "%10.5f  %10.5f  %10.5f  " % (q[i][j].vx, q[i][j].vy, q[i][j].omega)
      line += "%10.5f  %10.5f  %10.5f  " % (q[i][j].ax, q[i][j].ay, q[i][j].alpha)
      line += "\n"
      f.write(line)
    f.close()

  return
