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
import sys
import os
import argparse
import numpy as np
import cv2
from TrAQ.Trial import Trial
from TrAQ.Tracer import Tracer
from TrAQ.VideoCV import VideoCV


def arg_parse():
    parser = argparse.ArgumentParser(description="OpenCV2 Fish Tracker")
    parser.add_argument("trial", type=str, help="path to Trial object")
    parser.add_argument("-ts","--t_start", type=float, help="start time, in seconds", default=0)
    parser.add_argument("-te","--t_end", type=float, help="end time, in seconds", default=-1)
    parser.add_argument("-bs","--block_size", type=int, help="contour block size",default=15)
    parser.add_argument("-th","--thresh_offset", type=int, help="threshold offset for contour-finding",default=13) 
    parser.add_argument("-amin","--min_area", type=int, help="minimum area for threhold detection", default=20)
    parser.add_argument("-amax","--max_area", type=int, help="maximum area for threhold detection", default=500)
    parser.add_argument("-npix","--n_pixel_blur", type=float, help="square-root of n-pixels for threshold blurring", default=3)
    parser.add_argument("-on","--online_viewer", help="view tracking results in real time", action='store_true') 
    parser.add_argument("-RGB", help="generate movie in RGB (default is greyscale)", action='store_true')
    parser.add_argument("-GPU", help="use UMat file handling for OpenCV Transparent API", action='store_true') 
    return parser.parse_args()


args = arg_parse()

trial = Trial(args.trial)

frame_start = int(args.t_start * trial.fps)
frame_end   = int(args.t_end   * trial.fps)
videocv = VideoCV(trial, frame_start, frame_end, 
                  args.block_size, args.thresh_offset, 
                  args.min_area, args.max_area, n_pixel_blur = 3,
                  args.RGB, args.online_viewer, args.view_scale, args.GPU ) 

tracer = Tracer(trial, video)
tracer.print_title()

tracer.set_frame(frame_start)
for i_frame in range(frame_start,frame_end+1):
    
    # load current frame into the tracer video
    if tracer.videocv.get_frame():

        # detect and analyze contours in current frame
        tracer.videocv.detect_contours()
        tracer.videocv.analyze_contours()
        
        # estimate connection between last frame and current frame
        tracer.connect_frames()
        # store results from current frame in the trial object
        tracer.update_trial()
        # write frame with traces to new video file, show if online
        tracer.videocv.set_frame()

        # print time to terminal 
        tracer.videocv.print_current_frame()

 
tracer.videocv.release()
tracer.trial.save()