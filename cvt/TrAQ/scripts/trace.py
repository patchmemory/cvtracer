#!/usr/bin/python3
import argparse
from TrAQ.Trial import Trial
from TrAQ.CVTracer import CVTracer

def arg_parse():
    parser = argparse.ArgumentParser(description="OpenCV2 Fish Tracker")
    parser.add_argument("raw_video", type=str, help="path to raw video")
    parser.add_argument("-ts","--t_start", type=float, help="start time, in seconds", default=0)
    parser.add_argument("-te","--t_end", type=float, help="end time, in seconds", default=-1)
    parser.add_argument("-npix","--n_pixel_blur", type=float, help="square-root of n-pixels for threshold blurring", default=3)
    parser.add_argument("-bs","--block_size", type=int, help="contour block size",default=15)
    parser.add_argument("-th","--thresh_offset", type=int, help="threshold offset for contour-finding",default=13) 
    parser.add_argument("-amin","--min_area", type=int, help="minimum area for threhold detection", default=20)
    parser.add_argument("-amax","--max_area", type=int, help="maximum area for threhold detection", default=500)
    parser.add_argument("-lt","--trail_length", type=int, help="length of trail", default=3)
    parser.add_argument("-on","--online_viewer", help="view tracking results in real time", action='store_true') 
    parser.add_argument("-vs","--view_scale", help="to scale size of window for online viewing", default=1.) 
    parser.add_argument("-RGB", help="generate movie in RGB (default is greyscale)", action='store_true')
    parser.add_argument("-GPU", help="use UMat file handling for OpenCV Transparent API", action='store_true') 
    return parser.parse_args()


args = arg_parse()
# open up the Trial
trial       = Trial(args.raw_video)
# initialize a CVTracer
frame_start = int(args.t_start * trial.fps)
frame_end   = int(args.t_end   * trial.fps)
cvt         = CVTracer( trial, frame_start, frame_end, 
                        args.n_pixel_blur, args.block_size, args.thresh_offset, 
                        args.min_area, args.max_area, args.trail_length, 
                        args.RGB, args.online_viewer, args.view_scale, args.GPU ) 

cvt.print_title()
cvt.set_frame(frame_start)
for i_frame in range(cvt.frame_start, cvt.frame_end + 1):
    # load next frame into video tracer
    if cvt.get_frame():
        # first remove all information from outside of tank
        cvt.mask_tank()       
        # detect contours in current frame
        cvt.detect_contours()
        # analyze contours for center and long axis
        cvt.analyze_contours()
        # estimate connection between last frame and current frame
        cvt.connect_frames()
        # store results from current frame in the trial object
        cvt.update_trial()
        # write frame with traces to new video file, show if online
        cvt.draw()
        # write frame to video output
        cvt.write_frame()
        # post frame to screen, may accept user input to break loop
        if not cvt.post_frame(): break
        # print time to terminal
        cvt.print_current_frame()
# put things away
cvt.release()
cvt.trial.save()
