#!/usr/bin/python
import sys
cvhome="/home/patch/Code/cvtracer"
sys.path.insert(0, cvhome)
import argparse
from TrAQ.Trial import Trial

def arg_parse():
    parser = argparse.ArgumentParser(description="cv-tracer Trial")
    parser.add_argument("raw_video", type=str, help="path to raw video")
#    parser.add_argument("-ts","--t_start", type=float, help="start time, in seconds", default=0)
#    parser.add_argument("-te","--t_end", type=float, help="end time, in seconds", default=-1)
#    parser.add_argument("-nbf","--n_buffer_frames", type=float, help="number of buffer frames for cuts", default=2)
#    parser.add_argument("-oc", "--ocut", type = [float,float], help="range for occlusion cut", default=None)
#    parser.add_argument("-vc", "--ocut", type = [float,float], help="range for speed cut", default=None)
#    parser.add_argument("-wc", "--wcut", type = [float,float], help="range for omega cut", default=None)
#    parser.add_argument("-dwr", "--dwall_range", type = [float,float], help="range for dwall", default=None )
    return parser.parse_args()

# read arguments
args = arg_parse()

trial = Trial(args.raw_video)

#if args.t_end != -1:
#    frame_range = [args.t_start/float(trial.fps), args.t_end/float(trial.fps)]
#else:
#    frame_range = None
ti = 10 * 60 # min to sec
tf = 50 * 60 # min to sec
frame_range = [ int(ti * trial.fps), int(tf * trial.fps)]
ocut = 0
vcut = [1, 100]
wcut = [-25, 25]

#trial.group.clear_results()

trial.calculate_pairwise()

cut_tag = trial.evaluate_cuts(frame_range = frame_range, n_buffer_frames = 2, 
                              ocut = ocut, vcut = vcut, wcut = wcut)

r_dwall = [0, trial.tank.r_cm ]
trial.calculate_statistics( val_name  = [ 'dwall', 'speed', 'omega' ], 
                            val_range = [ r_dwall,   vcut,     wcut ], 
                            val_symm  = [   False,   False,    True ],
                            val_bins  = [     100,     100,     160 ],
                            frame_range = frame_range, 
                            ocut = True, vcut = True, wcut = True, tag = cut_tag)

trial.gather_pairwise(frame_range = frame_range, 
                         ocut = True, vcut = True, wcut = True, 
                         tag = cut_tag)

trial.save()
