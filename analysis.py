#!/usr/bin/python
import argparse
from Analysis.Archive import Archive

def arg_parse():
    parser = argparse.ArgumentParser(description="cv-tracer cross-trial analysis")
    parser.add_argument("trial_list", type=str, help="list of trials to combine")
#    parser.add_argument("-ts","--t_start", type=float, help="start time, in seconds", default=0)
#    parser.add_argument("-te","--t_end", type=float, help="end time, in seconds", default=-1)
#    parser.add_argument("-nbf","--n_buffer_frames", type=float, help="number of buffer frames for cuts", default=2)
#    parser.add_argument("-oc", "--ocut", type = [float,float], help="range for occlusion cut", default=None)
#    parser.add_argument("-vc", "--ocut", type = [float,float], help="range for speed cut", default=None)
#    parser.add_argument("-wc", "--wcut", type = [float,float], help="range for omega cut", default=None)
#    parser.add_argument("-dwr", "--dwall_range", type = [float,float], help="range for dwall", default=None )
    return parser.parse_args()

args = arg_parse()

ns   = [ 1, 2, 5, 10 ]
ts   = [ "SF", "Pa", "Ti", "Mo" ]

val_names = [ "dwall", "speed", "omega" ]

val_range = { "dwall": [  0.,  55.] ,
              "speed": [  1., 100.] ,
              "omega": [-40.,  40.]   }

val_bins =  { "dwall":  55 ,
              "speed": 100 ,
              "omega": 160   }


arc = Archive()
arc.load_trials(args.trial_list, ns, ts)
arc.print_sorted()