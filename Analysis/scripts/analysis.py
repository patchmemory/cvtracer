#!/usr/bin/python
import argparse
from TrAQ.Trial import Trial
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

t_start = 10*60 # min
t_end = 30*60 # min
fps = 30
frame_range = [ t_start*fps, t_end*fps  ]

ns   = [ 1, 2, 5, 10 ]
ts   = [ "SF", "Pa", "Ti", "Mo" ]

val_names = [ "dwall", "speed", "omega" ]

val_range = { "dwall": [  0.,  55.] ,
              "speed": [  1., 100.] ,
              "omega": [-40.,  40.]   }

val_bins =  { "dwall":  55 ,
              "speed": 100 ,
              "omega": 160   }

ocut_min = 0

arc = Archive()
arc.load_trials(args.trial_list, ns, ts)
arc.print_sorted()

ocut = True
vcut = True
wcut = True
for t in ts:
    for n in ns:
        tag = arc.calculate_statistics( t, n, val_name = val_names,
                                        val_range = [ val_range['dwall'], 
                                                      val_range['speed'], 
                                                      val_range['omega'] ],
                                        val_symm = [ False, False, True],
                                        val_bins = [ val_bins['dwall'], 
                                                     val_bins['speed'], 
                                                     val_bins['omega'] ],
                                        frame_range = frame_range,
                                        ocut = ocut, vcut = vcut, wcut = wcut )
        for val_name in val_names:
            arc.plot_hist(t, n, val_name, tag)
            arc.plot_hist_each_group(t, n, val_name, tag)
            arc.plot_valid(t, n, frame_range, tag)

fname = "analysis.archive"
arc.save(fname)
