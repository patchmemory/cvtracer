#!/usr/bin/python
import os
import argparse
from TrAQ.Trial import Trial

def arg_parse():
    parser = argparse.ArgumentParser(description="cv-tracer Trial")
    parser.add_argument("raw_video", type=str, help="path to raw video")
    parser.add_argument("n_individual", type=int, help="number of individuals in group")
    parser.add_argument("fish_type", type=str, help="type of fish in trial")
    parser.add_argument("-ts","--t_start", type=float, help="start time, in seconds", default=0)
    parser.add_argument("-te","--t_end", type=float, help="end time, in seconds", default=-1)
    parser.add_argument("-fps","--frames_per_second", type=float, help="frames per second in raw video", default=30)
    parser.add_argument("-td","--tank_diameter", type=float, help="tank diameter", default=111.)
    parser.add_argument("-ds","--datafile", type=str, help="data file-path, if not standard location") 
    parser.add_argument("-YYYYMMDD","--date", type=str, help="dat in (numeric YYYYMMDD format) of video collection", default=None)
    return parser.parse_args()

# read arguments
args = arg_parse()

trial = Trial(args.raw_video, n = args.n_individual, t = args.fish_type, date = args.date,
              fps = args.frames_per_second, tank_radius = args.tank_diameter/2., 
              t_start = args.t_start, t_end = args.t_end)

trial.save()