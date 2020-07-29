#!/usr/bin/python
import sys
import os
import argparse
cvhome=os.path.join(os.path.abspath('.'),'cvtracer')
sys.path.insert(0, cvhome)
from cvt.TrAQ.Trial import Trial

def arg_parse():
    parser = argparse.ArgumentParser(description="cv-tracer Trial")
    parser.add_argument("raw_video", type=str, help="path to raw video")
    return parser.parse_args()

# read arguments
args = arg_parse()

trial = Trial(args.raw_video)

# first convert pixels to centimeters
trial.convert_pixels_to_cm()
#trial.transform_lens()
trial.calculate_kinematics()

trial.save()
