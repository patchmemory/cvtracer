#!/usr/bin/python
import sys
cvhome="/home/patch/Code/cvtracer"
sys.path.insert(0, cvhome)
import os
import argparse
from TrAQ.Tank import Tank

def arg_parse():
    parser = argparse.ArgumentParser(description="cv-tracer tank init via older text format")
    parser.add_argument("raw_video", type=str, help="path to raw video")
    parser.add_argument("tank_file", type=str, help="path to tank.dat")
    parser.add_argument("-d", "--diameter", type=str, help="diameter of tank",default=111.)
    return parser.parse_args()

# read arguments
args = arg_parse()

tank = Tank(args.raw_video, args.diameter)
tank.load_txt(args.tank_file)
tank.save()
