#!/usr/bin/python

def arg_parse():
    parser = argparse.ArgumentParser(description="cv-tracer Trial output to text")
    parser.add_argument("raw_video", type=str, help="path to raw video")
    parser.add_argument("raw_video", type=str, help="path to raw video")
    return parser.parse_args()

trial = Trial(args.raw_video)

for  
