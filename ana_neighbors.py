#!/usr/bin/python3
import sys, math
import numpy as np
import matplotlib.pyplot as plt
from Group import Group

fname = sys.argv[1]
group_size = int(sys.argv[2])
group_type = sys.argv[3]

grp = Group(group_size,group_type,fname)
grp.print()
grp.print_frame(10)
print("\n\n  Calculating alignment...")
grp.calculate_alignment()
print("... done.\n\n")

fps = 30
ti = 10*60 #sec
tf = 30*60 #sec
framei = ti*fps
framef = tf*fps
print("\n\n  Calculating nearest neighbor distance and alignment...")
grp.nearest_neighbor_distance(framei,framef)
#grp.plot_nn_dist()
grp.neighbor_distance(framei,framef)
print("... done.\n\n")
