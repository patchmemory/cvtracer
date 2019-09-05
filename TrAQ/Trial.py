#!/usr/bin/python
import sys, os
import numpy as np
from Fish.Individual import Individual
from Fish.Group import Group


class Trial:
    
    def __init__(self,fpath):

        fpath  = line.rstrip("\n")
        fname = fpath.split('/')[-1]
        fishcount = int(fname.split('_n')[-1].split('_')[0])
        fishtype = fname.split('_')[0]
        fdir   = fpath.split('/')[-2]
        fyear  = fdir[ :4]
        fmonth = fdir[4:6]
        fday   = fdir[6:8]
        home_dir = fpath.split('/video/')[0]
        fdata = "%s/data/%s_%s_cv_kinematics.npy" % (home_dir,fdir,fname.split('.')[0])

        self.file   = fname
        self.group  = Group()
        self.tank   = Tank()
        self.date   = [year,month,day]
        self.stats_dict = {}
        self.issue  = []
        
        try:
          self.group = Group(int(n),t,fdata)
          print("  Loaded %s" % fdata)
        except FileNotFoundError:
          fish_data = None
          self.issue.append("Data file not found! ")
          print("  Error loaded %s" % fdata)
          
        
    def statistics(self,options):
        # basically use this to store average values of each fish in the group 
        # as well as the group average values across the group
        return
        
    def analyze_neighbors(self,n,t,d_min=0,n_buffer_frames=2):
        return
    