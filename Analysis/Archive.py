#!/usr/bin/python
import os
import copy
import numpy as np
from TrAQ.Trial import Trial
import Analysis.Math as ana_math


class Archive:

    def __init__(self):
        self.trials = {}
        self.result = {}

    # Functions for storing and recalling trials
    def trial_key(self, t, n):
        return "%s_%02i" % (t, n)
    
    def trial_list(self, t, n):
        return self.trials[self.trial_key(t,n)]

    def append_trial_list(self, trial):
        self.trials[self.trial_key(trial.group.t,trial.group.n)].append(trial)

    def load_trial_set(self, trial_list, ns, ts, fps = 30, t0 = 10, tf = 30):
        self.ns     = np.array(ns)
        self.ts     = np.array(ts)
        self.framei = t0 * 60. * fps
        self.framef = tf * 60. * fps

        for t in ts:
            for n in ns:
                self.trials[self.trial_key(t,n)] = []

        f_list = open(trial_list,'r')
        for f_trial in f_list:
            f_trial = f_trial.rstrip("\n")
            fpath  = os.path.abspath(f_trial)
            fraw = fpath + "/raw.mp4"
            _trial = Trial(fraw)
            if np.isin(_trial.group.n, ns) and np.isin(_trial.group.t, ts):
                self.append_trial_list(copy.deepcopy(_trial))


    def print_trial_list(self, t, n):
        for _trial in self.trials[self.trial_key(t, n)]:            
            _trial.print_info()

    def print_sorted(self):
        print("\n\n\n    Listing all trials by type and size... \n")
        for t in self.ts:
            for n in self.ns:
                self.print_trial_list(t, n)
            print("\n")



    # Functions for storing and recalling results    
    def result_key(self, t, n, val_name, stat_name, tag = None):
        return "%s_%02i_%s_%s_%s" % (t, n, val_name, stat_name, tag)
    
    def save_result(self, t, n, val_name, stat_name, tag = None):
        self.result[self.result_key(t, n, val_name, stat_name, tag)] = result

    def get_result(self, t, n, val_name, stat_name, tag = None):
        return self.result[self.result_key(t, n, val_name, stat_name, tag)]

    def print_result(self,key):
        print(key,self.result[key])

    def print_results(self):
        for key in self.result:
            self.print_result(key)

 
#    # If speed and occlusion cuts are to be made, be sure to run them first!
#    def combine_trial_stats(self, t, n, val_name, 
#                            valmin=None, valmax=None, nbins = 100, 
#                            vcut=False, ocut=False, symm = False):
#
#        stat_keys = [ "mean", "stdd", "kurt", "hist" ]
#        stat_list = {}
#        for key in stat_keys:
#            stat_list[key] = []
#
#        for trial in self.trials[self.trial_key(t, n)]:                      
#            trial.group.calculate_stats( val_name, valmin, valmax, 
#                                         self.framei, self.framef, 
#                                         vcut, ocut, symm)
#            
#            for key in stat_keys:
#                stat_list[key].append(self.group.get_result(val_name,key))
#
#        for key in stat_keys:
#            if key == 'hist':
#                stat_result = ana_math.mean_and_err_hist(stat_list[key], nbins)
#            else:
#                stat_result = ana_math.mean_and_err(stat_list[key])
#            self.get_result(t, n, val_name, key, tag = None) = stat_result
# 
#
