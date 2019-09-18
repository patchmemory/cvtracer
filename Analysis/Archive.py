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


    def print_trial_list(self, t, n):
        for _trial in self.trials[self.trial_key(t, n)]:            
            _trial.print_info()

    def print_sorted(self):
        print("\n\n\n    Listing all trials by type and size... \n")
        for t in self.ts:
            for n in self.ns:
                self.print_trial_list(t, n)
            print("\n")


    def load_trials(self, trial_list, ns, ts, fps = 30, t0 = 10, tf = 30):
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


    # Functions for storing and recalling results    
    def result_key(self, t, n, val_name, stat_name, tag = None):
        return "%s_%02i_%s_%s_%s" % (t, n, val_name, stat_name, tag)

    def get_result(self, t, n, val_name, stat_name, tag = None):
        return self.result[self.result_key(t, n, val_name, stat_name, tag)]

    def store_result(self, t, n, result, val_name, stat_name, tag = None):
        key = self.result_key(t, n, val_name, stat_name, tag)
        self.result[key] = result
        
    def print_result(self,key):
        print(key,self.result[key])

    def print_results(self):
        for key in self.result:
            self.print_result(key)

 
    # Combine results of val_name over val_range given a (t,n) pair
    def combine_stats(self, t, n, val_name, val_range = None, val_symm = False,
                        frame_range = None, nbins = 100,
                        ocut = False, vcut = False, wcut = False, tag = None):

        stat_keys = [ "mean", "stdd", "kurt", "hist" ]
        stat_list = {}
        for key in stat_keys:
            stat_list[key] = []

        for trial in self.trials[self.trial_key(t, n)]:
# for now, i will continue ensuring to only look at runs with statistics already
# calculated, but it would be good to find a way to automatically calculate
# stats for those trials that are not already prepared for this step... would 
# allow more seemless integration of new runs into the data...
#            if stats_not_calculated:
#                trial.group.calculate_stats( val_name, val_range, val_symm,
#                                             frame_range = frame_range, nbins = nbins,
#                                             ocut = ocut, vcut = vcut, wcut = wcut, tag = tag )          
            for key in stat_keys:
                stat_list[key].append(self.group.get_result(val_name,key,tag))

        for key in stat_keys:
            if key == 'hist':
                stat_result = ana_math.mean_and_err_hist(stat_list[key], nbins)
            else:
                stat_result = ana_math.mean_and_err(stat_list[key])
            #self.get_result(t, n, val_name, key, tag = None) = stat_result
            self.store_result(t, n, stat_result, val_name, key, tag)


