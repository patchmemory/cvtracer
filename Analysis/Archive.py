#!/usr/bin/python
import sys
import numpy as np
sys.path.insert(0, '/data1/cavefish/social/python/cv-tracer')
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
        
    def load_trials(self, trials, ns, ts, fps = 30, t0 = 10, tf = 30):
        self.ns     = np.array(ns)
        self.ts     = np.array(ts)
        self.framei = t0 * 60. * fps 
        self.framef = tf * 60. * fps 

        for t in ts:
            for n in ns:
                self.trials[self.trial_key(t,n)] = []

        f = open(trials,'r')
        for line in f:
            fpath  = line.rstrip("\n")
            _trial = Trial(fpath)
            if np.isin(_trial.n,ns) and np.isin(_trial.t,ts):
                self.trial_list(t,n).append(_trial)


    def print_trial_list(self, t, n):
        for trial in self.trials[self.trial_key(t, n)]:            
            print("\n  %s, %2i %s" % ( trial.date, t, n ) )
            print("     video: %s" % ( trial.video ) )
            print("      data: %s" % ( trial.data ) )
            if trial.issue:
                print("       Known issues: " )
                print("           %s\n" % trial.issue )

    def print_sorted(self):
        for t in self.ts:
            for n in self.ns:
                self.print_trial_list(t, n)



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

 
    # If speed and occlusion cuts are to be made, be sure to run them first!
    def combine_trial_stats(self, t, n, val_name, 
                            valmin=None, valmax=None, nbins = 100, 
                            vcut=False, ocut=False, symm = False):

        stat_keys = [ "mean", "stdd", "kurt", "hist" ]
        stat_list = {}
        for key in stat_keys:
            stat_list[key] = []

        for trial in self.trials[self.trial_key(t, n)]:                      
            trial.group.calculate_stats( val_name, valmin, valmax, 
                                         self.framei, self.framef, 
                                         vcut, ocut, symm)
            
            for key in stat_keys:
                stat_list[key].append(self.group.get_result(val_name,key))

        for key in stat_keys:
            if key == 'hist':
                stat_result = ana_math.mean_and_err_hist(stat_list[key], nbins)
            else:
                stat_result = ana_math.mean_and_err(stat_list[key])
            self.get_result(t, n, val_name, key, tag = None) = stat_result
 

    def trial_speed_cut(self,i_file,speed_cut=1.,n_buffer_frames=2):
        self.d['group'][i_file].speed_cut(speed_cut,n_buffer_frames)


    def summarize_valid_frames(self,t, n):
        keys = ['vcut', 'ocut', 'both']
        frac_valid = {}
        mean       = {}
        err        = {}
        for key in keys:
            frac_valid[key] = []
            mean[key] = [] 
            err[key]  = []

        for i_file in range(len(self.d['file'])):
            if self.d['n'][i_file] == n and self.d['type'][i_file] == t:
                print("\n\n  Determining valid portion of frames...")
                print(self.d['file'][i_file])
                frac_both, frac_both_err, \
                frac_vcut, frac_vcut_err, \
                frac_ocut, frac_ocut_err = self.d['group'][i_file].frac_valid(self.framei,self.framef) 
                frac_valid['both'].append(1-frac_both)
                frac_valid['vcut'].append(1-frac_vcut)
                frac_valid['ocut'].append(1-frac_ocut)

        for key in frac_valid:
            frac_valid[key] = np.array(frac_valid[key])
            mean[key] = np.mean(frac_valid[key])
            err[key] = np.std(frac_valid[key])/np.sqrt(len(frac_valid[key]))
            val = "frac_valid_%s" % key
            self.result[self.result_key(t, n,val)] = [mean[key],err[key]] 

        return mean['both'], err['both'], mean['vcut'], err['vcut'], mean['ocut'], err['ocut']