#!/usr/bin/python
import sys, os
import numpy as np
import copy
import scipy.stats as spstats 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as mpl_cm
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
sys.path.insert(0, '/data1/cavefish/social/python/track')
from Trial import Trial
import Math as am


class Archive:

    def __init__(self):
        self.trials = {}
        self.result = {}

    def trial_key(self, t, n):
        return "%s_%02i" % (t, n)
    
    def result_key(self, t, n, val, tag=None):
        return "%s_%02i_%s_%s" % (t, n, val, tag)
    
    def save_result(self, t, n, val_name, stat_name, result):
        self.result[self.result_key(t, n,"%s_%s" % (val_name,stat_name))] = result

    def load_result(self, t, n, val_name, stat_name, result):
        return self.result[self.result_key(t, n, "%s_%s" % (val_name,stat_name))]

    def print_results(self):
        for key in self.result:
            print(key,self.result[key])
    
    def load_list(self, trials, ns, ts, fps = 30, t0 = 10, tf = 30):
        self.framei = t0 * 60. * fps 
        self.framef = tf * 60. * fps 
        self.ns = ns
        self.ts = ts

        for t in ts:        
            for n in ns:
                self.trials[self.trial_key(t,n)] = []

        ns = np.array(ns)
        ts = np.array(ts)
        f = open(trials,'r')
        for line in f:
            fpath  = line.rstrip("\n")
            fname = fpath.split('/')[-1]
            t = fname.split('_')[0]
            n = int(fname.split('_n')[-1].split('_')[0])
            if np.isin(n,ns) and np.isin(t,ts):
                self.trials[self.trial_key(t,n)].append(Trial(fpath))

    def print_select(self, t, n):
        for trial in range(len(self.trials[self.trial_key(t, n)])):            
            print("\n  %s, %2i %s" % ( trial.date, t, n ) )
            print("     video: %s" % ( trial.file ) )
            print("      data: %s" % ( trial.data ) )
            if trial.issue:
                print("       Known issues: " )
                print("           %s\n" % trial.issue )

    def print_sorted(self):
        for t in self.ts:
            for n in self.ns:
                self.print_select(t, n)

 
    # If speed and occlusion cuts are to be made, be sure to run them first!
    def combine_trial_stats(self, t, n, 
                            val_name, valmin=None, valmax=None, 
                            vcut=False, ocut=False, symm=False):
        mean_arr = []
        stdd_arr = []
        kurt_arr = []
        for i_file in range(len(self.d['file'])):
            if self.d['n'][i_file] == n and self.d['type'][i_file] == t:

                mean_tmp, mean_err, stdd_tmp, stdd_err, kurt_tmp, kurt_err = \
                        self.d['group'][i_file].calculate_stats(val_name, valmin, valmax, 
                              self.framei, self.framef, vcut, ocut, symm)
                print("  %4.2e %4.2e %4.2e %4.2e %4.2e %4.2e" % 
                            (mean_tmp, mean_err, stdd_tmp, stdd_err, kurt_tmp, kurt_err))
                mean_arr.append(mean_tmp)
                stdd_arr.append(stdd_tmp)
                kurt_arr.append(kurt_tmp)

        mean_mean, mean_err = am.mean_and_err(mean_arr, val_name, 'mean', t, n)
        self.save_result(val_name, 'mean', t, n, [mean_mean, mean_err])
    
        stdd_mean, stdd_err = am.mean_and_err(mean_arr, val_name, 'stdd', t, n)
        self.save_result(val_name, 'stdd', t, n, [stdd_mean, stdd_err])
    
        kurt_mean, kurt_err = am.mean_and_err(mean_arr, val_name, 'kurt', t, n)
        self.save_result(val_name, 'kurt', t, n, [kurt_mean, kurt_err])
 
        return mean_mean, mean_err, stdd_mean, stdd_err, kurt_mean, kurt_err


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