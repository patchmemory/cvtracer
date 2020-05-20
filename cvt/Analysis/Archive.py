#!/usr/bin/python
import os
import sys
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from TrAQ.Trial import Trial
import Analysis.Math as ana_math


class Archive:

    def __init__(self):
        self.trials = {}
        self.result = {}

        self.valid_speed_min = 0.5

    def save(self, fname = None):
        if fname != None:
            self.fname = fname
        try:
            f = open(self.fname, 'wb')
            pickle.dump(self.__dict__, f, protocol = 3)
            sys.stdout.write("\n        Archive saved as %s \n" % self.fname)
            sys.stdout.flush()
            f.close()
            return True
        except:
            return False

    def load(self, fname = None):
        if fname != None:
            self.fname = fname
        try:
            f = open(self.fname, 'rb')
            tmp_dict = pickle.load(f)
            f.close()
            self.__dict__.update(tmp_dict) 
            sys.stdout.write("\n        Archive loaded from %s \n" % self.fname)
            sys.stdout.flush()
            return True
        except:
            return False


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

    ######################################################
    # some functions for storing and retrieving results
    #####################################################

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
    def calculate_statistics(self, t, n,
                             val_name  = [ 'dwall', 'speed', 'omega' ], 
                             val_range = [    None,    None,    None ], 
                             val_symm  = [   False,   False,    True ],
                             val_bins  = [     100,     100,     100 ],
                             frame_range = None, n_buffer_frames = 2,
                             ocut = False, vcut = False, wcut = False):

        
        stat_keys = [ "mean", "stdd", "kurt", "hist" ]
        self.stat_list = {}
        
        for val in val_name:
            for stat in stat_keys:
                key = val + stat
                self.stat_list[key] = []        

        inactive = []
        exceptions = []
        tag = ""
        for trial in self.trials[self.trial_key(t, n)]:
            print("  Working on..." )
            trial.print_info()
            try:
                tag = trial.evaluate_cuts(frame_range, n_buffer_frames,
                                              0, val_range[1], val_range[2])

                # test to make sure trial has enough active time
                if trial.cuts_stats['mean']['vcut'] >= self.valid_speed_min:
    
                    trial.calculate_statistics( val_name, val_range, val_symm, val_bins, 
                                            frame_range, ocut, vcut, wcut, tag)
                    for val in val_name:
                        for stat in stat_keys:
                            key = val + stat
                            self.stat_list[key].append(trial.group.get_result(val, stat, tag))
                            print(" key, len(stat_list): ", key, len(self.stat_list[key]))
    
                else:
                    print("\n\n")
                    print("    Warning! Trial has too few valid frames. ")
                    print("             Minimum fraction of valid frames = %5.3f" %
                                                              self.valid_speed_min)
                    inactive.append(trial.fvideo_raw)
                    print(" number inactive: ", len(exceptions))

            except:
                print("\n\n")
                print("    Warning! Issue evaluating trial cuts and/or stats. ")
                print("             Will skip for now.\n\n")
                exceptions.append(trial.fvideo_raw)
                print(" number of exceptions: ", len(exceptions))

        print(" t, n = ", t, n)
        print("  number of exceptions: ", len(exceptions))
        print("  number of inactive f: ", len(inactive))
        print(self.stat_list)

        for i in range(len(val_name)):
            val = val_name[i]
            for stat in stat_keys:

                key = val + stat
                print(" key, len(stat_list): ", key, len(self.stat_list[key]))

                if stat == 'hist':
                    stat_result = ana_math.mean_and_err_hist(self.stat_list[key], val_bins[i])
                else:
                    stat_result = ana_math.mean_and_err(self.stat_list[key])
                    
                self.store_result(t, n, stat_result, val, stat, tag)

        return tag, exceptions

    #################################################
    # plot functions
    #################################################


    def plot_hist(self, t, n, val_name, tag = None, save = True):
        h = self.get_result(t, n, val_name, 'hist', tag)
        plt.fill_between(h[:,0], h[:,1] - h[:,2], h[:,1] + h[:,2], color = 'blue', label='cross-fish error')
        plt.plot(h[:,0], h[:,1], color='red', linewidth=0.5, label='cross-fish mean')
        mean = self.get_result(t, n, val_name, 'mean', tag)
        print(" MEAN IS: ", mean)
        plt.axvline(x = mean[0], color = 'green', linewidth = 3, linestyle = '-', label = 'distribution mean')
        plt.axvline(x = mean[0] - mean[1], color = 'green', linewidth = 1, linestyle = '--', label = 'distribution error')
        plt.axvline(x = mean[0] + mean[1], color = 'green', linewidth = 1, linestyle = '--')
        plt.legend()
        plt.tight_layout()
        if save:
            fig_name = "results/%s_n%02i_%s_hist_%s.png" % (t, n, val_name, tag)
            plt.savefig(fig_name)
        else:
            plt.show()
        plt.clf()
        
        
    def plot_hist_each_group(self, t, n, val_name, tag = None, save = True):
        hist = []
        mean = []
        for trial in self.trials[self.trial_key(t, n)]:
            print("  Plotting ")
            trial.print_info()
            try:
                # test to make sure trial has enough active time
                if trial.cuts_stats['mean']['vcut'] >= self.valid_speed_min:
                    trial_hist = trial.get_group_result(val_name, 'hist', tag)
                    trial_mean = trial.get_group_result(val_name, 'mean', tag)
                    hist.append(trial_hist)
                    mean.append(trial_mean)
                else:
                    print("\n\n")
                    print("    Warning! Trial has too few valid frames. ")
                    print("             Minimum fraction of valid frames = %5.3f" %
                                                              self.valid_speed_min)
            except:
                    print("\n\n")
                    print("    Warning! Issue gathering results from: ")

                
        color_set = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i in range(len(hist)):
            h = hist[i]
            c = color_set[i%len(color_set)]
            plt.axvline(x = mean[i][0], color = c, linewidth = 1)
            plt.plot(h[:,0], h[:,1], color = c, linewidth = 1, label=i)
        plt.xlabel(val_name)
        plt.ylabel("normalized count")
        plt.legend()
        plt.tight_layout()
        if save:
            fig_name = "results/%s_n%02i_%s_hist_each_group_%s.png" % (t, n, val_name, tag)
            plt.savefig(fig_name)
        else:
            plt.show()
        plt.clf()
        
        
    def plot_valid(self, t, n, frame_range = None, tag = None, save = True):
        cuts = [ 'ocut', 'vcut', 'wcut', 'cut']
        valid = { cut: [] for cut in cuts}
        for trial in self.trials[self.trial_key(t, n)]:
            print("  Plotting ")
            trial.print_info()
            try:
                for cut in cuts:
                    valid_tmp = trial.group.valid_frame_fraction(frame_range, cut_name = cut)
                    valid[cut].append(np.nanmean(valid_tmp))

                # test to make sure trial has enough active time
                if trial.cuts_stats['mean']['vcut'] >= self.valid_speed_min:
                    print("\n\n")
                    print("    Trial has enough active frames for analysis. ")
                    trial.print_info()
                    # for cut in cuts:
                    #     valid_tmp = trial.group.valid_frame_fraction(frame_range, 
                    #                                                  cut_name = cut)
                    #     valid[cut].append(np.nanmean(valid_tmp))
                else:
                    print("\n\n")
                    print("    Warning! Trial has too few valid frames. ")
                    print("             Minimum fraction of valid frames = %5.3f" %
                                                              self.valid_speed_min)
                    trial.print_info()
            except:
                print("\n\n")
                print("    Warning! Issue gathering results from: ")
                trial.print_info()

                
        index = np.arange(len(cuts))
        #n_trials = len(self.trials[self.trial_key(t,n)])
        n_trials = len(valid['cut'])
        for i in range(n_trials):
            valid[i] = []
            for cut in cuts:
                valid[i].append(valid[cut][i])
            valid[i] = np.array(valid[i])
   
        gutter = 0.1
        bar_width = ( 1 - gutter ) / n_trials
        opacity = 0.8
        for i in range(n_trials):
            plt.bar(index + i*bar_width + 0.5*gutter, valid[i], bar_width, alpha = opacity, label=i)

        plt.xlabel('fraction valid after cuts')
        plt.xlabel('cut')
        plt.xticks(index + 0.5 - 0.5*gutter, cuts)
        plt.ylim([0,1])
        plt.legend()
        plt.tight_layout()        
        if save:
            fig_name = "results/%s_n%02i_valid_by_group_%s.png" % (t, n, tag)
            plt.savefig(fig_name)
        else:
            plt.show()
        plt.clf()

