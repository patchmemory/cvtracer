#!/usr/bin/python
import numpy as np


def mean_and_err(self, l, hist=False):
    l = np.array(l)
    mean_tmp = np.nanmean(l)
    err_tmp = np.nanstd(l) / np.sqrt(sum(~np.isnan(l)))
    return np.array([mean_tmp, err_tmp])

def mean_and_err_hist(self, l, nbins):
    l = np.array(l)
    # first test that all entries have desired number of bins
    valid = [ True for i in len(l) ]
    for i in range(len(l)):
        if len(l) != nbins:
            print("Incorrect number of bins in array %i." % i)
            valid[i] = False
        
    hist_mean = np.zeros((nbins,2),dtype=float)
    for i in range(len(l)):
        bin_mean = np.nanmean(l[valid,i])
        bin_err  = np.nanstd(l[valid,i]) / \
                        np.sqrt(sum(~np.isnan(l[valid,i])))
                        
        hist_mean[i][0] = bin_mean
        hist_mean[i][1] = bin_err
        
    return np.array(hist_mean)