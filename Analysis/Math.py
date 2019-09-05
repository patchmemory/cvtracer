#!/usr/bin/python
import numpy as np

def mean_and_err(self, l):
    l = np.array(l)   
    mean_tmp = np.nanmean(l)
    err_tmp = np.nanstd(l)/np.sqrt(sum(~np.isnan(l)))
    return mean_tmp, err_tmp