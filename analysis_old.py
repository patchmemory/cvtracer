#!/usr/bin/python
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
sys.path.insert(0, '/data1/cavefish/social/python/track')
from DataDictionary import DataDictionary 
from KinematicDataFrame import KinematicDataFrame
import pickle

folder = "paper-1"
fname = "%s/data.dd" % folder

try:
  f = open(fname,'rb')
  print("\n\n  Data dictionary located. Unpickling %s \n" % fname)
  unpickler = pickle.Unpickler(f)
  dd = unpickler.load()
  f.close()
  dd.print_sorted()

except FileNotFoundError:

  trials = sys.argv[1]
  ns = [ 1, 2, 5, 10 ]
  ts = [ "SF", "Pa", "Ti", "Mo" ]
  vals = [ "dw", "speed", "omega" ]
  obsv = { "dw":  0 , "speed":   1 , "omega":  2 }
  vmin = { "dw":  0 , "speed":   0 , "omega":  0 }
  vmax = { "dw": 55 , "speed": 100 , "omega": 20 }
  
  dd = DataDictionary()
  print("\n\n  No data dictionary found, so generating from %s \n" % trials)
  dd.load_list(trials,ns,ts)
  dd.print_sorted()
  
  print("  Analyzing data for speed and neighbor distance cuts... ")
  lines = []
  for t in ts:
    for n in ns:
  
      print("  Making speed cuts and combining data for %i %s... " % (n,t)) 
      dd.make_cuts(n,t,min_speed=1,d_min=0,n_buffer_frames=2)
      mean_cut, err_cut, mean_vcut, err_vcut, mean_ocut, err_ocut = dd.summarize_valid_frames(n,t)

      print("\n\n")
      print("# mean_cut err_cut mean_vcut err_vcut mean_ocut err_ocut")
      print("%4.2e %4.2e %4.2e %4.2e %4.2e %4.2e " % (mean_cut, err_cut, mean_vcut, err_vcut, mean_ocut, err_ocut))

      for val in vals:
        print("  ... %s" % val) 
        print(" statistics for %s of %i %s" % (val,n,t))
        if val == 'omega':
          mean_mean, mean_err, std_mean, std_err, kurt_mean, kurt_err = \
                       dd.combine_trial_stats_symm( n, t, val, 
                                                    valmin=vmin[val], valmax=vmax[val],
                                                    vcut=True, ocut=True )
          print("# mean_mean mean_err std_mean std_err kurt_mean kurt_err")
          print("%4.2e %4.2e %4.2e %4.2e %4.2e %4.2e " % (mean_mean, mean_err, std_mean, std_err, kurt_mean, kurt_err))

        else:
          print("# mean_mean mean_err")
          mean_mean, mean_err = dd.combine_trial_stats( n, t, val, 
                                                        valmin=vmin[val], valmax=vmax[val],
                                                        vcut=True, ocut=True )
          print("%4.2e %4.2e" % (mean_mean, mean_err))

      print("\n")
  
  
  print("  Pickling data dictionary in %s... " % fname)
  f = open(fname,'wb')
  pickle.dump(dd, f)
  f.close()
  print("  ... %s pickled. " % fname)
  


exit()

print("  Generating paper figures... ")
ns   = dd.ns
ts   = dd.ts
vals = [ "dw", "speed", "omega" ]

dwall_range = [0,55]
speed_range = [0,80]
omega_range = [0,10]

dwall_bins =  27
speed_bins =  40
omega_bins = 100

obsv      = { "dw":  0 , "speed":  1 , "omega":  2 }
binv      = { "dw": 40 , "speed": 40 , "omega": 50 }
xmin      = { "dw":  0 , "speed":  0 , "omega":  0 }
xmax      = { "dw": 55 , "speed": 90 , "omega": 10 }
xmin_logs = { "dw":  0 , "speed":  0 , "omega":  0 }
xmax_logs = { "dw": 55 , "speed": 90 , "omega": 20 } 
ftype="png"

ts_abbrev = ts[:2]
dd.plot_single_fish_distributions(ts_abbrev,ns,binv,xmin,xmax,speed_cut=True,save=True)
dd.plot_multi_fish_distributions(ts_abbrev,ns,binv,xmin,xmax,speed_cut=True,save=True)

ts = [ 'SF', 'Pa' ]
ns = [ 2, 5, 10 ]

d_max   = 105
d_bins  = 105
m_max   =   1
m_bins  = 100
d_range = [      0, d_max ]
m_range = [ -m_max, m_max ]

print(ts,ns)

for t in ts:
  for n in ns:
    k = dd.val_d_key(n,t,'dij_mij')
    print(dd.val_d[k])

dd.print_val_d_all()

dd.plot_dij_mij_log_one_cbar(ts,ns,save=True)
dd.plot_dij_mij_lin_one_cbar(ts,ns,save=True)
dd.plot_dij_mij_lin_multi_cbar(ts,ns,save=True)


ts = ["SF", "Pa", "Ti", "Mo"]
dd.plot_frac_valid_dcut(ts,ns,save=True)
ns = [ 1, 2, 5, 10 ]
dd.plot_frac_valid_vcut(ts,ns,save=True)
dd.plot_frac_valid_both(ts,ns,save=True)

def plot_mean(dd,ns,ts,val_name,val_range=None,save=False):
  plt.title(val_name)
  plt.ylabel(val_name)
  plt.xlabel("group size")
  for t in ts:
    mean_arr = []
    err_arr = []
    for n in ns:
      k = dd.val_d_key(n,t,val_name,tag='speed_cut')
      val_tmp = dd.val_d[k]
      if val_range != None:
        val_tmp = val_tmp[(val_tmp > val_range[0]) & (val_tmp < val_range[1])]
      mean_tmp = np.nanmean(np.array(val_tmp))
      err_tmp = np.nanstd(np.array(val_tmp))/np.sqrt(len(val_tmp)-1)
      print(t,n,mean_tmp,err_tmp)
      mean_arr.append(mean_tmp)
      err_arr.append(err_tmp)
    #plt.errorbar(ns,mean_arr,err_arr,fmt='o',label=t)
    plt.errorbar(ns,mean_arr,yerr=err_arr,linewidth=10,label=t)
  plt.legend()
  plt.tight_layout()

  if save:
    plt.savefig("paper01/mean_vals_%s.png" % val_name)
  else:
    plt.show()
  plt.clf()


def plot_std(dd,ns,ts,val_name,val_range=None,save=False):
  plt.title(val_name)
  plt.ylabel("standard deviation of %s" % val_name)
  plt.xlabel("group size")
  for t in ts:
    std_arr = []
    kurt_arr = []
    for n in ns:
      k = dd.val_d_key(n,t,val_name,tag='speed_cut')
      val_tmp = dd.val_d[k]
      vals = []
      vals.extend(list(val_tmp)) 
      vals.extend(list(-val_tmp)) 
      val_tmp = np.array(vals)
      if val_range != None:
        val_tmp = val_tmp[(val_tmp > val_range[0]) & (val_tmp < val_range[1])]
      std_tmp = np.nanstd(np.array(val_tmp))/np.sqrt(len(val_tmp)-1)
      std_arr.append(std_tmp)
    plt.plot(ns,std_arr,'o',markersize=20,label="%s" %t)
  plt.legend()
  plt.tight_layout()
  if save:
    plt.savefig("paper01/std_%s.png" % val_name)
  else:
    plt.show()
  plt.clf()

def plot_kurt(dd,ns,ts,val_name,val_range=None,save=False):
  plt.title(val_name)
  plt.ylabel("kurtosis of %s" % val_name)
  plt.xlabel("group size")
  for t in ts:
    std_arr = []
    kurt_arr = []
    for n in ns:
      k = dd.val_d_key(n,t,val_name,tag='speed_cut')
      val_tmp = dd.val_d[k]
      vals = []
      vals.extend(list(val_tmp)) 
      vals.extend(list(-val_tmp)) 
      val_tmp = np.array(vals)
      if val_range != None:
        val_tmp = val_tmp[(val_tmp > val_range[0]) & (val_tmp < val_range[1])]
      kurt_tmp = kurtosis(np.array(val_tmp))/np.sqrt(len(val_tmp)-1)
      kurt_arr.append(kurt_tmp)
    plt.plot(ns,kurt_arr,'o',markersize=20,label="%s" %t)
  plt.legend()
  plt.tight_layout()

  if save:
    plt.savefig("paper01/kurt_%s.png" % val_name)
  else:
    plt.show()
  plt.clf()


plot_mean(dd,ns,ts,'dw',val_range=[0,100],save=True)
plot_mean(dd,ns,ts,'speed',val_range=[0,100],save=True)
plot_std( dd,ns,ts,'omega',val_range=[-25,25],save=True)
plot_kurt(dd,ns,ts,'omega',val_range=[-25,25],save=True)

print("  ... figures generated.")
