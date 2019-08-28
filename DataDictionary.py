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
from Fish import KinematicDataFrame
from KinematicDataFrame import KinematicDataFrame
from Group import Group


def aspect_ratio(x_range,y_range):
  xlen = x_range[1] - x_range[0]
  ylen = y_range[1] - y_range[0]
  return xlen/ylen


class DataDictionary:

  def __init__(self):
    self.d = {'file':[], 'data':[], 'type':[], 'n':[], 'year':[], 'month':[], 'day':[], 'group':[], 'issue':[]}


  def load_list(self,trials,ns,ts,fps=30):
    start_minute = 10
    end_minute = 30
    self.framei = start_minute * 60 * fps 
    self.framef =   end_minute * 60 * fps 
    self.ns = ns
    self.ts = ts
    self.val_d = {}

    ns = np.array(ns)
    ts = np.array(ts)
    f = open(trials,'r')
    for line in f:
      fpath  = line.rstrip("\n")
      fname = fpath.split('/')[-1]
      n     = int(fname.split('_n')[-1].split('_')[0])
      fishtype = fname.split('_')[0]

      if np.isin(n,ns) and np.isin(fishtype,ts):

        fdir   = fpath.split('/')[-2]
        fyear  = fdir[ :4]
        fmonth = fdir[4:6]
        fday   = fdir[6:8]
        home_dir = fpath.split('/video/')[0]
        fdata = "%s/data/%s_%s_cv_kinematics.npy" % (home_dir,fdir,fname.split('.')[0])
        
        self.d['file'].append(fpath)
        self.d['data'].append(fdata)
        self.d['type'].append(fishtype)
        self.d['n'].append(int(n))
        self.d['day'].append(fday)
        self.d['month'].append(fmonth)
        self.d['year'].append(fyear)
    
        try:
          fish_data = Group(int(n),fishtype,fdata)
          self.d['issue'].append(None)
          print("  Loaded %s" % fdata)
        except FileNotFoundError:
          fish_data = None
          self.d['issue'].append("Data file not found! ")
          print("  Error loaded %s" % fdata)
          
        self.d['group'].append(fish_data)


  def print_select(self,n_select,t_select):
    for i in range(len(self.d['file'])):
      if self.d['n'][i] == n_select and self.d['type'][i] == t_select:
        print("")
        print("  %s/%s/%s, %2i %s" % ( 
                  self.d['month'][i],self.d['day'][i],self.d['year'][i], 
                  self.d['n'][i],self.d['type'][i]) )
        print("    video: %s" % ( self.d['file'][i] ) )
        print("     data: %s" % ( self.d['data'][i] ) )
        if self.d['issue'][i]:
          print("       Known issues: ")
          print("           %s" % self.d['issue'][i])
    print("")

 
  def print_sorted(self):
    for n in self.ns:
      for t in self.ts:
        self.print_select(n,t)

  def print_val_d_all(self):
    for key in self.val_d:
      print(key,self.val_d[key])


  def combined_trials_stats(self,n,t,val,vals,tb="",range=None):
    print("%i %s %s %s" % (n,t,val,tb))
    vals = vals[np.logical_not(np.isnan(vals))]
    if val == 'omega':
      vals_tmp = []
      vals_tmp.extend(list(vals))
      vals_tmp.extend(list(-vals))
      vals = vals_tmp

    if range != None:
      vals = vals[(vals > range[0]) & (vals < range[1])]

    self.val_d[self.val_d_key(n,t,val,tag=tb+'mean')]   = np.mean(vals)
    self.val_d[self.val_d_key(n,t,val,tag=tb+'median')] = np.median(vals)
    self.val_d[self.val_d_key(n,t,val,tag=tb+'std')]    = np.std(vals)
    if len(vals) > 0:
      self.val_d[self.val_d_key(n,t,val,tag=tb+'stderr')] = np.std(vals)/np.sqrt(len(vals))
    else:
      self.val_d[self.val_d_key(n,t,val,tag=tb+'stderr')] = 0 
    self.val_d[self.val_d_key(n,t,val,tag=tb+'kurtosis')] = spstats.kurtosis(vals,fisher=False)


  def combined_trials_stats_by_n(self,ns,t,val,val_stat,tb=""):
    stats_by_n = np.zeros((len(ns),2))
    for i in range(len(ns)): 
      key = self.val_d_key(ns[i],t,val,tag=tb+val_stat)
      if not key in self.val_d: 
        if tb == 'speed_cut':
          self.combined_trials_speed_cut(ns[i],t,val)
        else:
          self.combined_trials(ns[i],t,val)
      stats_by_n[i] = [ ns[i], self.val_d[key] ]
    return stats_by_n


  def combined_trials(self,n,t,val):
    vals = []
    for i in range(len(self.d['file'])):
      if self.d['n'][i] == n and self.d['type'][i] == t:
        for i_fish in range(n):
          try:
            vals.extend(
              self.d['group'][i].fish[i_fish].df[val][self.framei:self.framef].tolist() )
          except TypeError:
            fname = self.d['file'][i].split('/')[-1]
            fdate = self.d['file'][i].split('/')[-2]
            print("No data was found for %s/%s." % (fdate,fname))
    vals = np.array(vals) 
    self.combined_trials_stats(n,t,val,vals)
    if val == 'omega':
      vals = abs(vals)
    self.val_d[self.val_d_key(n,t,val)] = vals
    return vals


  def combined_trials(self,n,t,val):
    vals = []
    for i in range(len(self.d['file'])):
      if self.d['n'][i] == n and self.d['type'][i] == t:
        for i_fish in range(n):
          try:
            vals.extend(
              self.d['group'][i].fish[i_fish].df[val][self.framei:self.framef].tolist() )
          except TypeError:
            fname = self.d['file'][i].split('/')[-1]
            fdate = self.d['file'][i].split('/')[-2]
            print("No data was found for %s/%s." % (fdate,fname))
    vals = np.array(vals) 
    self.combined_trials_stats(n,t,val,vals)
    if val == 'omega':
      vals = abs(vals)
    self.val_d[self.val_d_key(n,t,val)] = vals
    return vals


  # If speed and occlusion cuts are to be made, be sure to run them first!
  def combine_trial_stats(self,n,t,val_name,valmin=None,valmax=None,vcut=None,ocut=None):
    # first check to be sure cuts
    mean_arr = []
    for i_file in range(len(self.d['file'])):
      if self.d['n'][i_file] == n and self.d['type'][i_file] == t:

        mean_tmp, err_tmp = self.d['group'][i_file].calculate_stats(val_name,
                                  valmin, valmax, self.framei, self.framef, vcut, ocut)
        print("  %4.2e %4.2e" % (mean_tmp, err_tmp))
        mean_arr.append(mean_tmp)

    mean_arr  = np.array(mean_arr)
    mean_mean = np.nanmean(mean_arr)
    mean_err  = np.nanstd(mean_arr)/np.sqrt(sum(~np.isnan(mean_arr)))

    self.val_d[self.val_d_key(n,t,"%s_mean" % val_name)] = [mean_mean, mean_err]
    return mean_mean, mean_err
    

  # If speed and occlusion cuts are to be made, be sure to run them first!
  def combine_trial_stats_symm(self,n,t,val_name,valmin=None,valmax=None,vcut=None,ocut=None):
    # first check to make sure cuts
    mean_arr = []
    std_arr = []
    kurt_arr = []
    for i_file in range(len(self.d['file'])):
      if self.d['n'][i_file] == n and self.d['type'][i_file] == t:

        mean_tmp, mean_err, std_tmp, std_err, kurt_tmp, kurt_err = \
                                  self.d['group'][i_file].calculate_stats_symm(val_name,
                                      valmin, valmax, self.framei, self.framef, vcut, ocut)
        print("  %4.2e %4.2e %4.2e %4.2e %4.2e %4.2e" % 
                            (mean_tmp, mean_err, std_tmp, std_err, kurt_tmp, kurt_err))

        mean_arr.append(mean_tmp)
        std_arr.append(std_tmp)
        kurt_arr.append(kurt_tmp)

    mean_arr  = np.array(mean_arr) 
    mean_mean = np.nanmean(mean_arr)
    mean_err  = np.nanstd(mean_arr) / np.sqrt(sum(~np.isnan(mean_arr)))
    self.val_d[self.val_d_key(n,t,"%s_mean" % val_name)] = [mean_mean, mean_err]
    
    std_arr  = np.array(std_arr) 
    std_mean = np.nanmean(std_arr)
    std_err  = np.nanstd(std_arr) / np.sqrt(sum(~np.isnan(std_arr)))
    self.val_d[self.val_d_key(n,t,"%s_std" % val_name)] = [std_mean, std_err]
    
    kurt_arr  = np.array(kurt_arr) 
    kurt_mean = np.nanmean(kurt_arr)
    kurt_err  = np.nanstd(kurt_arr) / np.sqrt(sum(~np.isnan(kurt_arr)))
    self.val_d[self.val_d_key(n,t,"%s_kurt" % val_name)] = [kurt_mean, kurt_err]

    return mean_mean, mean_err, std_mean, std_err, kurt_mean, kurt_err



  def trial_speed_cut(self,i_file,speed_cut=1.,n_buffer_frames=2):
    self.d['group'][i_file].speed_cut(speed_cut,n_buffer_frames)

  def combined_trials_speed_cut(self,n,t,val,speed_cut=1.,n_buffer_frames=2):
    vals_filtered = []
    frac_valid = []
    for i_file in range(len(self.d['file'])):
      if self.d['n'][i_file] == n and self.d['type'][i_file] == t:
        self.trial_speed_cut(i_file)
        for i_fish in range(n):
          try:
            vcut = np.array(self.d['group'][i_file].fish[i_fish].df['speed_cut'][self.framei:self.framef].tolist())
            dcut = np.array(self.d['group'][i_file].fish[i_fish].df['d_cut'][self.framei:self.framef].tolist())
            cut = dcut | vcut
            vals = np.array(self.d['group'][i_file].fish[i_fish].df[val][self.framei:self.framef].tolist())
          except TypeError:
            fname = self.d['file'][i_file].split('/')[-1]
            fdate = self.d['file'][i_file].split('/')[-2]
            print("No %s data was found for %s/%s." % (val,fdate,fname))

          vals_valid = list(vals[np.logical_not(cut)])
          n_total_tmp = len(vals)
          n_valid_tmp = len(vals_valid) 
          vals_filtered.extend(list(vals[np.logical_not(cut)]))
          frac_valid.append(n_valid_tmp/n_total_tmp)

    vals_filtered = np.array(vals_filtered)
    tb="speed_cut"
    self.combined_trials_stats(n,t,val,vals_filtered,tb=tb)
    if val == 'omega':
      vals_filtered = abs(vals_filtered)
    self.val_d[self.val_d_key(n,t,val,tag=tb)] = vals_filtered

    frac_valid = np.array(frac_valid)
    self.val_d[self.val_d_key(n,t,'vcut_frac_valid')] = frac_valid

    return vals_filtered


  def plot_hist_across_n(self,ns,t,val,val_title,nbins=10,hrange=None,norm=True,speed_cut=False):
    plt.title("Histograms of %s for %s across group size" % (val_title,t))
    plt.ylabel("Normalized Count") 
    plt.xlabel("%s" % val_title) 
    for i in range(len(ns)):  
      if speed_cut:
        hist, bin_edges = np.histogram(self.combined_trials_speed_cut(ns[i],t,val),range=hrange,bins=nbins,density=norm)
      else:
        hist, bin_edges = np.histogram(self.combined_trials(ns[i],t,val),range=hrange,bins=nbins,density=norm)
      binc = ( bin_edges[1:] + bin_edges[:-1] ) / 2 
      plt.plot(binc,hist,alpha=0.7,lw=2,label="%s fish"%ns[i])
    plt.legend()
    plt.show()


  def plot_hist_across_t(self,n,ts,val,val_title,nbins=10,hrange=None,norm=True,speed_cut=False):
    plt.title("Histograms of %s for group size %i across type" % (val_title,n))
    plt.ylabel("Normalized Count") 
    plt.xlabel("%s" % val_title) 
    for i in range(len(ts)):  
      if speed_cut:
        hist, bin_edges = np.histogram(self.combined_trials_speed_cut(n,ts[i],val),range=hrange,bins=nbins,density=norm)
      else:
        hist, bin_edges = np.histogram(self.combined_trials(n,ts[i],val),range=hrange,bins=nbins,density=norm)
      binc = ( bin_edges[1:] + bin_edges[:-1] ) / 2 
      plt.plot(binc,hist,alpha=0.7,lw=2,label="%s"%ts[i])
    plt.legend()
    plt.show()


  def val_d_key(self,n,t,val,tag=None):
    return "%i%s%s%s" % (n,t,val,tag)


  def plot_hist_singles_all(self,n,t,val,val_title,nbins=10,hrange=None,norm=True):
    plt.title("Histogram of %s for groups of %i %s" % (val_title,n,t))
    plt.ylabel("Normalized Count") 
    plt.xlabel("%s" % val_title) 
    for i in range(len(self.d['file'])):
      if self.d['n'][i] == n and self.d['type'][i] == t:
        for i_fish in range(n):
          try:
            plt.hist(self.d['group'][i].fish[i_fish].df[val][self.framei:self.framef],range=hrange,bins=nbins,density=norm,alpha=0.5,lw=3)
          except TypeError:
            fname = self.d['file'][i].split('/')[-1]
            fdate = self.d['file'][i].split('/')[-2]
            print("No data was found for %s/%s." % (fdate,fname))
    plt.show()


  def plot_hist_singles_each(self,n,t,val,val_title,nbins=10,hrange=None,norm=True):
    for i in range(len(self.d['file'])):
      if self.d['n'][i] == n and self.d['type'][i] == t:
        fname = self.d['file'][i].split('/')[-1]
        fdate = self.d['file'][i].split('/')[-2]

        plt.title("Histogram of %s for the group of %i %s in\n%s/%s" % 
                            (val_title,n,t,fdate,fname) )
        plt.ylabel("Normalized Count") 
        plt.xlabel("%s" % val_title) 
        for i_fish in range(n):
          try:
            plt.hist(self.d['group'][i].fish[i_fish].df[val][self.framei:self.framef],range=hrange,bins=nbins,density=norm,alpha=0.5,lw=3)
          except TypeError:
            try:
              print("No data was found for %s/%s." % (fdate,fname))
            except:
              print("No data was found for %s." % (fdate,fname))
        plt.show()

        
  def plot_hist_combined(self,n,t,val,val_title,nbins=10,hrange=None,norm=True):
    val_list = []
    plt.title("Combined histogram %s for groups of %i %s" % (val_title,n,t))
    plt.ylabel("Normalized Count") 
    plt.xlabel("%s" % val_title) 
    for i in range(len(self.d['file'])):
      if self.d['n'][i] == n and self.d['type'][i] == t:
        for i_fish in range(n):
          try:
            val_list.extend(
              self.d['group'][i].fish[i_fish].df[val][self.framei:self.framef].tolist() )
          except TypeError:
            fname = self.d['file'][i].split('/')[-1]
            fdate = self.d['file'][i].split('/')[-2]
            print("No data was found for %s/%s." % (fdate,fname))
    plt.hist(val_list,range=hrange,bins=nbins,density=norm,alpha=0.5,lw=3)
    plt.show()


  def plot_single_fish_distributions(self,ts,ns,binv,xmin,xmax,speed_cut=False,ftype='png',tag="",save=False):
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    aspect_ratio = 1
    n=1
    norm=True
    
    v = 'dw'
    v_label = 'distance to wall'
    hrange = [xmin[v],xmax[v]]
    nbins = binv[v]
    for t in ts:
      if speed_cut:
        hist, bin_edges = np.histogram(self.combined_trials_speed_cut(n,t,v),range=hrange,bins=nbins,density=norm)
      else:
        hist, bin_edges = np.histogram(self.combined_trials(n,t,v),range=hrange,bins=nbins,density=norm)
      binc = ( bin_edges[1:] + bin_edges[:-1] ) / 2 
      ax[0].plot(binc, hist/(2*np.pi*(111./2)- binc),label="%s" % (t))
    ax[0].set_xlabel("%s" % v_label)
    ax[0].set_ylabel("frequency")
    ax[0].set_xlim((xmin[v],xmax[v]))
    ax[0].set_ylim(bottom=0)
    ax[0].legend()
    ymin, ymax = ax[0].get_ylim() 
    ax[0].set_aspect((xmax[v]-xmin[v])/(ymax-ymin)/aspect_ratio)
    
    v = 'speed'
    v_label = 'speed'
    hrange = [xmin[v],xmax[v]]
    nbins = binv[v]
    for t in ts:
      if speed_cut:
        hist, bin_edges = np.histogram(self.combined_trials_speed_cut(n,t,v),range=hrange,bins=nbins,density=norm)
      else:
        hist, bin_edges = np.histogram(self.combined_trials(n,t,v),range=hrange,bins=nbins,density=norm)
      binc = ( bin_edges[1:] + bin_edges[:-1] ) / 2 
      ax[1].plot(binc, hist, label="%s" % (t))
    ax[1].set_xlabel("%s" % v_label)
    ax[1].set_xlim((xmin[v],xmax[v]))
    ax[1].set_ylim(bottom=0)
    ymin, ymax = ax[1].get_ylim() 
    ax[1].set_aspect((xmax[v]-xmin[v])/(ymax-ymin)/aspect_ratio)
  
    v = 'omega'
    v_label = 'angular velocity'
    hrange = [xmin[v],xmax[v]]
    nbins = binv[v]
    for t in ts:
      if speed_cut:
        hist, bin_edges = np.histogram(self.combined_trials_speed_cut(n,t,v),range=hrange,bins=nbins,density=norm)
      else:
        hist, bin_edges = np.histogram(self.combined_trials(n,t,v),range=hrange,bins=nbins,density=norm)
      binc = ( bin_edges[1:] + bin_edges[:-1] ) / 2 
      ax[2].plot(binc, hist, label="%s" % (t))
    ax[2].set_xlabel("%s" % v_label)
    ax[2].set_xlim((0,xmax[v]))
    ymin, ymax = ax[2].get_ylim() 
    ax[2].set_aspect((xmax[v]-0)/(ymax-ymin)/aspect_ratio)
  
    ax2inset = inset_axes(ax[2], width="43%", height="43%", loc=1, borderpad=1.4)
    hrange = [xmin[v],2*xmax[v]]
    nbins = binv[v]
    for t in ts:
      if speed_cut:
        hist, bin_edges = np.histogram(self.combined_trials_speed_cut(n,t,v),range=hrange,bins=nbins,density=norm)
      else:
        hist, bin_edges = np.histogram(self.combined_trials(n,t,v),range=hrange,bins=nbins,density=norm)
      binc = ( bin_edges[1:] + bin_edges[:-1] ) / 2 
      ax2inset.plot(binc, hist, label="%s" % (t))
    ax2inset.set_xlabel("%s" % v_label)
    ax2inset.set_xlim((0,2*xmax[v]))
    ax2inset.set_yscale('log')
  
    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.05,wspace=0.05)
    if save:
      plt.savefig("paper01/distribs_compare_singlefish%s.png" % tag)
    else:
      plt.show()

    plt.clf()


#  def distance(self,[x1,y1],[x2,y2]):
#    return np.sqrt(pow(x1-x2,2) + pow(y1-y2,2))
#
#  def direction(self,[x1,y1],[x2,y2]):
#    return x2-x1, y2-y1
#
#  def midpoint(self,[x1,y1],[x2,y2]):
#    return (x1+x2)/2 + (y1+y2)/2


  def neighbor_distance(self,n,t,dij_min=0):
    if n > 1:
      n_dist = []
      nn_dist = []
      mean_n_dist = []
      mean_nn_dist = []
      dij_mij = []
      count = 0
      for i in range(len(self.d['file'])):
        if self.d['n'][i] == n and self.d['type'][i] == t:
          print(self.d['file'][i])
          count += 1
          print("\n\n  Calculating neighbor distances...")
          self.d['group'][i].calculate_distance_alignment()
          print("... done.\n\n")
          n_dist_tmp  = self.d['group'][i].neighbor_distance(self.framei,self.framef)
          nn_dist_tmp = self.d['group'][i].nearest_neighbor_distance(self.framei,self.framef)
          dij_mij_tmp = self.d['group'][i].neighbor_distance_alignment(self.framei,self.framej)
          mean_n_dist.append(np.mean(n_dist_tmp))
          mean_nn_dist.append(np.mean(nn_dist_tmp))
          n_dist.extend(list(n_dist_tmp))
          nn_dist.extend(list(nn_dist_tmp))
          dij_mij.extend(list(dij_mij_tmp))
      n_dist       = np.array(n_dist      )
      nn_dist      = np.array(nn_dist     )
      mean_n_dist  = np.array(mean_n_dist )
      mean_nn_dist = np.array(mean_nn_dist)
      mean_n_dist_avg  = np.mean(mean_n_dist)
      err_n_dist_avg   = np.sqrt(np.std(self.mean_n_dist)/count)
      mean_nn_dist_avg = np.mean(self.mean_nn_dist)
      err_nn_dist_avg  = np.sqrt(np.std(self.mean_nn_dist)/count)

      val = 'dij_mij'
      dij_mij = np.array(dij_mij)
      self.val_d[self.val_d_key(n,t,val)] = dij_mij

      print("%s %i %e %e %e %e" % (t,n,mean_n_dist,err_n_dist,mean_nn_dist,err_nn_dist))
      return mean_n_dist,err_n_dist,mean_nn_dist,err_nn_dist
    else:
      print("  Only one fish in this trial, so no neighbors...")
      return 0,0,0,0

  def analyze_neighbors(self,n,t,d_min=0,n_buffer_frames=2):
    frac_valid_vcut = []
    frac_valid_dcut = []
    frac_valid_both = []
    dij_mij = []
    for i_file in range(len(self.d['file'])):
      if self.d['n'][i_file] == n and self.d['type'][i_file] == t:
        print(self.d['file'][i_file])

        if n > 1:
          print("\n\n  Calculating neighbor distance and alignment...")
          self.d['group'][i_file].calculate_distance_alignment()
          print("  ... done.\n\n")
         
          print("\n\n  Making neighbor distance cut...")
          self.d['group'][i_file].neighbor_distance_cut(d_min,n_buffer_frames)
          dij_mij_tmp, frac_valid_both_tmp, frac_valid_vcut_tmp, frac_valid_dcut_tmp = self.d['group'][i_file].valid_distance_alignment(self.framei,self.framef)
          dij_mij.extend(list(dij_mij_tmp))
        else:
          print("\n\n  Single fish, no neighbors... ")
          self.d['group'][i_file].neighbor_distance_cut(d_min,n_buffer_frames)
          dij_mij_tmp, frac_valid_both_tmp, frac_valid_vcut_tmp, frac_valid_dcut_tmp = self.d['group'][i_file].valid_distance_alignment(self.framei,self.framef)

        frac_valid_vcut.append(frac_valid_vcut_tmp)
        frac_valid_dcut.append(frac_valid_dcut_tmp)
        frac_valid_both.append(frac_valid_both_tmp)
        print("  ... done.\n\n")

    val = 'dij_mij'
    dij_mij = np.array(dij_mij)
    self.val_d[self.val_d_key(n,t,val)] = dij_mij

    val = 'frac_valid_vcut'
    frac_valid_vcut = np.array(frac_valid_vcut)
    self.val_d[self.val_d_key(n,t,val)] = frac_valid_vcut

    val = 'frac_valid_dcut'
    frac_valid_dcut = np.array(frac_valid_dcut)
    self.val_d[self.val_d_key(n,t,val)] = frac_valid_dcut

    val = 'frac_valid_both'
    frac_valid_both = np.array(frac_valid_both)
    self.val_d[self.val_d_key(n,t,val)] = frac_valid_both


  def make_cuts(self,n,t,min_speed=1,d_min=0,n_buffer_frames=2):
    for i_file in range(len(self.d['file'])):
      if self.d['n'][i_file] == n and self.d['type'][i_file] == t:
        print(self.d['file'][i_file])
        if n > 1:
          try:
            len(self.d['group'][i_file].d_M)
            print("\n\n Neighbor distance and alignment found.")
          except AttributeError:
            print("\n\n  Calculating neighbor distance and alignment...")
            self.d['group'][i_file].calculate_distance_alignment()
            print("  ... done.\n\n")

        self.d['group'][i_file].cut_occlusions(d_min,n_buffer_frames)
        self.d['group'][i_file].cut_inactive(min_speed,n_buffer_frames)

        print("  ... done.\n\n")


  def frac_valid(self,n,t):
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
      self.val_d[self.val_d_key(n,t,val)] = [mean[key],err[key]] 

    return mean['both'], err['both'], mean['vcut'], err['vcut'], mean['ocut'], err['ocut']


  def analyze_neighbors(self,n,t,d_min=0,n_buffer_frames=2):
    frac_valid_vcut = []
    frac_valid_dcut = []
    frac_valid_both = []
    dij_mij = []
    for i_file in range(len(self.d['file'])):
      if self.d['n'][i_file] == n and self.d['type'][i_file] == t:
        print(self.d['file'][i_file])

        if n > 1:
          print("\n\n  Calculating neighbor distance and alignment...")
          self.d['group'][i_file].calculate_distance_alignment()
          print("  ... done.\n\n")
         
          print("\n\n  Making neighbor distance cut...")
          self.d['group'][i_file].neighbor_distance_cut(d_min,n_buffer_frames)
          dij_mij_tmp, frac_valid_both_tmp, frac_valid_vcut_tmp, frac_valid_dcut_tmp = self.d['group'][i_file].valid_distance_alignment(self.framei,self.framef)
          dij_mij.extend(list(dij_mij_tmp))
        else:
          print("\n\n  Single fish, no neighbors... ")
          self.d['group'][i_file].neighbor_distance_cut(d_min,n_buffer_frames)
          dij_mij_tmp, frac_valid_both_tmp, frac_valid_vcut_tmp, frac_valid_dcut_tmp = self.d['group'][i_file].valid_distance_alignment(self.framei,self.framef)

        frac_valid_vcut.append(frac_valid_vcut_tmp)
        frac_valid_dcut.append(frac_valid_dcut_tmp)
        frac_valid_both.append(frac_valid_both_tmp)
        print("  ... done.\n\n")

    val = 'dij_mij'
    dij_mij = np.array(dij_mij)
    self.val_d[self.val_d_key(n,t,val)] = dij_mij

    val = 'frac_valid_vcut'
    frac_valid_vcut = np.array(frac_valid_vcut)
    self.val_d[self.val_d_key(n,t,val)] = frac_valid_vcut

    val = 'frac_valid_dcut'
    frac_valid_dcut = np.array(frac_valid_dcut)
    self.val_d[self.val_d_key(n,t,val)] = frac_valid_dcut

    val = 'frac_valid_both'
    frac_valid_both = np.array(frac_valid_both)
    self.val_d[self.val_d_key(n,t,val)] = frac_valid_both



  def plot_multi_fish_distributions(self,ts,ns,binv,xmin,xmax,speed_cut=False,ftype='png',tag="",save=False):
    fig, ax = plt.subplots(len(ts), 3, figsize=(15,len(ts)*5))
    aspect_ratio = 1
    norm=True
    
    i = 0
    for t in ts:
      v = 'dw'
      v_label = 'distance to wall'
      hrange = [xmin[v],2*xmax[v]]
      nbins = binv[v]
      for n in ns:
        if speed_cut:
          hist, bin_edges = np.histogram(self.combined_trials_speed_cut(n,t,v),range=hrange,bins=nbins,density=norm)
        else:
          hist, bin_edges = np.histogram(self.combined_trials(n,t,v),range=hrange,bins=nbins,density=norm)
        binc = ( bin_edges[1:] + bin_edges[:-1] ) / 2 
        ax[i][0].plot(binc, hist/(2*np.pi*(111./2)- binc), label="groups of %s" % (n))
      ax[i][0].set_xlabel("%s" % v_label)
      ax[i][0].set_ylabel("frequency")
      ax[i][0].set_xlim((xmin[v],xmax[v]))
      ax[i][0].set_ylim(bottom=0)
      ax[i][0].legend(loc=(0.57,0.13))
      ymin, ymax = ax[i][0].get_ylim() 
      ax[i][0].set_aspect((xmax[v]-xmin[v])/(ymax-ymin)/aspect_ratio)
      v_stat1 = "mean"
      ax0inset = inset_axes(ax[i][0], width="43%", height="43%", loc=1)
      ax0inset.set_ylabel("mean distance to wall")

      if speed_cut:
        mean = self.combined_trials_stats_by_n(ns,t,v,'mean',tb="speed_cut")
        sterr = self.combined_trials_stats_by_n(ns,t,v,'stderr',tb="speed_cut")
      else:
        mean = self.combined_trials_stats_by_n(ns,t,v,'mean',tb="speed_cut")
        sterr = self.combined_trials_stats_by_n(ns,t,v,'stderr',tb="speed_cut")
      ax0inset.errorbar(mean[:,0],mean[:,1],yerr=sterr[:,1],fmt='co',linewidth=5)

      #ax0inset.plot(mean[:,0],kin_stats[key_vt(v,t)][v_stat1][:,1],'c-',linewidth=2)
      ax0inset.set_xlabel("group size")
      ax0inset.set_xlim((0,12))
      ymin, ymax = ax0inset.get_ylim() 
      ylen = ymax-ymin
      ax0inset.set_ylim(ymin-0.1*ylen,ymax+0.1*ylen)
      
      v = 'speed'
      v_label = 'speed'
      hrange = [xmin[v],2*xmax[v]]
      nbins = binv[v]
      for n in ns:
        if speed_cut:
          hist, bin_edges = np.histogram(self.combined_trials_speed_cut(n,t,v),range=hrange,bins=nbins,density=norm)
        else:
          hist, bin_edges = np.histogram(self.combined_trials(n,t,v),range=hrange,bins=nbins,density=norm)
        binc = ( bin_edges[1:] + bin_edges[:-1] ) / 2 
        ax[i][1].plot(binc, hist, label="groups of %s" % (n))
      ax[i][1].set_xlabel("%s" % v_label)
      ax[i][1].set_xlim((xmin[v],xmax[v]))
      ax[i][1].set_ylim(bottom=0)
      ymin, ymax = ax[i][1].get_ylim() 
      ax[i][1].set_aspect((xmax[v]-xmin[v])/(ymax-ymin)/aspect_ratio)
      ax1inset = inset_axes(ax[i][1], width="43%", height="43%", loc=1)
      v_stat1 = "mean"
      ax1inset.set_ylabel("mean speed")

      if speed_cut:
        mean = self.combined_trials_stats_by_n(ns,t,v,'mean',tb="speed_cut")
        sterr = self.combined_trials_stats_by_n(ns,t,v,'stderr',tb="speed_cut")
      else:
        mean = self.combined_trials_stats_by_n(ns,t,v,'mean')
        sterr = self.combined_trials_stats_by_n(ns,t,v,'stderr')

      ax1inset.errorbar(mean[:,0],mean[:,1],yerr=sterr[:,1],fmt='co',linewidth=5)
      #ax1inset.errorbar(kin_stats[key_vt(v,t)][v_stat1][:,0],kin_stats[key_vt(v,t)][v_stat1][:,1],kin_stats[key_vt(v,t)]['stderr'][:,1],'c-',linewidth=2)
###      ax1inset.plot(kin_stats[key_vt(v,t)][v_stat1][:,0],kin_stats[key_vt(v,t)][v_stat1][:,1],'c-',linewidth=2)
      ax1inset.set_xlabel("group size")
      ax1inset.set_xlim((0,12))
      ymin, ymax = ax1inset.get_ylim() 
      ylen = ymax-ymin
      ax1inset.set_ylim(ymin-0.1*ylen,ymax+0.1*ylen)
  
      v = 'omega'
      v_label = 'angular velocity'
      hrange = [xmin[v],2*xmax[v]]
      nbins = binv[v]
      for n in ns:
        if speed_cut:
          hist, bin_edges = np.histogram(self.combined_trials_speed_cut(n,t,v),range=hrange,bins=nbins,density=norm)
        else:
          hist, bin_edges = np.histogram(self.combined_trials(n,t,v),range=hrange,bins=nbins,density=norm)
        binc = ( bin_edges[1:] + bin_edges[:-1] ) / 2 
        ax[i][2].plot(binc, hist, label="%s" % (t))
      ax[i][2].set_xlabel("%s" % v_label)
      ax[i][2].set_xlim((xmin[v],xmax[v]))
      ymin, ymax = ax[i][2].get_ylim() 
      ax[i][2].set_aspect((xmax[v]-0)/(ymax-ymin)/aspect_ratio)

      ax2inset = inset_axes(ax[i][2], width="53%", height="53%", loc=1, borderpad=1.4)
      hrange = [xmin[v],2*xmax[v]]
      nbins = binv[v]
      for n in ns:
        if speed_cut:
          hist, bin_edges = np.histogram(self.combined_trials_speed_cut(n,t,v),range=hrange,bins=nbins,density=norm)
        else:
          hist, bin_edges = np.histogram(self.combined_trials(n,t,v),range=hrange,bins=nbins,density=norm)
        binc = ( bin_edges[1:] + bin_edges[:-1] ) / 2 
        ax2inset.plot(binc, hist, label="%s" % (t))
      ax2inset.set_xlabel("%s" % v_label)
      ax2inset.set_xlim((0,2*xmax[v]))
      ax2inset.set_yscale('log')
  
#      v = "omega"
#      ax2inset = inset_axes(ax[i][2], width="43%", height="43%", bbox_to_anchor=(-.13,0,1.,1.), bbox_transform=ax[i][2].transAxes, loc=1)
#      v_stat1 = "std"
#      ax2inset.set_ylabel("std. dev.")
#
#      if speed_cut:
#        std = self.combined_trials_stats_by_n(ns,t,v,'std',tb="speed_cut")
#        kurt = self.combined_trials_stats_by_n(ns,t,v,'kurtosis',tb="speed_cut")
#
#
#        #############
#        ### TEMPORARY FIX
#        ##### find a better way!
#        val_range=[-20,20]
#        std_arr = []
#        kurt_arr = []
#        for n in ns: 
#          k = self.val_d_key(n,t,v,tag='speed_cut')
#          val_tmp = self.val_d[k]
#          vals = []
#          vals.extend(list(val_tmp)) 
#          vals.extend(list(-val_tmp)) 
#          val_tmp = np.array(vals)
#          if val_range != None:
#            val_tmp = val_tmp[(val_tmp >= val_range[0]) & (val_tmp <= val_range[1])]
#          std_tmp = np.nanstd(np.array(val_tmp))/np.sqrt(len(val_tmp)-1)
#          kurt_tmp = spstats.kurtosis(np.array(val_tmp))/np.sqrt(len(val_tmp)-1)
#          std_arr.append([n,std_tmp])
#          kurt_arr.append([n,kurt_tmp])
#        std = np.array(std_arr)
#        kurt = np.array(kurt_arr)
#        #############
#        #############
#
#      else:
#        std  = self.combined_trials_stats_by_n(ns,t,v,'std')
#        kurt = self.combined_trials_stats_by_n(ns,t,v,'kurtosis')
#
#      ax2inset.plot(std[:,0],std[:,1],'co',linewidth=5,label="std")
####      ax2inset.plot(kin_stats[key_vt(v,t)][v_stat1][:,0],kin_stats[key_vt(v,t)][v_stat1][:,1],'c-',linewidth=2,label="std. deviation")
#      ax2inset.set_xlabel("group size")
#      ax2inset.set_xlim((0,12))
#      ymin, ymax = ax2inset.get_ylim() 
#      ylen = ymax - ymin
#      ax2inset.set_ylim(ymin-0.1*ylen,ymax+0.1*ylen)
#      v_stat2 = "kurtosis"
#      ax2binset = ax2inset.twinx()
#      ax2binset.set_ylabel("kurtosis")
#      ax2binset.plot(kurt[:,0],kurt[:,1],'ro',linewidth=5,label="kurt")
####      ax2binset.plot(kin_stats[key_vt(v,t)][v_stat2][:,0],kin_stats[key_vt(v,t)][v_stat2][:,1],'r-',linewidth=2, label="kurtosis")
#      ymin, ymax = ax2binset.get_ylim() 
#      ax2binset.set_ylim(ymin-0.1*ylen,ymax+0.1*ylen)
#      ax2inset.legend()
 
      i+=1
 
    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.05,wspace=0.05)
    if save:
      plt.savefig("paper01/distribs_compare_multifish%s.png" % tag)
    else:
      plt.show()

    fig.clf()
  # new dij_mij plots
  #    val = 'dij_mij'
  #    dij_mij = np.array(dij_mij)
  #    self.val_d[self.val_d_key(n,t,val)] = dij_mij
  

  def plot_dij_mij_log_one_cbar(self,ts,ns,d_max=105,d_bins=105,m_max=1,m_bins=100,save=False):

    my_cmap = copy.copy(mpl_cm.get_cmap('viridis'))
    my_cmap.set_bad(my_cmap.colors[0])

#    my_cmap = copy.copy(mpl_cm.get_cmap('Blues'))

    d_range=[0,d_max]
    m_range=[-m_max,m_max]
    print(ts,ns)
    #fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
    fig = plt.figure(figsize=(15,10))
    grid = ImageGrid(fig, 111, nrows_ncols=(len(ts),len(ns)), cbar_mode='single', axes_pad=0.15)
    ims=[]
    dij_mij = {}
    for i in range(len(ts)):
      for j in range(len(ns)):
        t = ts[i]
        n = ns[j]
        k = self.val_d_key(n,t,'dij_mij')
        dij_mij[k] = self.val_d[k]
        i_grid = i*len(ns)+j
        if i == len(ts) - 1:
          grid[i_grid].set_xlabel("distance (cm)")
        if j == 0:
          grid[i_grid].set_ylabel("alignment")
        print("Binning %s..." %k)
        print("printing dij_mij",dij_mij[k])
        counts, xedges, yedges, im = grid[i_grid].hist2d(dij_mij[k][:,0], dij_mij[k][:,1],
                                                      bins  = [d_bins , m_bins ],
                                                      range = [d_range, m_range],
                                                      density = True,
                                                      norm = colors.LogNorm(),
                                                      cmap = my_cmap )
        grid[i_grid].set_aspect(aspect_ratio(d_range,m_range))
        ims.append(im)
  
    clims = [im.get_clim() for im in ims]
    vmin = min([clim[0] for clim in clims])
    vmax = max([clim[1] for clim in clims])
    #print(vmin,vmax)
    for im in ims:
      im.set_clim(vmin=vmin,vmax=vmax)
    cb = fig.colorbar(ims[0], cax=grid[0].cax)
    #grid[0].cax.colorbar(ims[0])
 
    if save:
      plt.savefig("paper01/dij_mij_singlecbar_log.png")
    else:
      plt.show()

    fig.clf()

  def plot_dij_mij_lin_one_cbar(self,ts,ns,d_max=105,d_bins=105,m_max=1,m_bins=100,save=False):
    d_range=[0,d_max]
    m_range=[-m_max,m_max]
    #fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
    fig = plt.figure(figsize=(15,10))
    grid = ImageGrid(fig, 111, nrows_ncols=(len(ts),len(ns)), cbar_mode='single', axes_pad=0.15)
    ims=[]
    dij_mij = {}
    for i in range(len(ts)):
      for j in range(len(ns)):
        t = ts[i]
        n = ns[j]
        k = self.val_d_key(n,t,'dij_mij')
        dij_mij[k] = self.val_d[k]
        i_grid = i*len(ns)+j
        if i == len(ts) - 1:
          grid[i_grid].set_xlabel("distance (cm)")
        if j == 0:
          grid[i_grid].set_ylabel("alignment")
        print("Binning %s..." %k)
        counts, xedges, yedges, im = grid[i_grid].hist2d(dij_mij[k][:,0], dij_mij[k][:,1],
                                                     bins  = [d_bins , m_bins ],
                                                     range = [d_range, m_range],
                                                     density = True)
        grid[i_grid].set_aspect(aspect_ratio(d_range,m_range))
        ims.append(im)
    
    clims = [im.get_clim() for im in ims]
    vmin = min([clim[0] for clim in clims])
    vmax = max([clim[1] for clim in clims])
    for im in ims:
      im.set_clim(vmin=vmin,vmax=vmax)
    grid[0].cax.colorbar(ims[0])
    
    if save:
      plt.savefig("paper01/dij_mij_singlebar_lin.png")
    else:
      plt.show()
    fig.clf()
  
  
  def plot_dij_mij_lin_multi_cbar(self,ts,ns,d_max=105,d_bins=105,m_max=1,m_bins=100,save=False):
    d_range=[0,d_max]
    m_range=[-m_max,m_max]
    fig, ax = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(15,10))
    dij_mij = {}
    for i in range(len(ts)):
      for j in range(len(ns)):
        if i == len(ts) - 1:
          ax[i,j].set_xlabel("distance (cm)")
        if j == 0:
          ax[i,j].set_ylabel("alignment")
        t = ts[i]
        n = ns[j]
        k = self.val_d_key(n,t,'dij_mij')
        dij_mij[k] = self.val_d[k]
        print("Binning %s..." %k)
        counts, xedges, yedges, im = ax[i,j].hist2d(dij_mij[k][:,0], dij_mij[k][:,1],
                                                     bins  = [d_bins,m_bins],
                                                     range = [d_range,m_range],
                                                     density = True)
        plt.colorbar(im, ax=ax[i,j])
        #ax[i,j].set_aspect(0.75*aspect_ratio(d_range,m_range))
    
    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.05,wspace=0.05)
  
    if save:
      plt.savefig("paper01/dij_mij_multicbar_lin.png")
    else:
      plt.show()
    fig.clf()


  def plot_frac_valid_vcut(self,ts,ns,save=False):
    plt.rcParams.update({'font.size': 36})
    frac_valid = {}
    avg = [ [] for t in ts ]
    err = [ [] for t in ts ]
    t_iter = 0
    plt.title("Fraction valid after speed cut")
    plt.xlabel("group size")
    plt.xlim([0,12])
    plt.ylim([0,1.05])
    for t in ts:
      for n in ns:
        k = self.val_d_key(n,t,'frac_valid_vcut')
        frac_valid_tmp = self.val_d[k]
        avg[t_iter].append(np.mean(frac_valid_tmp)) 
        err[t_iter].append(np.std(frac_valid_tmp)/np.sqrt(len(frac_valid_tmp))) 
      plt.errorbar(ns,avg[t_iter],yerr=err[t_iter],fmt='o',markersize='10',label=t)
      t_iter += 1
    plt.legend()
    plt.tight_layout()

    if save:
      plt.savefig("paper01/frac_valid_vcut.png")
    else:
      plt.show()
    plt.clf()


  def plot_frac_valid_dcut(self,ts,ns,save=True):
    plt.rcParams.update({'font.size': 36})
    frac_valid = {}
    avg = [ [] for t in ts ]
    err = [ [] for t in ts ]
    t_iter = 0
    plt.title("Fraction valid after occlusion cut")
    plt.xlabel("group size")
    plt.xlim([0,12])
    plt.ylim([0,1.05])
    for t in ts:
      for n in ns:
        k = self.val_d_key(n,t,'frac_valid_dcut')
        frac_valid_tmp = self.val_d[k]
        avg[t_iter].append(np.mean(frac_valid_tmp)) 
        err[t_iter].append(np.std(frac_valid_tmp)/np.sqrt(len(frac_valid_tmp))) 
      plt.errorbar(ns,avg[t_iter],yerr=err[t_iter],fmt='o',markersize='10',label=t)
      t_iter += 1
    plt.legend()
    plt.tight_layout()
    
    if save:
      plt.savefig("paper01/frac_valid_dcut.png")
    else:
      plt.show()
    plt.clf()

  def plot_frac_valid_both(self,ts,ns,save=True):
    plt.rcParams.update({'font.size': 22})
    frac_valid = {}
    avg = [ [] for t in ts ]
    err = [ [] for t in ts ]
    plt.title("Fraction valid after both cuts")
    plt.xlabel("group size")
    plt.xlim([0,12])
    plt.ylim([0,1.05])
    t_iter = 0
    for t in ts:
      for n in ns:
        k = self.val_d_key(n,t,'frac_valid_both')
        try:
          frac_valid_tmp = self.val_d[k]
        except ValueError:
          k = self.val_d_key(n,t,'frac_valid_both')
          frac_valid_tmp = self.val_d[k]
        avg[t_iter].append(np.mean(frac_valid_tmp)) 
        err[t_iter].append(np.std(frac_valid_tmp)/np.sqrt(len(frac_valid_tmp))) 
      plt.errorbar(ns,avg[t_iter],yerr=err[t_iter],fmt='o',markersize='10',label=t)
      t_iter += 1
    plt.legend()
    plt.tight_layout()
    
    if save:
      plt.savefig("paper01/frac_valid_both.png")
    else:
      plt.show()
    plt.clf()






def main():
  trials = sys.argv[1]
  ns = [1, 2, 5, 10]
  ts = ["SF", "Pa", "Ti", "Mo"]
  
  dd = DataDictionary(trials,ns,ts)
  dd.print_sorted(ns,ts)

  #dd.plot_hist_singles(1,"SF","x","x",nbins=400,hrange=[0,1200])
  #dd.plot_hist_singles(1,"SF","y","y",nbins=400,hrange=[0,1200])
  for t in ts:
    for n in ns:
      print("\n\n  type = %s, group size = %i \n" % (t,n))
      dd.plot_hist_singles_each(n,t,"speed","speed",nbins=100,hrange=[0,80])
      dd.plot_hist_singles_all(n,t,"speed","speed",nbins=100,hrange=[0,80])
      dd.plot_hist_combined(n,t,"speed","speed",nbins=100,hrange=[0,80])
  dd.plot_hist_across_n(ns,t,"speed","speed",nbins=100,hrange=[0,80])

#main()
