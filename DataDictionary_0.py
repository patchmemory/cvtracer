#!/usr/bin/python
import sys, os
import numpy as np
import scipy.stats as spstats 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
sys.path.insert(0, '/data1/cavefish/social/python/track')
from KinematicDataFrame import KinematicDataFrame
import pickle

class DataDictionary:

  def __init__(self):
    self.d = {'file':[], 'data':[], 'type':[], 'n':[], 'year':[], 'month':[], 'day':[], 'fish':[], 'issue':[]}

  def load_list(self,trials,ns,ts,fps=30):
    self.read_list(trials,ns,ts)
    self.framei = 10 * 60 * fps 
    self.framef = 30 * 60 * fps 
    self.ns = ns
    self.ts = ts

    self.val_d = {}

#  def save(self,fname):
#    with open(fname,'wb') as output:
#      pickle.dump(self,fname)
#
#  def load(self,fname):
#    try:
#      with open(fname,'rb') as fname:
#        self = pickle.load(fname)
#    except FileNotFoundError:
#      print("  %s could not be found." % fname)
#      exit()

  def read_list(self,trials,ns,ts):
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
          fish_data = np.load(fdata)
          self.d['issue'].append(None)
          print("  Loaded %s" % fdata)
        except FileNotFoundError:
          fish_data = None
          self.d['issue'].append("Data file not found! ")
          print("  Error loaded %s" % fdata)
          
        self.d['fish'].append(fish_data)

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

  def combined_trials_stats(self,n,t,val,vals,tb=""):
    print("%i %s %s %s" % (n,t,val,tb))
    self.val_d[self.val_d_key(n,t,val,tag=tb+'mean')] = np.mean(vals)
    self.val_d[self.val_d_key(n,t,val,tag=tb+'std')]  = np.std(vals)
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
              self.d['fish'][i][i_fish].df[val][self.framei:self.framef].tolist() )
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

  def trial_speed_cut(self,i_file,speed_cut=1.,n_buffer_frames=2):
    vals_filtered= []
    def valid_speed_window(speed,i_frame,speed_cut,n_buffer_frames):
      for j_frame in range(n_buffer_frames+1):
        if speed[i_frame+j_frame] < speed_cut or speed[i_frame-j_frame] < speed_cut: 
          return False
      else:
        return True

    n = self.d['n'][i_file]
    for i_fish in range(n):
      if len(self.d['speed_cut'][i_file][i_fish]) > 0:
        continue  

      try:
        speed = self.d['fish'][i_file][i_fish].df['speed'].tolist()
      except TypeError:
        fname = self.d['file'][i_file].split('/')[-1]
        fdate = self.d['file'][i_file].split('/')[-2]
        print("No %s data was found for %s/%s." % (val,fdate,fname))

      speed = np.array(speed)
      cut = np.zeros_like(speed)
      for frame in range(len(speed)):
        if frame < n_buffer_frames or frame >= len(speed) - n_buffer_frames:
          cut[frame] = True
        elif not valid_speed_window(speed,frame,speed_cut,n_buffer_frames):
          cut[frame] = True
      self.d['speed_cut'][i_file][i_fish] = cut

  def combined_trials_speed_cut(self,n,t,val,speed_cut=1.,n_buffer_frames=2):
    vals_filtered= []
    for i in range(len(self.d['file'])):
      if self.d['n'][i] == n and self.d['type'][i] == t:
        self.trial_speed_cut(i)
        for i_fish in range(n):
          try:
            cut = self.d['speed_cut'][i][i_fish][self.framei:self.framef]
            vals = np.array(self.d['fish'][i][i_fish].df[val][self.framei:self.framef].tolist())
          except TypeError:
            fname = self.d['file'][i].split('/')[-1]
            fdate = self.d['file'][i].split('/')[-2]
            print("No %s data was found for %s/%s." % (val,fdate,fname))
          vals_filtered.extend(list(vals[np.logical_not(cut)]))

    vals_filtered = np.array(vals_filtered)
    tb="speed_cut"
    self.combined_trials_stats(n,t,val,vals_filtered,tb=tb)
    if val == 'omega':
      vals_filtered = abs(vals_filtered)
    self.val_d[self.val_d_key(n,t,val,tag=tb)] = vals_filtered
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
            plt.hist(self.d['fish'][i][i_fish].df[val][self.framei:self.framef],range=hrange,bins=nbins,density=norm,alpha=0.5,lw=3)
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
            plt.hist(self.d['fish'][i][i_fish].df[val][self.framei:self.framef],range=hrange,bins=nbins,density=norm,alpha=0.5,lw=3)
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
              self.d['fish'][i][i_fish].df[val][self.framei:self.framef].tolist() )
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
      plt.savefig("plot/hist_compare_singlefish%s.png" % tag)
    else:
      plt.show()


  def distance(self,[x1,y1],[x2,y2]):
    return np.sqrt(pow(x1-x2,2) + pow(y1-y2,2))

  def direction(self,[x1,y1],[x2,y2]):
    return x2-x1, y2-y1

  def midpoint(self.[x1,y1],[x2,y2]):
    return (x1+x2)/2 + (y1+y2)/2


  def neighbor_distance(self,ts,ns,binv,xmin,xmax,speed_cut=False,ftype='png',tag="",save=False):
    dij = [] # 1st index for trial, 2nd index for time step, and third for neighbor distance 
    for i in range(len(self.d['file'])):
      if self.d['n'][i] == n and self.d['type'][i] == t:
        dij.append([[]])
        pos = []
        for i_fish in range(n):
          try:
            pos.append(np.array(self.d['fish'][i][i_fish].df[['x','y']][self.framei:self.framef].tolist()))
          except TypeError:
            fname = self.d['file'][i].split('/')[-1]
            fdate = self.d['file'][i].split('/')[-2]
            print("No data was found for %s/%s." % (fdate,fname))

        for frame in range(len(pos[0])):
          dij_tmp = []
          for i_fish in range(len(pos))):
            for j_fish in range(i_fish+1,len(pos[frame])):
              dij_tmp.append(self.distance(pos[frame][i_fish],pos[frame][j_fish]))
          dij.append(dij_tmp)
        
    dij = np.array(dij)
    dij_min = np.zeros(len(dij))
    
    for i in range(len(
    self.combined_trials_stats(n,t,val,vals)
    if val == 'omega':
      vals = abs(vals)
    self.val_d[self.val_d_key(n,t,val)] = vals
    return vals

  def trial_speed_cut(self,i_file,speed_cut=1.,n_buffer_frames=2):

    return


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
      ax0inset.errorbar(mean[:,0],mean[:,1],yerr=sterr[:,1],fmt='c-',linewidth=2)

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

      ax1inset.errorbar(mean[:,0],mean[:,1],yerr=sterr[:,1],fmt='c-',linewidth=2)
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
  
      v = "omega"
      ax2inset = inset_axes(ax[i][2], width="43%", height="43%", bbox_to_anchor=(-.13,0,1.,1.), bbox_transform=ax[i][2].transAxes, loc=1)
      v_stat1 = "std"
      ax2inset.set_ylabel("std. dev.")

      if speed_cut:
        std = self.combined_trials_stats_by_n(ns,t,v,'std',tb="speed_cut")
        kurt = self.combined_trials_stats_by_n(ns,t,v,'kurtosis',tb="speed_cut")
      else:
        std = self.combined_trials_stats_by_n(ns,t,v,'std')
        kurt = self.combined_trials_stats_by_n(ns,t,v,'kurtosis')
      ax2inset.plot(std[:,0],std[:,1],'c-',linewidth=2)
###      ax2inset.plot(kin_stats[key_vt(v,t)][v_stat1][:,0],kin_stats[key_vt(v,t)][v_stat1][:,1],'c-',linewidth=2,label="std. deviation")
      ax2inset.set_xlabel("group size")
      ax2inset.set_xlim((0,12))
      ymin, ymax = ax2inset.get_ylim() 
      ylen = ymax - ymin
      ax2inset.set_ylim(ymin-0.1*ylen,ymax+0.1*ylen)
      v_stat2 = "kurtosis"
      ax2binset = ax2inset.twinx()
      ax2binset.set_ylabel("kurtosis")
      ax2binset.plot(kurt[:,0],kurt[:,1],'r-',linewidth=2)
###      ax2binset.plot(kin_stats[key_vt(v,t)][v_stat2][:,0],kin_stats[key_vt(v,t)][v_stat2][:,1],'r-',linewidth=2, label="kurtosis")
      ymin, ymax = ax2binset.get_ylim() 
      ax2binset.set_ylim(ymin-0.1*ylen,ymax+0.1*ylen)
      ax2inset.legend()
 
      i+=1
 
    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.05,wspace=0.05)
    if save:
      plt.savefig("plot/distribs_compare_multifish%s.png" % tag)
    else:
      plt.show()


def make_dat_list(flist):
  f = open(flist,'r')
  dat_list = []
  dir_list = []
  for line in f:
    fdir = line.split()[0].split('.')[0]
    dir_list.append(fdir)
    fname = fdir + "/kinematic_scatter.npy"


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
