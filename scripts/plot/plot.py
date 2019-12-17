#!/usr/bin/env python3
import sys
cvhome="/disk1/astyanax-mexicanus/cv-tracer"
sys.path.insert(0, cvhome)
import numpy as np
import matplotlib.pyplot as plt
from Analysis.Archive import Archive

def hkey(t,n,val):
    return "%s_%02i_%s" % (t,n,val)

def load_hist(ts,ns,vals,tag):
    hist = {}
    for val in vals:
        for t in ts:
            for n in ns:
                hist[hkey(t,n,val)] = arc.get_result(t,n,val,'hist',tag)
    return hist


def skey(t,val,stat):
    return "%s_%s_%s" % (t,val,stat)

def load_stats(ts,ns,vals,stats,tag):
    s = {}
    for stat in stats: 
        for val in vals:
            for t in ts:
                s[skey(t,val,stat)] = []
                for n in ns:
                    result = arc.get_result(t,n,val,stat,tag)
                    s[skey(t,val,stat)].append([n,result[0],result[1]])
                s[skey(t,val,stat)] = np.array(s[skey(t,val,stat)])
    return s


def plot_hist_t_compare(h,ts,n,val,tag, t_name, val_name, save=True,
                        vrange = None, logs = False):
    plt.title("%s across type for groups of %i" % (val_name[val],n))
    for t in ts:
        plt.fill_between( h[hkey(t,n,val)][:,0],
                          h[hkey(t,n,val)][:,1] - h[hkey(t,n,val)][:,2], 
                          h[hkey(t,n,val)][:,1] + h[hkey(t,n,val)][:,2], 
                          alpha=0.5, label=t_name[t])
    plt.xlabel(val_name[val])
    plt.legend()

    if vrange != None:
        plt.xlim(vrange)

    if logs == True:
        plt.yscale('log')

    if save:
        if logs:
            plt.savefig("results/%s_n%02i_hist_across_t_log_%s.png" % (val, n, tag) )
        else:
            plt.savefig("results/%s_n%02i_hist_across_t_%s.png" % (val, n, tag) )
    else:
        plt.show()
    plt.clf()


def plot_hist_n_compare(h,t,ns,val,tag, t_name, val_name, save=True,
                        vrange = None, logs = False):
    plt.title("%s of %s" % (val_name[val],t_name[t]) )
    for n in ns:
        plt.fill_between( h[hkey(t,n,val)][:,0],
                          h[hkey(t,n,val)][:,1] - h[hkey(t,n,val)][:,2], 
                          h[hkey(t,n,val)][:,1] + h[hkey(t,n,val)][:,2], 
                          alpha=0.5, label=n )
    plt.xlabel(val_name[val])
    plt.legend()

    if vrange != None:
        plt.xlim(vrange)

    if logs == True:
        plt.yscale('log')

    if save:
        if logs:
            plt.savefig("results/%s_%s_hist_across_n_log_%s.png" % (t, val, tag) )
        else:
            plt.savefig("results/%s_%s_hist_across_n_%s.png" % (t, val, tag) )
    else:
        plt.show()
    plt.clf()


def plot_hist_all_compare(h,ts,ns,val,tag, t_name, val_name, colors, save=True,
                        vrange = None, logs = False):
    for t in ts:
        for n in ns:
           # plt.fill_between( h[hkey(t,n,val)][:,0],
           #                   h[hkey(t,n,val)][:,1] - h[hkey(t,n,val)][:,2], 
           #                   h[hkey(t,n,val)][:,1] + h[hkey(t,n,val)][:,2], 
           #                   alpha=0.5, label=t_name[t] , color = colors[t])
            if n == 1:
                _label = t_name[t]
            else:
                _label = None
            plt.plot( h[hkey(t,n,val)][:,0],
                              h[hkey(t,n,val)][:,1], label= _label , color = colors[t] )
    plt.xlabel(val_name[val])
    plt.legend()
    
    if vrange != None:
        plt.xlim(vrange)
    
    if logs == True:
        plt.yscale('log')

    if save:
        if logs:
            plt.savefig("results/%s_hist_across_all_log_%s.png" % (val, tag) )
        else:
            plt.savefig("results/%s_hist_across_all_%s.png" % (val, tag) )
    else:
        plt.show()
    plt.clf()




def plot_stat(m,ts,val,stat,tag, t_name, val_name, stat_name, save=True):
    plt.title("%s %s" % (stat_name[stat], val_name[val]) )
    for t in ts:
        plt.errorbar( s[skey(t,val,stat)][:,0], 
                      s[skey(t,val,stat)][:,1],
                      s[skey(t,val,stat)][:,2],
                      fmt = 'o',
                      capsize = 3,
                      label = t_name[t] )
    plt.ylim(bottom=0)
    plt.xlim([0,12])
    plt.xlabel("group size")
    plt.legend()
    if save:
        plt.savefig("results/%s_%s_vs_n_across_t_%s.png" % (val,stat,tag))
    else:
        plt.show()
    plt.clf()



stats = ['mean', 'stdd', 'kurt']
stat_name = { 'mean': 'Mean', 
              'stdd': 'Standard deviation',
              'kurt': 'Kurtosis'}

vals = [ 'dwall', 'speed', 'omega' ]
val_name = { 'dwall': 'distance to wall (cm)', 
             'speed': 'speed (cm/s)', 
             'omega': 'angular speed (rad/s)' }

ts = [ 'SF', 'Pa', 'Mo', 'Ti' ]
t_name = { 'SF': "Surface", 
           'Pa': "Pachon", 
           'Mo': "Molino", 
           'Ti': "Tinaja" }

colors = { 'SF': 'tab:blue', 
           'Pa': 'tab:orange', 
           'Mo': 'tab:green', 
           'Ti': 'tab:red' }

ns = [ 10, 5, 2, 1 ]

tag = "t10to30_o0.0_v001.0to100.0_w-25.0to025.0_nbf3"
#tag = "t10to30_o0.0_v000.0to100.0_w-25.0to025.0_nbf3"
fname = "analysis_%s.arc" % tag

arc = Archive()
arc.load(fname)
h = load_hist(ts,ns,vals,tag)
s = load_stats(ts,ns,vals,stats,tag)


save = True 

plot_hist_all_compare(h,ts,ns,'omega',tag, t_name, val_name, colors, save, vrange = [0,10], logs = False)
plot_hist_all_compare(h,ts,ns,'omega',tag, t_name, val_name, colors, save, vrange = [0,25], logs = True)

exit()

# figure 3
_ts = ['SF', 'Pa']
n = 1
plot_hist_t_compare(h,_ts,n,'dwall',tag, t_name, val_name, save)
plot_hist_t_compare(h,_ts,n,'speed',tag, t_name, val_name, save)

plot_hist_t_compare(h,_ts,n,'omega',tag, t_name, val_name, save, vrange = [0,10], logs = False)
plot_hist_t_compare(h,_ts,n,'omega',tag, t_name, val_name, save, vrange = [0,25], logs = True)

n = 10
plot_hist_t_compare(h,_ts,n,'omega',tag, t_name, val_name, save, vrange = [0,10], logs = False)
plot_hist_t_compare(h,_ts,n,'omega',tag, t_name, val_name, save, vrange = [0,25], logs = True)

n = 5
plot_hist_t_compare(h,_ts,n,'omega',tag, t_name, val_name, save, vrange = [0,10], logs = False)
plot_hist_t_compare(h,_ts,n,'omega',tag, t_name, val_name, save, vrange = [0,25], logs = True)

n = 2
plot_hist_t_compare(h,_ts,n,'omega',tag, t_name, val_name, save, vrange = [0,10], logs = False)
plot_hist_t_compare(h,_ts,n,'omega',tag, t_name, val_name, save, vrange = [0,25], logs = True)


# figure 4
plot_hist_n_compare(h,'SF',ns,'dwall',tag, t_name, val_name, save)
plot_hist_n_compare(h,'Pa',ns,'dwall',tag, t_name, val_name, save)
plot_hist_n_compare(h,'SF',ns,'speed',tag, t_name, val_name, save)
plot_hist_n_compare(h,'Pa',ns,'speed',tag, t_name, val_name, save)

plot_hist_n_compare(h,'SF',ns,'omega',tag, t_name, val_name, save, vrange = [0,10], logs = False)
plot_hist_n_compare(h,'SF',ns,'omega',tag, t_name, val_name, save, vrange = [0,25], logs = True)
plot_hist_n_compare(h,'Pa',ns,'omega',tag, t_name, val_name, save, vrange = [0,10], logs = False)
plot_hist_n_compare(h,'Pa',ns,'omega',tag, t_name, val_name, save, vrange = [0,25], logs = True)
plot_hist_n_compare(h,'Mo',ns,'omega',tag, t_name, val_name, save, vrange = [0,10], logs = False)
plot_hist_n_compare(h,'Mo',ns,'omega',tag, t_name, val_name, save, vrange = [0,25], logs = True)
plot_hist_n_compare(h,'Ti',ns,'omega',tag, t_name, val_name, save, vrange = [0,10], logs = False)
plot_hist_n_compare(h,'Ti',ns,'omega',tag, t_name, val_name, save, vrange = [0,25], logs = True)

# figure 6 
plot_stat(s,ts,'dwall','mean',tag, t_name, val_name, stat_name, save)
plot_stat(s,ts,'speed','mean',tag, t_name, val_name, stat_name, save)
plot_stat(s,ts,'omega','stdd',tag, t_name, val_name, stat_name, save)
plot_stat(s,ts,'omega','kurt',tag, t_name, val_name, stat_name, save)

