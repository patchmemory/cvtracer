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


def figure_single_fish_compare(h, ts, tag, t_name, val_name, save=True, omega_weighted = False):
    
    left_shift = 0.03
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(1,3, figsize=(17,5))

    val = 'dwall'
    vrange = [0,55]
    for t in ts:
        ax[0].fill_between( h[hkey(t,n,val)][:,0],
                            h[hkey(t,n,val)][:,1] - h[hkey(t,n,val)][:,2], 
                            h[hkey(t,n,val)][:,1] + h[hkey(t,n,val)][:,2], 
                            alpha=0.5, label=t_name[t])
    ax[0].set_ylabel("normalized count", fontsize = 16)
    ax[0].set_xlabel(val_name[val], fontsize = 16)
    ax[0].set_xlim(vrange)
    ax[0].set_ylim(bottom=0)

    val = 'speed'
    vrange = [0,75]
    for t in ts:
        ax[1].fill_between( h[hkey(t,n,val)][:,0],
                            h[hkey(t,n,val)][:,1] - h[hkey(t,n,val)][:,2], 
                            h[hkey(t,n,val)][:,1] + h[hkey(t,n,val)][:,2], 
                            alpha=0.5, label=t_name[t])
    ax[1].set_xlabel(val_name[val], fontsize = 16)
    ax[1].set_xlim(vrange)
    ax[1].set_ylim(bottom=0)


    val = 'omega'
    vrange1 = [0,10]
    if omega_weighted:
        weight = h[hkey(t,n,val)][:,0]
        vrange1 = [0,25]
    else:
        weight = np.ones_like(h[hkey(t,n,val)][:,0])
    for t in ts:
        ax[2].fill_between( h[hkey(t,n,val)][:,0],
                           weight* (h[hkey(t,n,val)][:,1] - h[hkey(t,n,val)][:,2]), 
                           weight* (h[hkey(t,n,val)][:,1] + h[hkey(t,n,val)][:,2]), 
                            alpha=0.5, label=t_name[t])
    ax[2].set_xlabel(val_name[val], fontsize = 16)
    ax[2].set_xlim(vrange1)
    ax[2].set_ylim(bottom=0)

    vrange2 = [0,25]
    inset_ax2 = plt.axes([0.75-left_shift,0.5,0.14,0.35])
    for t in ts:
        inset_ax2.fill_between( h[hkey(t,n,val)][:,0],
                           weight* (h[hkey(t,n,val)][:,1] - h[hkey(t,n,val)][:,2]), 
                           weight* (h[hkey(t,n,val)][:,1] + h[hkey(t,n,val)][:,2]), 
                            alpha=0.5, label=t_name[t])
    inset_ax2.set_xlabel(val_name[val], fontsize = 16)
    inset_ax2.set_xlim(vrange2)
    inset_ax2.set_yscale('log')
    #inset_ax2.set_ylabel("logarithmic", fontsize = 16)

    ax[2].legend(loc="center", bbox_to_anchor=(1.25,0.5), fontsize = 16)

    for axis in ax:
        box = axis.get_position()
        box.x0 = box.x0 - left_shift
        box.x1 = box.x1 - left_shift
        axis.set_position(box)

    if save:
        plt.savefig("paper/figures/03/single_fish_compare_%s.png" % tag )
    else:
        plt.show()
    plt.clf()


def cumulative(arr):
    arr = np.array(arr)
    acc = np.zeros_like(arr)
    for i in range(1,len(arr)):
        acc[i] = acc[i-1] + arr[i]
    return acc


def figure_multi_fish_compare(h, s, ts, ns, tag, t_name, val_name, save=True, omega_weighted = False):
    
    left_shift = 0.03
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(len(ts),3, figsize=(17,5*len(ts)))
    inset = np.empty_like(ax)

    row = 0
    for t in ts:

        val = 'dwall'
        vrange = [0,55]
        for n in ns:
            ax[row][0].fill_between( h[hkey(t,n,val)][:,0],
                                     h[hkey(t,n,val)][:,1] - h[hkey(t,n,val)][:,2], 
                                     h[hkey(t,n,val)][:,1] + h[hkey(t,n,val)][:,2], 
                                     alpha = 0.5, label=n )

        ax[row][0].set_ylabel("normalized count", fontsize = 16)
        ax[row][0].set_xlabel(val_name[val], fontsize = 16)
        ax[row][0].set_xlim(vrange)
        ax[row][0].set_ylim(bottom=0)

        ## inset 
        box = ax[row][0].get_position()
        center = [ ( box.x0 + box.x1 ) / 2 , ( box.y0 + box.y1 ) / 2 ]
        width = [ box.x1 - box.x0 , box.y1 - box.y0 ]
        w_frac = 0.49
        gutter = 0.04
        pos_frac = 0.5 - w_frac - gutter
        inset[row][0] = plt.axes([center[0]+pos_frac*width[0]-left_shift,
                                  center[1]+pos_frac*width[1],
                                  w_frac*width[0],
                                  w_frac*width[1]])


        for n in ns:
            c = cumulative(h[hkey(t,n,val)][:,1])
            cmax = c[-1]
            inset[row][0].plot( h[hkey(t,n,val)][:,0], c/cmax, 
                                alpha=0.5, linewidth=2, label=n)
        #inset[row][0].set_xlabel(val_name[val], fontsize = 16)
        inset[row][0].set_xlabel("distance to wall", fontsize = 16)
        inset[row][0].set_xlim(vrange)
        inset[row][0].set_ylim([0,1])
        #inset[row][0].set_yscale('log')
        inset[row][0].set_ylabel("cumulative", fontsize = 16)
        inset[row][0].set_yticks([0,0.2,0.4,0.6,0.8,1])
        
 
        val = 'speed'
        vrange = [0,75]
        for n in ns:
            ax[row][1].fill_between( h[hkey(t,n,val)][:,0],
                                h[hkey(t,n,val)][:,1] - h[hkey(t,n,val)][:,2], 
                                h[hkey(t,n,val)][:,1] + h[hkey(t,n,val)][:,2], 
                                alpha=0.5, label=n)
        ax[row][1].set_xlabel(val_name[val], fontsize = 16)
        ax[row][1].set_xlim(vrange)
        ax[row][1].set_ylim(bottom=0)

        ## inset 
        stat = 'stdd'
        box = ax[row][1].get_position()
        center = [ ( box.x0 + box.x1 ) / 2 , ( box.y0 + box.y1 ) / 2 ]
        width = [ box.x1 - box.x0 , box.y1 - box.y0 ]
        w_frac = 0.40
        gutter = 0.04
        pos_frac = 0.5 - w_frac - gutter
        inset[row][1] = plt.axes([center[0]+pos_frac*width[0]-left_shift,
                                  center[1]+pos_frac*width[1],
                                  w_frac*width[0],
                                  w_frac*width[1]])
        inset[row][1].errorbar( s[skey(t,val,stat)][:,0], 
                                s[skey(t,val,stat)][:,1],
                                s[skey(t,val,stat)][:,2],
                                fmt = 'o', capsize = 3      )
        inset[row][1].set_ylabel(r'$\sigma_{speed}$')
        #inset[row][1].errorbar( s[skey(t,val,stat)][:,0], 
        #                        s[skey(t,val,stat)][:,1]*s[skey(t,val,stat)][:,1],
        #                        2*s[skey(t,val,stat)][:,2]*s[skey(t,val,stat)][:,1],
        #                        fmt = 'o', capsize = 3      )
        #inset[row][1].set_ylabel(r'$\sigma_{speed}^2$')
        #inset[row][1].set_ylim(bottom=0)
        inset[row][1].set_xlim([0,12])
        inset[row][1].set_xlabel("group size")

        
 
        val = 'omega'
        vrange1 = [0,10]
        if omega_weighted:
            vrange1 = [0,25]
            weight = h[hkey(t,n,val)][:,0]
        else:
            weight = np.ones_like(h[hkey(t,n,val)][:,0])
        for n in ns:
            ax[row][2].fill_between( h[hkey(t,n,val)][:,0],
                                weight*(h[hkey(t,n,val)][:,1] - h[hkey(t,n,val)][:,2]), 
                                weight*(h[hkey(t,n,val)][:,1] + h[hkey(t,n,val)][:,2]), 
                                alpha=0.5, label=n)
        ax[row][2].set_xlabel(val_name[val], fontsize = 16)
        ax[row][2].set_xlim(vrange1)
        ax[row][2].set_ylim(bottom=0)
 
        ## inset 
        vrange2 = [0,25]
        box = ax[row][2].get_position()
        center = [ ( box.x0 + box.x1 ) / 2 , ( box.y0 + box.y1 ) / 2 ]
        width = [ box.x1 - box.x0 , box.y1 - box.y0 ]
        inset[row][2] = plt.axes([center[0]-left_shift-0.09*width[0],
                                  center[1]-0.09*width[1],
                                  0.55*width[0],
                                  0.55*width[1]])
        for n in ns:
            inset[row][2].fill_between( h[hkey(t,n,val)][:,0],
                                weight*(h[hkey(t,n,val)][:,1] - h[hkey(t,n,val)][:,2]), 
                                weight*(h[hkey(t,n,val)][:,1] + h[hkey(t,n,val)][:,2]), 
                                alpha=0.5, label=n)
        inset[row][2].set_xlabel(val_name[val], fontsize = 16)
        inset[row][2].set_xlim(vrange2)
        inset[row][2].set_yscale('log')
        inset[row][2].set_ylabel("log[count]", fontsize = 16)
        
        box = ax[row][0].get_position()
        y_center = ( box.y1 + box.y0 ) / 2 - ( box.y1 - box.y0 ) / 8 
        fig.text(0.01, y_center, t_name[t], fontsize = 24, rotation = 90, fontweight = 'bold')
        row += 1

    handles, labels = ax[0][2].get_legend_handles_labels()
    leg = ax[0][2].legend(handles[::-1], labels[::-1],loc="center", 
                          bbox_to_anchor=(1.25,0.5), fontsize = 24)
    leg.set_title("Group \nSize",prop={'size':24})
    plt.setp(leg.get_title(), multialignment='center')

    for ax_row in ax:
        for axis in ax_row:
             box = axis.get_position()
             box.x0 = box.x0 - left_shift
             box.x1 = box.x1 - left_shift
             axis.set_position(box)

    if save:
        plt.savefig("paper/figures/05/multi_fish_compare_%s.png" % tag )
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
           'Pa': 'Pachón', 
           'Mo': "Molino", 
           'Ti': "Tinaja" }

ns = [ 10, 5, 2, 1 ]

tag = "t10to30_o0.0_v001.0to100.0_w-25.0to025.0_nbf3"
#tag = "t10to30_o0.0_v000.0to100.0_w-25.0to025.0_nbf3"
fname = "analysis_%s.arc" % tag

arc = Archive()
arc.load(fname)
h = load_hist(ts,ns,vals,tag)
s = load_stats(ts,ns,vals,stats,tag)


save = True 

om_weight = False 
# figure 3
_ts = ['SF', 'Pa']
#_ts = ['SF', 'Pa', 'Mo', 'Ti']
n = 1
figure_single_fish_compare(h, _ts, tag, t_name, val_name, save, omega_weighted = om_weight)
figure_multi_fish_compare(h, s, _ts, ns, tag, t_name, val_name, save, omega_weighted = om_weight)
#plot_hist_t_compare(h,_ts,n,'dwall',tag, t_name, val_name, save)
#plot_hist_t_compare(h,_ts,n,'speed',tag, t_name, val_name, save)
#
#plot_hist_t_compare(h,_ts,n,'omega',tag, t_name, val_name, save, vrange = [0,10], logs = False)
#plot_hist_t_compare(h,_ts,n,'omega',tag, t_name, val_name, save, vrange = [0,25], logs = True)
#
#n = 10
#plot_hist_t_compare(h,_ts,n,'omega',tag, t_name, val_name, save, vrange = [0,10], logs = False)
#plot_hist_t_compare(h,_ts,n,'omega',tag, t_name, val_name, save, vrange = [0,25], logs = True)
#
#n = 5
#plot_hist_t_compare(h,_ts,n,'omega',tag, t_name, val_name, save, vrange = [0,10], logs = False)
#plot_hist_t_compare(h,_ts,n,'omega',tag, t_name, val_name, save, vrange = [0,25], logs = True)
#
#n = 2
#plot_hist_t_compare(h,_ts,n,'omega',tag, t_name, val_name, save, vrange = [0,10], logs = False)
#plot_hist_t_compare(h,_ts,n,'omega',tag, t_name, val_name, save, vrange = [0,25], logs = True)
#
#
## figure 4
#plot_hist_n_compare(h,'SF',ns,'dwall',tag, t_name, val_name, save)
#plot_hist_n_compare(h,'Pa',ns,'dwall',tag, t_name, val_name, save)
#plot_hist_n_compare(h,'SF',ns,'speed',tag, t_name, val_name, save)
#plot_hist_n_compare(h,'Pa',ns,'speed',tag, t_name, val_name, save)
#
#plot_hist_n_compare(h,'SF',ns,'omega',tag, t_name, val_name, save, vrange = [0,10], logs = False)
#plot_hist_n_compare(h,'SF',ns,'omega',tag, t_name, val_name, save, vrange = [0,25], logs = True)
#plot_hist_n_compare(h,'Pa',ns,'omega',tag, t_name, val_name, save, vrange = [0,10], logs = False)
#plot_hist_n_compare(h,'Pa',ns,'omega',tag, t_name, val_name, save, vrange = [0,25], logs = True)
#plot_hist_n_compare(h,'Mo',ns,'omega',tag, t_name, val_name, save, vrange = [0,10], logs = False)
#plot_hist_n_compare(h,'Mo',ns,'omega',tag, t_name, val_name, save, vrange = [0,25], logs = True)
#plot_hist_n_compare(h,'Ti',ns,'omega',tag, t_name, val_name, save, vrange = [0,10], logs = False)
#plot_hist_n_compare(h,'Ti',ns,'omega',tag, t_name, val_name, save, vrange = [0,25], logs = True)
#
## figure 6 
#plot_stat(s,ts,'dwall','mean',tag, t_name, val_name, stat_name, save)
#plot_stat(s,ts,'speed','mean',tag, t_name, val_name, stat_name, save)
#plot_stat(s,ts,'omega','stdd',tag, t_name, val_name, stat_name, save)
#plot_stat(s,ts,'omega','kurt',tag, t_name, val_name, stat_name, save)

