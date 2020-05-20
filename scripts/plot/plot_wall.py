#!/usr/bin/python3
import sys
cvhome="/home/patch/Code/cvtracer"
sys.path.insert(0, cvhome)
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from TrAQ.Trial import Trial
from Analysis.Archive import Archive
import matplotlib.cm as mpl_cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import ImageGrid
import scipy.stats as sps

def area_circle(r):
  return np.pi*r**2

def area_shell(r1,r2):
  return area_circle(r2) - area_circle(r1)

def next_radius(r0,area_shell):
  return np.sqrt( area_shell/np.pi + r0**2 )

def area_shell_from_bins(r, nbins = 10):
  return area_circle(r)/nbins

def bin_edges(r, nbins = 10):
    a_shell = area_shell_from_bins(r, nbins = nbins)
    edges = [ 0 ] 
    for i in range(nbins):
      edges.append(next_radius(edges[-1],a_shell))
    return np.array(edges)

def print_bins(edges):
    print("# bin  edge0  edge1   area ")
    for i in range(1,len(edges)):
        print("  %3i %6.2f %6.2f %6.2f "  % (i,edges[i-1],edges[i],area_shell(edges[i-1],edges[i])))


def aspect_ratio(x_range,y_range):
    xlen = x_range[1] - x_range[0]
    ylen = y_range[1] - y_range[0]
    return xlen/ylen

def filter_dw_thetaw(dw_thetaw_arr):
    cut1 = dw_thetaw_arr[:,1] !=  1 
    cut2 = dw_thetaw_arr[:,1] != -1 
    cut3 = dw_thetaw_arr[:,0] !=  0
    cuts = cut1 & cut2 & cut3 
    dw_thetaw_cut = dw_thetaw_arr[cuts]
    frac_cut = (len(dw_thetaw_arr)-len(dw_thetaw_cut))/len(dw_thetaw_arr)
    return dw_thetaw_cut, frac_cut 


def calculate_dw_thetaw_all_trials_in_set(t,n):
    for trial in arc.trial_list(t,n):
        try:
            trial.calculate_wall_distance_orientation()
        except:
            print("  Issue calculating wall distance and orientation for trial, ")
            trial.print_info()


def combine_all_trials_in_set(t,n,tag):
    dm = []
    fps=30
    frame_range = [10*60*fps, 30*60*fps]
    for trial in arc.trial_list(t,n):
        try:
            trial.gather_wall_distance_orientation(frame_range, 
                    ocut = True, vcut = True, wcut = True, tag = tag)
            #trial.group.collect_wall_distance_orientation(frame_range, ocut = True, 
            #                                       vcut = True, wcut = True)
            dm.extend(trial.group.dw_thetaw)
        except:
            print("  Trial data not accessible... ")
            trial.print_info()

    return np.array(dm)


def set_key(t,n):
    return "%s%02i" % (t,n)


def collect_all_sets(ts,ns,tag,calc=False):
    d_dm = {}
    #d_fc = {}
    for t in ts:
        for n in ns:
            print("  Collecting results for %s %2i" % (t,n))
            k = set_key(t,n)
            if calc:
                calculate_dw_thetaw_all_trials_in_set(t,n)
            #d_dm[k], d_fc[k] = combine_all_trials_in_set(t,n)
            d_dm[k] = combine_all_trials_in_set(t,n,tag)
            print("d_dm[k]",d_dm[k])
    #return d_dm, d_fc
    return d_dm


def plot_set(d_dm, t, n, tag, save = False):
    my_cmap = copy.copy(mpl_cm.get_cmap('viridis'))
    my_cmap.set_bad(my_cmap.colors[0])
    k = set_key(t,n)
    
    plt.title("%s group size %i" %(t,n))
    plt.ylabel(r"Alignment ($\cos\theta_{ij}$)")
    plt.xlabel("Distance (cm)")
    plt.hist2d(d_dm[k][:,0],d_dm[k][:,1],bins=100, range=[[0,111.],[-1,1]], 
               norm = colors.LogNorm(), cmap = my_cmap)
    plt.colorbar()
    plt.tight_layout()
    if save:
        plt.savefig("results/%s_%02i_dw_thetaw_%s.png" % (t,n,tag))
    else:
        plt.show()
    plt.clf()

def dhist_dx(binc,mean,err):
    _binc = ( binc[1:] + binc[:-1] ) / 2 
    dx = binc[1:] - binc[:-1]
    dmean = mean[1:] - mean[:-1]
    derr = np.sqrt(err[1:]**2 + err[:-1]**2)
    return _binc, dmean/dx, derr

def figure_wall_distance_mean_orientation( d_dm, t, ns, tag, d_bins = 100, m_bins = 100, 
                            d_range = [0, 55.5], m_range = [0,np.pi], save = False, bins = 10,  
                            orientation = True, dhistdx = False):

    plt.rcParams.update({'font.size': 18})

    plt.xlabel("Distance (cm)")
    if dhistdx:
        plt.title("%s wall turns" % t)
        if orientation:
            plt.ylabel(r"Derivative of Orientation ($d\theta_{i,w}/dx$)")
        else:
            plt.ylabel(r"Derivative of Alignment ($d\cos2\theta_{i,w}/dx$)")
    else:
        plt.title("%s wall coordinates" % t)
        if orientation:
            plt.ylabel(r"Mean Orientation ($\theta_{i,w}$)")
        else:
            plt.ylabel(r"Mean Alignment ($\cos2\theta_{i,w}$)")

    edges = bin_edges(d_range[1], nbins = bins)
    print_bins(edges)
    edges = -edges + d_range[1]
    print_bins(edges)
    edges = np.flipud(edges)
    edges = np.abs(edges)
    print_bins(edges)

    binc = ( edges[1:] + edges[:-1] ) / 2
    if dhistdx:
        flat = [ 0 for ibin in binc]
    else:
        if orientation:
            flat = [ np.pi/2 for ibin in binc]
        else:
            flat = [ 0 for ibin in binc]
    plt.plot(binc, flat, 'k')

    
    for j in range(len(ns)):
        k = set_key(t,ns[j])

        print("Binning %s..." %k) 
        count, edge = np.histogram(d_dm[k][:,0], bins=edges, range=[0,50], weights=None)
        if orientation:
            mean, edge, binn = sps.binned_statistic(d_dm[k][:,0],d_dm[k][:,1],statistic='mean',bins=edges,range=[0,50])
            stdd, edge, binn = sps.binned_statistic(d_dm[k][:,0],d_dm[k][:,1],statistic='std',bins=edges,range=[0,50])
        else:
            mean, edge, binn = sps.binned_statistic(d_dm[k][:,0],np.cos(2*d_dm[k][:,1]),statistic='mean',bins=edges,range=[0,50])
            stdd, edge, binn = sps.binned_statistic(d_dm[k][:,0],np.cos(2*d_dm[k][:,1]),statistic='std',bins=edges,range=[0,50])

        ax_label = "n = %i" % (ns[j])
        if dhistdx:
            dbin, dhdx, derr = dhist_dx(binc,mean,stdd/np.sqrt(count-1))
            plt.errorbar(dbin, dhdx, derr, label=ax_label)
            
        else:
            plt.errorbar(binc,mean, stdd/np.sqrt(count-1),label=ax_label)

    plt.legend()
    plt.tight_layout()
    if save:
        if dhistdx:
            if orientation:
                plt.savefig("results/%s_dw_dcos2thetawdx_mean_nbin%02i_%s.png" % (t,bins,tag))
            else:
                plt.savefig("results/%s_dw_dthetawdx_mean_nbin%02i_%s.png" % (t,bins,tag))
        else:
            if orientation:
                plt.savefig("results/%s_dw_cos2thetaw_mean_nbin%02i_%s.png" % (t,bins,tag))
            else:
                plt.savefig("results/%s_dw_thetaw_mean_nbin%02i_%s.png" % (t,bins,tag))
    else:
        plt.show()
    plt.clf()


def figure_wall_distance_orientation(d_dm, ts, ns, tag, d_bins = 100, m_bins = 100, 
                            d_range = [0, 55.5], m_range = [0,np.pi], save = False):
    #fig = plt.figure(figsize=(5*len(ns),5*len(ts)))
    plt.rcParams.update({'font.size': 18})

    my_cmap = copy.copy(mpl_cm.get_cmap('viridis'))
    my_cmap.set_bad(my_cmap.colors[0])

    fig = plt.figure(figsize=(5*len(ns)+1,5*len(ts)))
    #grid = ImageGrid(fig, 111, nrows_ncols=(len(ts),len(ns)), cbar_mode='single', axes_pad=0.15)
    grid = ImageGrid(fig, 111, nrows_ncols=(len(ts),len(ns)), axes_pad=0.3)
    ims=[]
    for i in range(len(ts)):
        for j in range(len(ns)):
            #k = key(i,j,ts[i],ns[j])
            k = set_key(ts[i],ns[j])

            i_grid = i*len(ns)+j
            if i == len(ts) - 1:
                grid[i_grid].set_xlabel("Distance (cm)")
            if j == 0:
                grid[i_grid].set_ylabel(r"Orientation ($\theta_{i,w}$)")
                #grid[i_grid].set_ylabel(r"Alignment ($\cos2\theta_{i,w}$)")
                grid[i_grid].set_yticks(np.linspace(m_range[0],m_range[1],5))

            print("Binning %s..." %k) 
            counts, xedges, yedges, im = grid[i_grid].hist2d(d_dm[k][:,0],
                                                             d_dm[k][:,1],
                                                bins  = [d_bins , m_bins ],
                                                range = [d_range, m_range],
                                                density = True,
                                                norm = colors.LogNorm(),
                                                cmap = my_cmap)
            grid[i_grid].set_aspect(aspect_ratio(d_range,m_range))
            grid_label = "%s, n = %i" % (ts[i], ns[j])
            grid[i_grid].text( 0.7, 0.9, grid_label, color='white',
                               horizontalalignment='center', verticalalignment='center', 
                               transform=grid[i_grid].transAxes )
            ims.append(im)

    print(len(grid))
  
    clims = [im.get_clim() for im in ims]
    vmin = min([clim[0] for clim in clims])
    vmax = max([clim[1] for clim in clims])
    #print(vmin,vmax)
    for im in ims:
        im.set_clim(vmin=vmin,vmax=vmax)

    fig.subplots_adjust(right=0.9)
    
    first = grid[0].get_position()
    last  = grid[-1].get_position()
    top_left = [ first.x0, first.y0 ]
    bottom_right = [ last.x1, last.y1 ]
    cbar_space = 0
    cbar_width = 0.03
    cbar_height = bottom_right[1] - top_left[1]
    print(cbar_height, top_left, bottom_right)
    cbar_ax = fig.add_axes([bottom_right[0] + cbar_space, 
                            top_left[1], 
                            cbar_width, 
                            cbar_height])
    cb = fig.colorbar(im, cax = cbar_ax)
    cb.set_label("normalized count", rotation=270, labelpad=20)

    if save:
        plt.savefig("results/figure_dw_thetaw_%s.png" % (tag))
    else:
        plt.show()
    plt.clf()



def plot_all_sets(d_dm,ts,ns,tag):
    for t in ts:
        for n in ns:
            plot_set(d_dm,t,n,tag,save=True)

arc = Archive()
tag = "t10to30_o0.0_v001.0to100.0_w-25.0to025.0_nbf3"
#tag = "t10to30_o0.0_v000.0to100.0_w-25.0to025.0_nbf3"
fname = "analysis_%s.arc" % tag
arc.load(fname)
calc=False

ts = [ "SF", "Pa", "Ti", "Mo" ]
#ts = [ "SF", "Pa" ]
ns = [ 1, 2, 5, 10 ]

import pickle
cwf = "wall_interaction.pik"
if os.path.isfile(cwf) and not calc:
    with open(cwf, 'rb') as handle:
        d_dm = pickle.load(handle)
else:
    d_dm = collect_all_sets(ts,ns,tag,calc=calc)
    if calc:
        arc.save()
    with open(cwf, 'wb') as handle:
        pickle.dump(d_dm, handle, protocol=pickle.HIGHEST_PROTOCOL)


for key in d_dm:
    print(key, d_dm[key].shape)
    print(d_dm[key][0:10])

plot_all_sets(d_dm,ts,ns,tag) 

nbins= [5,10,20]
#figure_wall_distance_orientation(d_dm, ts, ns, tag, save = True)
dhistdxs = [False, True]
orientations = [False, True]
for dhistdx in dhistdxs:
    for orientation in orientations:
        for nbin in nbins: 
            for t in ts:
                figure_wall_distance_mean_orientation(d_dm, t, ns, tag, bins = nbin, save = True, orientation = orientation, dhistdx = dhistdx)
            print("figure printed")







