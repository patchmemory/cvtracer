#!/usr/bin/python3
import sys
cvhome="/disk1/astyanax-mexicanus/cv-tracer"
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

def aspect_ratio(x_range,y_range):
    xlen = x_range[1] - x_range[0]
    ylen = y_range[1] - y_range[0]
    return xlen/ylen

def collect_diw_miw_trial(diw_miw):
    diw_miw_list = []
    for fish in diw_miw:
        for frame in fish:
            for neighbor in frame:
                diw_miw_list.append(neighbor)
    return diw_miw_list

def filter_diw_miw(diw_miw_arr):
    cut1 = diw_miw_arr[:,1] !=  1 
    cut2 = diw_miw_arr[:,1] != -1 
    cut3 = diw_miw_arr[:,0] !=  0
    cuts = cut1 & cut2 & cut3 
    diw_miw_cut = diw_miw_arr[cuts]
    frac_cut = (len(diw_miw_arr)-len(diw_miw_cut))/len(diw_miw_arr)
    return diw_miw_cut, frac_cut 


def calculate_diw_miw_all_trials_in_set(t,n):
    for trial in arc.trial_list(t,n):
        try:
            trial.calculate_wall_coordinates()
        except:
            print("  Issue calculating wall coordinates for trial, ")
            trial.print_info()


def combine_all_trials_in_set(t,n,tag):
    dm = []
    fps=30
    frame_range = [10*60*fps, 30*60*fps]
    for trial in arc.trial_list(t,n):
        try:
            trial.gather_wall_coordinates(frame_range, 
                    ocut = True, vcut = True, wcut = True, tag = tag)
            #trial.group.collect_wall_distance_alignment(frame_range, ocut = True, 
            #                                       vcut = True, wcut = True)
            dm.extend(trial.group.diw_miw)
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
                calculate_diw_miw_all_trials_in_set(t,n)
            #d_dm[k], d_fc[k] = combine_all_trials_in_set(t,n)
            d_dm[k] = combine_all_trials_in_set(t,n,tag)
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
        plt.savefig("results/%s_%02i_diw_miw_%s.png" % (t,n,tag))
    else:
        plt.show()
    plt.clf()


def figure_wall_distance_alignment(d_dm, ts, ns, tag, d_bins = 100, m_bins = 100, 
                            d_range = [0, 55.5], m_range = [-1,1], save = False):
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
                grid[i_grid].set_ylabel(r"Alignment ($\cos\theta_{ij}$)")
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
        plt.savefig("results/figure_diw_miw_%s.png" % (tag))
    else:
        plt.show()



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
ts = [ "SF", "Pa" ]
ns = [ 1, 2, 5, 10 ]

import pickle
cwf = "wall_interaction.pik"
if os.path.isfile(cwf):
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

figure_wall_distance_alignment(d_dm, ts, ns, tag, save = True)
print("figure printed")
