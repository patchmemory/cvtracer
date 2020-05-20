#!/usr/bin/python
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as mpl_cm
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

### HAVE ONLY MOVED THIS DIRECTLY FROM THE DATADICTIONARY / ARCHIVE MODULE
### NEED TO NOW MAKE IT WORK CORRECTLY

def aspect_ratio(x_range,y_range):
    xlen = x_range[1] - x_range[0]
    ylen = y_range[1] - y_range[0]
    return xlen/ylen


def plot_hist_across_n(arc,ns,t,val,val_title,nbins=10,hrange=None,norm=True,speed_cut=False):
    plt.title("Histograms of %s for %s across group size" % (val_title,t))
    plt.ylabel("Normalized Count") 
    plt.xlabel("%s" % val_title) 
    for i in range(len(ns)):  
        if speed_cut:
            hist, bin_edges = np.histogram(arc.combined_trials_speed_cut(ns[i],t,val),range=hrange,bins=nbins,density=norm)
        else:
            hist, bin_edges = np.histogram(arc.combined_trials(ns[i],t,val),range=hrange,bins=nbins,density=norm)
        binc = ( bin_edges[1:] + bin_edges[:-1] ) / 2 
        plt.plot(binc,hist,alpha=0.7,lw=2,label="%s fish"%ns[i])
    plt.legend()
    plt.show()


def plot_hist_across_t(arc,n,ts,val,val_title,nbins=10,hrange=None,norm=True,speed_cut=False):
    plt.title("Histograms of %s for group size %i across type" % (val_title,n))
    plt.ylabel("Normalized Count") 
    plt.xlabel("%s" % val_title) 
    for i in range(len(ts)):  
        if speed_cut:
            hist, bin_edges = np.histogram(arc.combined_trials_speed_cut(n,ts[i],val),range=hrange,bins=nbins,density=norm)
        else:
            hist, bin_edges = np.histogram(arc.combined_trials(n,ts[i],val),range=hrange,bins=nbins,density=norm)
        binc = ( bin_edges[1:] + bin_edges[:-1] ) / 2 
        plt.plot(binc,hist,alpha=0.7,lw=2,label="%s"%ts[i])
    plt.legend()
    plt.show()


def plot_hist_singles_all(arc,n,t,val,val_title,nbins=10,hrange=None,norm=True):
    plt.title("Histogram of %s for groups of %i %s" % (val_title,n,t))
    plt.ylabel("Normalized Count") 
    plt.xlabel("%s" % val_title) 
    for i in range(len(arc.d['file'])):
        if arc.d['n'][i] == n and arc.d['type'][i] == t:
            for i_fish in range(n):
                try:
                    plt.hist(arc.d['group'][i].fish[i_fish].df[val][arc.framei:arc.framef],range=hrange,bins=nbins,density=norm,alpha=0.5,lw=3)
                except TypeError:
                    fname = arc.d['file'][i].split('/')[-1]
                    fdate = arc.d['file'][i].split('/')[-2]
                    print("No data was found for %s/%s." % (fdate,fname))
    plt.show()


def plot_hist_singles_each(arc,n,t,val,val_title,nbins=10,hrange=None,norm=True):
    for i in range(len(arc.d['file'])):
        if arc.d['n'][i] == n and arc.d['type'][i] == t:
            fname = arc.d['file'][i].split('/')[-1]
            fdate = arc.d['file'][i].split('/')[-2]

            plt.title("Histogram of %s for the group of %i %s in\n%s/%s" % 
                            (val_title,n,t,fdate,fname) )
            plt.ylabel("Normalized Count") 
            plt.xlabel("%s" % val_title) 
            for i_fish in range(n):
                try:
                    plt.hist(arc.d['group'][i].fish[i_fish].df[val][arc.framei:arc.framef],range=hrange,bins=nbins,density=norm,alpha=0.5,lw=3)
                except TypeError:
                    try:
                        print("No data was found for %s/%s." % (fdate,fname))
                    except:
                        print("No data was found for %s." % (fdate,fname))
            plt.show()

        
def plot_hist_combined(arc,n,t,val,val_title,nbins=10,hrange=None,norm=True):
    val_list = []
    plt.title("Combined histogram %s for groups of %i %s" % (val_title,n,t))
    plt.ylabel("Normalized Count") 
    plt.xlabel("%s" % val_title) 
    for i in range(len(arc.d['file'])):
        if arc.d['n'][i] == n and arc.d['type'][i] == t:
            for i_fish in range(n):
                try:
                    val_list.extend(
                    arc.d['group'][i].fish[i_fish].df[val][arc.framei:arc.framef].tolist() )
                except TypeError:
                    fname = arc.d['file'][i].split('/')[-1]
                    fdate = arc.d['file'][i].split('/')[-2]
                    print("No data was found for %s/%s." % (fdate,fname))
    plt.hist(val_list,range=hrange,bins=nbins,density=norm,alpha=0.5,lw=3)
    plt.show()


def plot_single_fish_distributions(arc,ts,ns,binv,xmin,xmax,speed_cut=False,ftype='png',tag="",save=False):
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
            hist, bin_edges = np.histogram(arc.combined_trials_speed_cut(n,t,v),range=hrange,bins=nbins,density=norm)
        else:
            hist, bin_edges = np.histogram(arc.combined_trials(n,t,v),range=hrange,bins=nbins,density=norm)
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
            hist, bin_edges = np.histogram(arc.combined_trials_speed_cut(n,t,v),range=hrange,bins=nbins,density=norm)
        else:
            hist, bin_edges = np.histogram(arc.combined_trials(n,t,v),range=hrange,bins=nbins,density=norm)
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
            hist, bin_edges = np.histogram(arc.combined_trials_speed_cut(n,t,v),range=hrange,bins=nbins,density=norm)
        else:
            hist, bin_edges = np.histogram(arc.combined_trials(n,t,v),range=hrange,bins=nbins,density=norm)
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
            hist, bin_edges = np.histogram(arc.combined_trials_speed_cut(n,t,v),range=hrange,bins=nbins,density=norm)
        else:
            hist, bin_edges = np.histogram(arc.combined_trials(n,t,v),range=hrange,bins=nbins,density=norm)
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



def plot_multi_fish_distributions(arc,ts,ns,binv,xmin,xmax,speed_cut=False,ftype='png',tag="",save=False):
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
                hist, bin_edges = np.histogram(arc.combined_trials_speed_cut(n,t,v),range=hrange,bins=nbins,density=norm)
            else:
                hist, bin_edges = np.histogram(arc.combined_trials(n,t,v),range=hrange,bins=nbins,density=norm)
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
            mean = arc.combined_trials_stats_by_n(ns,t,v,'mean',tb="speed_cut")
            sterr = arc.combined_trials_stats_by_n(ns,t,v,'stderr',tb="speed_cut")
        else:
            mean = arc.combined_trials_stats_by_n(ns,t,v,'mean',tb="speed_cut")
            sterr = arc.combined_trials_stats_by_n(ns,t,v,'stderr',tb="speed_cut")
        ax0inset.errorbar(mean[:,0],mean[:,1],yerr=sterr[:,1],fmt='co',linewidth=5)

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
                hist, bin_edges = np.histogram(arc.combined_trials_speed_cut(n,t,v),range=hrange,bins=nbins,density=norm)
            else:
                hist, bin_edges = np.histogram(arc.combined_trials(n,t,v),range=hrange,bins=nbins,density=norm)
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
            mean = arc.combined_trials_stats_by_n(ns,t,v,'mean',tb="speed_cut")
            sterr = arc.combined_trials_stats_by_n(ns,t,v,'stderr',tb="speed_cut")
        else:
            mean = arc.combined_trials_stats_by_n(ns,t,v,'mean')
            sterr = arc.combined_trials_stats_by_n(ns,t,v,'stderr')

        ax1inset.errorbar(mean[:,0],mean[:,1],yerr=sterr[:,1],fmt='co',linewidth=5)
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
                hist, bin_edges = np.histogram(arc.combined_trials_speed_cut(n,t,v),range=hrange,bins=nbins,density=norm)
            else:
                hist, bin_edges = np.histogram(arc.combined_trials(n,t,v),range=hrange,bins=nbins,density=norm)
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
                hist, bin_edges = np.histogram(arc.combined_trials_speed_cut(n,t,v),range=hrange,bins=nbins,density=norm)
            else:
                hist, bin_edges = np.histogram(arc.combined_trials(n,t,v),range=hrange,bins=nbins,density=norm)
            binc = ( bin_edges[1:] + bin_edges[:-1] ) / 2 
            ax2inset.plot(binc, hist, label="%s" % (t))
        ax2inset.set_xlabel("%s" % v_label)
        ax2inset.set_xlim((0,2*xmax[v]))
        ax2inset.set_yscale('log') 
        i+=1
 
    plt.tight_layout()
    #plt.subplots_adjust(hspace=0.05,wspace=0.05)
    if save:
        plt.savefig("paper01/distribs_compare_multifish%s.png" % tag)
    else:
        plt.show()

    fig.clf()
  

def plot_dij_mij_log_one_cbar(arc,ts,ns,d_max=105,d_bins=105,m_max=1,m_bins=100,save=False):

    my_cmap = copy.copy(mpl_cm.get_cmap('viridis'))
    my_cmap.set_bad(my_cmap.colors[0])
    #my_cmap = copy.copy(mpl_cm.get_cmap('Blues'))

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
            k = arc.val_d_key(n,t,'dij_mij')
            dij_mij[k] = arc.val_d[k]
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


def plot_dij_mij_lin_one_cbar(arc,ts,ns,d_max=105,d_bins=105,m_max=1,m_bins=100,save=False):
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
            k = arc.val_d_key(n,t,'dij_mij')
            dij_mij[k] = arc.val_d[k]
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
  
  
def plot_dij_mij_lin_multi_cbar(arc,ts,ns,d_max=105,d_bins=105,m_max=1,m_bins=100,save=False):
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
            k = arc.val_d_key(n,t,'dij_mij')
            dij_mij[k] = arc.val_d[k]
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


def plot_frac_valid_vcut(arc,ts,ns,save=False):
    plt.rcParams.update({'font.size': 36})
    avg = [ [] for t in ts ]
    err = [ [] for t in ts ]
    t_iter = 0
    plt.title("Fraction valid after speed cut")
    plt.xlabel("group size")
    plt.xlim([0,12])
    plt.ylim([0,1.05])
    for t in ts:
        for n in ns:
            k = arc.val_d_key(n,t,'frac_valid_vcut')
            frac_valid_tmp = arc.val_d[k]
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


def plot_frac_valid_dcut(arc,ts,ns,save=True):
    plt.rcParams.update({'font.size': 36})
    avg = [ [] for t in ts ]
    err = [ [] for t in ts ]
    t_iter = 0
    plt.title("Fraction valid after occlusion cut")
    plt.xlabel("group size")
    plt.xlim([0,12])
    plt.ylim([0,1.05])
    for t in ts:
        for n in ns:
            k = arc.val_d_key(n,t,'frac_valid_dcut')
            frac_valid_tmp = arc.val_d[k]
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

def plot_frac_valid_both(arc,ts,ns,save=True):
    plt.rcParams.update({'font.size': 22})
    avg = [ [] for t in ts ]
    err = [ [] for t in ts ]
    plt.title("Fraction valid after both cuts")
    plt.xlabel("group size")
    plt.xlim([0,12])
    plt.ylim([0,1.05])
    t_iter = 0
    for t in ts:
        for n in ns:
            k = arc.val_d_key(n,t,'frac_valid_both')
            try:
                frac_valid_tmp = arc.val_d[k]
            except ValueError:
                k = arc.val_d_key(n,t,'frac_valid_both')
                frac_valid_tmp = arc.val_d[k]
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
