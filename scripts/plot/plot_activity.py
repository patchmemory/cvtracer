#!/usr/bin/env python3
import sys
cvhome="/disk1/astyanax-mexicanus/cv-tracer"
sys.path.insert(0, cvhome)
import numpy as np
import matplotlib.pyplot as plt
from Analysis.Archive import Archive

def key(t,n):
    return "%s_%02i" % (t,n)

def frac_active(arc, t, n, vmin=1, vmax=100):
    frac = []
    for trial in arc.trial_list(t,n):
        for fish in trial.group.fish:
            try:
                arr = np.array(fish.df['speed'])
            except KeyError:
                continue
            arr = arr[arr < vmax]
            if len(arr) > 0:
                active  = float(len(arr[arr > vmin]))
                total   = float(len(arr))
                print(active/total, active, total)
                frac.append(active/total)
            else:
                print("Length of arr = 0!")
    frac = np.array(frac)
    mean = np.mean(frac)
    stdd = np.std(frac)
    return mean, stdd/np.sqrt(len(frac)-1)


def load_frac_active(arc, ts, ns):
    d = {}
    for t in ts:
        for n in ns:
            print("\n  Loading trials %s %2i..." % (t,n))
            mean, err = frac_active(arc, t, n)
            d[key(t,n)] = [ mean, err ]
    return d

def plot_frac_active(d, ts, ns, t_name, tag, save=True):
    print("  Plotting activity across group size and types...")
#    width = 1. / len(ts)
#    i = 0
#    start = len(ts) / 2.
    plt.title("Fraction of active frames by group size for each type")
    for t in ts:
        frac_by_n = []
        for n in ns:
            result = d[key(t,n)]
            frac_by_n.append([n,result[0],result[1]])
        frac_by_n = np.array(frac_by_n)
        plt.errorbar(frac_by_n[:,0], frac_by_n[:,1], yerr=frac_by_n[:,2],
                     fmt = 'o',
                     capsize = 3,
                     label = t_name[t] )
        
#        plt.bar(frac_by_n[:,0] + (i - start)*width, frac_by_n[:,1],yerr=frac_by_n[:,2],
#                     width = width, capsize = 3, label = t_name[t] )
#        i += 1

    plt.ylim(bottom=0)
    plt.xlim([0,12])
    plt.xlabel("group size")
    plt.legend()
    if save:
        plt.savefig("results/frac_active_vs_n_across_t_%s.png" % (tag))
    else:
        plt.show()
    plt.clf()


ts = [ 'SF', 'Pa', 'Mo', 'Ti' ]
t_name = { 'SF': "Surface", 
           'Pa': "Pachon", 
           'Mo': "Molino", 
           'Ti': "Tinaja" }

ns = [ 10, 5, 2, 1 ]

tag = "t10to30_o0.0_v001.0to100.0_w-25.0to025.0_nbf3"
#tag = "t10to30_o0.0_v000.0to100.0_w-25.0to025.0_nbf3"
fname = "analysis_%s.arc" % tag

arc = Archive()
arc.load(fname)

d = load_frac_active(arc, ts, ns)
plot_frac_active(d, ts, ns, t_name, tag)

