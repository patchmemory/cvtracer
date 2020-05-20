#!/usr/bin/env python3
import sys
cvhome="/home/patch/Code/cvtracer"
sys.path.insert(0, cvhome)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from Analysis.Archive import Archive
import pickle

def key(t,n):
    return "%s_%02i" % (t,n)

def calculate_tank_crossing(arc, t, n):
    frac = []
    for trial in arc.trial_list(t,n):
        trial.calculate_tank_crossing()
        for fish in trial.group.fish:
            try:
                tc = np.array(fish.df['tc'])
                cut = np.array(fish.df['cut'])
            except KeyError:
                continue
            tc = tc[np.logical_not(cut)]
            if len(tc) > 0:
                center = len(tc[tc])
                total   = len(tc)
                print(center/total, center, total)
                frac.append(center/total)
            else:
                print("Length of arr = 0!")
    frac = np.array(frac)
    mean = np.mean(frac)
    stdd = np.std(frac)
    return mean, stdd/np.sqrt(len(frac)-1)

def save_pickle(d,fname):
    f = open('tank_crossing_dict.pik','wb')
    pickle.dump(d,f)
    f.close()

def load_pickle(fname):
    f = open(fname, 'rb')
    d = pickle.load(f)
    f.close()
    return d

def load_tank_crossing(arc, ts, ns):
    d = {}
    d_fname = 'tank_crossing_dict.pik'
    try:
        d = load_pickle(d_fname)
    except:
        for t in ts:
            for n in ns:
                print("\n  Loading trials %s %2i..." % (t,n))
                mean, err = calculate_tank_crossing(arc, t, n)
                d[key(t,n)] = [ mean, err ]
        save_pickle(d,d_fname)
    return d


def frac_center(q,fstr,n):
    total = len(q["%s_n%i" % (fstr, n)])
    center = len(q["%s_n%i" % (fstr, n)][q["%s_n%i" % (fstr, n)] == True])
    return center/total

def sim_key(fstr,n):
    return "%s_n%i" % (fstr, n)

def frac_center(q,fstr,n):
    total = len(q["%s_n%i" % (fstr, n)])
    center = len(q["%s_n%i" % (fstr, n)][q["%s_n%i" % (fstr, n)] == True])
    return center/total


def collect_center_crossing_fracs(home,fstrs,ns):
    q = {}
    for n in ns: 
        for fstr in fstrs:
            if fstr == "avoid":
                R=21
                subdir="avoid_n%02i_R%i_v01.00_Dr0.50_r1.0_k100.0_rt02.0_kt40.0_rw01.0_kw100.0_rwt02.0_kwt10.0" % (n,R)
            if fstr == "align":
                R=11
                subdir="align_n%02i_R%i_v01.00_Dr1.00_r1.0_k10.0_rt02.5_kt01.0_rw02.0_kw01.0_rwt02.0_kwt10.0" % (n,R)
            if fstr == "nointeract":
                R=21
                subdir="nointeract_n%02i_R%i_v01.00_Dr0.50_r1.0_k100.0_rt02.0_kt40.0_rw01.0_kw100.0_rwt02.0_kwt10.0" % (n,R)
        #    q[sim_key(fstr, n)] = np.load("%s/%s_n%02i_R50_Dr1.00/tank_crossings.npy" % (home, fstr, n) )
            q[sim_key(fstr, n)] = np.load("%s/%s/tank_crossings.npy" % (home, subdir) )

    fracs = {}
    for fstr in fstrs:
        fracs[fstr] = []
        for n in ns: 
            print(fstr, n, frac_center(q,fstr,n))
            fracs[fstr].append([n,frac_center(q,fstr,n)])
        fracs[fstr] = np.array(fracs[fstr])

    return fracs

    

ts = [ 'SF', 'Pa', 'Mo', 'Ti' ]
ts = [ 'Ti', 'Mo', 'Pa', 'SF' ]

t_name = { 'SF': "Surface", 
           'Pa': "Pachon", 
           'Mo': "Molino", 
           'Ti': "Tinaja" }

t_color = { 'SF': (114,158,206), 
            'Pa': (255,158, 74), 
            'Mo': (103,191, 92), 
            'Ti': (237,102, 93) }

ns = [ 1, 2, 5, 10 ]
tag = "t10to30_o0.0_v001.0to100.0_w-25.0to025.0_nbf3"
#tag = "t10to30_o0.0_v000.0to100.0_w-25.0to025.0_nbf3"
fname = "analysis_%s.arc" % tag

arc = Archive()
arc.load(fname)
print(" %s loaded." % fname)

print(" Collecting center crossings from simulations... ")
_ns = [ 1, 2, 5, 10, 20 ]
fstrs = [ "nointeract", "avoid", "align"]
fstrs = [ "nointeract", "avoid" ]
s_name = { "nointeract": "Ignore", 
           "avoid":      "Evade" } 
str_color = { "nointeract": (205,204, 93),
              "avoid"     : (237,151,202) }

for k in str_color:
    c1, c2, c3 = str_color[k]
    str_color[k] = (c1/255., c2/255., c3/255.)
for k in t_color:
    c1, c2, c3 = t_color[k]
    t_color[k] = (c1/255., c2/255., c3/255.)

#home="/disk1/astyanax-mexicanus/avoidance/circle_wall/comparisons"
home="/disk1/astyanax-mexicanus/avoidance/circle_wall_align/"
fracs = collect_center_crossing_fracs(home,fstrs,_ns)


print(" Calculating tank crossings from experiment...")
d = load_tank_crossing(arc, ts, ns)
print(d)


plt.rc('font', size=12)
plt.rc('axes',  labelsize=16)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=12)

fig, ax = plt.subplots()
#plt.title("Fraction in center")
arr = {}
for t in ts:
    arr[t] = []
    for n in ns:
        _d = d[key(t,n)]
        _mean = _d[0]
        _stde = _d[1]
        arr[t].append([n,_mean,_stde])
    arr[t] = np.array(arr[t])
    #plt.errorbar(arr[t][:,0],arr[t][:,1],yerr=arr[t][:,2], label=t_name[t], barsabove=True, fmt='o')
    ax.errorbar(arr[t][:,0],arr[t][:,1],yerr=arr[t][:,2], label=t_name[t], linewidth=2,capsize=5, capthick=2, barsabove=True, fmt='--', color=t_color[t])

for fstr in fstrs:
    ax.plot(fracs[fstr][:,0], fracs[fstr][:,1], label=s_name[fstr], color=str_color[fstr],linewidth=2)


ax.set_xlabel('Group size')
ax.set_ylabel('Fraction of time in center')
ax.set_ylim(bottom=0)
ax.set_yticks([0,0.1,0.2,0.3,0.4])
ax.set_xscale('log')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1],labels[::-1],loc='middle center', bbox_to_anchor=(0.89,0.76), fancybox=True, shadow=True)
ax.set_xticks([1,2,5,10,20])
fmat = ScalarFormatter()
fmat.set_scientific(False)
ax.xaxis.set_major_formatter(fmat)
plt.tight_layout()
plt.savefig('tank_crossing.png')
