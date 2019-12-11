#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from Analysis.Archive import Archive

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


def load_tank_crossing(arc, ts, ns):
    d = {}
    for t in ts:
        for n in ns:
            print("\n  Loading trials %s %2i..." % (t,n))
            mean, err = calculate_tank_crossing(arc, t, n)
            d[key(t,n)] = [ mean, err ]
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
			q[sim_key(fstr, n)] = np.load("%s/%s_n%02i_R50_Dr1.00/tank_crossings.npy" % (home, fstr, n) )

	fracs = {}
	for fstr in fstrs:
		fracs[fstr] = []
		for n in ns: 
			print(fstr, n, frac_center(q,fstr,n))
			fracs[fstr].append([n,frac_center(q,fstr,n)])
		fracs[fstr] = np.array(fracs[fstr])

	return fracs

	

ts = [ 'SF', 'Pa', 'Mo', 'Ti' ]
ts = [ 'Mo', 'Ti', 'Pa', 'SF' ]

t_name = { 'SF': "Surface", 
           'Pa': "Pachon", 
           'Mo': "Molino", 
           'Ti': "Tinaja" }

ns = [ 1, 2, 5, 10 ]
tag = "t10to30_o0.0_v001.0to100.0_w-25.0to025.0_nbf3"
#tag = "t10to30_o0.0_v000.0to100.0_w-25.0to025.0_nbf3"
fname = "analysis_%s.arc" % tag

arc = Archive()
arc.load(fname)
print(" %s loaded." % fname)

print(" Calculating tank crossings from experiment...")
d = load_tank_crossing(arc, ts, ns)
print(d)

print(" Collecting center crossings from simulations... ")
_ns = [ 2, 5, 10, 20 ]
fstrs = [ "nointeract", "avoid", "align"]
home="/disk1/astyanax-mexicanus/avoidance/circle_wall/comparisons"
fracs = collect_center_crossing_fracs(home,fstrs,_ns)

plt.title("Fraction in center")
for fstr in fstrs:
    plt.plot(fracs[fstr][:,0], fracs[fstr][:,1], label="sim, %s" % fstr)

arr = {}
for t in ts:
	arr[t] = []
	for n in ns:
		_d = d[key(t,n)]
		_mean = _d[0]
		_stde = _d[1]
		arr[t].append([n,_mean,_stde])
	arr[t] = np.array(arr[t])
	plt.errorbar(arr[t][:,0],arr[t][:,1],yerr=arr[t][:,2], label=t_name[t], barsabove=True, fmt='o')

plt.xlabel('Group size')
plt.ylabel('Fraction of frames in center')
plt.ylim(bottom=0)
plt.legend()
plt.savefig('tank_crossing.png')

