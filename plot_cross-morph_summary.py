#!/usr/bin/env python3
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
	buff = 0.1
	width = (1. - buff ) / len(ts)
	i = 0
	start = - ( ( len(ts) - 1 ) / 2.) * width
	plt.title("Fraction of active frames by group size for each type")
	_ns = ns
	_ns.sort()
	for t in ts:
	    frac_by_n = []
	    j = 1
	    for n in _ns:
	        result = d[key(t,n)]
	        frac_by_n.append([j,result[0],result[1]])
	        j += 1
	    frac_by_n = np.array(frac_by_n)
	    # plt.bar(frac_by_n[:,0] + (i*width + start), frac_by_n[:,1],yerr=frac_by_n[:,2],
	    #         width = width, capsize = 3, label = t_name[t], alpha = 0.7 )
	    plt.bar(frac_by_n[:,0] + (i*width + start), frac_by_n[:,1],yerr=frac_by_n[:,2],
	            width = width, capsize = 3, label = t_name[t], alpha = 0.7 )
	    i += 1

	plt.ylim(bottom=0)
	plt.xlim([0.5-buff/2,4.5+buff/2])
	plt.xlabel("group size")
	plt.xticks([1,2,3,4],_ns)
	plt.legend()
	if save:
	    plt.savefig("results/frac_active_vs_n_across_t_%s.png" % (tag))
	else:
	    plt.show()
	
	plt.clf()


def figure_morph_summary(arc, d, ts, ns, t_name, tag, save=True):

	plt.rcParams.update({'font.size': 20})
	fig, ax = plt.subplots(1,3, figsize=(17,5))

	buff = 0.1
	width = (1. - buff ) / len(ts)
	i = 0
	start = - ( ( len(ts) - 1 ) / 2.) * width
	ax[0].set_ylabel("Active Frame Fraction")
	_ns = ns
	_ns.sort()
	x = np.arange(len(_ns)) + 1
	for t in ts:
	    frac_by_n = []
	    for n in _ns:
	        result = d[key(t,n)]
	        frac_by_n.append([n, result[0],result[1]])
	    frac_by_n = np.array(frac_by_n)
	    # plt.bar(frac_by_n[:,0] + (i*width + start), frac_by_n[:,1],yerr=frac_by_n[:,2],
	    #         width = width, capsize = 3, label = t_name[t], alpha = 0.7 )
	    ax[0].bar(x + (i*width + start), frac_by_n[:,1],yerr=frac_by_n[:,2],
	            width = width, capsize = 3, label = t_name[t], alpha = 0.7 )
	    i += 1
	ax[0].set_ylim(bottom=0)
	ax[0].set_xlabel("group size")
	ax[0].set_xlim([0.5-buff/2,4.5+buff/2])
	ax[0].set_xticks(x)
	ax[0].set_xticklabels(_ns)



	speed_mean = {}
	omega_kurt = {}
	for t in ts:
	    speed_mean[t] = []
	    omega_kurt[t] = []
	    for n in _ns:
	        k = arc.result_key(t,n,'speed','mean',tag)
	        _mean = arc.result[k]
	        speed_mean[t].append([n, _mean[0], _mean[1]])

	        k = arc.result_key(t,n,'omega','kurt',tag)
	        _kurt = arc.result[k]
	        omega_kurt[t].append([n, _kurt[0], _kurt[1]])

	    speed_mean[t] = np.array(speed_mean[t])
	    omega_kurt[t] = np.array(omega_kurt[t])


	i = 0
	for t in ts:
	    ax[1].bar(x + (i*width + start), speed_mean[t][:,1], yerr=speed_mean[t][:,2],
	    		width = width, capsize = 3, label = t_name[t], alpha = 0.7 )
	    i += 1
	ax[1].set_ylabel("Mean Speed (cm/s)")
	ax[1].set_ylim(bottom=0)
	ax[1].set_xlabel("group size")
	ax[1].set_xlim([0.5-buff/2,4.5+buff/2])
	ax[1].set_xticks(x)
	ax[1].set_xticklabels(_ns)


	i = 0
	for t in ts:
	    ax[2].bar(x + (i*width + start), omega_kurt[t][:,1], yerr=omega_kurt[t][:,2],
	    		width = width, capsize = 3, label = t_name[t], alpha = 0.7 )
	    i += 1
	ax[2].set_ylabel("Kurtosis Angular Speed (rad/s)")
	ax[2].set_ylim(bottom=0)
	ax[2].set_xlabel("group size")
	ax[2].set_xlim([0.5-buff/2,4.5+buff/2])
	ax[2].set_xticks(x)
	ax[2].set_xticklabels(_ns)


	ax[2].legend(loc="center", bbox_to_anchor=(1.3,0.5), fontsize = 16)
	plt.tight_layout()
	left_shift = 0.005
	for axis in ax:
	    box = axis.get_position()
	    box.x0 = box.x0 - left_shift
	    box.x1 = box.x1 - left_shift
	    axis.set_position(box)

	


	if save:
	    plt.savefig("results/figure_morph_summary_%s.png" % (tag))
	else:
	    plt.show()
	

ts = [ 'SF', 'Pa', 'Mo', 'Ti' ]
t_name = { 'SF': "Surface", 
           'Pa': "Pachon", 
           'Mo': "Molino", 
           'Ti': "Tinaja" }

ns = [ 10, 5, 2, 1 ]

ns = [ 1, 2, 5, 10 ]
tag = "t10to30_o0.0_v001.0to100.0_w-25.0to025.0_nbf3"
#tag = "t10to30_o0.0_v000.0to100.0_w-25.0to025.0_nbf3"
fname = "analysis_%s.arc" % tag

arc = Archive()
arc.load(fname)

d = load_frac_active(arc, ts, ns)
#plot_frac_active(d, ts, ns, t_name, tag)
figure_morph_summary(arc, d, ts, ns, t_name, tag, save=True)
