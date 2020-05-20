from Analysis.Archive import Archive
import sys
cvhome="/home/patch/Code/cvtracer"
sys.path.insert(0, cvhome)
import numpy as np
import matplotlib.pyplot as plt

def collect_bouts(t,n,dcut=10):
    bouts = []
    for trial in arc.trial_list(t,n):
        bouts.extend(trial.group.neighbor_bouts(d_cut = dcut, frame_range = [10*60*30,30*60*30], ocut = True, vcut = True, wcut = True))
    bouts = np.array(bouts)
    return bouts/30

arc = Archive()
arc.load("../neighbor_plot.arc")

sf_n5 = collect_bouts('SF',5,20)
pa_n5 = collect_bouts('Pa',5,20)

plt.title("Duration of close proximity bouts") 
plt.ylabel("Normalized count")
#plt.xlabel("Duration (min)")
#plt.hist(pa_n5/60,bins=10, range=[0,20], alpha=0.5, density=True, label="Pachon")
#plt.hist(sf_n5/60,bins=10, range=[0,20], alpha=0.5, density=True, label="Surface")
plt.xlabel("Duration (sec)")
plt.hist(pa_n5,bins=60, range=[0,60], alpha=0.5, density=True, label="Pachon")
plt.legend()
#plt.show()
plt.savefig("prox_bouts_seconds.png")
#plt.savefig("prox_bouts_minutes.png")
