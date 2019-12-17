import numpy as np
import sys
cvhome="/disk1/astyanax-mexicanus/cv-tracer"
sys.path.insert(0, cvhome)
import matplotlib.pyplot as plt
from Analysis.Archive import Archive
savefig=True

arc = Archive()
fname="../analysis_t10to30_o0.0_v001.0to100.0_w-40.0to040.0_nbf3.arc"
tag = '.'.join('_'.join(fname.split('/')[-1].split('_')[1:]).split('.')[:-1])
arc.load(fname)

ts = ['SF','Pa','Mo','Ti']
ns = [1,2,5,10]

dwall_mean = {}
speed_mean = {}
omega_kurt = {}
omega_stdd = {}
omega_hist = {}
for t in ts:
    dwall_mean[t] = []
    speed_mean[t] = []
    omega_kurt[t] = []
    omega_stdd[t] = []
    omega_hist[t] = []
    for n in ns:

        k = arc.result_key(t,n,'dwall','mean',tag)
        _mean = arc.result[k]
        dwall_mean[t].append([n, _mean[0], _mean[1]])

        k = arc.result_key(t,n,'speed','mean',tag)
        _mean = arc.result[k]
        speed_mean[t].append([n, _mean[0], _mean[1]])

        k = arc.result_key(t,n,'omega','stdd',tag)
        _stdd = arc.result[k]
        omega_stdd[t].append([n, _stdd[0], _stdd[1]])

        k = arc.result_key(t,n,'omega','kurt',tag)
        _kurt = arc.result[k]
        omega_kurt[t].append([n, _kurt[0], _kurt[1]])
        
        k = arc.result_key(t,n,'omega','hist',tag)
        _hist = arc.result[k]
        histkey = "%s_%02i" % (t,n)
        omega_hist[histkey] = (_hist)


    dwall_mean[t] = np.array(dwall_mean[t])
    speed_mean[t] = np.array(speed_mean[t])
    omega_stdd[t] = np.array(omega_stdd[t])
    omega_kurt[t] = np.array(omega_kurt[t])
    omega_hist[t] = np.array(omega_hist[t])


plt.title("Mean distance to wall")
plt.xlabel("group size")
for t in ts:
    plt.errorbar(dwall_mean[t][:,0],dwall_mean[t][:,1], yerr=dwall_mean[t][:,2], fmt='o', label=t)
plt.ylim(bottom=0)
plt.xlim([0,12])
plt.legend()
if savefig:
    plt.savefig("summary_dwall_mean.png")
else:
    plt.show()
plt.clf()

plt.title("Mean speed")
plt.xlabel("group size")
for t in ts:
    plt.errorbar(speed_mean[t][:,0],speed_mean[t][:,1], yerr=speed_mean[t][:,2], fmt='o', label=t)
plt.ylim(bottom=0)
plt.xlim([0,12])
plt.legend()
if savefig:
    plt.savefig("summary_speed_mean.png")
else:
    plt.show()
plt.clf()

plt.title("Standard deviation of angular speed")
plt.xlabel("group size")
for t in ts:
    plt.errorbar(omega_stdd[t][:,0],omega_stdd[t][:,1], yerr=omega_stdd[t][:,2], fmt='o', label=t)
plt.ylim(bottom=0)
plt.xlim([0,12])
plt.legend()
if savefig:
    plt.savefig("summary_omega_stdd.png")
else:
    plt.show()
plt.clf()

plt.title("Kurtosis of angular speed")
plt.xlabel("group size")
for t in ts:
    plt.errorbar(omega_kurt[t][:,0],omega_kurt[t][:,1], yerr=omega_kurt[t][:,2], fmt='o', label=t)
plt.ylim(bottom=0)
plt.xlim([0,12])
plt.legend()
if savefig:
    plt.savefig("summary_omega_kurt.png")
else:
    plt.show()
plt.clf()

plt.title("Fourth moment of angular speed")
plt.xlabel("group size")
for t in ts:
    plt.errorbar(omega_kurt[t][:,0],omega_kurt[t][:,1]*np.power(omega_stdd[t][:,1],4), 
                    yerr=omega_kurt[t][:,1]*np.power(omega_stdd[t][:,1],4)*
                            np.sqrt(np.power(omega_kurt[t][:,2]/omega_kurt[t][:,1],2)
                                + np.power(omega_stdd[t][:,2]/omega_stdd[t][:,1],8)), 
                    fmt='o', label=t)
plt.ylim(bottom=0)
plt.xlim([0,12])
plt.legend()
if savefig:
    plt.savefig("summary_omega_mu4.png")
else:
    plt.show()
plt.clf()


exit()
for t in ts:
    plt.title("Histogram of angular speeds across groups of %s" % t)
    #plt.yscale("log")
    for n in ns:
        histkey = "%s_%02i" % (t,n)
        plt.plot(omega_hist[histkey][0][0],omega_hist[histkey][0][1], label=n)
        print(len(omega_hist[histkey]))
    plt.xlim(left=0)
    plt.legend()
    plt.show()

