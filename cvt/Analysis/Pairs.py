#!/usr/bin/python
import sys
import numpy as np
sys.path.insert(0, '/data1/cavefish/social/python/track')
from Analysis.Archive import Archive


def neighbor_distance(arc,n,t,dij_min=0):
    if n > 1:
        n_dist = []
        nn_dist = []
        mean_n_dist = []
        mean_nn_dist = []
        dij_mij = []
        count = 0
        for i in range(len(arc.d['file'])):
            if arc.d['n'][i] == n and arc.d['type'][i] == t:
                print(arc.d['file'][i])
                count += 1
                print("\n\n  Calculating neighbor distances...")
                arc.d['group'][i].calculate_distance_alignment()
                print("... done.\n\n")
                n_dist_tmp  = arc.d['group'][i].neighbor_distance(arc.framei,arc.framef)
                nn_dist_tmp = arc.d['group'][i].nearest_neighbor_distance(arc.framei,arc.framef)
                dij_mij_tmp = arc.d['group'][i].neighbor_distance_alignment(arc.framei,arc.framej)
                mean_n_dist.append(np.mean(n_dist_tmp))
                mean_nn_dist.append(np.mean(nn_dist_tmp))
                n_dist.extend(list(n_dist_tmp))
                nn_dist.extend(list(nn_dist_tmp))
                dij_mij.extend(list(dij_mij_tmp))
        n_dist       = np.array(n_dist      )
        nn_dist      = np.array(nn_dist     )
        mean_n_dist  = np.array(mean_n_dist )
        mean_nn_dist = np.array(mean_nn_dist)
        mean_n_dist_avg  = np.mean(mean_n_dist)
        err_n_dist_avg   = np.sqrt(np.std(arc.mean_n_dist)/count)
        mean_nn_dist_avg = np.mean(arc.mean_nn_dist)
        err_nn_dist_avg  = np.sqrt(np.std(arc.mean_nn_dist)/count)
        
        val = 'dij_mij'
        dij_mij = np.array(dij_mij)
        arc.result[arc.result_key(n,t,val)] = dij_mij

        print("%s %i %e %e %e %e" % (t, n, mean_n_dist_avg, err_n_dist_avg, 
                                           mean_nn_dist_avg, err_nn_dist_avg))
        return mean_n_dist_avg, err_n_dist_avg, mean_nn_dist_avg, err_nn_dist_avg
    else:
        print("  Only one fish in this trial, so no neighbors...")
        return 0,0,0,0

def analyze_neighbors(arc,n,t,d_min=0,n_buffer_frames=2):
    frac_valid_vcut = []
    frac_valid_dcut = []
    frac_valid_both = []
    dij_mij = []
    for i_file in range(len(arc.d['file'])):
        if arc.d['n'][i_file] == n and arc.d['type'][i_file] == t:
            print(arc.d['file'][i_file])

            if n > 1:
                print("\n\n  Calculating neighbor distance and alignment...")
                arc.d['group'][i_file].calculate_distance_alignment()
                print("  ... done.\n\n")
                print("\n\n  Making neighbor distance cut...")
                arc.d['group'][i_file].neighbor_distance_cut(d_min,n_buffer_frames)
                dij_mij_tmp, frac_valid_both_tmp, frac_valid_vcut_tmp, frac_valid_dcut_tmp = \
                            arc.d['group'][i_file].valid_distance_alignment(arc.framei,arc.framef)
                dij_mij.extend(list(dij_mij_tmp))
            else:
                print("\n\n  Single fish, no neighbors... ")
                arc.d['group'][i_file].neighbor_distance_cut(d_min,n_buffer_frames)
                dij_mij_tmp, frac_valid_both_tmp, frac_valid_vcut_tmp, frac_valid_dcut_tmp = \
                    arc.d['group'][i_file].valid_distance_alignment(arc.framei,arc.framef)

            frac_valid_vcut.append(frac_valid_vcut_tmp)
            frac_valid_dcut.append(frac_valid_dcut_tmp)
            frac_valid_both.append(frac_valid_both_tmp)
            print("  ... done.\n\n")

    val = 'dij_mij'
    dij_mij = np.array(dij_mij)
    arc.result[arc.result_key(n,t,val)] = dij_mij

    val = 'frac_valid_vcut'
    frac_valid_vcut = np.array(frac_valid_vcut)
    arc.result[arc.result_key(n,t,val)] = frac_valid_vcut

    val = 'frac_valid_dcut'
    frac_valid_dcut = np.array(frac_valid_dcut)
    arc.result[arc.result_key(n,t,val)] = frac_valid_dcut

    val = 'frac_valid_both'
    frac_valid_both = np.array(frac_valid_both)
    arc.result[arc.result_key(n,t,val)] = frac_valid_both
    