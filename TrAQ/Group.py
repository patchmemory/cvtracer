import numpy as np
import matplotlib.pyplot as plt
from TrAQ.Individual import Individual
from Analysis import Math as ana_math


class Group: 

    def __init__(self, n, t):

        self.t = t
        self.n = n

        self.fish = list(np.empty(n))
        for i in range(n):
            self.fish[i] = Individual()

        self.result = {}


    def n_fish(self):
        return len(self.fish)

    # store positions from a new tracked frame
    def add_entry(self, tstamp, coord):
        for i in range(self.n):
            self.fish[i].add_entry(tstamp, coord[i][0], coord[i][1], coord[i][2])

    def sort_by_time(self):
        for i_fish in range(self.n_fish()):
            self.fish[i_fish].sort_by_time()

    def print_info(self):
        print("  Group of %i %s fish." % (self.n, self.t) )
        print(self.fish)

    def print_frame(self,index):
        for i_fish in range(self.n_fish()):
            self.fish[i_fish].print_frame(index)

    def coordinates(self, index):
        coords = []
        for individual in self.fish:
            coords.append(individual.coordinates(index))
        return coords

    # Functions for storing and recalling results    
    def result_key(self, val_name, stat_name, tag = None):
        return "%s_%s_%s" % (val_name, stat_name, tag)
    
    def store_result(self, result, val_name, stat_name, tag = None):
        key = self.result_key(val_name, stat_name, tag)
        self.result[key] = result

    def get_result(self, val_name, stat_name, tag = None):
        key = self.result_key(val_name, stat_name, tag)
        return self.result[key]

    def print_result(self,key):
        print(key,self.result[key])

    def print_results(self):
        for key in self.result:
            self.print_result(key)

    def calculate_dwall(self,tank_radius):
        for i_fish in range(self.n_fish()):
            self.fish[i_fish].calculate_dwall(tank_radius)

    def calculate_velocity(self,fps):
        for i_fish in range(self.n_fish()):
            self.fish[i_fish].calculate_velocity(fps)

    def calculate_director(self,fps,theta_replace=False):
        for i_fish in range(self.n_fish()):
            self.fish[i_fish].calculate_director(fps,theta_replace)

    def calculate_acceleration(self,fps):
        for i_fish in range(self.n_fish()):
            self.fish[i_fish].calculate_acceleration(fps)

    def calculate_angular_velocity(self,fps):
        for i_fish in range(self.n_fish()):
            self.fish[i_fish].calculate_angular_velocity(fps)

    def calculate_angular_acceleration(self,fps):
        for i_fish in range(self.n_fish()):
            self.fish[i_fish].calculate_angular_acceleration(fps)

    def calculate_local_acceleration(self,fps):
        for i_fish in range(self.n_fish()):
            self.fish[i_fish].calculate_local_acceleration(fps)

    def convert_pixels(self, row_c, col_c, L_pix, L_m):
        for i_fish in range(self.n_fish()):
            self.fish[i_fish].convert_pixels(row_c, col_c, L_pix, L_m)

    def tstamp_correct(self,fps):
        for i_fish in range(self.n_fish()):
            self.fish[i_fish].t_stamp_reformat(fps)

    def speed_cut(self,speed_cut=1.,n_buffer_frames=2):
        for i_fish in range(self.n_fish()):
            self.fish[i_fish].speed_cut(speed_cut,n_buffer_frames)

    def cut_inactive(self,speed_cut=1.,n_buffer_frames=2):
        print("\n\n  Making speed cut at %4.2f cm/s with %i buffer frames" 
                                               % (speed_cut,n_buffer_frames))
        for i_fish in range(self.n_fish()):
            print("    ... for fish %i, " % i_fish)
            self.fish[i_fish].speed_cut(speed_cut,n_buffer_frames)

    def alignment(self,i,j,theta):
        return np.cos(theta[i])*np.cos(theta[j]) + np.sin(theta[i])*np.sin(theta[j])

    def distance(self,i,j,x,y):
        return np.sqrt(pow(x[j]-x[i],2) +  pow(y[j]-y[i],2))

    def calculate_distance_alignment(self):
        self.d_M = [ [ [] for j in range(self.fish[i].n_frames()) ] for i in range(self.n_fish()) ]

        x = [ [] for i in range(self.n_fish()) ]
        y = [ [] for i in range(self.n_fish()) ]
        e = [ [] for i in range(self.n_fish()) ]

        for i_fish in range(self.n_fish()):
            x[i_fish] = np.array(self.fish[i_fish].df['x'].tolist())
            y[i_fish] = np.array(self.fish[i_fish].df['y'].tolist())
            e[i_fish] = np.array(self.fish[i_fish].df['etheta'].tolist())
            x = np.array(x)
            y = np.array(y)
            e = np.array(e)

        for frame in range(len(e[0])):
            xs_tmp     = x[:,frame]
            ys_tmp     = y[:,frame]
            thetas_tmp = e[:,frame]
            for i_fish in range(self.n_fish()):
                dist_align_tmp = []
                for j_fish in range(self.n_fish()):
                    if i_fish != j_fish:
                        dij = self.distance(i_fish,j_fish,xs_tmp,ys_tmp)
                        mij = self.alignment(i_fish,j_fish,thetas_tmp)
                        dist_align_tmp.append([dij,mij])
                dist_align_tmp.sort()
                self.d_M[i_fish][frame] = dist_align_tmp

        self.d_M = np.array(self.d_M)


    def nearest_neighbor_distance_frame(self,i_frame): 
        print("i-frame=",i_frame)
        for i_fish in range(self.n_fish()):
            print("i-fish=",i_fish)
            print(self.d_M[i_fish][i_frame][0])
  

    def print_nn_dist_stats(self):
        print("  Mean NN Distance = %f cm" % np.mean(self.nn_dist))
        print("                     %f std. body length" % (np.mean(self.nn_dist)/self.std_body_length))
        print("  Median NN Distance = %f cm" % np.median(self.nn_dist))
        print("                     %f std. body length" % (np.median(self.nn_dist)/self.std_body_length))


    def nearest_neighbor_distance(self,framei,framef): 
        self.nn_dist = []
        for i_frame in range(framei,framef):
            for i_fish in range(self.n_fish()):
                self.nn_dist.append(self.d_M[i_fish][i_frame][0][0])
        self.nn_dist = np.array(self.nn_dist)
        print(self.nn_dist)
        print(" length of nn_dist array = ",len(self.nn_dist))
        self.nn_dist = self.nn_dist[self.nn_dist > 0]
        print(" length of nn_dist array, zeros removed = ",len(self.nn_dist))
        self.std_body_length = 5
        self.print_nn_dist_stats()
        return self.nn_dist

    def plot_nn_dist(self):
        print("  Plotting nearest neighbor distance... ")
        plt.title("Histogram of nearest neighbor distance")  
        plt.ylabel("Normalized count")  
        plt.xlabel("Nearest neighbor distance (cm)") 
        plt.hist(self.nn_dist/self.std_body_length,bins=100,density=True)
        plt.show()

    def neighbor_distance(self,framei,framef):
        n_dist = [] 
        for i_frame in range(framei,framef):
            for i_fish in range(self.n_fish()):
                n_list = self.d_M[i_fish][i_frame]
                for j_fish in range(len(n_list)):
                    n_dist_tmp = n_list[j_fish][0]
                    # if fish are overlapping, tag appropriately
                    if n_dist_tmp > 0:
                        self.overlap[i_frame][i_fish] = True
                        n_dist.append(n_list[j_fish][0])
        n_dist = np.array(n_dist)
        print(n_dist)
        print(" length of n_dist array =",len(n_dist))
        n_dist = n_dist[n_dist > 0]
        print(" length of n_dist array, zeros removed= ",len(n_dist))
        self.std_body_length = 5
        print("  Mean N Distance = %f cm" % np.mean(n_dist))
        print("                     %f std. body length" % 
                                        (np.mean(n_dist)/self.std_body_length))
        print("  Median N Distance = %f cm" % np.median(n_dist))
        print("                     %f std. body length" % 
                                      (np.median(n_dist)/self.std_body_length))
        return n_dist


    def cut_occlusions(self,d_min=0,n_buffer_frames=2):
        if self.n_fish() > 1:
            for i_fish in range(self.n_fish()):
                distance_nn = self.d_M[i_fish][:,0]
                distance_nn = distance_nn[:,0]
                #print("distance_nn",distance_nn)
                print("\n\n  Making occlusion cuts for fish %i via zeroed distances..." % i_fish)
                self.fish[i_fish].distance_cut(distance_nn,d_min,n_buffer_frames)
        else:
            print("\n\n  No occlusion cuts to make for single fish.")
            self.fish[0].distance_cut_null()
    
    def frac_valid(self,framei,framef):
        frac_ocut = []
        frac_vcut = []
        frac_both = []
        for i_fish in range(self.n_fish()):
            frac_ocut.append(self.fish[i_fish].total_frames_occlusion(framei,framef))
            frac_vcut.append(self.fish[i_fish].total_frames_inactive(framei,framef))
            frac_both.append(self.fish[i_fish].total_frames_cut(framei,framef))
        frac_ocut = np.array(frac_ocut)
        frac_vcut = np.array(frac_vcut)  
        frac_both = np.array(frac_both) 
        return np.mean(frac_both), np.std(frac_both)/np.sqrt(len(frac_both)), \
               np.mean(frac_vcut), np.std(frac_vcut)/np.sqrt(len(frac_vcut)), \
               np.mean(frac_ocut), np.std(frac_ocut)/np.sqrt(len(frac_ocut))


    # should move the cuts to their own routine
    def valid_distance_alignment(self,framei,framef):
        dij_mij = []
        d_M_cut = []
        n_total = 0
        n_valid_dcut = 0
        n_valid_vcut = 0
        n_valid_both = 0
        n_valid_dcut_tmp = 0
        n_valid_vcut_tmp = 0
        n_valid_both_tmp = 0
        for i_fish in range(self.n_fish()):
            vcut_tmp = np.array(self.fish[i_fish].df['speed_cut'].tolist())
            vcut_tmp = vcut_tmp[framei:framef]
            n_total_tmp = len(vcut_tmp)
            n_valid_vcut_tmp = len(vcut_tmp[np.logical_not(vcut_tmp)])
            dcut_tmp = np.array(self.fish[i_fish].df['d_cut'].tolist())
            dcut_tmp = dcut_tmp[framei:framef]
            n_valid_dcut_tmp = len(dcut_tmp[np.logical_not(dcut_tmp)])
            both_tmp = dcut_tmp | vcut_tmp
            n_valid_both_tmp = len(both_tmp[np.logical_not(both_tmp)])
            n_total += n_total_tmp
            n_valid_vcut += n_valid_vcut_tmp
            n_valid_dcut += n_valid_dcut_tmp
            n_valid_both += n_valid_both_tmp 
            print(i_fish,"fraction vcut",(n_total_tmp-n_valid_vcut_tmp)/n_total_tmp)
            print(i_fish,"fraction dcut",(n_total_tmp-n_valid_dcut_tmp)/n_total_tmp)
            print(i_fish,"fraction both",(n_total_tmp-n_valid_both_tmp)/n_total_tmp)
            print("")
            if self.n_fish() > 1:
                d_M_cut_tmp = self.d_M[i_fish][framei:framef]
                d_M_cut_tmp = list(d_M_cut_tmp[np.logical_not(both_tmp)])
                d_M_cut.extend(d_M_cut_tmp)

        if self.n_fish() > 1:
            for i in range(len(d_M_cut)):
                for j in range(len(d_M_cut[i])):
                    dij_mij.append(d_M_cut[i][j])

        print("length of dij_mij",len(dij_mij))
        return np.array(dij_mij), \
               n_valid_both_tmp/n_total_tmp, \
               n_valid_vcut_tmp/n_total_tmp, \
               n_valid_dcut_tmp/n_total_tmp


    def neighbor_distance_cut(self,d_min=0,n_buffer_frames=2):
        if self.n_fish() > 1:
            for i_fish in range(self.n_fish()):
                distance_nn = self.d_M[i_fish][:,0]
                distance_nn = distance_nn[:,0]
                #print("distance_nn",distance_nn)
                self.fish[i_fish].distance_cut(distance_nn,d_min,n_buffer_frames)
        else:
            self.fish[0].distance_cut_null()   


    # calculate_stats(...) takes the name of a value and calculates its 
    # statistics across individuals in the group. User can specify a range of 
    # values and a range of time, along with whether or not to use speed and 
    # occlusion cuts. Also has option to make data symmetric about the origin, 
    # for use with angular speed statistics.
    def calculate_stats(self, val_name, valmin = None, valmax = None, 
                        nbins = 100, framei = 0, framef = None, 
                        vcut = False, ocut = False, symm = False):
        
        stat_keys = [ "mean", "stdd", "kurt", "hist" ]
        stat_list = {}
        for key in stat_keys:
            stat_list[key] = []

        for i_fish in range(self.n_fish()):
            self.fish[i_fish].calculate_stats( val_name, valmin, valmax, nbins,
                                               framei, framef, vcut, ocut, symm )
            for key in stat_keys:
                stat_list[key].append(self.fish[i_fish].get_result(val_name,key))
    
        for key in stat_keys:
            if key == 'hist':
                stat_result = ana_math.mean_and_err_hist(stat_list[key], nbins)
            else:
                stat_result = ana_math.mean_and_err(stat_list[key])
            self.store_result(val_name, key, stat_result)
            