import numpy as np
import matplotlib.pyplot as plt
from cvt.TrAQ.Individual import Individual
from cvt.Analysis import Math as ana_math


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

    def clear_results(self, tag = None):
        for key in self.result:
            tag_tmp = '_'.join(key.split('_')[2:])
            if tag == tag_tmp:
                del self.result[key]

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

    def calculate_tank_crossing(self, R):
        for i_fish in range(self.n_fish()):
            self.fish[i_fish].calculate_tank_crossing(R)

    def convert_pixels(self, row_c, col_c, L_pix, L_m):
        for i_fish in range(self.n_fish()):
            self.fish[i_fish].convert_pixels(row_c, col_c, L_pix, L_m)

    def tstamp_correct(self,fps):
        for i_fish in range(self.n_fish()):
            self.fish[i_fish].t_stamp_reformat(fps)


    def valid_frame_fraction(self, frame_range = None, cut_name = 'cut'):
        valid = []
        for i_fish in range(self.n_fish()):    
            valid.append(self.fish[i_fish].valid_frame_fraction(frame_range, cut_name))
        return np.array(valid)
    
    
    # cut_occlusion(...) generates a list of frames to be cut for being too close 
    # (occluded fish are set with same position, so d_ij = 0)
    def cut_occlusion(self, d_min = 0, n_buffer_frames = 2):
        print("\n\n  Generating occlusion cut using min d_nn = %4.2f cm with %i buffer frames"
                                               % (d_min, n_buffer_frames))
        if self.n_fish() > 1:
            try:
                self.dij_mij
            except:
                self.calculate_distance_alignment()
                
            for i_fish in range(self.n_fish()):
                distance_nn = np.array(self.fish[i_fish].get_dij_mij(1))
                distance_nn = distance_nn[:,0]
                #print("distance_nn",distance_nn)
                print("    ... for fish %i, " % i_fish)
                self.fish[i_fish].cut_occlusion(d_min, n_buffer_frames, distance_nn)
        else:
            print("\n\n  No occlusion cuts to make for single fish.")
            self.fish[0].cut_occlusion_null()

    # cut_omega(...) generates list of frames to be cut based on their angular speed
    def cut_omega(self, fps = 30, omega_range = [ -40., 40. ], n_buffer_frames = 2):
        print("\n\n  Generating angular speed cut using range [%4.2f, %4.2f] rad/s with %i buffer frames" 
                                               % (omega_range[0], omega_range[1], n_buffer_frames))
        for i_fish in range(self.n_fish()):
            print("    ... for fish %i, " % i_fish)
            self.fish[i_fish].cut_omega(fps, omega_range, n_buffer_frames)

    # cut_speed(...) generates list of frames to be cut based on their speed
    def cut_speed(self, fps = 30, speed_range = [ 1., 100. ], n_buffer_frames=2):
        print("\n\n  Generating speed cut using range [%4.2f, %4.2f] cm/s with %i buffer frames" 
                                               % (speed_range[0], speed_range[1], n_buffer_frames))
        for i_fish in range(self.n_fish()):
            print("    ... for fish %i, " % i_fish)
            self.fish[i_fish].cut_speed(fps, speed_range, n_buffer_frames)

    # cut_speed(...) generates list of frames to be cut based on their speed
    def cut_combine(self):
        print("\n\n  Combining all cuts.... ")
        for i_fish in range(self.n_fish()):
            print("    ... for fish %i, " % i_fish)
            self.fish[i_fish].cut_all()




    def cut_stats(self, framei, framef):
        valid = {'ocut': [], 'vcut': [], 'wcut': [], 'cut': [] }
        for i_fish in range(self.n_fish()):
            for key in valid:
                valid[key].append(
                        self.fish[i_fish].valid_frame_fraction(
                                                [framei, framef], key))
        mean = {}
        err  = {}
        for key in valid:
            valid[key] = np.array(valid[key])
            mean[key]  = np.mean(valid[key])
            err[key]   = np.std(valid[key])/np.sqrt(len(valid[key]))
        
        return mean, err



    def alignment(self,i,j,theta):
        return np.cos(theta[i])*np.cos(theta[j]) + np.sin(theta[i])*np.sin(theta[j])

    def distance(self,i,j,x,y):
        return np.sqrt(pow(x[j]-x[i],2) +  pow(y[j]-y[i],2))


    def wall_distance(self, i, x, y, r_tank = 55.5):
        return r_tank - np.sqrt(pow(x[i],2) + pow(y[i],2))

    def wall_alignment(self,i, x, y, theta):
        r_pos = np.sqrt(x[i]**2 + y[i]**2)
        n_hat = np.column_stack((x[i]/r_pos, y[i]/r_pos))
        return np.cos(theta[i])*n_hat[:,0] + np.sin(theta[i])*n_hat[:,1]

    def calculate_wall_distance_orientation(self, r_tank = 55.5):
        _dw_thetaw = [ [] for i in range(self.n_fish()) ]
        for i_fish in range(self.n_fish()):
            dwall = np.array(self.fish[i_fish].df.dwall.tolist())
            self.fish[i_fish].calculate_theta_wall()
            theta_wall = np.array(self.fish[i_fish].df.theta_wall.tolist())
            _dw_thetaw[i_fish] = np.column_stack((dwall, theta_wall))

        _dw_thetaw = np.array(_dw_thetaw)
        self.dw_thetaw = []
        for i_fish in range(self.n_fish()):
            #self.fish[i_fish].set_diw_miw(_diw_miw[i_fish])
            self.dw_thetaw.extend(_dw_thetaw[i_fish])
        self.dw_thetaw = np.array(self.dw_thetaw)

    def collect_wall_distance_orientation(self, frame_range = None, 
                                   ocut = False, vcut = False, wcut = False):
        self.dw_thetaw = []
        for i_fish in range(self.n_fish()):
            _dw_thetaw = self.fish[i_fish].get_dw_thetaw(frame_range, ocut, vcut, wcut)
            self.dw_thetaw.extend(_dw_thetaw)
        self.dw_thetaw = np.array(self.dw_thetaw)

    def calculate_wall_distance_alignment(self, r_tank = 55.5):

        x = [ [] for i in range(self.n_fish()) ]
        y = [ [] for i in range(self.n_fish()) ]
        e = [ [] for i in range(self.n_fish()) ]
        for i_fish in range(self.n_fish()):
            x[i_fish] = np.array(self.fish[i_fish].df['x'].tolist())
            y[i_fish] = np.array(self.fish[i_fish].df['y'].tolist())
            e[i_fish] = np.array(self.fish[i_fish].df['theta'].tolist())
        x = np.array(x)
        y = np.array(y)
        e = np.array(e)

        _diw_miw = [ [] for i in range(self.n_fish()) ]
        for i_fish in range(self.n_fish()):
            diw = self.wall_distance(i_fish, x, y, r_tank = r_tank)
            miw = self.wall_alignment(i_fish, x, y, e)
            _diw_miw[i_fish] = np.column_stack((diw, miw))

        _diw_miw = np.array(_diw_miw)
        self.diw_miw = []
        for i_fish in range(self.n_fish()):
            self.fish[i_fish].set_diw_miw(_diw_miw[i_fish])
            self.diw_miw.extend(_diw_miw[i_fish])
        self.diw_miw = np.array(self.diw_miw)

    def collect_wall_distance_alignment(self, frame_range = None, 
                                   ocut = False, vcut = False, wcut = False):
        self.diw_miw = []
        for i_fish in range(self.n_fish()):
            _diw_miw = self.fish[i_fish].get_diw_miw(frame_range, ocut, vcut, wcut)
            self.diw_miw.extend(_diw_miw)
        self.diw_miw = np.array(self.diw_miw)


    def calculate_distance_alignment(self):
        _dij_mij = [ [ [] for j in range(self.fish[i].n_frames()) ] for i in range(self.n_fish()) ]

        x = [ [] for i in range(self.n_fish()) ]
        y = [ [] for i in range(self.n_fish()) ]
        e = [ [] for i in range(self.n_fish()) ]
        for i_fish in range(self.n_fish()):
            x[i_fish] = np.array(self.fish[i_fish].df['x'].tolist())
            y[i_fish] = np.array(self.fish[i_fish].df['y'].tolist())
            e[i_fish] = np.array(self.fish[i_fish].df['theta'].tolist())
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
                _dij_mij[i_fish][frame] = dist_align_tmp

        _dij_mij = np.array(_dij_mij)
        self.dij_mij = []
        for i_fish in range(self.n_fish()):
            self.fish[i_fish].set_dij_mij(_dij_mij[i_fish])
            self.dij_mij.extend(_dij_mij[i_fish])
        self.dij_mij = np.array(self.dij_mij)
            
            
    def collect_distance_alignment(self, frame_range = None, 
                                   ocut = False, vcut = False, wcut = False):
        self.dij_mij = []
        for i_fish in range(self.n_fish()):
            _dij_mij = self.fish[i_fish].get_dij_mij(self.n_fish()-1, frame_range, 
                                                     ocut, vcut, wcut)
            self.dij_mij.extend(_dij_mij)
        self.dij_mij = np.array(self.dij_mij)
        

    def neighbor_bouts(self, d_cut = 10, frame_range = None,
                       ocut = False, vcut = False, wcut = False):
        neighbor_bouts = []
        for i_fish in range(self.n_fish()):
            #try:
            neighbor_bouts.extend(self.fish[i_fish].nearest_distance_bout(
                                         d_cut, frame_range, ocut, vcut, wcut))
            #except:
                #print("  Issue gathering nearest neighbor bouts.")
        return neighbor_bouts

    def nearest_neighbor_distance_frame(self,i_frame): 
        print("i-frame=",i_frame)
        for i_fish in range(self.n_fish()):
            print("i-fish=",i_fish)
            print(self.dij_mij[i_fish][i_frame][0])
  

    def print_nn_dist_stats(self):
        print("  Mean NN Distance = %f cm" % np.mean(self.nn_dist))
        print("                     %f std. body length" % (np.mean(self.nn_dist)/self.std_body_length))
        print("  Median NN Distance = %f cm" % np.median(self.nn_dist))
        print("                     %f std. body length" % (np.median(self.nn_dist)/self.std_body_length))


    def nearest_neighbor_distance(self,framei,framef): 
        self.nn_dist = []
        for i_frame in range(framei,framef):
            for i_fish in range(self.n_fish()):
                self.nn_dist.append(self.dij_mij[i_fish][i_frame][0][0])
        self.nn_dist = np.array(self.nn_dist)
        print(self.nn_dist)
        print(" length of nn_dist array = ",len(self.nn_dist))
        self.nn_dist = self.nn_dist[self.nn_dist > 0]
        print(" length of nn_dist array, zeros removed = ",len(self.nn_dist))
        self.std_body_length = 5
        self.print_nn_dist_stats()

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
                n_list = self.dij_mij[i_fish][i_frame]
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


    # calculate_stats(...) takes the name of a value and calculates its 
    # statistics across individuals in the group. User can specify a range of 
    # values and a range of time, along with whether or not to use speed and 
    # occlusion cuts. Also has option to make data symmetric about the origin, 
    # for use with angular speed statistics.
    def calculate_stats(self, val_name, val_range = None, val_symm = False,
                        frame_range = None, nbins = 100,
                        ocut = False, vcut = False, wcut = False, tag = None ):
        
        stat_keys = [ "mean", "stdd", "kurt", "hist" ]
        stat_list = {}
        for key in stat_keys:
            stat_list[key] = []

        for i_fish in range(self.n_fish()):
            self.fish[i_fish].calculate_stats( val_name, val_range, val_symm,
                                               frame_range = frame_range, nbins = nbins,
                                               ocut = ocut, vcut = vcut, wcut = wcut, tag = tag)
            for key in stat_keys:
                stat_list[key].append(self.fish[i_fish].get_result(val_name,key,tag))
    
        for key in stat_keys:
            if key == 'hist':
                stat_result = ana_math.mean_and_err_hist(stat_list[key], nbins)
            else:
                stat_result = ana_math.mean_and_err(stat_list[key])
            self.store_result(stat_result, val_name, key, tag)
            
