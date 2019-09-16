import math
import numpy as np
import pandas as pd
import scipy.stats as spstats


class Individual:

    def __init__(self):
        d = {'t': [], 'x': [], 'y': [], 'theta': [] }    
        self.df = pd.DataFrame( data=d, dtype=np.float64 )
        self.result = {}
        self.val_range = {}

    def n_frames(self):
        return len(self.df.index)

    def print(self):
        print(self.df)
    
    def add_entry(self,t,x,y,theta):
        #df_add = pd.DataFrame({'t':[t], 'row':[x],'col':[y],'theta_pix':[theta]})
        df_add = pd.DataFrame({'t':[t], 'x':[x],'y':[y],'theta':[theta]})
        self.df = self.df.append(df_add, ignore_index=True)

    def print_frame(self,index):
        print(self.df.loc[index])

    def sort_by_time(self):
        self.df = self.df.sort_values(by='t')

    def loc(self,index,val):
        return self.df.loc[index,val]
    
    def coordinates(self, index):
        return [ self.loc(index,'x'), 
                 self.loc(index,'y'), 
                 self.loc(index,'theta') ]

    def convert_pixels(self, row_c, col_c, L_pix, L_m):
        A_convert = L_m / L_pix
        self.pixel_reformat()
        self.df['x'] =  A_convert*(self.df['row'] - row_c)
        self.df['y'] =  A_convert*(self.df['col'] - col_c)

    def pixel_reformat(self):
        if 'col' not in self.df.columns:
            self.df['col'] = self.df['x']
            self.df.drop(columns=['x'])
        if 'row' not in self.df.columns:
            self.df['row'] = self.df['y']
            self.df.drop(columns=['y'])
        if 'theta_pix' not in self.df.columns:
            self.df['theta_pix'] = self.df['theta']
            self.df.drop(columns=['theta'])

    def calculate_dwall(self,tank_radius):
        self.df['dwall'] = self.df.apply(lambda row: tank_radius - 
                           ( np.sqrt(pow(row.x,2) + pow(row.y,2)) ), axis=1)

    def calculate_velocity(self,fps):
        if 'vx' not in self.df.columns or 'vy' not in self.df.columns:
            self.df['vx'] = ( self.df.x.shift(-1) - self.df.x.shift(1) ) / 2 * fps
            self.df['vy'] = ( self.df.y.shift(-1) - self.df.y.shift(1) ) / 2 * fps
        if 'speed' not in self.df.columns:
            self.df['speed'] = np.sqrt(pow(self.df.vx,2) + pow(self.df.vy,2)) 

    def calculate_director(self,fps,theta_replace=False):
        if 'vx' not in self.df.columns or 'vy' not in self.df.columns:
            self.calculate_velocity()
        if 'ex' not in self.df.columns or 'ey' not in self.df.columns:
            self.df['ex'] = self.df.vx / self.df.speed 
            self.df['ey'] = self.df.vy / self.df.speed
        if theta_replace:
            self.df['theta'] = np.arctan2(self.df.ey,self.df.ex)
            self.df['etheta'] = np.arctan2(self.df.ey,self.df.ex)
        else:
            self.df['theta_pix'] = self.df.theta
            self.df['etheta'] = np.arctan2(self.df.ey,self.df.ex)

    def calculate_acceleration(self,fps):
        self.df['ax'] = ( self.df.vx.shift(-1) - self.df.vx.shift(1) ) / 2 * fps
        self.df['ay'] = ( self.df.vy.shift(-1) - self.df.vy.shift(1) ) / 2 * fps

    def angle_diff(self,q2,q1):
        return min(q2-q1,q2-q1,q2-q1+2*np.pi,q2-q1-2*np.pi, key=abs)

    def calculate_angular_velocity(self,fps):
        self.df['omega1'] = self.df.theta.shift(-1) - self.df.theta.shift(1) 
        self.df['omega2'] = self.df.theta.shift(-1) - self.df.theta.shift(1) + 2*np.pi 
        self.df['omega3'] = self.df.theta.shift(-1) - self.df.theta.shift(1) - 2*np.pi
        self.df['omega'] = self.df[['omega1', 'omega2', 'omega3']].apply(lambda row: min(row[0],row[1],row[2],key=abs), axis=1)
        self.df['omega'] *= (fps/2.)
        self.df.drop(columns=['omega1','omega2','omega3'], inplace=True)

    def calculate_angular_acceleration(self,fps):
        self.df['alpha1'] = self.df.theta.shift(-1) - self.df.theta.shift(1) 
        self.df['alpha2'] = self.df.theta.shift(-1) - self.df.theta.shift(1) + 2*np.pi 
        self.df['alpha3'] = self.df.theta.shift(-1) - self.df.theta.shift(1) - 2*np.pi
        self.df['alpha'] = self.df[['alpha1', 'alpha2', 'alpha3']].apply(lambda row: min(row[0],row[1],row[2],key=abs), axis=1)
        self.df['alpha'] *= (fps/2.)
        self.df.drop(columns=['alpha1','alpha2','alpha3'], inplace=True)

    def calculate_local_acceleration(self,fps):
        if 'ax' not in self.df.columns or 'ay' not in self.df.columns:
            self.calculate_acceleration()
        self.df['af'] =   np.cos(self.df.etheta)*self.df.ax + np.sin(self.df.etheta)*self.df.ay
        self.df['al'] = - np.sin(self.df.etheta)*self.df.ax + np.cos(self.df.etheta)*self.df.ay

    def tstamp_reformat(self,fps):
        mean_dt = (self.df.t.shift(-1) - self.df.t.shift(0)).mean()
        expected_dt = 1./fps
        if math.isclose(mean_dt,expected_dt,rel_tol=0.01/fps):
            factor = expected_dt / mean_dt 
            self.df.t *= factor
            
    def set_distance_nn(self, d_nn):
        self.df['d_nn'] = d_nn.tolist()
    
    # cut_omega(...) generates list of frames to be cut based on their angular speed
    def cut_omega(self, fps = 30, omega_range = [ -40., 40. ], n_buffer_frames = 2):
        if 'omega' not in self.df.columns:
            self.calculate_angular_velocity(fps)
    
        self.df['wcut'] = self.df[['omega']].apply(lambda row: 
                    (row[0] < omega_range[0] or row[0] > omega_range[1]), axis=1)
        for j_frame in range(1,n_buffer_frames+1):
            self.df['wcut'] = ( self.df['wcut'] 
                        | self.df.wcut.shift(-j_frame) 
                        | self.df.wcut.shift( j_frame) )

    # cut_speed(...) generates list of frames to be cut based on their speed
    def cut_speed(self, fps = 30, speed_range = [ 1., 100. ], n_buffer_frames = 2 ):
        if 'speed' not in self.df.columns:
            self.calculate_velocity(fps)

        self.df['vcut'] = self.df[['speed']].apply(lambda row: 
                    (row[0] < speed_range[0] or row[0] > speed_range[1]), axis=1)
        for j_frame in range(1,n_buffer_frames+1):
            self.df['vcut'] = ( self.df['vcut'] 
                        | self.df.vcut.shift(-j_frame) 
                        | self.df.vcut.shift( j_frame) )

    # cut_occlusion(...) generates a list of frames to be cut for being too 
    # close (occluded fish are set with same position, so d_ij = 0)
    def cut_occlusion(self, d_min = 0, n_buffer_frames = 2, d_nn = None):
        try:
            if d_nn == None and 'd_nn' not in self.df.columns:
                print("  Nearest neighbor distance not available, and cannot be calculated by an Individual. Please first calculate and store neighbor distances")
                return
        except:
            self.set_distance_nn(d_nn)
            
        self.df['ocut'] = np.logical_not(self.df['d_nn'] > d_min)

        for j_frame in range(1, n_buffer_frames + 1):
            self.df['ocut'] = ( self.df['ocut'] 
                              | self.df.ocut.shift(-j_frame) 
                              | self.df.ocut.shift( j_frame) )
    
    def cut_occlusion_null(self):
        self.df['ocut'] = False


    def valid_frame_fraction(self, frame_range = None, cut_name = 'cut'):
        if frame_range == None:
            framei = 0
            framef = self.n_frames()
        else:
            framei = frame_range[0]
            framef = frame_range[1]
        n_cut = 1.*sum(self.df[cut_name].values[framei:framef])
        n_tot = 1.*len(self.df[cut_name].values[framei:framef])
        return 1. - n_cut / n_tot


    def total_frames_occlusion(self,framei,framef):
        return float(sum(self.df['ocut'][framei:framef]))/len(self.df['ocut'][framei:framef])

    def total_frames_inactive(self,framei,framef):
        return float(sum(self.df['vcut'][framei:framef]))/len(self.df['vcut'][framei:framef])

    def total_frames_cut(self,framei,framef):
        n_cut = sum(self.df['vcut'][framei:framef] | self.df['ocut'][framei:framef])
        return float(n_cut)/len(self.df['vcut'][framei:framef])

    # cut_array(...) generates a mask based on cuts, providing boolean values
    # that represent frames to be cut. User specifies cuts which must be 
    # completed prior to running this function.
    def cut_array(self, arr, ocut = False, vcut = False, wcut = False ):
        
        self.cut_arr = np.zeros_like(arr,dtype=bool)

        if ocut:
            if 'ocut' not in self.df.columns:
                print("\n No occlusion cut column found in DataFrame, generating now...\n")

            ocut_arr = self.df['ocut']
            self.cut_arr = self.cut_arr | ocut_arr
        
        if vcut:
            if 'vcut' not in self.df.columns:
                print("\n No speed cut column found in DataFrame, generating now... \n")

            vcut_arr = self.df['vcut']
            self.cut_arr = self.cut_arr | vcut_arr
            
        if wcut:
            if 'wcut' not in self.df.columns:
                print("\n No angular speed cut column found in DataFrame, generating now... \n")

            wcut_arr = self.df['wcut']
            self.cut_arr = self.cut_arr | wcut_arr

    def cut_all(self):
        self.df['cut'] = self.df['ocut'] | self.df['vcut'] | self.df['wcut']

    # calculate_stats(...) takes the name of a value and calculates its 
    # statistics across valid frames. User can specify a range of values and a 
    # range of time, along with whether or not to use speed and occlusion cuts. 
    # Also has option to make data symmetric about the origin, for use with 
    # angular speed statistics.
    def calculate_stats(self, val_name, val_range = None, val_symm = False,
                        frame_range = None, nbins = 100,
                        ocut = False, vcut = False, wcut = False, tag = None):

        if val_name not in self.df.columns:
            print("\n  %s not found in DataFrame. Skipping statistics...\n" % val_name)
      
        else:
            arr = np.array(self.df[val_name])
            self.cut_array(arr, ocut, vcut, wcut)
            arr = arr[np.logical_not(self.cut_arr)]
            if frame_range == None:
                frame_range = [0, len(arr)]

            arr = arr[frame_range[0]:frame_range[1]]
            arr = arr[~np.isnan(arr)]
            if val_range != None:
                if val_range[0] != None:
                    arr = arr[arr >= val_range[0]]
                if val_range[1] != None:
                    arr = arr[arr <= val_range[1]]
                
            if val_symm: 
                arr = np.concatenate((arr,-arr))
      
            mean = np.mean(arr)
            stdd = np.std(arr)
            kurt = spstats.kurtosis(arr,fisher=False)
            h, bin_edges = np.histogram(arr, bins=nbins, range=val_range, density=True)
            binc = ( bin_edges[1:] + bin_edges[:-1] ) / 2.
            hist = []
            for i in range(len(binc)):
                hist.append([binc[i], h[i]])
            hist = np.array(hist)
            
            self.store_result(mean, val_name, 'mean', tag)
            self.store_result(stdd, val_name, 'stdd', tag) 
            self.store_result(kurt, val_name, 'kurt', tag)
            self.store_result(hist, val_name, 'hist', tag)


    def result_key(self, val_name, stat_name, tag = None):
        return "%s_%s_%s" % (val_name, stat_name, tag)
        
    def get_result(self, val_name, stat_name, tag = None):
        k = self.result_key(val_name, stat_name, tag)
        return self.result[k]
    
    def store_result(self, result, val_name, stat_name, tag = None):
        k = self.result_key(val_name, stat_name, tag)
        #print("Storing result with key... %s" % k)
        self.result[k] = result
        
    def clear_results(self, tag = None):
        for key in self.result:
            tag_tmp = '_'.join(key.split('_')[2:])
            if tag == tag_tmp:
                del self.result[key]