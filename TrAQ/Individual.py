#!/usr/bin/python3
import math
import numpy as np
import pandas as pd
import scipy.stats as spstats


class Individual:

    def __init__(self):
        d = {'t': [], 'x': [], 'y': [], 'theta': [] }    
        self.df = pd.DataFrame( data=d, dtype=np.float64 )
        self.result = {}

    def n_entries(self):
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
        self.df['dw'] = self.df.apply(lambda row: tank_radius - 
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


    # speed_cut(...) is used to determine frames to be cut for being out of the 
    # valid range of speeds
    def speed_cut(self,speed_min=1.,speed_max=100,n_buffer_frames=2):
        if 'speed_cut' not in self.df.columns:
            if 'speed' not in self.df.columns:
                self.calculate_velocity()
        
            self.df['speed_cut'] = self.df[['speed']].apply(lambda row: 
                        (row[0] < speed_min or row[0] > speed_max ), axis=1)
            for j_frame in range(1,n_buffer_frames+1):
                self.df['speed_cut'] = ( self.df['speed_cut'] \
                            | self.df.speed_cut.shift(-j_frame) \
                            | self.df.speed_cut.shift(j_frame) )

    # distance_cut(...) is used to determine frames to be cut at too short a 
    # range, default of zero. This is a way to remove occlusions in post-
    # processing under the assumption that the tracking module outputs occlusions
    # as two fish with the same center-point
    def distance_cut(self,distance_nn,d_min=0,n_buffer_frames=2):
        if 'd_cut' not in self.df.columns:
            distance_nn = np.array(distance_nn)
            d_cut_bool = np.logical_not(distance_nn > d_min)
            self.df['d_cut'] = d_cut_bool 

        for j_frame in range(1,n_buffer_frames+1):
            self.df['d_cut'] = ( self.df['d_cut'] \
                                | self.df.d_cut.shift(-j_frame) \
                                | self.df.d_cut.shift(j_frame) )
    
    def distance_cut_null(self):
        if 'd_cut' not in self.df.columns:
            self.df['d_cut'] = False

    def total_frames_occlusion(self,framei,framef):
        return float(sum(self.df['d_cut'][framei:framef]))/len(self.df['d_cut'][framei:framef])

    def total_frames_inactive(self,framei,framef):
        return float(sum(self.df['speed_cut'][framei:framef]))/len(self.df['speed_cut'][framei:framef])

    def total_frames_cut(self,framei,framef):
        n_cut = sum(self.df['speed_cut'][framei:framef] | self.df['d_cut'][framei:framef])
        return float(n_cut)/len(self.df['speed_cut'][framei:framef])

    # cut_array(...) generates a mask based on cuts, providing boolean values
    # that represent frames to be cut. User specifies cuts which must be 
    # completed prior to running this function.
    def cut_array(self,arr,vcut=False,ocut=False):
        
        cut_arr = np.zeros_like(arr,dtype=bool)
        if vcut:
            if 'speed_cut' not in self.df.columns:
                print("\n No speed cut column found in DataFrame. Skipping speed cut.\n")
        else:
            vcut_arr = self.df['speed_cut']
            cut_arr = cut_arr | vcut_arr
            
        if ocut:
            if 'd_cut' not in self.df.columns:
                print("\n No occlusion cut column found in DataFrame. Skipping occlusion cut.\n")
            else:
                ocut_arr = self.df['d_cut']
                cut_arr = cut_arr | ocut_arr
                
        return cut_arr 


    # calculate_stats(...) takes the name of a value and calculates its 
    # statistics across valid frames. User can specify a range of values and a 
    # range of time, along with whether or not to use speed and occlusion cuts. 
    # Also has option to make data symmetric about the origin, for use with 
    # angular speed statistics.
    def calculate_stats(self, val_name, valmin = None, valmax = None,
                        framei = 0, framef = None, vcut = False, ocut = False, 
                        nbins = 100, hrange = None, symm = False):

        if val_name not in self.df.columns:
            print("\n  %s not found in DataFrame. Skipping calculation of mean.\n" % val_name)
      
        else:
            arr = np.array(self.df[val_name])
            cut_arr = self.cut_array(arr,vcut,ocut)
            arr = arr[np.logical_not(cut_arr)]
            if framef == None:
                framef = len(arr)
        
            arr = arr[framei:framef]
            if valmin != None:
                arr = arr[arr >= valmin]
            if valmax != None:
                arr = arr[arr <= valmax]
                
            if symm: arr = np.concatenate((arr,-arr))
      
        self.result["mean_%s" % val_name] = np.mean(arr) 
        self.result["stdd_%s" % val_name] = np.std(arr)
        self.result["kurt_%s" % val_name] = spstats.kurtosis(arr,fisher=False)
        self.result["hist_%s" % val_name] = np.histogram(arr, bins=nbins, range=hrange, density=True)
      
        
    def get_result(self,val_name,stat_name):
        return self.result["%s_%s" % (stat_name,val_name)]
  