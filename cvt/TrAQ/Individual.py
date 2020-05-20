import math
import numpy as np
import pandas as pd
import scipy.stats as spstats
import itertools
import sys
cvhome="/disk1/astyanax-mexicanus/cv-tracer"
sys.path.insert(0, cvhome)
import cvt.Analysis.Math as ana_math 


class Individual:

    def __init__(self):

        d = {'t': [], 'x': [], 'y': [], 'theta': [] }    
        self.df = pd.DataFrame( data=d, dtype=np.float64 )
        self.result = {}


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


    def loc(self,index,val):
        return self.df.loc[index,val]

    
    def coordinates(self, index):
        return [ self.loc(index,'x'), 
                 self.loc(index,'y'), 
                 self.loc(index,'theta') ]


    def sort_by_time(self):
        self.df = self.df.sort_values(by='t')


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


    def tstamp_reformat(self,fps):
        mean_dt = (self.df.t.shift(-1) - self.df.t.shift(0)).mean()
        expected_dt = 1./fps
        if math.isclose(mean_dt,expected_dt,rel_tol=0.01/fps):
            factor = expected_dt / mean_dt 
            self.df.t *= factor



    ##############################
    # Calculation functions
    ##############################


    def calculate_dwall(self,tank_radius):
        self.df['dwall'] = self.df.apply(lambda row: tank_radius - 
                           ( np.sqrt(pow(row.x,2) + pow(row.y,2)) ), axis=1)

    def calculate_theta_wall(self):
        r_pos = np.sqrt(self.df.x**2 + self.df.y**2)
        n_hat = np.column_stack((self.df.x/r_pos, self.df.y/r_pos))
        self.df['theta_wall'] = np.arccos(np.cos(self.df.theta)*n_hat[:,0] + np.sin(self.df.theta)*n_hat[:,1])
        print(self.df.theta_wall)


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


    def calculate_tank_crossing(self, R):
        self.df['tc'] = np.sqrt( self.df.x**2 + self.df.y**2 ) < R/np.sqrt(2)


    def set_distance_nn(self, d_nn):
        self.df['d_nn'] = d_nn.tolist()


    def set_diw_miw(self, diw_miw):
        self.df['diw'] = diw_miw[:,0].tolist()
        self.df['miw'] = diw_miw[:,1].tolist()

    def get_diw_miw(self, frame_range = None, 
                    ocut = False, vcut = False, wcut = False):
        
        diw_miw = []
        if 'diw' not in self.df.columns:
            print("\n  'diw' not found in DataFrame. Skipping statistics...\n")
        elif 'miw' not in self.df.columns:
            print("\n  'miw' not found in DataFrame. Skipping statistics...\n")
        else:                
            _diw = np.array(self.df['diw'])
            _miw = np.array(self.df['miw'])
            _diw_miw = np.column_stack((_diw,_miw))
            print(" Shape before cuts: ", _diw_miw.shape)
            _diw_miw = self.apply_cuts(_diw_miw, frame_range = frame_range, 
                                       ocut = ocut, vcut = vcut, wcut = wcut )                
            print(" Shape after cuts: ", _diw_miw.shape)
            diw_miw = _diw_miw.tolist()

        return diw_miw

    def get_dw_thetaw(self, frame_range = None, 
                    ocut = False, vcut = False, wcut = False):
        
        dw_thetaw = []
        if 'dwall' not in self.df.columns:
            print("\n  'dwall' not found in DataFrame. Skipping statistics...\n")
        elif 'theta_wall' not in self.df.columns:
            print("\n  'theta_wall' not found in DataFrame. Skipping statistics...\n")
        else:                
            _dw = np.array(self.df['dwall'])
            _thetaw = np.array(self.df['theta_wall'])
            _dw_thetaw = np.column_stack((_dw,_thetaw))
            print(" Shape before cuts: ", _dw_thetaw.shape)
            _dw_thetaw = self.apply_cuts(_dw_thetaw, frame_range = frame_range, 
                                       ocut = ocut, vcut = vcut, wcut = wcut )                
            print(" Shape after cuts: ", _dw_thetaw.shape)
            dw_thetaw = _dw_thetaw.tolist()

        return dw_thetaw



    def set_dij_mij(self, dij_mij):
        for j in range(len(dij_mij[0])):
            d_key = "d_n%i" % j
            m_key = "m_n%i" % j

            self.df[d_key] = dij_mij[:,j,0].tolist()
            self.df[m_key] = dij_mij[:,j,1].tolist()


    def get_dij_mij(self, n_neighbor, frame_range = None, 
                    ocut = False, vcut = False, wcut = False):
        
        dij_mij = []
        for j in range(n_neighbor):
            d_key = "d_n%i" % j
            m_key = "m_n%i" % j

            if d_key not in self.df.columns:
                print("\n  %s not found in DataFrame. Skipping statistics...\n" % d_key)
            elif m_key not in self.df.columns:
                print("\n  %s not found in DataFrame. Skipping statistics...\n" % m_key)
            else:                
                _dij = np.array(self.df[d_key])
                _mij = np.array(self.df[m_key])
                _dij_mij = np.column_stack((_dij,_mij))
                print(" Shape before cuts: ", _dij_mij.shape)
                _dij_mij = self.apply_cuts(_dij_mij, frame_range = frame_range, 
                                           ocut = ocut, vcut = vcut, wcut = wcut )                
                print(" Shape after cuts: ", _dij_mij.shape)
                dij_mij.extend(_dij_mij.tolist())

        return dij_mij
    
    def nearest_distance_bout(self, d_cut=10, frame_range = None,
                              ocut = False, vcut = False, wcut = False):
        bouts = []
        key = 'd_n0'
        if key not in self.df.columns:
            print("\n  %s not found in DataFrame. Skipping statistics...\n" % key)
        else:
            d_nn = np.array(self.df[key])
            self.cut_array(d_nn, ocut, vcut, wcut)
            d_nn = d_nn[frame_range[0]:frame_range[1]]
            d_nn = d_nn[np.logical_not(self.cut_arr[frame_range[0]:frame_range[1]])]
            bouts = [ sum( 1 for _ in group ) for key, group in itertools.groupby( d_nn <= d_cut ) if key ]
        return bouts


    #########################
    # Cut functions
    #########################
    

    # cut_all() can be called after all other cuts have been calculated    
    def cut_all(self):
        self.df['cut'] = self.df['ocut'] | self.df['vcut'] | self.df['wcut']

    
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

    # cut_array(...) generates a mask based on cuts, providing boolean values
    # that represent frames to be cut. User specifies cuts which must be 
    # completed prior to running this function.
    def cut_array(self, arr, ocut = False, vcut = False, wcut = False ):
        
        self.cut_arr = np.zeros(arr.shape[0],dtype=bool)

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


    def apply_cuts(self, arr, 
                   val_range = None, val_symm = False, frame_range = None, 
                   ocut = False, vcut = False, wcut = False):

        # first combine all cuts in self.cut_arr
        self.cut_array(arr, ocut, vcut, wcut)
        # then select frames from desired range of times
        if frame_range == None:
            frame_range = [0, len(arr)]
        arr = arr[frame_range[0]:frame_range[1]]
        arr = arr[np.logical_not(self.cut_arr[frame_range[0]:frame_range[1]])]
        # remove any NaN values
        not_nan = ~np.isnan(arr)
        if len(not_nan.shape) == 1:
            arr = arr[not_nan]
        elif len(not_nan.shape) == 2:
            s_not_nan = np.zeros(not_nan.shape[0], dtype=bool)
            for i in range(len(s_not_nan)):
                if sum(not_nan[i]) == len(not_nan[i]):
                    s_not_nan[i] = True
                else:
                    s_not_nan[i] = False
            arr = arr[s_not_nan]
            
        #print("   apply_cuts ", arr.shape, 6)
        # exclude values outside desired range of values
        if val_range != None:
            if val_range[0] != None:
                arr = arr[arr >= val_range[0]]
            if val_range[1] != None:
                arr = arr[arr <= val_range[1]]
        # if symmetric (e.g. omega), symmetrize to improve stats like kurtosis
        if val_symm: 
            arr = np.concatenate((arr,-arr))

        return arr
    


    #########################################
    # statistics and results functions
    #########################################


    def result_key(self, val_name, stat_name, tag = None):
        return "%s_%s_%s" % (val_name, stat_name, tag)


    def store_result(self, result, val_name, stat_name, tag = None):
        k = self.result_key(val_name, stat_name, tag)
        #print("Storing result with key... %s" % k)
        self.result[k] = result

        
    def get_result(self, val_name, stat_name, tag = None):
        k = self.result_key(val_name, stat_name, tag)
        return self.result[k]
  
        
    def clear_results(self, tag = None):
        for key in self.result:
            tag_tmp = '_'.join(key.split('_')[2:])
            if tag == tag_tmp:
                del self.result[key]



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
            arr = self.apply_cuts(arr, 
                                  val_range = val_range, val_symm = val_symm, 
                                  frame_range = frame_range, 
                                  ocut = ocut, vcut = vcut, wcut = wcut )

            mean = np.nanmean(arr)
            stdd = np.nanstd(arr)
            kurt = spstats.kurtosis(arr,fisher=False)

            if val_name == 'dwall':
                bin_edges, binc = ana_math.bin_edges_centers_circular(nbins, hrange = val_range)
                h, bin_edges = np.histogram(arr, bins=bin_edges, range=val_range, density=True)

            else:
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
