#!/usr/bin/python3
import sys, math
import numpy as np
import pandas as pd
import scipy.stats as spstats

class KinematicDataFrame:

  def __init__(self):
    d = {'t': [], 'x': [], 'y': [], 'theta': [] }
    self.stats = {}
    self.df = pd.DataFrame(data=d, dtype=np.float64)

  def n_entries(self):
    return len(self.df.index)
    
  def add_entry(self,t,x,y,theta):
    #df_add = pd.DataFrame({'t':[t], 'row':[x],'col':[y],'theta_pix':[theta]})
    df_add = pd.DataFrame({'t':[t], 'x':[x],'y':[y],'theta':[theta]})
    self.df = self.df.append(df_add, ignore_index=True)

  def sort_by_time(self):
    self.df = self.df.sort_values(by='t')

  def print(self):
    print(self.df)

  def print_frame(self,index):
    print(self.df.loc[index])

  def loc(self,index,val):
    return self.df.loc[index,val]

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
    

  def calculate_stats(self, val_name, valmin = None, valmax = None,
                                     framei = 0, framef = None,
                                     vcut = False, ocut = False):

    try: 
      self.stats
    except AttributeError:
      self.stats = {}
      
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

      self.stats["mean_%s" % val_name] = np.mean(arr) 
      self.stats["std_%s" % val_name]  = np.std(arr)
      return self.stats["mean_%s" % val_name], self.stats["std_%s" % val_name]


  def calculate_stats_symm(self, val_name, valmin = None, valmax = None,
                                     framei = 0, framef = None,
                                     vcut = False, ocut = False):
    try: 
      self.stats
    except AttributeError:
      self.stats = {}

    if val_name not in self.df.columns:
      print("\n  %s not found in DataFrame. Skipping calculation of symmetric stats.\n" % val_name)
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
      arr = np.concatenate((arr,-arr))
      self.stats["mean_%s" % val_name] = np.mean(arr) 
      self.stats["std_%s" % val_name]  = np.std(arr)
      self.stats["kurt_%s" % val_name] = spstats.kurtosis(arr,fisher=False)
      return self.stats["mean_%s" % val_name], self.stats["std_%s" % val_name], self.stats["kurt_%s" % val_name]

  def calculate_dwall(self,tank_radius):
    self.df['dw'] = self.df.apply(lambda row: tank_radius - ( np.sqrt(pow(row.x,2) + pow(row.y,2)) ), axis=1)

  def calculate_velocity(self,fps):
    if 'vx' not in self.df.columns or 'vy' not in self.df.columns:
      self.df['vx'] = ( self.df.x.shift(-1) - self.df.x.shift(1) ) / 2 * fps
      self.df['vy'] = ( self.df.y.shift(-1) - self.df.y.shift(1) ) / 2 * fps
    if 'speed' not in self.df.columns:
      self.df['speed'] = np.sqrt(pow(self.df.vx,2) + pow(self.df.vy,2)) 

  def calculate_mean(self,val_name,vmin=None,vmax=None):
    arr = np.array(self.df[val_name])
    mean[val_name] = np.mean(arr[(arr > vmin) & (arr <= vmax)])

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

  def speed_cut(self,speed_cut=1.,n_buffer_frames=2):
    if 'speed_cut' not in self.df.columns:
      if 'speed' not in self.df.columns:
        self.calculate_velocity()
        
      self.df['speed_cut'] = self.df[['speed']].apply(lambda row: (row[0] < speed_cut or row[0] > 100.), axis=1)
      for j_frame in range(1,n_buffer_frames+1):
        self.df['speed_cut'] = ( self.df['speed_cut'] \
                            | self.df.speed_cut.shift(-j_frame) \
                            | self.df.speed_cut.shift(j_frame) )


  def distance_cut(self,distance_nn,d_min=0,n_buffer_frames=2):

    if 'd_cut' not in self.df.columns:
      distance_nn = np.array(distance_nn)
      #print("entries in distance_nn",len(distance_nn))
      #print("dim of distance_nn entry",len(distance_nn[0]))
      #print("entries in DF", self.n_entries())
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


def main():
  nvals = int(sys.argv[1])
  diam = float(sys.argv[2])
  q = KinematicDataFrame()
  fps = 30
  for i in range(nvals):
    if i == 0:
      x_tmp = y_tmp = theta_tmp = 0
    else:
      x_tmp     = q.loc(i-1,'x') + np.random.normal(0,0.1)
      y_tmp     = q.loc(i-1,'y') + np.random.normal(0,0.1)
      theta_tmp = np.fmod(q.loc(i-1,'theta') + np.random.normal(0,0.1) + np.pi ,2*np.pi) - np.pi
      q.add_entry(i*dt,x_tmp,y_tmp,theta_tmp)
  
  dt = 1/fps
  
  q.calculate_dwall(diam)
  q.print_frame(nvals-3)
  
  q.calculate_velocity(fps)
  q.calculate_acceleration(fps)
  q.print_frame(nvals-3)
  
  q.calculate_director(fps)
  q.print_frame(nvals-3)
  q.calculate_local_acceleration(fps)
  q.print_frame(nvals-3)

#main()
