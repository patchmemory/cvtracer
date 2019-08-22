#!/usr/bin/python3
import sys
import numpy as np
import pandas as pd

class KinematicDataFrame:

  def __init__(self):
    d = {'t': [], 'x': [], 'y': [], 'theta': [] }
    self.df = pd.DataFrame(data=d, dtype=np.float64)

  def n_entries(self):
    return len(self.df.index)
    
  def add_entry(self,t,x,y,theta):
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

  def calculate_dwall(self,diam):
    self.df['dw'] = self.df.apply(lambda row: diam/2 - ( np.sqrt(pow(row.x,2)) + np.sqrt(pow(row.y,2)) ), axis=1)

  def calculate_velocity(self,fps):
    self.df['vx'] = ( self.df.x.shift(-1) - self.df.x.shift(1) ) / 2 * fps
    self.df['vy'] = ( self.df.y.shift(-1) - self.df.y.shift(1) ) / 2 * fps
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
    self.df.drop(columns=['omega1','omega2','omega3'])

  def calculate_angular_acceleration(self,fps):
    self.df['alpha1'] = self.df.theta.shift(-1) - self.df.theta.shift(1) 
    self.df['alpha2'] = self.df.theta.shift(-1) - self.df.theta.shift(1) + 2*np.pi 
    self.df['alpha3'] = self.df.theta.shift(-1) - self.df.theta.shift(1) - 2*np.pi
    self.df['alpha'] = self.df[['alpha1', 'alpha2', 'alpha3']].apply(lambda row: min(row[0],row[1],row[2],key=abs), axis=1)
    self.df['alpha'] *= (fps/2.)
    self.df.drop(columns=['alpha1','alpha2','alpha3'])

  def calculate_local_acceleration(self,fps):
    if 'ax' not in self.df.columns or 'ay' not in self.df.columns:
      self.calculate_acceleration()
    self.df['af'] =   np.cos(self.df.etheta)*self.df.ax + np.sin(self.df.etheta)*self.df.ay
    self.df['al'] = - np.sin(self.df.etheta)*self.df.ax + np.cos(self.df.etheta)*self.df.ay

  def convert_pixels(self, xcom, ycom, L_pix, L_m):
    A_convert = L_m / L_pix
    self.df['x'] -= xcom
    self.df['y'] -= ycom
    self.df['x'] *= A_convert
    self.df['y'] *= A_convert


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
