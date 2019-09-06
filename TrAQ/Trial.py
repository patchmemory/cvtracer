#!/usr/bin/python
import numpy as np
import pickle
from TrAQ.Group import Group
from TrAQ.Tank import Tank


class Trial:
    
    def __init__(self,fpath):

        self.parse(fpath)
        try:
            self.load()
        except:          
            self.tank   = Tank()
            self.group  = Group(self.n, self.t)        
            self.result = {}
            self.issue  = {}
  
    def save(self):
        f = open(self.filename, 'wb')
        pickle.dump(self.__dict__, f, protocol = 3)
        f.close()

        
    def load(self):
        f = open(self.data, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict) 


    def parse(self,fpath):
        # store name of raw video
        self.video  = fpath
        # store number and type in group trial
        fname = fpath.split('/')[-1]
        self.n = int(fname.split('_n')[-1].split('_')[0])
        self.t = fname.split('_')[0]
        # store the data file
        home_dir = fpath.split('/video/')[0]
        fdir   = fpath.split('/')[-2]
        self.data = "%s/data/%s_%s_cv_kinematics.npy" % (home_dir,fdir,fname.split('.')[0])      
        # store the date of video
        year  = fdir[ :4]
        month = fdir[4:6]
        day   = fdir[6:8]
        self.date = [year,month,day]
        
        
    def summarize_statistics(self, tag = None):
        vals = [ 'dw', 'speed', 'omega' ]
        stat_names = [ 'mean', 'stdd', 'kurt', 'hist' ]

        for val in vals:
            print("  Summary of %s statistics " % (val))
            for stat in stat_names:
                result = self.group_result(val,stat,tag)
                if stat == 'hist':
                    for i in range(len(result)):
                        print( "    %i \t%4.2e \t%4.2e " % 
                                              (i, result[i][0], result[i][1]) )
                else:
                    print( "    %s \t%4.2e \t%4.2e " % (stat, result[0], result[1]) )
                print("\n")
        print("\n")
        
    
    def get_group_result(self, val_name, stat_name, tag = None):
        return self.group.get_result(val_name, stat_name, tag)
    
    def get_individual_result(self, i_fish, val_name, stat_name, tag = None):
        return self.group.fish[i_fish].get_result(val_name, stat_name, tag)
    
    def get_individual_results(self, i_fish, val_name, stat_name, result, tag = None):
        results = []
        for i in range(self.n):
            results.append(self.group.fish[i_fish].get_result(val_name, stat_name, tag))
        return np.array(results)
    
    