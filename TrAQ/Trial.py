import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from TrAQ.Group import Group
from TrAQ.Tank import Tank


class Trial:
    
    def __init__(self, fvideo, n = 0, t = None, date = None, 
                 fps = 30, tank_radius = 111./2, t_start = 0, t_end = -1):

        self.result = {}
        self.issue  = {}

        self.fname_std = 'trial.pik'
        self.fvideo_raw_std = 'raw.mp4'
        self.fvideo_out_std = 'traced.mp4'
                    
        self.fvideo_raw = os.path.abspath(fvideo)
        if self.fvideo_raw.split('/')[-1] != self.fvideo_raw_std:
            print("  Reorganizing directory")
            self.reorganize_files()
        else:
            print("  Directory organized properly.")
            
        self.parse_fname(date)
        if not self.load():
            sys.stdout.write("\n        Generating new Trial object.\n")
            self.n           = int(n)
            self.t           = t
            self.fps         = fps
            self.tank        = Tank(self.fvideo_raw, tank_radius)
            self.tank.locate()
            self.tank.r_cm   = tank_radius
            self.frame_start = t_start * fps
            self.frame_end   = t_end   * fps
            self.group       = Group(self.n, self.t)        
            self.date        = date

        self.save()

    def parse_fname(self, date = None):
        fname_tmp = self.fvideo_raw.split('/')[:-1]
        fname_tmp.append(self.fname_std)
        self.fname = '/'.join(fname_tmp)
        
        # store the date of video
        if date != None:
            year  = int(date[0:4])
            month = int(date[4:6])
            day   = int(date[6:8])
        else:
            fdir  = self.fvideo_raw.split('/')[-2]
            year  = int(fdir[ :4])
            month = int(fdir[4:6])
            day   = int(fdir[6:8])
        
        self.date = [ year, month, day ]
        self.fdir = '/'.join(self.fvideo_raw.split('/')[:-1])
    
    def print_info(self):
        date_str = "%4i %2i %2i" % (self.date[0], self.date[1], self.date[2])
        print("\n  %s, %2i %s" % ( date_str, self.t, self.n ) )
        print("     video: %s" % ( self.fvideo_raw ) )
        print("     trial: %s" % ( self.fname ) )
        if self.issue:
            print("       Known issues: " )
            for key in self.issue:
                print("           %s" % (key, self.issue[key]))
    
    def save(self, fname = None):
        if fname != None:
            self.fname = fname
        f = open(self.fname, 'wb')
        pickle.dump(self.__dict__, f, protocol = 3)
        sys.stdout.write("\n        Trial object saved as %s \n" % self.fname)
        sys.stdout.flush()
        f.close()

    def load(self, fname = None):
        if fname != None:
            self.fname = fname
        try:
            f = open(self.fname, 'rb')
            tmp_dict = pickle.load(f)
            f.close()
            self.__dict__.update(tmp_dict) 
            sys.stdout.write("\n        Trial object loaded from %s \n" % self.fname)
            sys.stdout.flush()
            return True
        except:
            sys.stdout.write("\n        Unable to load Trial from %s \n" % self.fname)
            sys.stdout.flush()
            return False

    def reorganize_files(self):
        vfile = self.fvideo_raw.split('/')[-1]
        vodir = self.fvideo_raw.split('/')[-2]
        # store path up until current subdirectory
        vpath_remain = '/'.join(self.fvideo_raw.split('/')[:-2])
        # strip redundant fish type from filename
        vdir_ext = '_'.join(vfile.split('.')[0].split('_')[1:])
        # test to make sure there is indeed a file to move
        if os.path.isfile(self.fvideo_raw):
            #print(os.path.getmtime(self.fvideo_raw))
            new_dir = "%s_%s" % (vodir,vdir_ext)
            print(new_dir)
            if not os.path.isdir(new_dir):
                print("  Making new directory \n    %s" % new_dir)
                os.mkdir(new_dir)
                f_new = "%s/%s/%s" % (vpath_remain,new_dir,self.fvideo_raw_std)                
                print("  Moving\n    %s" % self.fvideo_raw)
                print("  to\n    %s" % f_new)
                os.rename( os.path.abspath(self.fvideo_raw),
                           os.path.abspath(f_new) )
                self.fvideo_raw = f_new
            else:
                f_traced = "%s/%s/%s" % (vpath_remain,new_dir,self.fvideo_out_std)
                if os.path.isfile(f_traced):
                    exit()


    ######################################################
    # some functions for storing and retrieving results
    #####################################################
        
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
    
    def get_individual_results(self, val_name, stat_name, tag = None):
        results = []
        for i in range(self.n):
            results.append(self.get_individual_result(i, val_name, stat_name, tag))
        return np.array(results)
        
    def get_individual_result(self, i_fish, val_name, stat_name, tag = None):
        return self.group.fish[i_fish].get_result(val_name, stat_name, tag)
    
    
    def convert_pixels_to_cm(self):
        sys.stdout.write("\n       Converting pixels to (x,y) space in (cm,cm).\n")
        sys.stdout.flush()
        self.group.convert_pixels(self.tank.row_c, self.tank.col_c, self.tank.r, self.tank.r_cm)
        sys.stdout.write("\n       %s converted according to tank size and location" % self.fname )
        sys.stdout.write("\n       as specified in %s" % self.tank.fname )
        sys.stdout.flush()

#    def transform_for_lens(self):
#        sys.stdout.write("\n       Transforming to account for wide-angle lens.\n")
#        sys.stdout.flush()
#        self.group.lens_trasnformation(A,B,C)

    def calculate_kinematics(self):           
        sys.stdout.write("\n       Calculating kinematics...\n")
        sys.stdout.flush()
        self.group.calculate_dwall(self.tank.r_cm)
        self.group.calculate_velocity(self.fps)
        self.group.calculate_acceleration(self.fps)
        self.group.calculate_director(self.fps)
        self.group.calculate_angular_velocity(self.fps)
        self.group.calculate_angular_acceleration(self.fps)
        self.group.calculate_local_acceleration(self.fps)
    
        self.save()          
        sys.stdout.write("\n")
        sys.stdout.write("       ... kinematics calculated for Trial and saved in\n")
        sys.stdout.write("             %s \n" % self.fname)
        sys.stdout.flush()

    def calculate_pairwise(self):
        sys.stdout.write("\n")
        sys.stdout.write("       Calculating neighbor distance and alignment across group... \n")
        self.group.calculate_distance_alignment()
        sys.stdout.write("\n")
        sys.stdout.write("       ... done \n")
    
    def evaluate_cuts(self, frame_range = None, n_buffer_frames = 2, 
                      ocut = None, vcut = None, wcut = None ):
        
        if ocut != None:
            self.group.cut_occlusion(ocut, n_buffer_frames)
        if vcut != None:
            self.group.cut_speed(self.fps, vcut, n_buffer_frames)
        if wcut != None:
            self.group.cut_omega(self.fps, wcut, n_buffer_frames)
        
        self.group.cut_combine()
        
        if frame_range == None:
            frame_range = [ int(self.frame_start), int(self.frame_end) ]
            
        mean, err = self.group.cut_stats(frame_range[0], frame_range[1])
        self.cuts_stats = { 'mean': mean, 'err': err }


    def calculate_statistics(self, 
                             val_name  = [ 'dwall', 'speed', 'omega' ], 
                             val_range = [    None,    None,    None ], 
                             val_symm  = [   False,   False,    True ],
                             val_bins  = [     100,     100,     100 ],
                             frame_range = None, 
                             ocut = False, vcut = False, wcut = False, tag = None):
        
        for i in range(len(val_name)):
            self.group.calculate_stats(val_name[i], val_range[i], val_symm[i],
                        frame_range = frame_range, nbins = val_bins[i],
                        ocut = ocut, vcut = vcut, wcut = wcut )
            self.plot_hist(val_name[i], tag)
            self.plot_hist_each(val_name[i], tag)
            
    def plot_hist(self, val_name, tag = None, save = True):
        h = self.get_group_result(val_name, 'hist', tag)
        plt.fill_between(h[:,0], h[:,1] - h[:,2], h[:,1] + h[:,2], color = 'blue')
        plt.plot(h[:,0], h[:,1], color='red', linewidth=0.5 )
        if save:
            fig_name = "%s/%s_hist_%s.png" % (self.fdir, val_name, tag)
            plt.savefig(fig_name)
        else:
            plt.show()
        plt.clf()
        
    def plot_hist_each(self, val_name, tag = None, save = True):
        hist = self.get_individual_results(val_name, 'hist', tag)
        i = 0
        for h in hist: 
            i += 1
            plt.plot(h[:,0], h[:,1], linewidth=1, label=i)
        plt.legend()
        plt.tight_layout()
        if save:
            fig_name = "%s/%s_hist_each_%s.png" % (self.fdir, val_name, tag)
            plt.savefig(fig_name)
        else:
            plt.show()
        plt.clf()