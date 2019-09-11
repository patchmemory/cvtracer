import os
import sys
import numpy as np
import pickle
from TrAQ.Group import Group
from TrAQ.Tank import Tank


class Trial:
    
    def __init__(self, fvideo, n = None, t = None, date = ["YYYY","MM","DD"], 
                 fps = 30, tank_radius = 111./2, t_start = 0, t_end = -1):

        self.result = {}
        self.issue  = {}

        self.fname_std = 'trial.pik'
        self.fvideo_raw_std = 'raw.mp4'
                    
        self.fvideo_raw = os.path.abspath(fvideo)
        if self.fvideo_raw.split('/')[-1] != self.fvideo_raw_std:
            print("  Reorganizing directory")
            self.reorganize_files()
        else:
            print("  Directory organized properly.")
            
        self.parse_fname()
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


    def parse_fname(self):
        fname_tmp = self.fvideo_raw.split('/')[:-1]
        fname_tmp.append(self.fname_std)
        self.fname = '/'.join(fname_tmp)
        
        # store the date of video
        fdir   = self.fvideo_raw.split('/')[-2]
        year  = int(fdir[ :4])
        month = int(fdir[4:6])
        day   = int(fdir[6:8])
        self.date = [ year, month, day ]
    
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
    
    def get_individual_result(self, i_fish, val_name, stat_name, tag = None):
        return self.group.fish[i_fish].get_result(val_name, stat_name, tag)
    
    def get_individual_results(self, i_fish, val_name, stat_name, result, tag = None):
        results = []
        for i in range(self.n):
            results.append(self.group.fish[i_fish].get_result(val_name, stat_name, tag))
        return np.array(results)
    



def organize_filenames(home_path, input_loc, video_output_dir, data_output_dir, output_str):
    home_path = os.path.abspath(home_path)
    video_output_dir = os.path.abspath(video_output_dir)
    data_output_dir = os.path.abspath(data_output_dir)
    if home_path in input_loc:
        input_loc = input_loc.replace(home_path, '')
    
    video = input_loc.split('/')[-1]
    video_dir = input_loc.split('/')[-2]
    video_str = '.'.join(video.split('.')[:-1])
    video_ext = '.' + video.split('.')[-1]
    input_vidpath   = home_path + input_loc
    output_vidpath  = home_path + video_output_dir + video_dir + "_" + video_str + '_' + output_str + video_ext 
    output_filepath = home_path + data_output_dir + video_dir + "_" + video_str + '_' + output_str + "_kinematics.npy" 
    output_text     = home_path + data_output_dir + video_dir + "_" + video_str + '_' + output_str + ".dat" 
    if ( video_ext == ".avi" ):
        codec = 'DIVX' 
    else:
        codec = 'mp4v'
    # try other codecs if the default doesn't work (e.g. 'DIVX', 'avc1', 'XVID') 
    return input_vidpath, output_vidpath, output_filepath, output_text, codec


def calculate_kinematics():
        q = np.load(output_filepath)
        sys.stdout.write("\n")
        sys.stdout.write("       Converting pixels to (x,y) space in (cm,cm).\n")
        sys.stdout.flush()
        for i in range(len(q)):
            q[i].convert_pixels(tank.row_c,tank.col_c,tank.r,tank_radius_cm)
            q[i].tstamp_reformat(fps)
        sys.stdout.write("\n")
        sys.stdout.write("       %s converted according to tank size and location.\n" % output_filepath)
        sys.stdout.flush()
        
        sys.stdout.write("\n")
        sys.stdout.write("       Calculating kinematics...\n")
        sys.stdout.flush()
        for i in range(len(q)):
            print("         Fish %2i" % (i+1)) 
            q[i].calculate_dwall(tank_radius_cm)
            q[i].calculate_velocity(fps)
            q[i].calculate_acceleration(fps)
            q[i].calculate_director(fps)
            q[i].calculate_angular_velocity(fps)
            q[i].calculate_angular_acceleration(fps)
            q[i].calculate_local_acceleration(fps)
    
        np.save(output_filepath,q)
      
        sys.stdout.write("\n")
        sys.stdout.write("       %s kinematic quantities have been calculated.\n" % output_filepath)
        sys.stdout.flush()
