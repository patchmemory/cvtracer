import os
import sys
import pickle
import numpy as np
import copy
import matplotlib.cm as mpl_cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from cvt.TrAQ.Group import Group
from cvt.TrAQ.Tank import Tank


class Trial:
    
    def __init__(self, fvideo = None, n = 0, t = None, date = None, 
                 fps = 30, tank_radius = 111./2, t_start = 0, t_end = -1):

        self.result = {}
        self.issue  = {}
        
        self.cut_stats = {}

        self.fname_std = 'trial.pik'
        self.fvideo_raw_std = 'raw.mp4'
        self.fvideo_out_std = 'traced.mp4'
    
        if fvideo != None:
            self.setup(fvideo, n, t, date, fps, tank_radius, t_start, t_end)
    
    def setup(self, fvideo = None, n = 0, t = None, date = None, 
              fps = 30, tank_radius = 111./2, t_start = 0, t_end = -1):
        self.fvideo_raw = os.path.abspath(fvideo)
        if self.fvideo_raw.split('/')[-1] != self.fvideo_raw_std:
            print("        Reorganizing directory...")
            self.reorganize_files()
        else:
            print("        Directory organized properly.")
            
        self.parse_fname(date)
        if not self.load():
            sys.stdout.write("\n        Generating new Trial object.\n")
            self.tank        = Tank(self.fvideo_raw, tank_radius)
            self.tank.locate()
            self.tank.r_cm   = tank_radius
            self.group       = Group(int(n), t) 
            self.fps         = fps
            self.frame_start = t_start * fps
            self.frame_end   = t_end   * fps


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

        self._date = [ year, month, day ]
        self.fdir = '/'.join(self.fvideo_raw.split('/')[:-1])
    
    def print_info(self):
        date_str = "%02i/%02i/%4i" % ( self._date[1], self._date[2], self._date[0] )
        print("\n  %s, %2i %s" % ( date_str, self.group.n, self.group.t ) )
        print("       trial: %s" % ( self.fname ) )
        print("       input: %s" % ( self.fvideo_raw ) )
        if self.issue:
            print("       Known issues: " )
            for key in self.issue:
                print("           %s: %s" % (key, self.issue[key]))
    
    def save(self, fname = None):
        try:
            if fname != None:
                self.fname = fname
            f = open(self.fname, 'wb')
            pickle.dump(self.__dict__, f, protocol = 3)
            sys.stdout.write("\n        Trial object saved as %s \n" % self.fname)
            sys.stdout.flush()
            f.close()
            return True
        except:
            return False

    def load(self, fname = None):
        if fname != None:
            self.fname = fname
        try:
            f = open(self.fname, 'rb')
            tmp_dict = pickle.load(f)
            f.close()
            self.__dict__.update(tmp_dict) 
            sys.stdout.write("\n        Trial loaded from %s \n" % self.fname)
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



    #################################################
    # experimental transformation functions
    #################################################

    
    def convert_pixels_to_cm(self):
        sys.stdout.write("\n       Converting pixels to (x,y) space in (cm,cm).\n")
        sys.stdout.flush()
        self.group.convert_pixels(self.tank.row_c, self.tank.col_c, self.tank.r, self.tank.r_cm)
        sys.stdout.write("\n       %s converted according to tank size and location" % self.fname )
        sys.stdout.write("\n       as specified in %s" % self.tank.fname )
        sys.stdout.flush()


    def transform_for_lens(self):
        sys.stdout.write("\n       Transforming to account for wide-angle lens.\n")
        sys.stdout.flush()
        #self.group.lens_transformation(A,B,C)


    def generate_tag(self, frame_range = None, n_buffer_frames = 2, 
                     ocut = None, vcut = None, wcut = None ):

        if frame_range == None:
            frame_range = [ int(self.frame_start), int(self.frame_end) ]           
        
        tag = [ "t%02ito%02i" % ( int(frame_range[0]/self.fps/60.), 
                                  int(frame_range[1]/self.fps/60.)  ) ]
        if ocut != None:
            tag.append("o%03.1f" % ocut)
        if vcut != None:
            tag.append("v%05.1fto%05.1f" % (vcut[0],vcut[1]))
        if wcut != None:
            tag.append("w%05.1fto%05.1f" % (wcut[0],wcut[1]))
        if ocut != None or vcut != None or wcut != None:
            tag.append("nbf%i" % n_buffer_frames)
        
        tag = '_'.join(tag)

        return tag
    
    
    def parse_tag_range(self, tag = "", tag_key = ""):
        split_tag = tag.split('_')
        val = None
        for entry in split_tag:
            if len(tag_key) <= len(entry) and tag_key == entry[0:len(tag_key)]:
                entry = entry[1:]
                if 'to' in entry:
                    val = np.array(entry.split('to'))
                    val = [ float(v) for v in val ]
                else:
                    val = float(entry)
        return val

    
    def read_tag(self, tag):
        
        tag_key = { 'time_range': 't',
                    'ocut': 'o', 
                    'vcut': 'v', 
                    'wcut': 'w', 
                    'n_buffer_frames': 'nbf' }
        
        tag_val = {}
        for key in tag_key:
            tag_val[key] = self.parse_tag_range(tag, tag_key[key])
        
        tag_val['frame_range'] = [0,0]
        tag_val['frame_range'][0] = tag_val['time_range'][0]*self.fps*60
        tag_val['frame_range'][1] = tag_val['time_range'][1]*self.fps*60

        return tag_val
    
    
    def evaluate_cuts(self, frame_range = None, n_buffer_frames = 2, 
                      ocut = None, vcut = None, wcut = None ):
        
        if ocut != None:
            self.group.calculate_distance_alignment()
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
        tag = self.generate_tag(frame_range, n_buffer_frames, ocut, vcut, wcut)
        self.plot_valid(frame_range = frame_range, tag = tag)
        return tag


    ######################################################
    # some functions for storing and retrieving results
    #####################################################


    def get_group_result(self, val_name, stat_name, tag = None):
        return self.group.get_result(val_name, stat_name, tag)

    
    def get_individual_results(self, val_name, stat_name, tag = None):
        results = []
        for i in range(self.group.n):
            results.append(self.get_individual_result(i, val_name, stat_name, tag))
        return np.array(results)

        
    def get_individual_result(self, i_fish, val_name, stat_name, tag = None):
        return self.group.fish[i_fish].get_result(val_name, stat_name, tag)

    
    def clear_results(self, tag = None):
        self.group.clear_results(tag)
        for i in range(self.group.n):
            self.group.fish.clear_results(tag)

        
    def summarize_statistics(self, tag = None):
        vals = [ 'dw', 'speed', 'omega' ]
        stat_names = [ 'mean', 'stdd', 'kurt', 'hist' ]

        for val in vals:
            print("  Summary of %s statistics " % (val))
            for stat in stat_names:
                result = self.get_group_result(val,stat,tag)
                if stat == 'hist':
                    for i in range(len(result)):
                        print( "    %i \t%4.2e \t%4.2e " % 
                                              (i, result[i][0], result[i][1]) )
                else:
                    print( "    %s \t%4.2e \t%4.2e " % (stat, result[0], result[1]) )
                print("\n")
        print("\n")      



    #################################################
    # calculation functions
    #################################################


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


    def calculate_tank_crossing(self):           
        sys.stdout.write("\n       Calculating tank crossings...\n")
        sys.stdout.flush()
        self.group.calculate_tank_crossing(self.tank.r_cm)
        self.save()          
        sys.stdout.write("\n")
        sys.stdout.write("       ... tank crossings calculated for Trial and saved in\n")
        sys.stdout.write("             %s \n" % self.fname)
        sys.stdout.flush()


    def calculate_wall_distance_orientation(self):
        sys.stdout.write("\n")
        sys.stdout.write("       Calculating wall distance and alignment across group... \n")
        self.group.calculate_wall_distance_orientation()
        sys.stdout.write("\n")
        sys.stdout.write("       ... done \n")


    def gather_wall_distance_orientation(self, frame_range = None, 
                           ocut = False, vcut = False, wcut = False,
                           tag = None):
        sys.stdout.write("       Collecting wall info according to cuts... \n")
        self.group.collect_wall_distance_orientation(frame_range = frame_range, 
                                           ocut = ocut, vcut = vcut, wcut = wcut)
        sys.stdout.write("\n")
        sys.stdout.write("       ... done \n")
        self.plot_wall_distance_orientation(tag)


    def calculate_wall_distance_alignment(self):
        sys.stdout.write("\n")
        sys.stdout.write("       Calculating wall distance and orientation across group... \n")
        self.group.calculate_wall_distance_alignment()
        sys.stdout.write("\n")
        sys.stdout.write("       ... done \n")


    def gather_wall_distance_alignment(self, frame_range = None, 
                           ocut = False, vcut = False, wcut = False,
                           tag = None):
        sys.stdout.write("       Collecting wall info according to cuts... \n")
        self.group.collect_wall_distance_alignment(frame_range = frame_range, 
                                           ocut = ocut, vcut = vcut, wcut = wcut)
        sys.stdout.write("\n")
        sys.stdout.write("       ... done \n")
        self.plot_wall_distance_alignment(tag)


    def calculate_pairwise(self):
        sys.stdout.write("\n")
        sys.stdout.write("       Calculating pair distance and alignment across group... \n")
        self.group.calculate_distance_alignment()
        sys.stdout.write("\n")
        sys.stdout.write("       ... done \n")

        
    def gather_pairwise(self, frame_range = None, 
                           ocut = False, vcut = False, wcut = False,
                           tag = None):
        sys.stdout.write("       Collecting pair info according to cuts... \n")
        self.group.collect_distance_alignment(frame_range = frame_range, 
                                              ocut = ocut, vcut = vcut, wcut = wcut)
        sys.stdout.write("\n")
        sys.stdout.write("       ... done \n")
        self.plot_distance_alignment(tag)


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
                        ocut = ocut, vcut = vcut, wcut = wcut, tag = tag )
            self.plot_hist(val_name[i], tag)
            self.plot_hist_each(val_name[i], tag)
        
    
    
    #################################################
    # plot functions
    #################################################


    def plot_hist(self, val_name, tag = None, save = True):
        h = self.get_group_result(val_name, 'hist', tag)
        plt.fill_between(h[:,0], h[:,1] - h[:,2], h[:,1] + h[:,2], color = 'blue', label='cross-fish error')
        plt.plot(h[:,0], h[:,1], color='red', linewidth=0.5, label='cross-fish mean')
        mean = self.get_group_result(val_name, 'mean', tag)
        plt.axvline(x = mean[0], color = 'green', linewidth = 3, linestyle = '-', label = 'distribution mean')
        plt.axvline(x = mean[0] - mean[1], color = 'green', linewidth = 1, linestyle = '--', label = 'distribution error')
        plt.axvline(x = mean[0] + mean[1], color = 'green', linewidth = 1, linestyle = '--')
        plt.legend()
        plt.tight_layout()
        if save:
            fig_name = "%s/%s_hist_%s.png" % (self.fdir, val_name, tag)
            plt.savefig(fig_name)
        else:
            plt.show()
        plt.clf()
        
        
    def plot_hist_each(self, val_name, tag = None, save = True):
        hist = self.get_individual_results(val_name, 'hist', tag)
        color_set = plt.rcParams['axes.prop_cycle'].by_key()['color']
        i = 0
        for h in hist: 
            mean = self.get_individual_result(i, val_name, 'mean', tag)
            i += 1
            c = color_set[i%len(color_set)]
            plt.axvline(x = mean, color = c, linewidth = 1)
            plt.plot(h[:,0], h[:,1], color = c, linewidth = 1, label=i)
        plt.xlabel(val_name)
        plt.ylabel("normalized count")
        plt.legend()
        plt.tight_layout()
        if save:
            fig_name = "%s/%s_hist_each_%s.png" % (self.fdir, val_name, tag)
            plt.savefig(fig_name)
        else:
            plt.show()
        plt.clf()
  

    def plot_wall_distance_orientation(self, tag = None, save = True):
        my_cmap = copy.copy(mpl_cm.get_cmap('viridis'))
        my_cmap.set_bad(my_cmap.colors[0])
        
        plt.ylabel(r"Orientation ($\theta_{i,w}$)")
        plt.xlabel("Distance (cm)")
        plt.hist2d( self.group.dw_thetaw[:,0], self.group.dw_thetaw[:,1],
                    bins=100, range=[[0,self.tank.r_cm],[-np.pi,np.pi]], 
                    norm = colors.LogNorm(), cmap = my_cmap )
        plt.colorbar()
        plt.tight_layout()
        if save:
            fig_name = "%s/dw_thetaw_%s.png" % (self.fdir, tag)
            plt.savefig(fig_name)
        else:
            plt.show()
        plt.clf()      


    def plot_wall_distance_alignment(self, tag = None, save = True):
        my_cmap = copy.copy(mpl_cm.get_cmap('viridis'))
        my_cmap.set_bad(my_cmap.colors[0])
        
        plt.ylabel(r"Alignment ($\cos\theta_{ij}$)")
        plt.xlabel("Distance (cm)")
        plt.hist2d( self.group.diw_miw[:,0], self.group.diw_miw[:,1],
                    bins=100, range=[[0,self.tank.r_cm],[-1,1]], 
                    norm = colors.LogNorm(), cmap = my_cmap )
        plt.colorbar()
        plt.tight_layout()
        if save:
            fig_name = "%s/diw_miw_%s.png" % (self.fdir, tag)
            plt.savefig(fig_name)
        else:
            plt.show()
        plt.clf()      


    def plot_distance_alignment(self, tag = None, save = True):
        my_cmap = copy.copy(mpl_cm.get_cmap('viridis'))
        my_cmap.set_bad(my_cmap.colors[0])
        
        plt.ylabel(r"Alignment ($\cos\theta_{ij}$)")
        plt.xlabel("Distance (cm)")
        plt.hist2d(self.group.dij_mij[:,0], self.group.dij_mij[:,1],
                   bins=100, range=[[0,self.tank.r_cm],[-1,1]], 
                   norm = colors.LogNorm(), cmap = my_cmap)
        plt.colorbar()
        plt.tight_layout()
        if save:
            fig_name = "%s/dij_mij_%s.png" % (self.fdir, tag)
            plt.savefig(fig_name)
        else:
            plt.show()
        plt.clf()      
       

    def plot_valid(self, frame_range = None, tag = None, save = True):
        cuts = [ 'ocut', 'vcut', 'wcut', 'cut']
        valid = {}
        for cut in cuts:
            valid[cut] = self.group.valid_frame_fraction(frame_range, cut_name = cut)
            if np.mean(valid[cut]) < 0.5:
                self.issue[cut] = "Less than half of frames are valid after %s." % cut
        index = np.arange(len(cuts))
        for i in range(self.group.n):
            valid[i] = []
            for cut in cuts:
                valid[i].append(valid[cut][i])
            valid[i] = np.array(valid[i])
   
        gutter = 0.1
        bar_width = ( 1 - gutter ) / self.group.n
        opacity = 0.8
        for i in range(self.group.n):
            plt.bar(index + i*bar_width + 0.5*gutter, valid[i], bar_width, alpha = opacity, label=i)

        plt.xlabel('fraction valid after cuts')
        plt.xlabel('cut')
        plt.xticks(index + 0.5 - 0.5*gutter, cuts)
        plt.ylim([0,1])
        plt.legend()
        plt.tight_layout()        
        if save:
            fig_name = "%s/valid_%s.png" % (self.fdir, tag)
            plt.savefig(fig_name)
        else:
            plt.show()
        plt.clf()