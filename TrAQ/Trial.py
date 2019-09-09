import numpy as np
import pickle
import argparse
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

        
    def parse_args(self):
        args        = arg_parse()
        n_ind       = args.n_individual
        fps         = args.frames_per_second
        tank_R_cm   = args.tank_diameter/2.
        frame_start = int(args.t_start * fps)
        frame_end   = int(args.t_end   * fps)
        block_size  = args.block_size 
        offset      = args.thresh_offset
        gpu_on      = args.gpu_on

        home_path = os.path.realpath(args.work_dir)
        input_loc = os.path.realpath(args.raw_video)
        data_output_dir = "data/"
        video_output_dir = "video/"
        output_str = "cv"
        input_vidpath, output_vidpath, output_filepath, output_text, codec = Trace.organize_filenames(
                                home_path,input_loc,video_output_dir,data_output_dir,output_str)

        
        
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
    
def arg_parse():
    parser = argparse.ArgumentParser(description="OpenCV2 Fish Tracker")
    parser.add_argument("raw_video", type=str, help="path to raw video")
    parser.add_argument("n_individual", type=int, help="number of individuals in group")
    parser.add_argument("-ts","--t_start", type=float, help="start time, in seconds", default=0)
    parser.add_argument("-te","--t_end", type=float, help="end time, in seconds", default=-1)
    parser.add_argument("-td","--tank_diameter", type=float, help="tank diameter", default=111.)
    parser.add_argument("-ds","--datafile", type=str, help="data file-path, if not standard location") 
    parser.add_argument("-bs","--block_size", type=int, help="contour block size",default=15) 
    parser.add_argument("-th","--thresh_offset", type=int, help="thresholding offset",default=13) 
    parser.add_argument("-sf","--sample_frames", type=int, help="number of frames to locate tank center and radius", default=50)
    parser.add_argument("-fps","--frames_per_second", type=float, help="frames per second in raw video",default=30)
    parser.add_argument("-wd","--work_dir", type=str, help="root working directory if not current",default=os.getcwd())
    parser.add_argument("-vs","--view_scale", type=float, help="factor to scale viewer to fit window", default=1)
    parser.add_argument("-RGB", help="generate movie in RGB (default is greyscale)", action='store_true')
    parser.add_argument("-dir","--directors", help="calculate directors", action='store_true')
    parser.add_argument("-on","--online_viewer", help="view tracking results in real time", action='store_true') 
    parser.add_argument("-gpu","--gpu_on", help="use UMat file handling for OpenCV Transparent API", action='store_true') 
    return parser.parse_args()


def organize_filenames(home_path, input_loc, video_output_dir, data_output_dir, output_str):
    home_path = path_slash(home_path)
    video_output_dir = path_slash(video_output_dir)
    data_output_dir = path_slash(data_output_dir)
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

