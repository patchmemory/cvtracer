import sys
from TrAQ.Trial import Trial
from TrAQ.VideoCV import VideoCV


class Tracer:
    
    def __init__(self, trial, videocv):
        
        self.trial   = trial
        self.videocv = videocv

    def print_title(self):
        sys.stdout.write("\n\n")
        sys.stdout.write("   ###########################################################\n")
        sys.stdout.write("   #                                                         #\n")
        sys.stdout.write("   #      ___ __   __  _____  ___    _     ___  ___  ___     #\n")
        sys.stdout.write("   #     / __\\\ \ / / |__ __|| _ \  / \   / __|| __|| _ \    #\n")
        sys.stdout.write("   #    | (__  \ V / -  | |  |   / / ^ \ | (__ | _| |   /    #\n")
        sys.stdout.write("   #     \___|  \_/     |_|  |_|_\/_/ \_\ \___||___||_|_\    #\n")
        sys.stdout.write("   #                                      v2.0, sept 2019    #\n")
        sys.stdout.write("   #                                                         #\n")
        sys.stdout.write("   #                                  press <esc> to exit    #\n")
        sys.stdout.write("   #                                                         #\n")
        sys.stdout.write("   ###########################################################\n")
        sys.stdout.write("\n")
        sys.stdout.write("                                 adam patch, fau, jupiter 2019\n")
        sys.stdout.write("                              github.com/patchmemory/cv-tracer\n")
        sys.stdout.write(" \n\n")
        sys.stdout.write("       Tracing %i fish using video, \n" % self.trial.n)
        sys.stdout.write("         %s \n" % (self.trial.fvideo_raw))
        sys.stdout.write(" \n\n")
        sys.stdout.write("       Writing output to \n")
        sys.stdout.write("\n")
        sys.stdout.write("         video: \n" )
        sys.stdout.write("           %s \n" % (self.trial.fvideo_out))
        sys.stdout.write("         data: \n" )
        sys.stdout.write("           %s \n" % (self.trial.fname))
        sys.stdout.write(" \n\n")
        sys.stdout.flush()
    
    
    def draw(self):
        self.videocv.draw_tank(self.trial.tank)
        if ( len(self.videocv.contour_list) != self.trial.n ):
            self.videocv.draw_contour_repeat()
        self.videocv.draw_points()
        self.videocv.draw_directors()
        self.videocv.draw_tstamp()
    
    def mask_tank(self):
        self.videocv.mask_tank(self.trial.tank)

    def connect_frames(self):
        n_contours = len(self.videocv.contours)
            
        # first make sure we have enough frames to do projections
        if self.videocv.tracked_frames() > self.videocv.len_trail:    
            # if tracer found correct number of contours, assume proper tracing 
            # and connect to previous frame based on hugarian min-dist algorithm
            if n_contours == self.videocv.n_ind:
                self.videocv.reorder_hungarian()            
            # if tracer has not found correct number of contours, consider the
            # situation for special handling of frame-to-frame connections
            else:
                # if no contours found, make guess based on trail prediction
                if n_contours == 0:
                    self.videocv.guess()
                # for all other cases, use the contour_connect function
                # attempts to work with occlusions and incorrectly identified contours
                # handles
                else:
                    self.videocv.contour_connect()
                    self.videocv.reorder_connected()

        # for initial frames, make "educated guesses" because tracking
        # needs to start somewhere
        else:
            self.videocv.kmeans_contours()
            if self.videocv.tracked_frames() > 1:
                self.videocv.reorder_hungarian()

        # regardless of method, check for misdirection
        self.videocv.correct_theta()
        
        # once new coordinates have been determined, update trail
        self.videocv.trail_update()

    def update_trial(self):
        tstamp = 1.*self.videocv.frame_num / self.videocv.fps
        self.trial.group.add_entry(tstamp, self.videocv.coord_now)