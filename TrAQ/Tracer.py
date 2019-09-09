import sys, os
import argparse
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


class Tracer:
    
    def __init__(self, n_ind):
        self.n_ind              = n_ind
        self.fvideo_in          = ""
        self.fvideo_out         = ""
        self.fdata              = ""
        self.RGB                = False
        self.block_size         = 15
        self.threshhold_offset  = 13
        self.fps                = 30
        self.gpu                = False
        self.colors             = [ (   0,   0, 255),
                                    (   0, 255, 255),
                                    ( 255,   0, 255),
                                    ( 255, 255, 255),
                                    ( 255, 255,   0),
                                    ( 255,   0,   0),
                                    (   0, 255,   0),
                                    (   0,   0,   0) ]
        
        
    def random_color_list(self):
        colors = []
        low = 0
        high = 255
        for i in range(self.n_ind):
            color = []
            for i in range(3):
                color.append(np.uint8(np.random.random_integers(low,high)))
            colors.append((color[0],color[1],color[2]))
        print(colors)
        return colors


    def print_title(self, trial):
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
        sys.stdout.write("       Tracing %i fish using video, \n" % trial.n)
        sys.stdout.write("         %s \n" % (trial.fvideo_in))
        sys.stdout.write(" \n\n")
        sys.stdout.write("       Writing output to \n")
        sys.stdout.write("\n")
        sys.stdout.write("         video: \n" )
        sys.stdout.write("           %s \n" % (trial.fvideo_out))
        sys.stdout.write("         data: \n" )
        sys.stdout.write("           %s \n" % (trial.fdata))
        sys.stdout.write(" \n\n")
        sys.stdout.flush()


    def print_current_frame(self, frame_num, fps):
        t_csc = int(frame_num/fps*100)
        t_sec = int(t_csc/100) 
        t_min = t_sec/60
        t_hor = t_min/60
        sys.stdout.write("       Current tracking time: %02i:%02i:%02i:%02i \r" % (t_hor,t_min%60,t_sec%60,t_csc%100))
        sys.stdout.flush()


    def kmeans_contours(self, contours, meas_now):
        # convert contour points to arrays
        clust_points = np.vstack(contours)
        clust_points = clust_points.reshape(clust_points.shape[0], clust_points.shape[2])
        # run KMeans clustering
        kmeans_init = 50
        kmeans = KMeans(n_clusters=self.n_ind, random_state=0, n_init = kmeans_init).fit(clust_points)
        l = len(kmeans.cluster_centers_)
        theta = -13
        del meas_now[:]
        for i in range(l):
            x = int(tuple(kmeans.cluster_centers_[i])[0])
            y = int(tuple(kmeans.cluster_centers_[i])[1])
            meas_now.append([x,y,theta])
        return meas_now
    
    # in the absence of contours, assuming there was something previously, make a
    # guess should only be done for a limited number of frames, hence "temporary"
    def temporary_guess(self, meas_last, meas_now, i_frame):
        prev = i_frame - 1 
        # use meas_last to calculate predicted trajectory based on previous three points
        #   (meas_now contains contours, meas_last contains guess) 
        meas_now = meas_last.copy()
        for i in range(len(meas_last)):
            meas_now[i][0] = q[i].loc(prev,'x') + ( 3*q[i].loc(prev,'x') - 2*q[i].loc(prev-1,'x') - q[i].loc(prev-2,'x') ) / 4.
            meas_now[i][1] = q[i].loc(prev,'y') + ( 3*q[i].loc(prev,'y') - 2*q[i].loc(prev-1,'y') - q[i].loc(prev-2,'y') ) / 4.
    
        return meas_last, meas_now
    
    
    def contour_connect(self, n_ind, meas_now, meas_last, contours, i_frame):
        prev = i_frame - 1
        # use meas_last to calculate predicted trajectory based on previous three points
        #   (meas_now contains contours, meas_last contains guess) 
        for i in range(len(meas_last)):
            meas_last[i][0] = q[i].loc(prev,'x') + ( 3*q[i].loc(prev,'x') - 2*q[i].loc(prev-1,'x') - q[i].loc(prev-2,'x') ) / 4.
            meas_last[i][1] = q[i].loc(prev,'y') + ( 3*q[i].loc(prev,'y') - 2*q[i].loc(prev-1,'y') - q[i].loc(prev-2,'y') ) / 4.
    
        # calculate the distance matrix and then look  
        dist_arr = cdist(meas_last,meas_now)
        ind_last_now = [ 0 for i in range(len(meas_last)) ]
        for i in range(len(meas_last)):
            index = np.argmin(dist_arr[i]) # i'th array has distances to predictions from last
            ind_last_now[i] = index # for i, what is corresponding index of last (j)
    
        # identify any unclaimed contours
        contourID_unclaimed = []
        for i in range(len(meas_now)):
            for j in range(len(meas_last)):
                if i == ind_last_now[j]:
                    break # exits current for loop, contour is located
                if j == len(meas_last) - 1:
                    contourID_unclaimed.append(i)
    
        # choose the closest contour to guess even if two choose the same
        for i in range(len(contourID_unclaimed)):
            index = np.argmin(dist_arr[:,contourID_unclaimed[i]])
            ind_last_now[index] = contourID_unclaimed[i]
    
        # identify contours containing multiple points
        contourID_repeat = []
        contour_deficit = len(meas_last) - len(meas_now)
        sorted_ind_last_now = sorted(ind_last_now)
        i = 0  
        while len(contourID_repeat) < contour_deficit:
            i += 1
            if sorted_ind_last_now[i] == sorted_ind_last_now[i-1]:
                contourID_repeat.append(sorted_ind_last_now[i])
    
        return ind_last_now, contourID_repeat, contourID_unclaimed


    def reorder_hungarian(self, meas_last, meas_now):
        # reorder contours based on results of the hungarian algorithm 
        # (originally in tracktor, but i've reorganized)
        row_ind, col_ind = self.hungarian_algorithm(meas_last,meas_now)
        equal = np.array_equal(col_ind, list(range(len(col_ind))))
        if equal == False:
            current_ids = col_ind.copy()
            reordered = [i[0] for i in sorted(enumerate(current_ids), key=lambda x:x[1])]
            meas_now = [x for (y,x) in sorted(zip(reordered,meas_now))]
            
        return meas_last, meas_now
    
    
    def hungarian_algorithm(self, meas_last, meas_now):
        """
        The hungarian algorithm is a combinatorial optimisation algorithm used
        to solve assignment problems. Here, we use the algorithm to reduce noise
        due to ripples and to maintain individual identity. This is accomplished
        by minimising a cost function; in this case, euclidean distances between 
        points measured in previous and current step. The algorithm here is written
        to be flexible as the number of contours detected between successive frames
        changes. However, an error will be returned if zero contours are detected.
       
        Parameters
        ----------
        meas_last: array_like, dtype=float
            individual's location on previous frame
        meas_now: array_like, dtype=float
            individual's location on current frame
            
        Returns
        -------
        row_ind: array, dtype=int64
            individual identites arranged according to input ``meas_last``
        col_ind: array, dtype=int64
            individual identities rearranged based on matching locations from 
            ``meas_last`` to ``meas_now`` by minimising the cost function
        """
        xy_last = np.array(meas_last)[:,[0,1]]
        xy_now = np.array(meas_now)[:,[0,1]]
        #if meas_now.shape != meas_last.shape:
        #    if meas_now.shape[0] < meas_last.shape[0]:
        #        while meas_now.shape[0] != meas_last.shape[0]:
        #            meas_last = np.delete(meas_last, meas_last.shape[0]-1, 0)
        #    else:
        #        result = np.zeros(meas_now.shape)
        #        result[:meas_last.shape[0],:meas_last.shape[1]] = meas_last
        #        meas_last = result
    
        xy_last = list(xy_last)
        xy_now = list(xy_now)
        cost = cdist(xy_last, xy_now)
        row_ind, col_ind = linear_sum_assignment(cost)
        return row_ind, col_ind


    def reorder_connected(self, meas_last, meas_now, ind_last_now):
        meas_now_tmp = meas_now
        meas_now = meas_last
        for i in range(len(meas_now)):
            meas_now[i] = meas_now_tmp[ind_last_now[i]]
            
        return meas_last, meas_now

    
    def reorient_directors(self):
        sys.stdout.write("\n\n")
        sys.stdout.write("       Reorienting directors with velocities...")
        sys.stdout.flush()
        # rotate directors (only nematic on pi until this point)
        frame_gap = 1 
        for i in range(frame_gap,len(q)-frame_gap):
            for j in range(len(q[i])):
                cos_psi = q[i][j].vx*np.cos(q[i][j].theta) + q[i][j].vy*np.sin(q[i][j].theta)
                if ( cos_psi < 0 ): 
                    q[i][j].theta = np.fmod(q[i][j].theta + np.pi,2*np.pi) 
        sys.stdout.write("         ... done reorienting directors.")
        sys.stdout.write("\n\n")
        return q