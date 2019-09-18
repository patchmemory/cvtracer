import sys
import numpy as np
import cv2
import pickle


class Tank:

    def __init__(self, fvideo, r_cm = 111./2):
        self.points  = np.zeros((3,2))    
        self.n_point = 0
        self.row_c   = 0
        self.col_c   = 0
        self.r       = 0
        self.r_cm    = r_cm
        self.found   = False
        self.frame   = None

        self.fvideo  = fvideo
        self.fname = '/'.join(self.fvideo.split('/')[0:-1]) + "/tank.pik"


    def print_info(self):
        print("")
        print("        Filenames")
        print("           Tank: %s" % self.fname)
        print("          Video: %s" % self.fvideo)
        print("")
        print("        Tank information (pixels)")
        print("            row: %4.2e " % self.row_c )
        print("            col: %4.2e " % self.col_c )
        print("              R: %4.2e " % self.r     )
        print("")


    def save(self, fname = None):
        if fname != None:
            self.fname = fname
        f = open(self.fname, 'wb')
        pickle.dump(self.__dict__, f, protocol = 3)
        sys.stdout.write("\n        Tank object saved as %s \n" % self.fname)
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
            sys.stdout.write("\n        Tank object loaded from %s \n" % self.fname)
            sys.stdout.flush()
            return True
        except:
            sys.stdout.write("\n        Tank not found %s \n" % self.fname)
            sys.stdout.flush()
            return False


    def load_txt(self, fname_txt = None):
        try:
            f = open(fname_txt,'r')
            f.readline()
            vals = f.readline().split(' ')
            vals = np.array(vals, dtype = float)
            self.row_c = vals[0]
            self.col_c = vals[1]
            self.r     = vals[2]
        except:
            print("    Cannot locate %s!" % fname_txt)
            exit()



    #########################
    # Tank locator GUI
    #########################
    
    
    def add_circle_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_point(x,y)


    def add_point(self, x, y):
        if self.n_point > 2:
            if self.n_point == 3: 
                print("    Note: Only green points are used for calculation.")
            x_tmp, y_tmp = self.points[self.n_point%3][0], self.points[self.n_point%3][1]
            cv2.circle(self.frame, (int(x_tmp), int(y_tmp)), 4, (0, 0, 255), -1)
            cv2.imshow('image',self.frame)
            cv2.waitKey(cv2.EVENT_LBUTTONUP)
        self.points[self.n_point % 3] = [x, y]
        self.n_point += 1
        cv2.circle(self.frame, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.imshow('image',self.frame)
        cv2.waitKey(cv2.EVENT_LBUTTONUP)
        if self.n_point > 2:
            self.calculate_circle()
            print("    Locating tank edges... ")


    def select_circle(self, event, x, y, flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            px,py = x,y
            if np.sqrt(pow(px-self.row_c,2)+pow(py-self.col_c,2)) > self.r:
                self.found = False
                cv2.circle(self.frame, (int(self.row_c), int(self.col_c)), int(self.r), (0, 0, 255), -1)
                cv2.imshow('image',self.frame)
                cv2.waitKey(0)
            else:
                cv2.circle(self.frame, (int(self.row_c), int(self.col_c)), int(self.r), (0, 255, 0), -1)
                cv2.imshow('image',self.frame)
                cv2.waitKey(0)    


    def calculate_circle(self):
        midpoint = []
        m = []
        b = []
        for i in range(2):
            midpoint.append([(self.points[i+1][0]+self.points[i][0])/2,
                             (self.points[i+1][1]+self.points[i][1])/2])
            slope = ((self.points[i+1][1]-self.points[i][1])/
                     (self.points[i+1][0]-self.points[i][0]))
            m.append(-1./slope)
            b.append(midpoint[i][1]-m[i]*midpoint[i][0])
 
        self.row_c = (b[1]-b[0])/(m[0]-m[1])
        self.col_c = m[0]*self.row_c + b[0]
        self.r = np.sqrt(pow(self.row_c-self.points[0][0],2) + pow(self.col_c-self.points[0][1],2))
        self.found = True


    def locate(self):
        frame_start = 0
        frame_end   = 0
        if not self.load():
            cap = cv2.VideoCapture(self.fvideo)
            if cap.isOpened() == False:
                sys.exit("  Video cannot be opened! Ensure proper video file specified.")         
            if frame_end < 0:
                frame_end = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1          
            while True:
                # open frame
                i_frame = int(np.random.uniform(frame_start,frame_end))
                cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame-1)
                ret, self.frame = cap.read()
                if ret == True:
                    # try locating the tank
                    cv2.namedWindow('image')
                    cv2.setMouseCallback('image', self.add_circle_point)
                    cv2.imshow('image', self.frame)
                    cv2.waitKey(0)
                    # show results and allow user to choose if it looks right
                    cv2.circle(self.frame, (int(self.row_c), int(self.col_c)), int(self.r), (0, 255, 0), 4)
                    cv2.circle(self.frame, (int(self.row_c), int(self.col_c)), 5, (0, 255, 0), -1)
                    cv2.setMouseCallback('image', self.select_circle)
                    cv2.imshow('image', self.frame)
                    cv2.waitKey(0)
                    # if the user decides the tank location is good, then exit loop
                    if self.found:
                        break
                    else:
                        continue
          
            self.frame = None
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
          
            sys.stdout.write("\n       Tank detection complete.\n")
            sys.stdout.flush()
            
            self.save()
     