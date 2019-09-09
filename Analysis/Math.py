import numpy as np


class Coordinate:
    
    def __init__(x,y,theta)
        self.x = x
        self.y = y
        self.theta = theta
   
    def distance(x,y):
        return np.sqrt(np.pow(self.x - x,2) + np.pow(self.y - y,2))
    
    def angle_diff(theta):
        return min(self.theta - theta,
                   theta - self.theta,
                   self.theta - theta + 2*np.pi,
                   self.theta - theta - 2*np.pi, key=abs)

    

def mean_and_err(self, l, hist=False):
    l = np.array(l)
    mean_tmp = np.nanmean(l)
    err_tmp = np.nanstd(l) / np.sqrt(sum(~np.isnan(l)))
    return np.array([mean_tmp, err_tmp])

def mean_and_err_hist(self, l, nbins):
    l = np.array(l)
    # first test that all entries have desired number of bins
    valid = [ True for i in len(l) ]
    for i in range(len(l)):
        if len(l) != nbins:
            print("Incorrect number of bins in array %i." % i)
            valid[i] = False
        
    hist_mean = np.zeros((nbins,2),dtype=float)
    for i in range(len(l)):
        bin_mean = np.nanmean(l[valid,i])
        bin_err  = np.nanstd(l[valid,i]) / \
                        np.sqrt(sum(~np.isnan(l[valid,i])))
                        
        hist_mean[i][0] = bin_mean
        hist_mean[i][1] = bin_err
        
    return np.array(hist_mean)


def distance(pos1,pos2):
    return np.sqrt(pow(pos1[0]-pos2[0],2)+pow(pos1[1]-pos2[1],2))


def angle_diff(q2,q1):
    return min(q2-q1,q2-q1,q2-q1+2*np.pi,q2-q1-2*np.pi, key=abs)


# rotate a point xp,yp about x0,y0 by angle
def transform_point(x0,y0,xp,yp,angle):
    cos_theta, sin_theta = np.cos(angle), np.sin(angle)
    x, y = xp - x0, yp - y0
    xf =   x * cos_theta + y * sin_theta 
    yf = - x * sin_theta + y * cos_theta 
    return xf,yf


def reject_outliers(data, m):
    """
    This function removes any outliers from presented data.
    
    Parameters
    ----------
    data: pandas.Series
        a column from a pandas dataframe that needs smoothing
    m: float
        standard deviation cutoff beyond which, datapoint is considered as an outlier
        
    Returns
    -------
    index: ndarray
        an array of indices of points that are not outliers
    """
    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d/mdev if mdev else 0.
    return np.where(s < m)