import numpy as np


class Coordinate:
    
    def __init__(self,x,y,theta):
        self.x = x
        self.y = y
        self.theta = theta
   
    def distance(self,x,y):
        return np.sqrt(np.pow(self.x - x,2) + np.pow(self.y - y,2))
    
    def angle_diff(self,theta):
        return min(self.theta - theta,
                   theta - self.theta,
                   self.theta - theta + 2*np.pi,
                   self.theta - theta - 2*np.pi, key=abs)

def area_circle(r):
    return np.pi*r**2

def area_shell(r1,r2):
    return area_circle(r2) - area_circle(r1)

def next_radius(r0,area_shell):
    return np.sqrt( area_shell/np.pi + r0**2 )

def area_shell_from_bins(r, nbins = 10):
    return area_circle(r)/nbins

def bin_edges_circular(r, nbins = 10):
    a_shell = area_shell_from_bins(r, nbins = nbins)
    edges = [ 0 ] 
    for i in range(nbins):
      edges.append(next_radius(edges[-1],a_shell))
    return np.array(edges)

def bin_edges_centers_circular(nbins, hrange = [0, 55.5]):
    edges = bin_edges_circular(hrange[1], nbins = nbins)
    edges = -edges + hrange[1]
    edges = np.flipud(edges)
    edges = np.abs(edges)
    binc = ( edges[1:] + edges[:-1] ) / 2
    return edges, binc

def mean_and_err(l):
    l = np.array(l)
    if len(l.shape) > 1: l = l[:,0]
    mean_tmp = np.nanmean(l)
    err_tmp = np.nanstd(l) / np.sqrt(sum(~np.isnan(l)))
    print(" mean and err", mean_tmp, err_tmp)
    return np.array([mean_tmp, err_tmp])

def mean_and_err_hist(l, nbins):
    l = np.array(l)
    # first test that all entries have desired number of bins
    valid = [ True for entry in l ]
    for i in range(len(l)):
        if len(l[i]) != nbins:
            print("  Incorrect number of bins in array %i." % i)
            print("  Found n = %i but expected n = %i." % (len(l[i]), nbins))
            valid[i] = False
        
    # note: l should be an array containing a sets of bins l[i], and
    # each bin l[i][j] containing a bin center l[i][j][0], 
    #                     and normalized count l[i][j][1]
    hist_mean = np.zeros((nbins, 3),dtype=float)
    for i in range(len(l[0])):
        bin_mean = np.nanmean(l[valid,i,1])
        bin_err  = np.nanstd(l[valid,i,1]) / \
                        np.sqrt(sum(~np.isnan(l[valid,i,1])))
                      
        hist_mean[i][0] = l[0][i][0]
        hist_mean[i][1] = bin_mean
        hist_mean[i][2] = bin_err
        
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
