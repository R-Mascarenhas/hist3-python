# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 22:54:21 2017

@author: Rafael

V 1.1.2
"""

def histogram3d(x,bins = 10, normed = False, color = 'blue', alpha = 1, hold = False):
    
    """
    
    Plotting a 3D histogram
    Parameters
    ----------
    
    sample : array_like.		
        The data to be histogrammed. It must be an (N,2) array or data 
        that can be converted to such. The rows of the resulting array 
        are the coordinates of points in a 2 dimensional polytope.
    
    bins : sequence or int, optional, default: 10.
        The bin specification:
        
        * A sequence of arrays describing the bin edges along each dimension.
        * The number of bins for each dimension (bins =[binx,biny])
        * The number of bins for all dimensions (bins = bins).
    
    normed : bool, optional, default: False.
        If False, returns the number of samples in each bin. 
        If True, returns the bin density bin_count / sample_count / bin_volume.
    
    color: string, matplotlib color arg, default = 'blue'
    
    alpha: float, optional, default: 1.
        0.0 transparent through 1.0 opaque
    
    hold: boolean, optional, default: False
    
    Returns   
    --------
    H : ndarray.
        The bidimensional histogram of sample x.
    
    edges : list.
        A list of 2 arrays describing the bin edges for each dimension.
        
    See Also 
    --------
    histogram: 1-D histogram
    histogram2d: 2-D histogram
    histogramdd: N-D histogram
    
    Examples
    --------
    >>> r = np.random.randn(1000,2)
    >>> H, edges = np.histogram3d(r,bins=[10,15])
    """
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    if np.size(bins) == 1:
        bins = [bins,bins]
    
    if(len(x) == 2):
        x = x.T;
        
    
    H, edges = np.histogramdd(x, bins, normed = normed)

    H = H.T
    X = np.array(list(np.linspace(min(edges[0]),max(edges[0]),bins[0]))*bins[1])   
    Y = np.sort(list(np.linspace(min(edges[1]),max(edges[1]),bins[1]))*bins[0])    
    
    dz = np.array([]);

    for i in range(bins[1]):
        for j in range(bins[0]):
            dz = np.append(dz, H[i][j])
    
    Z = np.zeros(bins[0]*bins[1])

    dx = X[1] - X[0]   
    dy = Y[bins[0]] - Y[0]
    
    if (not hold):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.bar3d(X,Y,Z,dx,dy,dz, alpha = alpha, color = color);
    else:
        try:
            ax = plt.gca();
            ax.bar3d(X,Y,Z,dx,dy,dz, alpha = alpha, color = color);
        except:
            plt.close(plt.get_fignums()[-1])
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.bar3d(X,Y,Z,dx,dy,dz, alpha = alpha, color = color);
            
            
    plt.xlabel('X');
    plt.ylabel('Y');
    
    edges = [X,Y];
    H = dz.reshape(bins[0],bins[1]);
    
    return H, edges;
