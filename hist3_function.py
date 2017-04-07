# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 22:54:21 2017

@author: Rafael

V 1.1.2
"""

import numpy as np
import matplotlib.pyplot as plt



def hist3(x,bins = 10, normed = False, color = 'blue', alpha = 1, hold = False):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pylab
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
    hist = dz.reshape(bins[0],bins[1]);
    
    return hist, edges
#==============================================================================
# 
# 
# mux = 4
# sigmax = 7
# 
# muy = 6
# sigmay = 5
# 
# events = 1000
# 
# x = np.random.randn(events,2)*sigmax + mux;
# y = np.random.normal(muy,sigmay,(2,events));
# 
# 
# 
# hist3(x,[10,10], normed = True, hold = False, alpha = 0.3)
# hist3(y,[10,10], normed = True, hold = True, color = 'g', alpha = 0.5)
# 
# 
# 
#==============================================================================
