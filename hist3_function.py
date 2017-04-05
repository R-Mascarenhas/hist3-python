# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 22:54:21 2017

@author: Rafael

V 1.1.1
"""

import numpy as np
import matplotlib.pyplot as plt



def hist3(x,bins = 10, normed = False, color = 'blue', alpha = 1, *args,**kwargs):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pylab
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if np.size(bins) == 1:
        bins = [bins,bins]
    
    H, edges = np.histogramdd(x, bins = bins, normed = normed)

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

    
    ax.bar3d(X,Y,Z,dx,dy,dz, alpha = alpha, color = color);
    plt.xlabel('X');
    plt.ylabel('Y');
    
    edges = [X,Y];
    hist = dz.reshape(bins[0],bins[1]);
    
    return hist, edges



def hist32(x,y,binx,biny):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pylab
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    bins = (binx,biny)
    
    H, edges = np.histogramdd(x, bins, normed = False)
    H = H.T

    X = list(np.linspace(min(x[0]),max(x[0]),binx))*binx
    Y = np.sort(list(np.linspace(min(x[1]),max(x[1]),biny))*biny)
    dz = []

    for i in range(binx):
        for j in range(biny):
            dz.append(H[i][j])
    
    Z = np.zeros(binx**2)

    dx = X[1] - X[0]
    dy = Y[biny]-Y[0]

    ax.bar3d(X,Y,Z,dx,dy,dz, alpha = 0.5)
    
    ###########################################################
    
    H, edges = np.histogramdd(y, bins, normed = True)

    H = H.T

    X = np.array(list(np.linspace(min(y[0]),max(y[0]),binx))*binx)
    Y = np.sort(list(np.linspace(min(y[1]),max(y[1]),biny))*biny)
    dz = np.array([]);

    for i in range(binx):
        for j in range(biny):
            dz= np.append(dz, H[i][j])
    
    Z = np.zeros(binx**2)

    
    dx = X[1] - X[0]
    dy = Y[biny]-Y[0]

    ax.bar3d(X,Y,Z,dx,dy,dz, alpha = 0.5, color = 'g')
    
    plt.show()
    
    
##################################

mux = 4
sigmax = 7

muy = 0
sigmay = 5

events = 1000

#x = np.random.randn(events,2)
x = np.random.randn(events,2)*sigmax + mux;
y = np.random.normal(muy,sigmay,(events,2));



hist3(x,[50,20], normed = True)






