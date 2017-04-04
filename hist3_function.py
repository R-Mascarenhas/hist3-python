# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 22:54:21 2017

@author: Rafael

V 1.0.2
"""

import numpy as np
import matplotlib.pyplot as plt



def hist3(x,bins = [10,10], normed = False, color = 'blue'):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pylab
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    
    H, xedges = np.histogramdd(x, bins, normed = normed)

    H = H.T

    X = np.array(list(np.linspace(min(xedges[0]),max(xedges[0]),bins[0]))*bins[0])     #np.array(list(np.linspace(min(x[0]),max(x[0]),binx))*binx)
    Y = np.sort(list(np.linspace(min(xedges[1]),max(xedges[1]),bins[1]))*bins[1])      #np.sort(list(np.linspace(min(x[1]),max(x[1]),biny))*biny)
    dz = np.array([]);

    for i in range(bins[0]):
        for j in range(bins[1]):
            dz = np.append(dz, H[i][j])
    
    Z = np.zeros(bins[0]*bins[1])

    #dx = np.linspace(min(x[0]),max(x[0]),binx)
    dx = xedges[0][1] - xedges[0][0]
    
    #dy = np.linspace(min(x[1]),max(x[1]),biny)
    dy = xedges[1][1] - xedges[1][0]

    
    ax.bar3d(X,Y,Z,dx,dy,dz, alpha = 0.5, color = color)
    
    

def hist32(x,y,binx,biny):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pylab
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    bins = (binx,biny)
    
    H, xedges = np.histogramdd(x, bins, normed = False)
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
    
    H, xedges = np.histogramdd(y, bins, normed = True)

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



hist3(x,[10,10])






