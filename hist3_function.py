# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 22:54:21 2017

@author: Rafael
"""

import numpy as np
import matplotlib.pyplot as plt



def hist3(x,binx,biny):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pylab
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    H, xedges, yedges = np.histogram2d(x[0],x[1], bins =(binx,biny))

    H = H.T

    X = list(np.linspace(min(x[0]),max(x[0]),binx))*binx
    Y = np.sort(list(np.linspace(min(x[1]),max(x[1]),biny))*biny)
    dz = []

    for i in range(binx):
        for j in range(biny):
            dz.append(H[i][j])
    
    Z = np.zeros(binx**2)

    dx = np.linspace(min(x[0]),max(x[0]),binx)
    dx = dx[1] - dx[0]
    dy = np.linspace(min(x[1]),max(x[1]),biny)
    dy = dy[1] - dy[0]

    ax.bar3d(X,Y,Z,dx,dy,dz, alpha = 0.5)


def hist32(x,y,binx,biny):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pylab
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    H, xedges, yedges = np.histogram2d(x[0],x[1], bins =(binx,biny), normed = True)

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
    
    H, xedges, yedges = np.histogram2d(y[0],y[1], bins = (binx,biny), normed = True)

    H = H.T

    X = list(np.linspace(min(y[0]),max(y[0]),binx))*binx
    Y = np.sort(list(np.linspace(min(y[1]),max(y[1]),biny))*biny)
    dz = []

    for i in range(binx):
        for j in range(biny):
            dz.append(H[i][j])
    
    Z = np.zeros(binx**2)

    
    dx = X[1] - X[0]
    dy = Y[biny]-Y[0]

    ax.bar3d(X,Y,Z,dx,dy,dz, alpha = 0.5, color = 'g')
    
    plt.show()

#==============================================================================
# mux = 4
# sigmax = 2
# 
# muy = 0
# sigmay = 1
# events = 10000
# 
# x = np.random.normal(mux,sigmax,(2,events));
# y = np.random.normal(muy,sigmay,(2,events));
# 
# 
# 
# hist32(x,y,100,100)
#==============================================================================






