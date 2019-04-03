# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 12:13:49 2018

@author: Thanh Tung Khuat

Functions to draw hyperboxes

"""

import numpy as np

def drawbox(low, up, plt, color = ['k']):
    """
    Drawing rectangular (2 dimensional inputs) or cube (3 and more dimensional inputs)
          han = drawbox(low,up)
  
    INPUT
      low       Min point coordinates
      up        Max point coordinates
      plt       plot object in pyplot
      color     hyperboxes border colors
  
    OUTPUT
      han		Handles of the plotted objects
    """

    ySizeUp = up.shape[0]
    han = np.empty(ySizeUp, dtype=object)
    
    default_color = 'k'
    if (len(color) > 0):
        default_color = color[0]
        
    for i in range(ySizeUp):
        selected_color = default_color
        if i < len(color):
            selected_color = color[i]
            
        if low[i].size == 2:
            # plot actually returns a list of artists, hence the ,
            if low[i, 0] == up[i, 0] and low[i, 1] == up[i, 1]:
                han[i], = plt.plot(low[i, 0], low[i, 1], color=selected_color, marker='+')
            else:
                han[i], = plt.plot([low[i, 0], low[i, 0], up[i, 0], up[i, 0], low[i, 0]], [low[i, 1], up[i, 1], up[i, 1], low[i, 1], low[i, 1]], color=selected_color)
        else:
            if low[i, 0] == up[i, 0] and low[i, 1] == up[i, 1] and low[i, 2] == up[i, 2]:
                han[i], = plt.plot([low[i, 0]], [low[i, 1]], [low[i, 2]], color=selected_color, marker='+')
            else:
                han[i], = plt.plot([low[i, 0], low[i, 0], up[i, 0], up[i, 0], low[i, 0], low[i, 0], low[i, 0], low[i, 0], up[i, 0], up[i, 0], low[i, 0], up[i, 0], up[i, 0], up[i, 0], up[i, 0], low[i, 0], low[i, 0]], \
                                   [low[i, 1], up[i, 1], up[i, 1], low[i, 1], low[i, 1], low[i, 1], low[i, 1], up[i, 1], up[i, 1], low[i, 1], low[i, 1], low[i, 1], low[i, 1], up[i, 1], up[i, 1], up[i, 1], up[i, 1]], \
                                   [low[i, 2], low[i, 2], low[i, 2], low[i, 2], low[i, 2], up[i, 2], up[i, 2], up[i, 2], up[i, 2], up[i, 2], up[i, 2], up[i, 2], low[i, 2], low[i, 2], up[i, 2], up[i, 2], low[i, 2]], color=selected_color)
            
    return han
