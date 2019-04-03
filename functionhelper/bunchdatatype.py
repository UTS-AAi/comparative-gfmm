# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:13:08 2018

@author: Thanh Tung Khuat

Collecting a bunch of named items

"""

class Bunch(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)