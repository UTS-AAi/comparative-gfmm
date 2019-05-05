# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:22:08 2018

@author: Thanh Tung Khuat

GFMM Predictor

"""

import numpy as np
from functionhelper.membershipcalc import membership_gfmm
from functionhelper.bunchdatatype import Bunch

def predict(V, W, classId, XlT, XuT, patClassIdTest, gama = 1, oper = 'min'):
    """
    GFMM classifier (test routine)

      result = predict(V,W,classId,XlT,XuT,patClassIdTest,gama,oper)

    INPUT
      V                 Tested model hyperbox lower bounds
      W                 Tested model hyperbox upper bounds
      classId	          Input data (hyperbox) class labels (crisp)
      XlT               Test data lower bounds (rows = objects, columns = features)
      XuT               Test data upper bounds (rows = objects, columns = features)
      patClassIdTest    Test data class labels (crisp)
      gama              Membership function slope (default: 1)
      oper              Membership calculation operation: 'min' or 'prod' (default: 'min')

   OUTPUT
      result           A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + sumamb           Number of objects with maximum membership in more than one class
                          + out              Soft class memberships
                          + mem              Hyperbox memberships

    """

    #initialization
    yX = XlT.shape[0]
    misclass = np.zeros(yX)
    classes = np.unique(classId)
    noClasses = classes.size
    ambiguity = np.zeros((yX, 1))
    mem = np.zeros((yX, V.shape[0]))
    out = np.zeros((yX, noClasses))

    # classifications
    for i in range(yX):
        mem[i, :] = membership_gfmm(XlT[i, :], XuT[i, :], V, W, gama, oper) # calculate memberships for all hyperboxes
        bmax = mem[i,:].max()	                                          # get max membership value
        maxVind = np.nonzero(mem[i,:] == bmax)[0]                         # get indexes of all hyperboxes with max membership

        for j in range(noClasses):
            out[i, j] = mem[i, classId == classes[j]].max()            # get max memberships for each class

        ambiguity[i, :] = np.sum(out[i, :] == bmax) 						  # number of different classes with max membership

        if bmax == 0:
            print('zero maximum membership value')                     # this is probably bad...

        if len(np.unique(classId[maxVind])) > 1:
            misclass[i] = True
        else:
            misclass[i] = ~(np.any(classId[maxVind] == patClassIdTest[i]) | (patClassIdTest[i] == 0))
            
    # results
    sumamb = np.sum(ambiguity[:, 0] > 1)
    summis = np.sum(misclass).astype(np.int64)

    result = Bunch(summis = summis, misclass = misclass, sumamb = sumamb, out = out, mem = mem)
    return result