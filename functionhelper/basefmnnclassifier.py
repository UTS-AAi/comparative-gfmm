# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 08:39:15 2018

@author: Thanh Tung Khuat

Base class for FMNN Classifier
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from functionhelper.matrixhelper import delete_const_dims, pca_transform
from functionhelper.preprocessinghelper import normalize
from functionhelper.baseclassification import predict 
from functionhelper.membershipcalc import simpson_membership


class BaseFMNNClassifier(object):
    
    def __init__(self, gamma = 1, teta = 1, isDraw = False, isNorm = True, norm_range = [0, 1]):
        self.gamma = gamma
        self.teta = teta
        self.isDraw = isDraw
        self.isNorm = isNorm
        
        self.V = np.array([])
        self.W = np.array([])
        self.classId = np.array([], dtype = np.int64)
      
        # parameters for data normalization
        self.loLim = norm_range[0]
        self.hiLim = norm_range[1]
        self.mins = np.array([])
        self.maxs = np.array([])
        self.delayConstant = 0.001 # delay time period to display hyperboxes on the canvas
    
    
    def data_preprocessing(self, Xh):
        """
        Preprocess data: delete constant dimensions, Normalize input samples if needed
        
        INPUT:
            Xh      Input data lower bounds (rows = objects, columns = features)
        
        OUTPUT
            Xh was preprocessed
        """
        
        # delete constant dimensions
        #Xh = delete_const_dims(Xh)
        
        # Normalize input samples if needed
        if Xh.min() < self.loLim or Xh.max() > self.hiLim:
            self.mins = Xh.min(axis = 0) # get min value of each feature
            self.maxs = Xh.max(axis = 0) # get max value of each feature
            Xh = normalize(Xh, [self.loLim, self.hiLim])
        else:
            self.isNorm = False
            self.mins = []
            self.maxs = []
            
        return Xh
    
    
    def pcatransform(self):
        """
        Perform PCA transform of V and W if the dimensions are larger than 3
        
        OUTPUT:
            V and W in the new space
        """
        yX, xX = self.V.shape
                
        if (xX > 3):
            Vt = pca_transform(self.V, 3)
            Wt = pca_transform(self.W, 3)
            mins = Vt.min(axis = 0)
            maxs = Wt.max(axis = 0)
            Vt = self.loLim + (self.hiLim - self.loLim) * (Vt - np.ones((yX, 1)) * mins) / (np.ones((yX, 1)) * (maxs - mins))
            Wt = self.loLim + (self.hiLim - self.loLim) * (Wt - np.ones((yX, 1)) * mins) / (np.ones((yX, 1)) * (maxs - mins))
        else:
            Vt = self.V
            Wt = self.W
            
        return (Vt, Wt)
    
    
    def initialize_canvas_graph(self, figureName, numDims):
        """
        Initialize canvas to draw hyperbox
        
            INPUT
                figureName          Title name of windows containing hyperboxes
                numDims             The number of dimensions of hyperboxes
                
            OUTPUT
                drawing_canvas      Plotting object of python
        """
        fig = plt.figure(figureName)
        plt.ion()
        if numDims == 2:
            drawing_canvas = fig.add_subplot(1, 1, 1)
            drawing_canvas.axis([0, 1, 0, 1])
        else:
            drawing_canvas = Axes3D(fig)
            drawing_canvas.set_xlim3d(0, 1)
            drawing_canvas.set_ylim3d(0, 1)
            drawing_canvas.set_zlim3d(0, 1)
            
        return drawing_canvas
    
    
    def delay(self):
        """
        Delay a time period to display hyperboxes
        """
        plt.pause(self.delayConstant)
        
    
    def predict(self, X_Test, patClassIdTest):
        """
        Perform classification
        
            result = predict(Xl_Test, Xu_Test, patClassIdTest)
        
        INPUT:
            X_Test             Test data (rows = objects, columns = features)
            patClassIdTest	     Test data class labels (crisp)
            
        OUTPUT:
            result        A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + sumamb           Number of objects with maximum membership in more than one class
                          + out              Soft class memberships
                          + mem              Hyperbox memberships
        """
        #X_Test = delete_const_dims(X_Test)
        # Normalize testing dataset if training datasets were normalized
        if len(self.mins) > 0:
            noSamples = X_Test.shape[0]
            X_Test = self.loLim + (self.hiLim - self.loLim) * (X_Test - np.ones((noSamples, 1)) * self.mins) / (np.ones((noSamples, 1)) * (self.maxs - self.mins))          
            
            if X_Test.min() < self.loLim or X_Test.max() > self.hiLim:
                print('Test sample falls outside', self.loLim, '-', self.hiLim, 'interval')
                print('Number of original samples = ', noSamples)
                
                # only keep samples within the interval loLim-hiLim
                indX_Keep = np.where((X_Test >= self.loLim).all(axis = 1) & (X_Test <= self.hiLim).all(axis = 1))[0]
                
                X_Test = X_Test[indX_Keep, :]
                
                print('Number of kept samples =', X_Test.shape[0])
            
        # do classification
        result = None
        
        if X_Test.shape[0] > 0:
            result = predict(self.V, self.W, self.classId, X_Test, patClassIdTest, self.gamma)
        
        return result
    
    
    def pruning_val(self, XTest, patClassIdTest, accuracy_threshold = 0.5):
        """
        pruning handling based on validation (validation routine) with hyperboxes stored in self. V, W, classId
    
          result = pruning_val(XlT,XuT,patClassIdTest)
    
            INPUT
              XlT               Test data lower bounds (rows = objects, columns = features)
              XuT               Test data upper bounds (rows = objects, columns = features)
              patClassIdTest    Test data class labels (crisp)
              accuracy_threshold  The minimum accuracy for each hyperbox
          
        """
    
        #initialization
        yX = XTest.shape[0]
        mem = np.zeros((yX, self.V.shape[0]))
        no_predicted_samples_hyperboxes = np.zeros((len(self.classId), 2))
        # classifications
        for i in range(yX):
            mem[i, :] = simpsonMembership(XTest[i, :], self.V, self.W, self.gamma) # calculate memberships for all hyperboxes
            bmax = mem[i,:].max()	                                          # get max membership value
            maxVind = np.nonzero(mem[i,:] == bmax)[0]                         # get indexes of all hyperboxes with max membership
            
            if len(maxVind) == 1:
                # Only one hyperbox with the highest membership function
                
                if self.classId[maxVind[0]] == patClassIdTest[i]:
                    no_predicted_samples_hyperboxes[maxVind[0], 0] = no_predicted_samples_hyperboxes[maxVind[0], 0] + 1                 
                else:
                    no_predicted_samples_hyperboxes[maxVind[0], 1] = no_predicted_samples_hyperboxes[maxVind[0], 1] + 1
            else:
                # More than one hyperbox with highest membership => random choosing
                id_min = maxVind[np.random.randint(len(maxVind))]
                        
                if self.classId[id_min] != patClassIdTest[i] and patClassIdTest[i] != 0:
                    no_predicted_samples_hyperboxes[id_min, 1] = no_predicted_samples_hyperboxes[id_min, 1] + 1
                else:
                    no_predicted_samples_hyperboxes[id_min, 0] = no_predicted_samples_hyperboxes[id_min, 0] + 1
                    
        # pruning handling based on the validation results
        tmp_no_box = no_predicted_samples_hyperboxes.shape[0]
        accuracy_larger_half = np.zeros(tmp_no_box).astype(np.bool)
        for i in range(tmp_no_box):
            if (no_predicted_samples_hyperboxes[i, 0] + no_predicted_samples_hyperboxes[i, 1] != 0) and no_predicted_samples_hyperboxes[i, 0] / (no_predicted_samples_hyperboxes[i, 0] + no_predicted_samples_hyperboxes[i, 1]) >= accuracy_threshold:
                accuracy_larger_half[i] = True
                
        # keep one hyperbox for class prunned all
        current_classes = np.unique(self.classId)
        class_tmp = self.classId[accuracy_larger_half]
        for c in current_classes:
            if c not in class_tmp:
                pos = np.nonzero(self.classId == c)
                id_kept = np.random.randint(len(pos))
                # keep pos[id_kept]
                accuracy_larger_half[pos[id_kept]] = True
        # Pruning
        self.V = self.V[accuracy_larger_half]
        self.W = self.W[accuracy_larger_half]
        self.classId = self.classId[accuracy_larger_half]