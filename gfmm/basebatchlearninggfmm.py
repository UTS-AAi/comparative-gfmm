# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 10:43:55 2018

@author: Thanh Tung Khuat

Base class for batch learning GFMM
"""

import numpy as np
from gfmm.basegfmmclassifier import BaseGFMMClassifier
from gfmm.classification import predict
from functionhelper.hyperboxadjustment import hyperbox_overlap_test, hyperbox_contraction
from functionhelper.membershipcalc import membership_gfmm

class BaseBatchLearningGFMM(BaseGFMMClassifier):
    
    def __init__(self, gamma = 1, teta = 1, isDraw = False, oper = 'min', isNorm = True, norm_range = [0, 1]):
        BaseGFMMClassifier.__init__(self, gamma, teta, isDraw, oper, isNorm, norm_range)
        
        self.cardin = np.array([])
        self.clusters = np.empty(None, dtype=object)
    
    
    def remove_contained_hyperboxes(self):
        """
        Remove all hyperboxes contained in other hyperboxes
        """
        numBoxes = len(self.classId)
        indtokeep = np.ones(numBoxes, dtype=np.bool)
        
        for i in range(numBoxes):
            memValue = membership_gfmm(self.V[i], self.W[i], self.V, self.W, self.gamma, self.oper)
            isInclude = (self.classId[memValue == 1] == self.classId[i]).all()
            
            # memValue always has one value being 1 because of self-containing
            if np.sum(memValue == 1) > 1 and isInclude == True:
                indtokeep[i] = False
                
        self.V = self.V[indtokeep, :]
        self.W = self.W[indtokeep, :]
        self.classId = self.classId[indtokeep]
        
    
    def overlap_resolve(self):
        """
        Resolve overlapping hyperboxes with bounders contained in self.V and self.W
        """
        yX = self.V.shape[0]
        # Contraction process does not cause overlappling regions => No need to check from the first hyperbox for each hyperbox
        for i in np.arange(yX - 1):
            j = i + 1
            while j < yX:
                if self.classId[i] != self.classId[j]:
                    caseDim = hyperbox_overlap_test(self.V, self.W, i, j)
                    if len(caseDim) > 0:
                        self.V, self.W = hyperbox_contraction(self.V, self.W, caseDim, j, i)
                
                j = j + 1
                
        return (self.V, self.W)
    

    def pruning(self, X_Val, classId_Val):
        """
        prunning routine for GFMM classifier - Hyperboxes having the number of corrected patterns lower than that of uncorrected samples are prunned
        
        INPUT
            X_Val           Validation data
            ClassId_Val     Validation data class labels (crisp)
            
        OUTPUT
            Lower and upperbounds (V and W), classId, cardin are retained
        """
        # test the model on validation data
        result = predict(self.V, self.W, self.classId, X_Val, X_Val, classId_Val, self.gamma, self.oper)
        mem = result.mem
        
        # find indexes of hyperboxes corresponding to max memberships for all validation patterns
        indmax = mem.argmax(axis = 1)
        
        numBoxes = self.V.shape[0]
        corrinc = np.zeros((numBoxes, 2))
        
        # for each hyperbox calculate the number of validation patterns classified correctly and incorrectly
        for ii in range(numBoxes):
            sampleLabelsInBox = classId_Val[indmax == ii]
            if len(sampleLabelsInBox) > 0:
                corrinc[ii, 0] = np.sum(sampleLabelsInBox == self.classId[ii])
                corrinc[ii, 1] = len(sampleLabelsInBox) - corrinc[ii, 0]
                
        # retain only the hyperboxes which classify at least the same number of patterns correctly as incorrectly
        indRetainedBoxes = np.nonzero(corrinc[:, 0] > corrinc[:, 1])[0]
        
        self.V = self.V[indRetainedBoxes, :]
        self.W = self.W[indRetainedBoxes, :]
        self.classId = self.classId[indRetainedBoxes]
        self.cardin = self.cardin[indRetainedBoxes]
        
        return self
        
        
        
        
