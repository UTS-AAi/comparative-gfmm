# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 14:57:55 2018

@author: Thanh Tung Khuat

Batch GFMM classifier (training core) - Faster version by only computing similarity among hyperboxes with the same label

    BatchGFMMV1(gamma, teta, bthres, simil, sing, isDraw, oper, isNorm, norm_range, cardin, clusters)
  
    INPUT
        gamma       Membership function slope (default: 1)
        teta        Maximum hyperbox size (default: 1)
        bthres		Similarity threshold for hyperbox concatenation (default: 0.5)
        simil       Similarity measure: 'short', 'long' or 'mid' (default: 'mid')
        sing        Use 'min' or 'max' (default) memberhsip in case of assymetric similarity measure (simil='mid')
        oper        Membership calculation operation: 'min' or 'prod' (default: 'min')
        isDraw      Progress plot flag (default: 1)
        oper        Membership calculation operation: 'min' or 'prod' (default: 'min')
        isNorm      Do normalization of input training samples or not?
        norm_range  New ranging of input data after normalization, for example: [0, 1]
        cardin      Input hyperbox cardinalities
        clusters    Identifiers of objects in each input hyperbox 
        
    ATTRIBUTES
        V               Hyperbox lower bounds
        W               Hyperbox upper bounds
        classId         Hyperbox class labels (crisp)
        # Currently, two attributes are not yet used, so comment code to accelerate the training time
        cardin          Hyperbox cardinalities (the number of training samples is covered by corresponding hyperboxes)
        clusters        Identifiers of input objects in each hyperbox (indexes of training samples covered by corresponding hyperboxes)

"""

import sys, os
sys.path.insert(0, os.path.pardir)

import ast
import numpy as np
import time
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

from GFMM.basebatchlearninggfmm import BaseBatchLearningGFMM
from functionhelper.membershipcalc import membership_gfmm
from functionhelper.drawinghelper import drawbox
from functionhelper.hyperboxadjustment import is_overlap
from functionhelper.preprocessinghelper import load_dataset, string_to_boolean

class BatchGFMMV1(BaseBatchLearningGFMM):
    
    def __init__(self, gamma = 1, teta = 1, bthres = 0.5, simil = 'mid', sing = 'max', isDraw = False, oper = 'min', isNorm = True, norm_range = [0, 1], cardin = np.array([], dtype=np.int64), clusters = np.array([], dtype=object)):
        BaseBatchLearningGFMM.__init__(self, gamma, teta, isDraw, oper, isNorm, norm_range)
        
        self.bthres = bthres
        self.simil = simil
        
        if simil == 'mid':
            self.sing = sing
        else:
            self.sing = 'max'
        
#        self.cardin = cardin
#        self.clusters = clusters       
        
    
    def fit(self, X_l, X_u, patClassId):
        """
        X_l          Input data lower bounds (rows = objects, columns = features)
        X_u          Input data upper bounds (rows = objects, columns = features)
        patClassId  Input data class labels (crisp)
        """
        
        if self.isNorm == True:
            X_l, X_u = self.dataPreprocessing(X_l, X_u)
            
        time_start = time.perf_counter()
         
        self.V = X_l
        self.W = X_u
        self.classId = patClassId
        
        yX, xX = X_l.shape
        
#        if len(self.cardin) == 0 or len(self.clusters) == 0:
#            self.cardin = np.ones(yX)
#            self.clusters = np.empty(yX, dtype=object)
#            for i in range(yX):
#                self.clusters[i] = np.array([i], dtype = np.int32)
        
        if self.isDraw:
            mark_col = np.array(['r', 'g', 'b', 'y', 'c', 'm', 'k'])
            drawing_canvas = self.initialize_canvas_graph("GFMM - AGGLO-SM-Fast version", xX)
                
            # plot initial hyperbox
            Vt, Wt = self.pcatransform()
            color_ = np.empty(len(self.classId), dtype = object)
            for c in range(len(self.classId)):
                color_[c] = mark_col[self.classId[c]]
            drawbox(Vt, Wt, drawing_canvas, color_)
            self.delay()
        
        # training
        isTraining = True
        while isTraining:
            isTraining = False
            
            # calculate class masks
            yX, xX = self.V.shape
            labList = np.unique(self.classId)[::-1]
            clMask = np.zeros(shape = (yX, len(labList)), dtype = np.bool)
            for i in range(len(labList)):
                clMask[:, i] = self.classId == labList[i]
        
        	# calculate pairwise memberships *ONLY* within each class (faster!)
            b = np.zeros(shape = (yX, yX))
            
            for i in range(len(labList)):
                Vi = self.V[clMask[:, i]] # get bounds of patterns with class label i
                Wi = self.W[clMask[:, i]]
                clSize = np.sum(clMask[:, i]) # get number of patterns of class i
                clIdxs = np.nonzero(clMask[:, i])[0] # get position of patterns with class label i in the training set
                
                if self.simil == 'short':
                    for j in range(clSize):
                        b[clIdxs[j], clIdxs] = membership_gfmm(Wi[j], Vi[j], Vi, Wi, self.gamma, self.oper)
                elif self.simil == 'long':
                    for j in range(clSize):
                        b[clIdxs[j], clIdxs] = membership_gfmm(Vi[j], Wi[j], Wi, Vi, self.gamma, self.oper)
                else:
                    for j in range(clSize):
                        b[clIdxs[j], clIdxs] = membership_gfmm(Vi[j], Wi[j], Vi, Wi, self.gamma, self.oper)
                
            if yX == 1:
                maxb = np.array([])
            else:
                maxb = self.split_similarity_maxtrix(b, self.sing, False)
                if len(maxb) > 0:
                    maxb = maxb[(maxb[:, 2] >= self.bthres), :]
                    
                    if len(maxb) > 0:
                        # sort maxb in the decending order following the last column
                        idx_smaxb = np.argsort(maxb[:, 2])[::-1]
                        maxb = np.hstack((maxb[idx_smaxb, 0].reshape(-1, 1), maxb[idx_smaxb, 1].reshape(-1, 1), maxb[idx_smaxb, 2].reshape(-1, 1)))
                        #maxb = maxb[idx_smaxb]
            
            while len(maxb) > 0:
                curmaxb = maxb[0, :] # current position handling
                
                # calculate new coordinates of curmaxb(0)-th hyperbox by including curmaxb(1)-th box, scrap the latter and leave the rest intact
                row1 = int(curmaxb[0])
                row2 = int(curmaxb[1])
                newV = np.concatenate((self.V[0:row1, :], np.minimum(self.V[row1, :], self.V[row2, :]).reshape(1, -1), self.V[row1 + 1:row2, :], self.V[row2 + 1:, :]), axis=0)
                newW = np.concatenate((self.W[0:row1, :], np.maximum(self.W[row1, :], self.W[row2, :]).reshape(1, -1), self.W[row1 + 1:row2, :], self.W[row2 + 1:, :]), axis=0)
                newClassId = np.concatenate((self.classId[0:row2], self.classId[row2 + 1:]))
                
#                index_remain = np.ones(len(self.classId)).astype(np.bool)
#                index_remain[row2] = False
#                newV = self.V[index_remain]
#                newW = self.W[index_remain]
#                newClassId = self.classId[index_remain]
#                if row1 < row2:
#                    tmp_row = row1
#                else:
#                    tmp_row = row1 - 1
#                newV[tmp_row] = np.minimum(self.V[row1], self.V[row2])
#                newW[tmp_row] = np.maximum(self.W[row1], self.W[row2])
                       
                
                # adjust the hyperbox if no overlap and maximum hyperbox size is not violated
                if (not is_overlap(newV, newW, int(curmaxb[0]), newClassId)) and (((newW[int(curmaxb[0])] - newV[int(curmaxb[0])]) <= self.teta).all() == True):
                    isTraining = True
                    self.V = newV
                    self.W = newW
                    self.classId = newClassId
                    
#                    self.cardin[int(curmaxb[0])] = self.cardin[int(curmaxb[0])] + self.cardin[int(curmaxb[1])]
#                    self.cardin = np.append(self.cardin[0:int(curmaxb[1])], self.cardin[int(curmaxb[1]) + 1:])
#                            
#                    self.clusters[int(curmaxb[0])] = np.append(self.clusters[int(curmaxb[0])], self.clusters[int(curmaxb[1])])
#                    self.clusters = np.append(self.clusters[0:int(curmaxb[1])], self.clusters[int(curmaxb[1]) + 1:])
#                    
                    # remove joined pair from the list as well as any pair with lower membership and consisting of any of joined boxes
                    mask = (maxb[:, 0] != int(curmaxb[0])) & (maxb[:, 1] != int(curmaxb[0])) & (maxb[:, 0] != int(curmaxb[1])) & (maxb[:, 1] != int(curmaxb[1])) & (maxb[:, 2] >= curmaxb[2])
                    maxb = maxb[mask, :]
                    
                    # update indexes to accomodate removed hyperbox
                    # indices of V and W larger than curmaxb(1,2) are decreased 1 by the element whithin the location curmaxb(1,2) was removed 
                    if len(maxb) > 0:
                        maxb[maxb[:, 0] > int(curmaxb[1]), 0] = maxb[maxb[:, 0] > int(curmaxb[1]), 0] - 1
                        maxb[maxb[:, 1] > int(curmaxb[1]), 1] = maxb[maxb[:, 1] > int(curmaxb[1]), 1] - 1
                            
                    if self.isDraw:
                        Vt, Wt = self.pcatransform()
                        color_ = np.empty(len(self.classId), dtype = object)
                        for c in range(len(self.classId)):
                            color_[c] = mark_col[self.classId[c]]
                        drawing_canvas.cla()
                        drawbox(Vt, Wt, drawing_canvas, color_)
                        self.delay()
                else:
                    maxb = maxb[1:, :]  # scrap examined pair from the list
        
        time_end = time.perf_counter()
        self.elapsed_training_time = time_end - time_start
         
        return self

                  
if __name__ == '__main__':
    """
    INPUT parameters from command line
    arg1: + 1 - training and testing datasets are located in separated files
          + 2 - training and testing datasets are located in the same files
    arg2: path to file containing the training dataset (arg1 = 1) or both training and testing datasets (arg1 = 2)
    arg3: + path to file containing the testing dataset (arg1 = 1)
          + percentage of the training dataset in the input file
    arg4: + True: drawing hyperboxes during the training process
          + False: no drawing
    arg5: + Maximum size of hyperboxes (teta, default: 1)
    arg6: + gamma value (default: 1)
    arg7: + Similarity threshold (default: 0.5)
    arg8: + Similarity measure: 'short', 'long' or 'mid' (default: 'mid')
    arg9: + operation used to compute membership value: 'min' or 'prod' (default: 'min')
    arg10: + do normalization of datasets or not? True: Normilize, False: No normalize (default: True)
    arg11: + range of input values after normalization (default: [0, 1])   
    arg12: + Use 'min' or 'max' (default) membership in case of assymetric similarity measure (simil='mid')
    """
    # Init default parameters
    if len(sys.argv) < 5:
        isDraw = False
    else:
        isDraw = string_to_boolean(sys.argv[4])
    
    if len(sys.argv) < 6:
        teta = 1    
    else:
        teta = float(sys.argv[5])
    
    if len(sys.argv) < 7:
        gamma = 1
    else:
        gamma = float(sys.argv[6])
    
    if len(sys.argv) < 8:
        bthres = 0.5
    else:
        bthres = float(sys.argv[7])
    
    if len(sys.argv) < 9:
        simil = 'mid'
    else:
        simil = sys.argv[8]
    
    if len(sys.argv) < 10:
        oper = 'min'
    else:
        oper = sys.argv[9]
    
    if len(sys.argv) < 11:
        isNorm = True
    else:
        isNorm = string_to_boolean(sys.argv[10])
    
    if len(sys.argv) < 12:
        norm_range = [0, 1]
    else:
        norm_range = ast.literal_eval(sys.argv[11])
        
    if len(sys.argv) < 13:
        sing = 'max'
    else:
        sing = sys.argv[12]
        
    if sys.argv[1] == '1':
        training_file = sys.argv[2]
        testing_file = sys.argv[3]

        # Read training file
        Xtr, X_tmp, patClassIdTr, pat_tmp = load_dataset(training_file, 1, False)
        # Read testing file
        X_tmp, Xtest, pat_tmp, patClassIdTest = load_dataset(testing_file, 0, False)
    
    else:
        dataset_file = sys.argv[2]
        percent_Training = float(sys.argv[3])
        Xtr, Xtest, patClassIdTr, patClassIdTest = load_dataset(dataset_file, percent_Training, False)
    
    classifier = BatchGFMMV1(gamma, teta, bthres, simil, sing, isDraw, oper, isNorm, norm_range)
    classifier.fit(Xtr, Xtr, patClassIdTr)
    
    # Testing
    print("-- Testing --")
    result = classifier.predict(Xtest, Xtest, patClassIdTest)
    if result != None:
        print("Number of wrong predicted samples = ", result.summis)
        numTestSample = Xtest.shape[0]
        print("Error Rate = ", np.round(result.summis / numTestSample * 100, 2), "%")
        print("Training Time = ", classifier.elapsed_training_time)