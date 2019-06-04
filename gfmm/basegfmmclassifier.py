# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 10:22:19 2018

@author: Thanh Tung Khuat

Base GFMM classifier
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from gfmm.classification import predict
from functionhelper.matrixhelper import delete_const_dims, pca_transform
from functionhelper.preprocessinghelper import normalize
from functionhelper.membershipcalc import memberG

class BaseGFMMClassifier(object):

    def __init__(self, gamma = 1, teta = 1, isDraw = False, oper = 'min', isNorm = True, norm_range = [0, 1]):
        self.gamma = gamma
        self.teta = teta
        self.isDraw = isDraw
        self.oper = oper
        self.isNorm = isNorm

        self.V = np.array([])
        self.W = np.array([])
        self.classId = np.array([])

        # parameters for data normalization
        self.loLim = norm_range[0]
        self.hiLim = norm_range[1]
        self.mins = []
        self.maxs = []
        self.delayConstant = 0.001 # delay time period to display hyperboxes on the canvas

    def dataPreprocessing(self, X_l, X_u):
        """
        Preprocess data: delete constant dimensions, Normalize input samples if needed

        INPUT:
            X_l          Input data lower bounds (rows = objects, columns = features)
            X_u          Input data upper bounds (rows = objects, columns = features)

        OUTPUT
            X_l, X_u were preprocessed
        """

        # delete constant dimensions
        #X_l, X_u = delete_const_dims(X_l, X_u)

        # Normalize input samples if needed
        if X_l.min() < self.loLim or X_u.min() < self.loLim or X_u.max() > self.hiLim or X_l.max() > self.hiLim:
            self.mins = X_l.min(axis = 0) # get min value of each feature
            self.maxs = X_u.max(axis = 0) # get max value of each feature
            X_l = normalize(X_l, [self.loLim, self.hiLim])
            X_u = normalize(X_u, [self.loLim, self.hiLim])
        else:
            self.isNorm = False
            self.mins = []
            self.maxs = []

        return (X_l, X_u)


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


    def split_similarity_maxtrix(self, A, asimil_type = 'max', isSort = True):
        """
        Split the similarity matrix A into the maxtrix with three columns:
            + First column is row indices of A
            + Second column is column indices of A
            + Third column is the values corresponding to the row and column

        if isSort = True, the third column is sorted in the descending order

            INPUT
                A               Degrees of membership of input patterns (each row is the output from memberG function)
                asimil_type     Use 'min' or 'max' (default) memberhsip in case of assymetric similarity measure (simil='mid')
                isSort          Sorting flag

            OUTPUT
                The output as mentioned above
        """

        # get min/max memberships from triu and tril of memberhsip matrix which might not be symmetric (simil=='mid')
        if asimil_type == 'min':
            transformedA = np.minimum(np.flipud(np.rot90(np.tril(A, -1))), np.triu(A, 1))  # rotate tril to align it with triu for min (max) operation
        else:
            transformedA = np.maximum(np.flipud(np.rot90(np.tril(A, -1))), np.triu(A, 1))

        ind_rows, ind_columns = np.nonzero(transformedA)
        values = transformedA[ind_rows, ind_columns]

        if isSort == True:
            ind_SortedTransformedA = np.argsort(values)[::-1]
            sortedTransformedA = values[ind_SortedTransformedA]
            result = np.concatenate((ind_rows[ind_SortedTransformedA][:, np.newaxis], ind_columns[ind_SortedTransformedA][:, np.newaxis], sortedTransformedA[:, np.newaxis]), axis=1)
        else:
            result = np.concatenate((ind_rows[:, np.newaxis], ind_columns[:, np.newaxis], values[:, np.newaxis]), axis=1)

        return result


    def predict(self, Xl_Test, Xu_Test, patClassIdTest):
        """
        Perform classification

            result = predict(Xl_Test, Xu_Test, patClassIdTest)

        INPUT:
            Xl_Test             Test data lower bounds (rows = objects, columns = features)
            Xu_Test             Test data upper bounds (rows = objects, columns = features)
            patClassIdTest	     Test data class labels (crisp)

        OUTPUT:
            result        A object with Bunch datatype containing all results as follows:
                          + summis           Number of misclassified objects
                          + misclass         Binary error map
                          + sumamb           Number of objects with maximum membership in more than one class
                          + out              Soft class memberships
                          + mem              Hyperbox memberships
        """
        #Xl_Test, Xu_Test = delete_const_dims(Xl_Test, Xu_Test)
        # Normalize testing dataset if training datasets were normalized
        if len(self.mins) > 0:
            noSamples = Xl_Test.shape[0]
            Xl_Test = self.loLim + (self.hiLim - self.loLim) * (Xl_Test - np.ones((noSamples, 1)) * self.mins) / (np.ones((noSamples, 1)) * (self.maxs - self.mins))
            Xu_Test = self.loLim + (self.hiLim - self.loLim) * (Xu_Test - np.ones((noSamples, 1)) * self.mins) / (np.ones((noSamples, 1)) * (self.maxs - self.mins))

            if Xl_Test.min() < self.loLim or Xu_Test.min() < self.loLim or Xl_Test.max() > self.hiLim or Xu_Test.max() > self.hiLim:
                print('Test sample falls outside', self.loLim, '-', self.hiLim, 'interval')
                print('Number of original samples = ', noSamples)

                # only keep samples within the interval loLim-hiLim
                indXl_good = np.where((Xl_Test >= self.loLim).all(axis = 1) & (Xl_Test <= self.hiLim).all(axis = 1))[0]
                indXu_good = np.where((Xu_Test >= self.loLim).all(axis = 1) & (Xu_Test <= self.hiLim).all(axis = 1))[0]
                indKeep = np.intersect1d(indXl_good, indXu_good)

                Xl_Test = Xl_Test[indKeep, :]
                Xu_Test = Xu_Test[indKeep, :]

                print('Number of kept samples =', Xl_Test.shape[0])
                #return

        # do classification
        result = None

        if Xl_Test.shape[0] > 0:
            result = predict(self.V, self.W, self.classId, Xl_Test, Xu_Test, patClassIdTest, self.gamma, self.oper)

        return result
    
    def pruning_val(self, XlT, XuT, patClassIdTest, accuracy_threshold = 0.5):
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
        yX = XlT.shape[0]
        mem = np.zeros((yX, self.V.shape[0]))
        no_predicted_samples_hyperboxes = np.zeros((len(self.classId), 2))
        # classifications
        for i in range(yX):
            mem[i, :] = memberG(XlT[i, :], XuT[i, :], self.V, self.W, self.gamma, self.oper) # calculate memberships for all hyperboxes
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
        accuracy_larger_half_keep_nojoin = np.zeros(tmp_no_box).astype(np.bool)
        for i in range(tmp_no_box):
            if (no_predicted_samples_hyperboxes[i, 0] + no_predicted_samples_hyperboxes[i, 1] != 0) and no_predicted_samples_hyperboxes[i, 0] / (no_predicted_samples_hyperboxes[i, 0] + no_predicted_samples_hyperboxes[i, 1]) >= accuracy_threshold:
                accuracy_larger_half[i] = True
                accuracy_larger_half_keep_nojoin[i] = True
            if (no_predicted_samples_hyperboxes[i, 0] + no_predicted_samples_hyperboxes[i, 1] == 0):
                accuracy_larger_half_keep_nojoin[i] = True
        
        # keep one hyperbox for class prunned all
        current_classes = np.unique(self.classId)
        class_tmp = self.classId[accuracy_larger_half]
        for c in current_classes:
            if c not in class_tmp:
                pos = np.nonzero(self.classId == c)
                id_kept = np.random.randint(len(pos))
                # keep pos[id_kept]
                accuracy_larger_half[pos[id_kept]] = True
        
        V_prun_remove = self.V[accuracy_larger_half]
        W_prun_remove = self.W[accuracy_larger_half]
        classId_prun_remove = self.classId[accuracy_larger_half]
        
        W_prun_keep = self.W[accuracy_larger_half_keep_nojoin]
        V_prun_keep = self.V[accuracy_larger_half_keep_nojoin]
        classId_prun_keep = self.classId[accuracy_larger_half_keep_nojoin]
        
        result_prun_remove = predict(V_prun_remove, W_prun_remove, classId_prun_remove, XlT, XuT, patClassIdTest, self.gamma, self.oper)
        result_prun_keep_nojoin = predict(V_prun_keep, W_prun_keep, classId_prun_keep, XlT, XuT, patClassIdTest, self.gamma, self.oper)
        
		# Pruning
        if (result_prun_remove.summis <= result_prun_keep_nojoin.summis):
            self.V = V_prun_remove
            self.W = W_prun_remove
            self.classId = classId_prun_remove
        else:
            self.V = V_prun_keep
            self.W = W_prun_keep
            self.classId = classId_prun_keep
