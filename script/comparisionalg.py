import sys, os
from os.path import dirname
root_path = dirname(dirname(os.getcwd()))
sys.path.insert(0, root_path) # insert root directory to environmental variables

import time
import numpy as np

from gfmm.accelbatchgfmm import AccelBatchGFMM
from gfmm.onlinegfmm import OnlineGFMM
from efmnn.efmnnclassification import EFMNNClassification
from efmnn.knefmnnclassification import KNEFMNNClassification
from fmnn.fmnnclassification import FMNNClassification

from functionhelper.preprocessinghelper import loadDataset
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

tetaRange = [0.06, 0.1, 0.16, 0.2, 0.26, 0.3, 0.36, 0.4, 0.46, 0.5, 0.56, 0.6, 0.66, 0.7, 0.76, 0.8]

def running_knn(n_neighbors, xTr, patClassIdTr, xVal, patClassIdVal):
    knnClassifier = KNeighborsClassifier(n_neighbors = n_neighbors)
    
    knnClassifier.fit(xTr, patClassIdTr)
    predictedKNN = knnClassifier.predict(xVal)     
    errorRateKNN = (patClassIdVal != predictedKNN).sum() / len(patClassIdVal)
    
    return errorRateKNN

def running_gaussian_naive_bayes(xTr, patClassIdTr, xVal, patClassIdVal):
    gnb = GaussianNB()
    gnb.fit(xTr, patClassIdTr)
    
    predictedGNB = gnb.predict(xVal)          
    errorGNB = (patClassIdVal != predictedGNB).sum() / len(patClassIdVal)
    
    return errorGNB

def running_svm(xTr, patClassIdTr, xVal, patClassIdVal, gamma = 0.0001, C = 1):
    svclassifier = SVC(kernel='rbf', decision_function_shape='ovo', C = C, gamma = gamma) 
    svclassifier.fit(xTr, patClassIdTr)
    
    predictedSVC = svclassifier.predict(xVal)
    errorSVC = (patClassIdVal != predictedSVC).sum() / len(patClassIdVal)
    
    return errorSVC

def running_decision_tree(xTr, patClassIdTr, xVal, patClassIdVal, max_depth = None):
    decisionTree = DecisionTreeClassifier(max_depth = max_depth)
    decisionTree.fit(xTr, patClassIdTr) 
    
    predictedDT = decisionTree.predict(xVal)
    errorDT = (patClassIdVal != predictedDT).sum() / len(patClassIdVal) 
    
    return errorDT

def running_online_gfmm(xTr, patClassIdTr, xVal, patClassIdVal, teta = 0.26):
    olnClassifier = OnlineGFMM(gamma = 1, teta = teta, tMin = teta, isDraw = False, oper = 'min', isNorm = False)
    olnClassifier.fit(xTr, xTr, patClassIdTr)
            
    result = olnClassifier.predict(xVal, xVal, patClassIdVal)
    err = 1
    if result != None:
        err = result.summis / len(patClassIdVal)
    
    return err

def running_agglo_2(xTr, patClassIdTr, xVal, patClassIdVal, teta = 0.26, sigma = 0.5, simil = 'short'):
    accelClassifier = AccelBatchGFMM(gamma = 1, teta = teta, bthres = sigma, simil = simil, sing = 'max', isDraw = False, oper = 'min', isNorm = False)
    accelClassifier.fit(xTr, xTr, patClassIdTr)
            
    result = accelClassifier.predict(xVal, xVal, patClassIdVal)
    err = 1
    if result != None:
        err = result.summis / len(patClassIdVal)
    
    return err

def running_fmnn(xTr, patClassIdTr, xVal, patClassIdVal, teta = 0.26):
    fmnnClassifier = FMNNClassification(gamma = 1, teta = teta, isDraw = False, isNorm = False)
    fmnnClassifier.fit(xTr, patClassIdTr)
            
    result = fmnnClassifier.predict(xVal, patClassIdVal)
    err = 1
    if result != None:
        err = result.summis / len(patClassIdVal)
    
    return err

def running_efmnn(xTr, patClassIdTr, xVal, patClassIdVal, teta = 0.26):
    efmnnClassifier = EFMNNClassification(gamma = 1, teta = teta, isDraw = False, isNorm = False)
    efmnnClassifier.fit(xTr, patClassIdTr)
            
    result = efmnnClassifier.predict(xVal, patClassIdVal)
    err = 1
    if result != None:
        err = result.summis / len(patClassIdVal)
    
    return err

def running_knefmnn (xTr, patClassIdTr, xVal, patClassIdVal, teta = 0.26, K = 10):
    knefmnnClassifier = KNEFMNNClassification(gamma = 1, teta = teta, isDraw = False, isNorm = False)
    knefmnnClassifier.fit(xTr, patClassIdTr, K)
            
    err = 1
    result = knefmnnClassifier.predict(xVal, patClassIdVal)
    if result != None:
        err = result.summis / len(patClassIdVal)
    
    return err

def find_optimal_k_knn(fold1Data, fold1Label, fold2Data, fold2Label, fold3Data, fold3Label, kRange = [3, 30]):
    dicErrorK = {key:0 for key in range(kRange[0], kRange[1] + 1)}
    
    for k_value in range(kRange[0], kRange[1] + 1):
        errorK = 0
        # using fold1 as validation, fold2 and fold3 are training sets
        trainingData = np.vstack((fold2Data, fold3Data))
        trainingLabel = np.append(fold2Label, fold3Label)
        
        errorK = errorK + running_knn(k_value, trainingData, trainingLabel, fold1Data, fold1Label)
        
        # using fold2 as validation, fold1 and fold3 are training sets
        trainingData = np.vstack((fold1Data, fold3Data))
        trainingLabel = np.append(fold1Label, fold3Label)
        
        errorK = errorK + running_knn(k_value, trainingData, trainingLabel, fold2Data, fold2Label)
        
        # using fold3 as validation, fold1 and fold2 are training sets
        trainingData = np.vstack((fold1Data, fold2Data))
        trainingLabel = np.append(fold1Label, fold2Label)
        
        errorK = errorK + running_knn(k_value, trainingData, trainingLabel, fold3Data, fold3Label)
        
        dicErrorK[k_value] = errorK
    
    # Find K value (key in dicErrorK) with the minimum value
    k = min(dicErrorK, key=dicErrorK.get)
    
    return k
    
def find_optimal_param_svm(fold1Data, fold1Label, fold2Data, fold2Label, fold3Data, fold3Label):
    cRange = [2**-5, 2**-3, 2**-1, 2**1, 2**3, 2**5, 2**7, 2**9, 2**11, 2**13, 2**15]
    gammaRange = [2**-15, 2**-13, 2**-11, 2**-9, 2**-7, 2**-5, 2**-3, 2**-1, 2**1, 2**3]
    
    minErr = 2
    cOpt = 2**-5
    gammaOpt = 2**-15
    
    for c in cRange:
        for gamma in gammaRange:
            error = 0
            # using fold1 as validation, fold2 and fold3 are training sets
            trainingData = np.vstack((fold2Data, fold3Data))
            trainingLabel = np.append(fold2Label, fold3Label)
        
            error = error + running_svm(trainingData, trainingLabel, fold1Data, fold1Label, gamma = gamma, C = c)
        
            # using fold2 as validation, fold1 and fold3 are training sets
            trainingData = np.vstack((fold1Data, fold3Data))
            trainingLabel = np.append(fold1Label, fold3Label)
            
            error = error + running_svm(trainingData, trainingLabel, fold2Data, fold2Label, gamma = gamma, C = c)
            
            # using fold3 as validation, fold1 and fold2 are training sets
            trainingData = np.vstack((fold1Data, fold2Data))
            trainingLabel = np.append(fold1Label, fold2Label)
            
            error = error + running_svm(trainingData, trainingLabel, fold3Data, fold3Label, gamma = gamma, C = c)
            
            error = error / 3
            
            if minErr > error:
                minErr = error
                cOpt = c
                gammaOpt = gamma
                
    return (cOpt, gammaOpt)

def find_optimal_param_decision_tree(fold1Data, fold1Label, fold2Data, fold2Label, fold3Data, fold3Label, maxDepthRange = [3, 30]):
    maxDepthRange = list(range(maxDepthRange[0], maxDepthRange[1] + 1))
    maxDepthRange.append(None)
    
    minErr = 2
    maxDepthOpt = None
    
    for maxDepth in maxDepthRange:      
        error = 0
        # using fold1 as validation, fold2 and fold3 are training sets
        trainingData = np.vstack((fold2Data, fold3Data))
        trainingLabel = np.append(fold2Label, fold3Label)
    
        error = error + running_decision_tree(trainingData, trainingLabel, fold1Data, fold1Label, max_depth = maxDepth)
    
        # using fold2 as validation, fold1 and fold3 are training sets
        trainingData = np.vstack((fold1Data, fold3Data))
        trainingLabel = np.append(fold1Label, fold3Label)
        
        error = error + running_decision_tree(trainingData, trainingLabel, fold2Data, fold2Label, max_depth = maxDepth)
        
        # using fold3 as validation, fold1 and fold2 are training sets
        trainingData = np.vstack((fold1Data, fold2Data))
        trainingLabel = np.append(fold1Label, fold2Label)
        
        error = error + running_decision_tree(trainingData, trainingLabel, fold3Data, fold3Label, max_depth = maxDepth)
        
        error = error / 3
        
        if minErr > error:
            minErr = error
            maxDepthOpt = maxDepth
    
    return maxDepthOpt    
 
def find_optimal_param_online_gfmm(fold1Data, fold1Label, fold2Data, fold2Label, fold3Data, fold3Label):
    minErr = 2
    tetaOpt = 0.26
    
    for teta in tetaRange:
        error = 0
        # using fold1 as validation, fold2 and fold3 are training sets
        trainingData = np.vstack((fold2Data, fold3Data))
        trainingLabel = np.append(fold2Label, fold3Label)
    
        error = error + running_online_gfmm(trainingData, trainingLabel, fold1Data, fold1Label, teta)
    
        # using fold2 as validation, fold1 and fold3 are training sets
        trainingData = np.vstack((fold1Data, fold3Data))
        trainingLabel = np.append(fold1Label, fold3Label)
        
        error = error + running_online_gfmm(trainingData, trainingLabel, fold2Data, fold2Label, teta)
        
        # using fold3 as validation, fold1 and fold2 are training sets
        trainingData = np.vstack((fold1Data, fold2Data))
        trainingLabel = np.append(fold1Label, fold2Label)
        
        error = error + running_online_gfmm(trainingData, trainingLabel, fold3Data, fold3Label, teta)
        
        error = error / 3
        
        if minErr > error:
            minErr = error
            tetaOpt = teta
    
    return tetaOpt

def find_optimal_param_agglo_2(fold1Data, fold1Label, fold2Data, fold2Label, fold3Data, fold3Label):
    minErr = 2
    tetaOpt = 0.26
    
    for teta in tetaRange:
        error = 0
        # using fold1 as validation, fold2 and fold3 are training sets
        trainingData = np.vstack((fold2Data, fold3Data))
        trainingLabel = np.append(fold2Label, fold3Label)
    
        error = error + running_agglo_2(trainingData, trainingLabel, fold1Data, fold1Label, teta)
    
        # using fold2 as validation, fold1 and fold3 are training sets
        trainingData = np.vstack((fold1Data, fold3Data))
        trainingLabel = np.append(fold1Label, fold3Label)
        
        error = error + running_agglo_2(trainingData, trainingLabel, fold2Data, fold2Label, teta)
        
        # using fold3 as validation, fold1 and fold2 are training sets
        trainingData = np.vstack((fold1Data, fold2Data))
        trainingLabel = np.append(fold1Label, fold2Label)
        
        error = error + running_agglo_2(trainingData, trainingLabel, fold3Data, fold3Label, teta)
        
        error = error / 3
        
        if minErr > error:
            minErr = error
            tetaOpt = teta
    
    return tetaOpt

def find_optimal_param_fmnn(fold1Data, fold1Label, fold2Data, fold2Label, fold3Data, fold3Label):
    minErr = 2
    tetaOpt = 0.26
    
    for teta in tetaRange:
        error = 0
        # using fold1 as validation, fold2 and fold3 are training sets
        trainingData = np.vstack((fold2Data, fold3Data))
        trainingLabel = np.append(fold2Label, fold3Label)
    
        error = error + running_fmnn(trainingData, trainingLabel, fold1Data, fold1Label, teta)
    
        # using fold2 as validation, fold1 and fold3 are training sets
        trainingData = np.vstack((fold1Data, fold3Data))
        trainingLabel = np.append(fold1Label, fold3Label)
        
        error = error + running_fmnn(trainingData, trainingLabel, fold2Data, fold2Label, teta)
        
        # using fold3 as validation, fold1 and fold2 are training sets
        trainingData = np.vstack((fold1Data, fold2Data))
        trainingLabel = np.append(fold1Label, fold2Label)
        
        error = error + running_fmnn(trainingData, trainingLabel, fold3Data, fold3Label, teta)
        
        error = error / 3
        
        if minErr > error:
            minErr = error
            tetaOpt = teta
    
    return tetaOpt

def find_optimal_param_efmnn(fold1Data, fold1Label, fold2Data, fold2Label, fold3Data, fold3Label):
    minErr = 2
    tetaOpt = 0.26
    
    for teta in tetaRange:
        error = 0
        # using fold1 as validation, fold2 and fold3 are training sets
        trainingData = np.vstack((fold2Data, fold3Data))
        trainingLabel = np.append(fold2Label, fold3Label)
    
        error = error + running_efmnn(trainingData, trainingLabel, fold1Data, fold1Label, teta)
    
        # using fold2 as validation, fold1 and fold3 are training sets
        trainingData = np.vstack((fold1Data, fold3Data))
        trainingLabel = np.append(fold1Label, fold3Label)
        
        error = error + running_efmnn(trainingData, trainingLabel, fold2Data, fold2Label, teta)
        
        # using fold3 as validation, fold1 and fold2 are training sets
        trainingData = np.vstack((fold1Data, fold2Data))
        trainingLabel = np.append(fold1Label, fold2Label)
        
        error = error + running_efmnn(trainingData, trainingLabel, fold3Data, fold3Label, teta)
        
        error = error / 3
        
        if minErr > error:
            minErr = error
            tetaOpt = teta
    
    return tetaOpt

def find_optimal_param_knefmnn(fold1Data, fold1Label, fold2Data, fold2Label, fold3Data, fold3Label):
    minErr = 2
    tetaOpt = 0.26
    kOpt = 10
    kRange = list(range(2, 11))
    
    for k in kRange:
        for teta in tetaRange:
            error = 0
            # using fold1 as validation, fold2 and fold3 are training sets
            trainingData = np.vstack((fold2Data, fold3Data))
            trainingLabel = np.append(fold2Label, fold3Label)
        
            error = error + running_knefmnn(trainingData, trainingLabel, fold1Data, fold1Label, teta, k)
        
            # using fold2 as validation, fold1 and fold3 are training sets
            trainingData = np.vstack((fold1Data, fold3Data))
            trainingLabel = np.append(fold1Label, fold3Label)
            
            error = error + running_knefmnn(trainingData, trainingLabel, fold2Data, fold2Label, teta, k)
            
            # using fold3 as validation, fold1 and fold2 are training sets
            trainingData = np.vstack((fold1Data, fold2Data))
            trainingLabel = np.append(fold1Label, fold2Label)
            
            error = error + running_knefmnn(trainingData, trainingLabel, fold3Data, fold3Label, teta, k)
            
            error = error / 3
            
            if minErr > error:
                minErr = error
                tetaOpt = teta
                kOpt = k
    
    return (tetaOpt, kOpt)


if __name__ == '__main__':
    
    save_online_gfmm_result_folder_path = root_path + '/experiment/fmm/online_gfmm'
    save_accel_agglo_result_folder_path = root_path + '/experiment/fmm/accel_agglo'
    save_efmnn_result_folder_path = root_path + '/experiment/fmm/efmnn'
    save_knefmnn_result_folder_path = root_path + '/experiment/fmm/knefmnn'
    save_fmnn_result_folder_path = root_path + '/experiment/fmm/fmnn'
    
    save_result_KNN_folder_path = root_path + '/experiment/otherml/knn'
    save_result_SVM_folder_path = root_path + '/experiment/otherml/svm'
    save_result_DT_folder_path = root_path + '/experiment/otherml/decision_tree'
    save_result_NB_folder_path = root_path + '/experiment/otherml/naive_bayes'
    
    dataset_path = root_path + '/dataset'
    
    dataset_names = ['circle_dps', 'complex9_dps', 'DiagnosticBreastCancer_dps', 'glass_dps', 'ionosphere_dps', 'iris_dps', 'segmentation_dps', 'spherical_5_2_dps', 'spiral_dps', 'thyroid_dps', 'wine_dps', 'yeast_dps', 'zelnik6_dps', 'ringnorm_dps', 'twonorm_dps', 'waveform_dps']

    fold_index = np.array([1, 2, 3, 4])
    
    for dt in range(len(dataset_names)):
        #try:
        print('Current dataset: ', dataset_names[dt])
        fold1File = dataset_path + '/' + dataset_names[dt] + '_1.dat'
        fold2File = dataset_path + '/' + dataset_names[dt] + '_2.dat'
        fold3File = dataset_path + '/' + dataset_names[dt] + '_3.dat'
        fold4File = dataset_path + '/' + dataset_names[dt] + '_4.dat'
        
        # Read data file
        fold1Data, _, fold1Label, _ = loadDataset(fold1File, 1, False)
        fold2Data, _, fold2Label, _ = loadDataset(fold2File, 1, False)
        fold3Data, _, fold3Label, _ = loadDataset(fold3File, 1, False)
        fold4Data, _, fold4Label, _ = loadDataset(fold4File, 1, False)
        
        numhyperbox_online_gfmm_save = np.array([])
        training_time_online_gfmm_save = np.array([])
        testing_error_online_gfmm_save = np.array([])
        optimization_value_online_gfmm_save = np.array([], dtype=np.str)
        optimization_time_online_gfmm_save = np.array([])
        
        numhyperbox_fmnn_save = np.array([])
        training_time_fmnn_save = np.array([])
        testing_error_fmnn_save = np.array([])
        optimization_value_fmnn_save = np.array([], dtype=np.str)
        optimization_time_fmnn_save = np.array([])
        
        numhyperbox_efmnn_save = np.array([])
        training_time_efmnn_save = np.array([])
        testing_error_efmnn_save = np.array([])
        optimization_value_efmnn_save = np.array([], dtype=np.str)
        optimization_time_efmnn_save = np.array([])
        
        numhyperbox_knefmnn_save = np.array([])
        training_time_knefmnn_save = np.array([])
        testing_error_knefmnn_save = np.array([])
        optimization_value_knefmnn_save = np.array([], dtype=np.str)
        optimization_time_knefmnn_save = np.array([])
        
        numhyperbox_accel_agglo_save = np.array([], dtype=np.int64)
        training_time_accel_agglo_save = np.array([])
        testing_error_accel_agglo_save = np.array([])
        optimization_value_accel_agglo_save = np.array([], dtype=np.str)
        optimization_time_accel_agglo_save = np.array([])
        
        training_time_knn_save = np.array([])
        testing_error_knn_save = np.array([])
        optimization_value_knn_save = np.array([], dtype=np.str)
        optimization_time_knn_save = np.array([])
        
        training_time_svm_save = np.array([])
        testing_error_svm_save = np.array([])
        optimization_value_svm_save = np.array([], dtype=np.str)
        optimization_time_svm_save = np.array([])
        
        training_time_gnb_save = np.array([])
        testing_error_gnb_save = np.array([])
        
        training_time_dt_save = np.array([])
        testing_error_dt_save = np.array([])
        optimization_value_dt_save = np.array([], dtype=np.str)
        optimization_time_dt_save = np.array([])
        
        # loop through 4 folds
        for fo in range(4):
            if fo == 0:
                # fold 1 is testing set
                trainingData = np.vstack((fold2Data, fold3Data, fold4Data))
                testingData = fold1Data
                trainingLabel = np.hstack((fold2Label, fold3Label, fold4Label))
                testingLabel = fold1Label
                numTestSample = len(testingLabel)
                f1dt, f2dt, f3dt = fold2Data, fold3Data, fold4Data
                f1lb, f2lb, f3lb = fold2Label, fold3Label, fold4Label
            elif fo == 1:
                # fold 2 is testing set
                trainingData = np.vstack((fold1Data, fold3Data, fold4Data))
                testingData = fold2Data
                trainingLabel = np.hstack((fold1Label, fold3Label, fold4Label))
                testingLabel = fold2Label
                numTestSample = len(testingLabel)
                f1dt, f2dt, f3dt = fold1Data, fold3Data, fold4Data
                f1lb, f2lb, f3lb = fold1Label, fold3Label, fold4Label
            elif fo == 2:
                # fold 3 is testing set
                trainingData = np.vstack((fold1Data, fold2Data, fold4Data))
                testingData = fold3Data
                trainingLabel = np.hstack((fold1Label, fold2Label, fold4Label))
                testingLabel = fold3Label
                numTestSample = len(testingLabel)
                f1dt, f2dt, f3dt = fold1Data, fold2Data, fold4Data
                f1lb, f2lb, f3lb = fold1Label, fold2Label, fold4Label
            else:
                # fold 4 is testing set
                trainingData = np.vstack((fold1Data, fold2Data, fold3Data))
                testingData = fold4Data
                trainingLabel = np.hstack((fold1Label, fold2Label, fold3Label))
                testingLabel = fold4Label
                numTestSample = len(testingLabel)
                f1dt, f2dt, f3dt = fold1Data, fold2Data, fold3Data
                f1lb, f2lb, f3lb = fold1Label, fold2Label, fold3Label
        
            # online GFMM
            time_start = time.perf_counter()
            tetaOnlnGFMM = find_optimal_param_online_gfmm(f1dt, f1lb, f2dt, f2lb, f3dt, f3lb)
            time_end = time.perf_counter()
            optimization_time_online_gfmm_save = np.append(optimization_time_online_gfmm_save, time_end - time_start)
            optimization_value_online_gfmm_save = np.append(optimization_value_online_gfmm_save, 'teta = ' + str(tetaOnlnGFMM))
            
            olnClassifier = OnlineGFMM(gamma = 1, teta = tetaOnlnGFMM, tMin = tetaOnlnGFMM, isDraw = False, oper = 'min', isNorm = False)
            olnClassifier.fit(trainingData, trainingData, trainingLabel)
            
            training_time_online_gfmm_save = np.append(training_time_online_gfmm_save, olnClassifier.elapsed_training_time)
            numhyperbox_online_gfmm_save = np.append(numhyperbox_online_gfmm_save, len(olnClassifier.classId))            
                
            result = olnClassifier.predict(testingData, testingData, testingLabel)
            if result != None:
                err = np.round(result.summis / numTestSample * 100, 3)
                testing_error_online_gfmm_save = np.append(testing_error_online_gfmm_save, err)
            
            # agglo-2
            time_start = time.perf_counter()
            tetaAGGLO2 = find_optimal_param_agglo_2(f1dt, f1lb, f2dt, f2lb, f3dt, f3lb)
            time_end = time.perf_counter()
            optimization_time_accel_agglo_save = np.append(optimization_time_accel_agglo_save, time_end - time_start)
            optimization_value_accel_agglo_save = np.append(optimization_value_accel_agglo_save, 'teta = ' + str(tetaAGGLO2))
            
            accelClassifier = AccelBatchGFMM(gamma = 1, teta = tetaAGGLO2, bthres = 0.5, simil = 'short', sing = 'max', isDraw = False, oper = 'min', isNorm = False)
            accelClassifier.fit(trainingData, trainingData, trainingLabel)
               
            training_time_accel_agglo_save = np.append(training_time_accel_agglo_save, accelClassifier.elapsed_training_time)
            numhyperbox_accel_agglo_save = np.append(numhyperbox_accel_agglo_save, len(accelClassifier.classId))
                
            result = accelClassifier.predict(testingData, testingData, testingLabel)
        
            if result != None:
                err = np.round(result.summis / numTestSample * 100, 3)
                testing_error_accel_agglo_save = np.append(testing_error_accel_agglo_save, err)
                
            # fmnn
            time_start = time.perf_counter()
            tetaFMNN = find_optimal_param_fmnn(f1dt, f1lb, f2dt, f2lb, f3dt, f3lb)
            time_end = time.perf_counter()
            optimization_time_fmnn_save = np.append(optimization_time_fmnn_save, time_end - time_start)
            optimization_value_fmnn_save = np.append(optimization_value_fmnn_save, 'teta = ' + str(tetaFMNN))
            
            fmnnClassifier = FMNNClassification(gamma = 1, teta = tetaFMNN, isDraw = False, isNorm = False)
            fmnnClassifier.fit(trainingData, trainingLabel)
            
            training_time_fmnn_save = np.append(training_time_fmnn_save, fmnnClassifier.elapsed_training_time)
            numhyperbox_fmnn_save = np.append(numhyperbox_fmnn_save, len(fmnnClassifier.classId))
                       
            result = fmnnClassifier.predict(testingData, testingLabel)

            if result != None:
                err = np.round(result.summis / numTestSample * 100, 3)
                testing_error_fmnn_save = np.append(testing_error_fmnn_save, err)
            
            # efmnn
            time_start = time.perf_counter()
            tetaEFMNN = find_optimal_param_efmnn(f1dt, f1lb, f2dt, f2lb, f3dt, f3lb)
            time_end = time.perf_counter()
            optimization_time_efmnn_save = np.append(optimization_time_efmnn_save, time_end - time_start)
            optimization_value_efmnn_save = np.append(optimization_value_efmnn_save, 'teta = ' + str(tetaEFMNN))
            
            efmnnClassifier = EFMNNClassification(gamma = 1, teta = tetaEFMNN, isDraw = False, isNorm = False)
            efmnnClassifier.fit(trainingData, trainingLabel)
            
            training_time_efmnn_save = np.append(training_time_efmnn_save, efmnnClassifier.elapsed_training_time)
            numhyperbox_efmnn_save = np.append(numhyperbox_efmnn_save, len(efmnnClassifier.classId))
                       
            result = efmnnClassifier.predict(testingData, testingLabel)

            if result != None:
                err = np.round(result.summis / numTestSample * 100, 3)
                testing_error_efmnn_save = np.append(testing_error_efmnn_save, err)
            
            # knefmnn
            time_start = time.perf_counter()
            tetaKNEFMNN, kOpt = find_optimal_param_knefmnn(f1dt, f1lb, f2dt, f2lb, f3dt, f3lb)
            time_end = time.perf_counter()
            optimization_time_knefmnn_save = np.append(optimization_time_knefmnn_save, time_end - time_start)
            optimization_value_knefmnn_save = np.append(optimization_value_knefmnn_save, 'teta = ' + str(tetaKNEFMNN) + '; K = ' + str(kOpt))
            
            knefmnnClassifier = KNEFMNNClassification(gamma = 1, teta = tetaKNEFMNN, isDraw = False, isNorm = False)
            knefmnnClassifier.fit(trainingData, trainingLabel, kOpt)
            
            training_time_knefmnn_save = np.append(training_time_knefmnn_save, knefmnnClassifier.elapsed_training_time)
            numhyperbox_knefmnn_save = np.append(numhyperbox_knefmnn_save, len(knefmnnClassifier.classId))
                       
            result = knefmnnClassifier.predict(testingData, testingLabel)

            if result != None:
                err = np.round(result.summis / numTestSample * 100, 3)
                testing_error_knefmnn_save = np.append(testing_error_knefmnn_save, err)
        
            # KNN
            time_start = time.perf_counter()
            kOptKNN = find_optimal_k_knn(f1dt, f1lb, f2dt, f2lb, f3dt, f3lb)
            time_end = time.perf_counter()
            optimization_time_knn_save = np.append(optimization_time_knn_save, time_end - time_start)
            optimization_value_knn_save = np.append(optimization_value_knn_save, 'K = ' + str(kOptKNN))
                       
            knnClassifier = KNeighborsClassifier(n_neighbors = kOptKNN)
    
            time_start = time.perf_counter()
            knnClassifier.fit(trainingData, trainingLabel)
            time_end = time.perf_counter()
            training_time_knn_save = np.append(training_time_knn_save, time_end - time_start)
            
            predictedKNN = knnClassifier.predict(testingData)     
            errorRateKNN = np.round((testingLabel != predictedKNN).sum() / numTestSample * 100, 3)
            testing_error_knn_save = np.append(testing_error_knn_save, errorRateKNN)
            
            # Gaussian Naive Bayes
            gnb = GaussianNB()
            time_start = time.perf_counter()
            gnb.fit(trainingData, trainingLabel)
            time_end = time.perf_counter()
            
            training_time_gnb_save = np.append(training_time_gnb_save, time_end - time_start)
            predicted_gnb = gnb.predict(testingData)
            
            error_rate_gnb = np.round((testingLabel != predicted_gnb).sum() / numTestSample * 100, 3)
            testing_error_gnb_save = np.append(testing_error_gnb_save, error_rate_gnb)           
            
            # svm
            time_start = time.perf_counter()
            cOpt, gammaOpt = find_optimal_param_svm(f1dt, f1lb, f2dt, f2lb, f3dt, f3lb)
            time_end = time.perf_counter()
            optimization_time_svm_save = np.append(optimization_time_svm_save, time_end - time_start)
            optimization_value_svm_save = np.append(optimization_value_svm_save, 'C = ' + str(cOpt) + '; gamma = ' + str(gammaOpt))
            
            svclassifier = SVC(kernel='rbf', decision_function_shape='ovo', C = cOpt, gamma = gammaOpt) 
            
            time_start = time.perf_counter()
            svclassifier.fit(trainingData, trainingLabel)
            time_end = time.perf_counter()
            
            training_time_svm_save = np.append(training_time_svm_save, time_end - time_start)
            
            predictedSVC = svclassifier.predict(testingData)
            errorSVC = np.round((testingLabel != predictedSVC).sum() / numTestSample * 100, 3)
            testing_error_svm_save = np.append(testing_error_svm_save, errorSVC)
            
            # decision tree
            time_start = time.perf_counter()
            maxDeptOpt = find_optimal_param_decision_tree(f1dt, f1lb, f2dt, f2lb, f3dt, f3lb)
            time_end = time.perf_counter()
            optimization_time_dt_save = np.append(optimization_time_dt_save, time_end - time_start)
            optimization_value_dt_save = np.append(optimization_value_dt_save, 'max_depth = ' + str(maxDeptOpt))
            
            decisionTree = DecisionTreeClassifier(max_depth = maxDeptOpt)
          
            time_start = time.perf_counter()
            decisionTree.fit(trainingData, trainingLabel) 
            time_end = time.perf_counter()
            
            training_time_dt_save = np.append(training_time_dt_save, time_end - time_start)
            
            predictedDT = decisionTree.predict(testingData)
            errorDT = np.round((testingLabel != predictedDT).sum() / numTestSample * 100, 3)
            testing_error_dt_save = np.append(testing_error_dt_save, errorDT)
        
        # save online gfmm result to file
        data_online_save = np.hstack((fold_index.reshape(-1, 1), numhyperbox_online_gfmm_save.reshape(-1, 1), training_time_online_gfmm_save.reshape(-1, 1), testing_error_online_gfmm_save.reshape(-1, 1), optimization_time_online_gfmm_save.reshape(-1, 1), optimization_value_online_gfmm_save.reshape(-1, 1)))
        filename_online = save_online_gfmm_result_folder_path + '/' + dataset_names[dt] + '.csv'
        
        open(filename_online, 'w').close() # make existing file empty
        
        with open(filename_online,'a') as f_handle:
            f_handle.writelines('Fold, No hyperboxes, Training time, Testing error, Optimization time, Optimal param\n')
            np.savetxt(f_handle, data_online_save, fmt='%s', delimiter=', ')
        
        # Save results of accelerated batch learning
        data_accel_agglo_save = np.hstack((fold_index.reshape(-1, 1), numhyperbox_accel_agglo_save.reshape(-1, 1), training_time_accel_agglo_save.reshape(-1, 1), testing_error_accel_agglo_save.reshape(-1, 1), optimization_time_accel_agglo_save.reshape(-1, 1), optimization_value_accel_agglo_save.reshape(-1, 1)))
        filename_accel_agglo = save_accel_agglo_result_folder_path + '/' + dataset_names[dt] + '.csv'
        
        open(filename_accel_agglo, 'w').close() # make existing file empty
        
        with open(filename_accel_agglo,'a') as f_handle:
            f_handle.write('simil_thres = 0.5, measure = short\n')
            f_handle.writelines('Fold, No hyperboxes, Training time, Testing error, Optimization time, Optimal param\n')
            np.savetxt(f_handle, data_accel_agglo_save, fmt='%s', delimiter=', ')
            
        # save FMNN
        data_fmnn_save = np.hstack((fold_index.reshape(-1, 1), numhyperbox_fmnn_save.reshape(-1, 1), training_time_fmnn_save.reshape(-1, 1), testing_error_fmnn_save.reshape(-1, 1), optimization_time_fmnn_save.reshape(-1, 1), optimization_value_fmnn_save.reshape(-1, 1)))
        filename_fmnn = save_fmnn_result_folder_path + '/' + dataset_names[dt] + '.csv'
        
        open(filename_fmnn, 'w').close() # make existing file empty
        
        with open(filename_fmnn,'a') as f_handle:
            f_handle.writelines('Fold, No hyperboxes, Training time, Testing error, Optimization time, Optimal param\n')
            np.savetxt(f_handle, data_fmnn_save, fmt='%s', delimiter=', ')
        
        # Save EFMNN
        data_efmnn_save = np.hstack((fold_index.reshape(-1, 1), numhyperbox_efmnn_save.reshape(-1, 1), training_time_efmnn_save.reshape(-1, 1), testing_error_efmnn_save.reshape(-1, 1), optimization_time_efmnn_save.reshape(-1, 1), optimization_value_efmnn_save.reshape(-1, 1)))
        filename_efmnn = save_efmnn_result_folder_path + '/' + dataset_names[dt] + '.csv'
        
        open(filename_efmnn, 'w').close() # make existing file empty
        
        with open(filename_efmnn,'a') as f_handle:
            f_handle.writelines('Fold, No hyperboxes, Training time, Testing error, Optimization time, Optimal param\n')
            np.savetxt(f_handle, data_efmnn_save, fmt='%s', delimiter=', ')
        
        # Save KNFMNN
        data_knefmnn_save = np.hstack((fold_index.reshape(-1, 1), numhyperbox_knefmnn_save.reshape(-1, 1), training_time_knefmnn_save.reshape(-1, 1), testing_error_knefmnn_save.reshape(-1, 1), optimization_time_knefmnn_save.reshape(-1, 1), optimization_value_knefmnn_save.reshape(-1, 1)))
        filename_knefmnn = save_knefmnn_result_folder_path + '/' + dataset_names[dt] + '.csv'
        
        open(filename_knefmnn, 'w').close() # make existing file empty
        
        with open(filename_knefmnn,'a') as f_handle:
            f_handle.writelines('Fold, No hyperboxes, Training time, Testing error, Optimization time, Optimal param\n')
            np.savetxt(f_handle, data_knefmnn_save, fmt='%s', delimiter=', ')
        
        data_knn_save = np.hstack((fold_index.reshape(-1, 1), training_time_knn_save.reshape(-1, 1), testing_error_knn_save.reshape(-1, 1), optimization_time_knn_save.reshape(-1, 1), optimization_value_knn_save.reshape(-1, 1)))
        filename_knn = save_result_KNN_folder_path + '/' + dataset_names[dt] + '.csv'
        
        open(filename_knn, 'w').close() # make existing file empty
        
        with open(filename_knn,'a') as f_handle:
            f_handle.writelines('Fold, Training time, Testing error, Optimization time, Optimal param\n')
            np.savetxt(f_handle, data_knn_save, fmt='%s', delimiter=', ')
            
        data_svm_save = np.hstack((fold_index.reshape(-1, 1), training_time_svm_save.reshape(-1, 1), testing_error_svm_save.reshape(-1, 1), optimization_time_svm_save.reshape(-1, 1), optimization_value_svm_save.reshape(-1, 1)))
        filename_svm = save_result_SVM_folder_path + '/' + dataset_names[dt] + '.csv'
        
        open(filename_svm, 'w').close() # make existing file empty
        
        with open(filename_svm,'a') as f_handle:
            f_handle.write('Kernel = RBF, decision = ovo \n')
            f_handle.writelines('Fold, Training time, Testing error, Optimization time, Optimal param\n')
            np.savetxt(f_handle, data_svm_save, fmt='%s', delimiter=', ')
            
        data_dt_save = np.hstack((fold_index.reshape(-1, 1), training_time_dt_save.reshape(-1, 1), testing_error_dt_save.reshape(-1, 1), optimization_time_dt_save.reshape(-1, 1), optimization_value_dt_save.reshape(-1, 1)))
        filename_dt = save_result_DT_folder_path + '/' + dataset_names[dt] + '.csv'
        
        open(filename_dt, 'w').close() # make existing file empty
        
        with open(filename_dt,'a') as f_handle:
            f_handle.writelines('Fold, Training time, Testing error, Optimization time, Optimal param\n')
            np.savetxt(f_handle, data_dt_save, fmt='%s', delimiter=', ')
            
        data_gnb_save = np.hstack((fold_index.reshape(-1, 1), training_time_gnb_save.reshape(-1, 1), testing_error_gnb_save.reshape(-1, 1)))
        filename_gnb = save_result_NB_folder_path + '/' + dataset_names[dt] + '.csv'
        
        open(filename_gnb, 'w').close() # make existing file empty
        
        with open(filename_gnb,'a') as f_handle:
            f_handle.writelines('Fold, Training time, Testing error\n')
            np.savetxt(f_handle, data_gnb_save, fmt='%s', delimiter=', ')
       
#        except:
#            pass
        
    print('---Finish---')