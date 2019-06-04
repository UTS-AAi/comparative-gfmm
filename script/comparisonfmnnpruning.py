# Run 10 times with different input

import sys, os
from os.path import dirname
root_path = dirname(dirname(os.getcwd()))
sys.path.insert(0, root_path) # insert root directory to environmental variables

from GFMM.accelbatchgfmm import AccelBatchGFMM
from GFMM.onlinegfmm import OnlineGFMM
import time
from EFMN.efmnnclassification import EFMNNClassification
from EFMN.knefmnnclassification import KNEFMNNClassification
from FMNN.fmnnclassification import FMNNClassification
import numpy as np
from functionhelper.preprocessinghelper import loadDataset
#from sklearn.naive_bayes import GaussianNB
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier


if __name__ == '__main__':
    
    save_online_gfmm_result_folder_path = root_path + '/Experiment/FMM/Online_GFMM/Pruning/'
    save_accel_agglo_result_folder_path = root_path + '/Experiment/FMM/Accel_Agglo/Pruning/'
    save_efmnn_result_folder_path = root_path + '/Experiment/FMM/EFMNN/Pruning/'
    save_knefmnn_result_folder_path = root_path + '/Experiment/FMM/KNEFMNN/Pruning/'
    save_fmnn_result_folder_path = root_path + '/Experiment/FMM/FMNN/Pruning/'
    
    dataset_path = root_path + '/Dataset/train_test/dps/'
    
    dataset_names = ['circle_dps', 'complex9_dps', 'DiagnosticBreastCancer_dps', 'glass_dps', 'ionosphere_dps', 'iris_dps', 'ringnorm_dps', 'segmentation_dps', 'spherical_5_2_dps', 'spiral_dps', 'thyroid_dps', 'twonorm_dps', 'waveform_dps', 'wine_dps', 'yeast_dps', 'zelnik6_dps']
    teta_onln_f1 = [0.06, 0.06, 0.46, 0.1, 0.06, 0.06, 0.36, 0.06, 0.2, 0.06, 0.06, 0.36, 0.36, 0.76, 0.1, 0.3]
    teta_onln_f2 = [0.06, 0.06, 0.26, 0.1, 0.16, 0.16, 0.36, 0.06, 0.2, 0.06, 0.06, 0.26, 0.36, 0.66, 0.1, 0.06]
    teta_onln_f3 = [0.16, 0.06, 0.4, 0.06, 0.16, 0.06, 0.36, 0.1, 0.16, 0.06, 0.1, 0.4, 0.4, 0.46, 0.06, 0.06]
    teta_onln_f4 = [0.06, 0.06, 0.4, 0.06, 0.26, 0.3, 0.36, 0.16, 0.2, 0.06, 0.26, 0.36, 0.36, 0.06, 0.1, 0.2]
    
    teta_agglo2_f1 = [0.66, 0.06, 0.46, 0.1, 0.06, 0.06, 0.36, 0.06, 0.3, 0.06, 0.06, 0.46, 0.5, 0.8, 0.16, 0.06]
    teta_agglo2_f2 = [0.06, 0.06, 0.8, 0.1, 0.7, 0.16, 0.36, 0.06, 0.36, 0.06, 0.56, 0.46, 0.76, 0.76, 0.16, 0.06]
    teta_agglo2_f3 = [0.06, 0.06, 0.3, 0.06, 0.76, 0.06, 0.36, 0.1, 0.06, 0.06, 0.1, 0.4, 0.76, 0.66, 0.1, 0.06]
    teta_agglo2_f4 = [0.3, 0.06, 0.26, 0.06, 0.6, 0.6, 0.36, 0.16, 0.36, 0.06, 0.4, 0.4, 0.66, 0.3, 0.1, 0.06]
    
    teta_fmnn_f1 = [0.1, 0.1, 0.06, 0.06, 0.06, 0.26, 0.8, 0.06, 0.3, 0.06, 0.1, 0.06, 0.06, 0.16, 0.06, 0.1]
    teta_fmnn_f2 = [0.06, 0.06, 0.06, 0.06, 0.06, 0.3, 0.8, 0.06, 0.3, 0.06, 0.06, 0.06, 0.56, 0.66, 0.06, 0.06]
    teta_fmnn_f3 = [0.26, 0.06, 0.06, 0.06, 0.3, 0.3, 0.8, 0.06, 0.3, 0.06, 0.1, 0.06, 0.5, 0.66, 0.06, 0.06]
    teta_fmnn_f4 = [0.1, 0.06, 0.06, 0.06, 0.06, 0.26, 0.8, 0.06, 0.2, 0.06, 0.1, 0.06, 0.4, 0.6, 0.06, 0.06]
    
    teta_efmnn_f1 = [0.06, 0.06, 0.2, 0.1, 0.1, 0.2, 0.66, 0.06, 0.36, 0.06, 0.06, 0.06, 0.06, 0.3, 0.1, 0.06]
    teta_efmnn_f2 = [0.06, 0.06, 0.06, 0.1, 0.1, 0.16, 0.7, 0.06, 0.3, 0.06, 0.06, 0.16, 0.1, 0.6, 0.1, 0.4]
    teta_efmnn_f3 = [0.06, 0.06, 0.36, 0.06, 0.1, 0.1, 0.6, 0.06, 0.16, 0.06, 0.06, 0.1, 0.06, 0.2, 0.1, 0.06]
    teta_efmnn_f4 = [0.06, 0.06, 0.1, 0.1, 0.1, 0.36, 0.06, 0.06, 0.26, 0.06, 0.36, 0.06, 0.06, 0.7, 0.1, 0.06]
    
    teta_knefmnn_f1 = [0.16, 0.06, 0.16, 0.1, 0.1, 0.36, 0.76, 0.06, 0.36, 0.06, 0.06, 0.26, 0.36, 0.6, 0.1, 0.06]
    k_knefmnn_f1 = [3, 2, 7, 3, 2, 2, 5, 2, 2, 2, 2, 7, 10, 2, 10, 2]
    teta_knefmnn_f2 = [0.16, 0.06, 0.2, 0.1, 0.1, 0.2, 0.6, 0.16, 0.3, 0.06, 0.06, 0.26, 0.3, 0.46, 0.1, 0.06]
    k_knefmnn_f2 = [3, 2, 3, 3, 2, 2, 10, 2, 2, 2, 2, 9, 9, 4, 10, 2]
    teta_knefmnn_f3 = [0.1, 0.06, 0.2, 0.06, 0.1, 0.1, 0.4, 0.06, 0.16, 0.06, 0.06, 0.26, 0.36, 0.5, 0.1, 0.06]
    k_knefmnn_f3 = [3, 2, 10, 2, 2, 2, 10, 2, 2, 2, 2, 7, 10, 4, 7, 2]
    teta_knefmnn_f4 = [0.06, 0.06, 0.26, 0.1, 0.1, 0.5, 0.46, 0.06, 0.26, 0.06, 0.06, 0.26, 0.3, 0.56, 0.16, 0.2]
    k_knefmnn_f4 = [3, 2, 6, 2, 2, 2, 8, 2, 2, 2, 2, 10, 10, 6, 3, 2]
    
    #dataset_names = ['yeast_dps']
    fold_index = np.array([1, 2, 3, 4])
    
    for dt in range(len(dataset_names)):
        #try:
        print('Current dataset: ', dataset_names[dt])
        fold1File = dataset_path + dataset_names[dt] + '_1.dat'
        fold2File = dataset_path + dataset_names[dt] + '_2.dat'
        fold3File = dataset_path + dataset_names[dt] + '_3.dat'
        fold4File = dataset_path + dataset_names[dt] + '_4.dat'
        
        # Read data file
        fold1Data, _, fold1Label, _ = loadDataset(fold1File, 1, False)
        fold2Data, _, fold2Label, _ = loadDataset(fold2File, 1, False)
        fold3Data, _, fold3Label, _ = loadDataset(fold3File, 1, False)
        fold4Data, _, fold4Label, _ = loadDataset(fold4File, 1, False)
        
        numhyperbox_online_gfmm_save = np.array([])
        numhyperbox_before_prun_online_gfmm_save = np.array([])
        training_time_online_gfmm_save = np.array([])
        testing_error_online_gfmm_save = np.array([])
        testing_error_before_prun_online_gfmm_save = np.array([])
        
        numhyperbox_fmnn_save = np.array([])
        numhyperbox_before_prun_fmnn_save = np.array([])
        training_time_fmnn_save = np.array([])
        testing_error_fmnn_save = np.array([])
        testing_error_before_prun_fmnn_save = np.array([])
        
        numhyperbox_efmnn_save = np.array([])
        numhyperbox_before_prun_efmnn_save = np.array([])
        training_time_efmnn_save = np.array([])
        testing_error_efmnn_save = np.array([])
        testing_error_before_prun_efmnn_save = np.array([])
        
        numhyperbox_knefmnn_save = np.array([])
        numhyperbox_before_prun_knefmnn_save = np.array([])
        training_time_knefmnn_save = np.array([])
        testing_error_knefmnn_save = np.array([])
        testing_error_before_prun_knefmnn_save = np.array([])
        
        numhyperbox_accel_agglo_save = np.array([], dtype=np.int64)
        numhyperbox_before_prun_accel_agglo_save = np.array([])
        training_time_accel_agglo_save = np.array([])
        testing_error_accel_agglo_save = np.array([])
        testing_error_before_prun_accel_agglo_save = np.array([])
        
        # loop through 4 folds
        for fo in range(4):
            if fo == 0:
                # fold 1 is testing set
                trainingData = np.vstack((fold2Data, fold3Data))
                testingData = fold1Data
                validationData = fold4Data
                trainingLabel = np.hstack((fold2Label, fold3Label))
                testingLabel = fold1Label
                validationLabel = fold4Label
                numTestSample = len(testingLabel)
                tetaOnlnGFMM = teta_onln_f1[dt]
                tetaAGGLO2 = teta_agglo2_f1[dt]
                tetaFMNN = teta_fmnn_f1[dt]
                tetaEFMNN = teta_efmnn_f1[dt]
                tetaKNEFMNN = teta_knefmnn_f1[dt]
                kOpt = k_knefmnn_f1[dt]
            elif fo == 1:
                # fold 2 is testing set
                trainingData = np.vstack((fold3Data, fold4Data))
                testingData = fold2Data
                validationData = fold1Data
                trainingLabel = np.hstack((fold3Label, fold4Label))
                testingLabel = fold2Label
                validationLabel = fold1Label
                numTestSample = len(testingLabel)
                tetaOnlnGFMM = teta_onln_f2[dt]
                tetaAGGLO2 = teta_agglo2_f2[dt]
                tetaFMNN = teta_fmnn_f2[dt]
                tetaEFMNN = teta_efmnn_f2[dt]
                tetaKNEFMNN = teta_knefmnn_f2[dt]
                kOpt = k_knefmnn_f2[dt]
            elif fo == 2:
                # fold 3 is testing set
                trainingData = np.vstack((fold1Data, fold4Data))
                testingData = fold3Data
                validationData = fold2Data
                trainingLabel = np.hstack((fold1Label, fold4Label))
                testingLabel = fold3Label
                validationLabel = fold2Label
                numTestSample = len(testingLabel)
                tetaOnlnGFMM = teta_onln_f3[dt]
                tetaAGGLO2 = teta_agglo2_f3[dt]
                tetaFMNN = teta_fmnn_f3[dt]
                tetaEFMNN = teta_efmnn_f3[dt]
                tetaKNEFMNN = teta_knefmnn_f3[dt]
                kOpt = k_knefmnn_f3[dt]
            else:
                # fold 4 is testing set
                trainingData = np.vstack((fold1Data, fold2Data))
                testingData = fold4Data
                validationData = fold3Data
                trainingLabel = np.hstack((fold1Label, fold2Label))
                testingLabel = fold4Label
                validationLabel = fold3Label
                numTestSample = len(testingLabel)
                tetaOnlnGFMM = teta_onln_f4[dt]
                tetaAGGLO2 = teta_agglo2_f4[dt]
                tetaFMNN = teta_fmnn_f4[dt]
                tetaEFMNN = teta_efmnn_f4[dt]
                tetaKNEFMNN = teta_knefmnn_f4[dt]
                kOpt = k_knefmnn_f4[dt]
                
        
            # online GFMM
            olnClassifier = OnlineGFMM(gamma = 1, teta = tetaOnlnGFMM, tMin = tetaOnlnGFMM, isDraw = False, oper = 'min', isNorm = False)
            olnClassifier.fit(trainingData, trainingData, trainingLabel)
            
            numhyperbox_before_prun_online_gfmm_save = np.append(numhyperbox_before_prun_online_gfmm_save, len(olnClassifier.classId)) 
            result = olnClassifier.predict(testingData, testingData, testingLabel)
            if result != None:
                err = np.round(result.summis / numTestSample * 100, 3)
                testing_error_before_prun_online_gfmm_save = np.append(testing_error_before_prun_online_gfmm_save, err)            
            
            start_t = time.perf_counter()
            olnClassifier.pruning_val(validationData, validationData, validationLabel)
            end_t = time.perf_counter()
            
            training_time_online_gfmm_save = np.append(training_time_online_gfmm_save, olnClassifier.elapsed_training_time + (end_t - start_t))
            numhyperbox_online_gfmm_save = np.append(numhyperbox_online_gfmm_save, len(olnClassifier.classId))            
                
            result = olnClassifier.predict(testingData, testingData, testingLabel)
            if result != None:
                err = np.round(result.summis / numTestSample * 100, 3)
                testing_error_online_gfmm_save = np.append(testing_error_online_gfmm_save, err)
            
            # agglo-2
            accelClassifier = AccelBatchGFMM(gamma = 1, teta = tetaAGGLO2, bthres = 0, simil = 'long', sing = 'min', isDraw = False, oper = 'min', isNorm = False)
            accelClassifier.fit(trainingData, trainingData, trainingLabel)
            
            numhyperbox_before_prun_accel_agglo_save = np.append(numhyperbox_before_prun_accel_agglo_save, len(accelClassifier.classId))
             
            result = accelClassifier.predict(testingData, testingData, testingLabel)
        
            if result != None:
                err = np.round(result.summis / numTestSample * 100, 3)
                testing_error_before_prun_accel_agglo_save = np.append(testing_error_before_prun_accel_agglo_save, err)                
            
            start_t = time.perf_counter()
            accelClassifier.pruning_val(validationData, validationData, validationLabel)
            end_t = time.perf_counter()
            
            training_time_accel_agglo_save = np.append(training_time_accel_agglo_save, accelClassifier.elapsed_training_time + (end_t - start_t))
            numhyperbox_accel_agglo_save = np.append(numhyperbox_accel_agglo_save, len(accelClassifier.classId))
                
            result = accelClassifier.predict(testingData, testingData, testingLabel)
        
            if result != None:
                err = np.round(result.summis / numTestSample * 100, 3)
                testing_error_accel_agglo_save = np.append(testing_error_accel_agglo_save, err)
                
            # fmnn
            fmnnClassifier = FMNNClassification(gamma = 1, teta = tetaFMNN, isDraw = False, isNorm = False)
            fmnnClassifier.fit(trainingData, trainingLabel)
            
            numhyperbox_before_prun_fmnn_save = np.append(numhyperbox_before_prun_fmnn_save, len(fmnnClassifier.classId))
             
            result = fmnnClassifier.predict(testingData, testingLabel)
        
            if result != None:
                err = np.round(result.summis / numTestSample * 100, 3)
                testing_error_before_prun_fmnn_save = np.append(testing_error_before_prun_fmnn_save, err)                
            
            start_t = time.perf_counter()
            fmnnClassifier.pruning_val(validationData, validationLabel)
            end_t = time.perf_counter()
            
            training_time_fmnn_save = np.append(training_time_fmnn_save, fmnnClassifier.elapsed_training_time + (end_t - start_t))
            numhyperbox_fmnn_save = np.append(numhyperbox_fmnn_save, len(fmnnClassifier.classId))
                       
            result = fmnnClassifier.predict(testingData, testingLabel)

            if result != None:
                err = np.round(result.summis / numTestSample * 100, 3)
                testing_error_fmnn_save = np.append(testing_error_fmnn_save, err)
            
#            # efmnn
            efmnnClassifier = EFMNNClassification(gamma = 1, teta = tetaEFMNN, isDraw = False, isNorm = False)
            efmnnClassifier.fit(trainingData, trainingLabel)
            
            numhyperbox_before_prun_efmnn_save = np.append(numhyperbox_before_prun_efmnn_save, len(efmnnClassifier.classId))
            
            result = efmnnClassifier.predict(testingData, testingLabel)

            if result != None:
                err = np.round(result.summis / numTestSample * 100, 3)
                testing_error_before_prun_efmnn_save = np.append(testing_error_before_prun_efmnn_save, err)
            
            
            start_t = time.perf_counter()
            efmnnClassifier.pruning_val(validationData, validationLabel)
            end_t = time.perf_counter()
            
            training_time_efmnn_save = np.append(training_time_efmnn_save, efmnnClassifier.elapsed_training_time + (end_t - start_t))
            numhyperbox_efmnn_save = np.append(numhyperbox_efmnn_save, len(efmnnClassifier.classId))
                       
            result = efmnnClassifier.predict(testingData, testingLabel)

            if result != None:
                err = np.round(result.summis / numTestSample * 100, 3)
                testing_error_efmnn_save = np.append(testing_error_efmnn_save, err)
            
#            # knefmnn
            knefmnnClassifier = KNEFMNNClassification(gamma = 1, teta = tetaKNEFMNN, isDraw = False, isNorm = False)
            knefmnnClassifier.fit(trainingData, trainingLabel, kOpt)
            
            numhyperbox_before_prun_knefmnn_save = np.append(numhyperbox_before_prun_knefmnn_save, len(knefmnnClassifier.classId))
            
            result = knefmnnClassifier.predict(testingData, testingLabel)

            if result != None:
                err = np.round(result.summis / numTestSample * 100, 3)
                testing_error_before_prun_knefmnn_save = np.append(testing_error_before_prun_knefmnn_save, err)       
            
            start_t = time.perf_counter()
            knefmnnClassifier.pruning_val(validationData, validationLabel)
            end_t = time.perf_counter()
            
            training_time_knefmnn_save = np.append(training_time_knefmnn_save, knefmnnClassifier.elapsed_training_time + (end_t - start_t))
            numhyperbox_knefmnn_save = np.append(numhyperbox_knefmnn_save, len(knefmnnClassifier.classId))
                       
            result = knefmnnClassifier.predict(testingData, testingLabel)

            if result != None:
                err = np.round(result.summis / numTestSample * 100, 3)
                testing_error_knefmnn_save = np.append(testing_error_knefmnn_save, err)
        
#            # KNN
#            time_start = time.perf_counter()
#            kOptKNN = find_optimal_k_knn(f1dt, f1lb, f2dt, f2lb, f3dt, f3lb)
#            time_end = time.perf_counter()
#            optimization_time_knn_save = np.append(optimization_time_knn_save, time_end - time_start)
#            optimization_value_knn_save = np.append(optimization_value_knn_save, 'K = ' + str(kOptKNN))
#                       
#            knnClassifier = KNeighborsClassifier(n_neighbors = kOptKNN)
#    
#            time_start = time.perf_counter()
#            knnClassifier.fit(trainingData, trainingLabel)
#            time_end = time.perf_counter()
#            training_time_knn_save = np.append(training_time_knn_save, time_end - time_start)
#            
#            predictedKNN = knnClassifier.predict(testingData)     
#            errorRateKNN = np.round((testingLabel != predictedKNN).sum() / numTestSample * 100, 3)
#            testing_error_knn_save = np.append(testing_error_knn_save, errorRateKNN)
#            
#            # Gaussian Naive Bayes
#            gnb = GaussianNB()
#            time_start = time.perf_counter()
#            gnb.fit(trainingData, trainingLabel)
#            time_end = time.perf_counter()
#            
#            training_time_gnb_save = np.append(training_time_gnb_save, time_end - time_start)
#            predicted_gnb = gnb.predict(testingData)
#            
#            error_rate_gnb = np.round((testingLabel != predicted_gnb).sum() / numTestSample * 100, 3)
#            testing_error_gnb_save = np.append(testing_error_gnb_save, error_rate_gnb)           
#            
#            # svm
#            time_start = time.perf_counter()
#            cOpt, gammaOpt = find_optimal_param_svm(f1dt, f1lb, f2dt, f2lb, f3dt, f3lb)
#            time_end = time.perf_counter()
#            optimization_time_svm_save = np.append(optimization_time_svm_save, time_end - time_start)
#            optimization_value_svm_save = np.append(optimization_value_svm_save, 'C = ' + str(cOpt) + '; gamma = ' + str(gammaOpt))
#            
#            svclassifier = SVC(kernel='rbf', decision_function_shape='ovo', C = cOpt, gamma = gammaOpt) 
#            
#            time_start = time.perf_counter()
#            svclassifier.fit(trainingData, trainingLabel)
#            time_end = time.perf_counter()
#            
#            training_time_svm_save = np.append(training_time_svm_save, time_end - time_start)
#            
#            predictedSVC = svclassifier.predict(testingData)
#            errorSVC = np.round((testingLabel != predictedSVC).sum() / numTestSample * 100, 3)
#            testing_error_svm_save = np.append(testing_error_svm_save, errorSVC)
#            
#            # decision tree
#            time_start = time.perf_counter()
#            maxDeptOpt = find_optimal_param_decision_tree(f1dt, f1lb, f2dt, f2lb, f3dt, f3lb)
#            time_end = time.perf_counter()
#            optimization_time_dt_save = np.append(optimization_time_dt_save, time_end - time_start)
#            optimization_value_dt_save = np.append(optimization_value_dt_save, 'max_depth = ' + str(maxDeptOpt))
#            
#            decisionTree = DecisionTreeClassifier(max_depth = maxDeptOpt)
#          
#            time_start = time.perf_counter()
#            decisionTree.fit(trainingData, trainingLabel) 
#            time_end = time.perf_counter()
#            
#            training_time_dt_save = np.append(training_time_dt_save, time_end - time_start)
#            
#            predictedDT = decisionTree.predict(testingData)
#            errorDT = np.round((testingLabel != predictedDT).sum() / numTestSample * 100, 3)
#            testing_error_dt_save = np.append(testing_error_dt_save, errorDT)
#        
#        # save online gfmm result to file
        data_online_save = np.hstack((fold_index.reshape(-1, 1), numhyperbox_before_prun_online_gfmm_save.reshape(-1, 1), numhyperbox_online_gfmm_save.reshape(-1, 1), training_time_online_gfmm_save.reshape(-1, 1), testing_error_before_prun_online_gfmm_save.reshape(-1, 1), testing_error_online_gfmm_save.reshape(-1, 1)))
        filename_online = save_online_gfmm_result_folder_path + dataset_names[dt] + '.csv'
        
        open(filename_online, 'w').close() # make existing file empty
        
        with open(filename_online,'a') as f_handle:
            f_handle.writelines('Fold, No hyperboxes before pruning, No hyperboxes after pruning, Training time, Testing error before pruning, Testing error after pruning\n')
            np.savetxt(f_handle, data_online_save, fmt='%s', delimiter=', ')
        
        # Save results of accelerated batch learning
        data_accel_agglo_save = np.hstack((fold_index.reshape(-1, 1), numhyperbox_before_prun_accel_agglo_save.reshape(-1, 1), numhyperbox_accel_agglo_save.reshape(-1, 1), training_time_accel_agglo_save.reshape(-1, 1), testing_error_before_prun_accel_agglo_save.reshape(-1, 1), testing_error_accel_agglo_save.reshape(-1, 1)))
        filename_accel_agglo = save_accel_agglo_result_folder_path + dataset_names[dt] + '.csv'
        
        open(filename_accel_agglo, 'w').close() # make existing file empty
        
        with open(filename_accel_agglo,'a') as f_handle:
            f_handle.writelines('Fold, No hyperboxes before pruning, No hyperboxes after pruning, Training time, Testing error before pruning, Testing error after pruning\n')
            np.savetxt(f_handle, data_accel_agglo_save, fmt='%s', delimiter=', ')
            
#        # save FMNN
        data_fmnn_save = np.hstack((fold_index.reshape(-1, 1), numhyperbox_before_prun_fmnn_save.reshape(-1, 1), numhyperbox_fmnn_save.reshape(-1, 1), training_time_fmnn_save.reshape(-1, 1), testing_error_before_prun_fmnn_save.reshape(-1, 1), testing_error_fmnn_save.reshape(-1, 1)))
        filename_fmnn = save_fmnn_result_folder_path + dataset_names[dt] + '.csv'
        
        open(filename_fmnn, 'w').close() # make existing file empty
        
        with open(filename_fmnn,'a') as f_handle:
            f_handle.writelines('Fold, No hyperboxes before pruning, No hyperboxes after pruning, Training time, Testing error before pruning, Testing error after pruning\n')
            np.savetxt(f_handle, data_fmnn_save, fmt='%s', delimiter=', ')
        
        # Save EFMNN
        data_efmnn_save = np.hstack((fold_index.reshape(-1, 1), numhyperbox_before_prun_efmnn_save.reshape(-1, 1), numhyperbox_efmnn_save.reshape(-1, 1), training_time_efmnn_save.reshape(-1, 1), testing_error_before_prun_efmnn_save.reshape(-1, 1), testing_error_efmnn_save.reshape(-1, 1)))
        filename_efmnn = save_efmnn_result_folder_path + dataset_names[dt] + '.csv'
        
        open(filename_efmnn, 'w').close() # make existing file empty
        
        with open(filename_efmnn,'a') as f_handle:
            f_handle.writelines('Fold, No hyperboxes before pruning, No hyperboxes after pruning, Training time, Testing error before pruning, Testing error after pruning\n')
            np.savetxt(f_handle, data_efmnn_save, fmt='%s', delimiter=', ')
        
        # Save KNFMNN
        data_knefmnn_save = np.hstack((fold_index.reshape(-1, 1), numhyperbox_before_prun_knefmnn_save.reshape(-1, 1), numhyperbox_knefmnn_save.reshape(-1, 1), training_time_knefmnn_save.reshape(-1, 1), testing_error_before_prun_knefmnn_save.reshape(-1, 1), testing_error_knefmnn_save.reshape(-1, 1)))
        filename_knefmnn = save_knefmnn_result_folder_path + dataset_names[dt] + '.csv'
        
        open(filename_knefmnn, 'w').close() # make existing file empty
        
        with open(filename_knefmnn,'a') as f_handle:
            f_handle.writelines('Fold, No hyperboxes before pruning, No hyperboxes after pruning, Training time, Testing error before pruning, Testing error after pruning\n')
            np.savetxt(f_handle, data_knefmnn_save, fmt='%s', delimiter=', ')
        
#        data_knn_save = np.hstack((fold_index.reshape(-1, 1), training_time_knn_save.reshape(-1, 1), testing_error_knn_save.reshape(-1, 1), optimization_time_knn_save.reshape(-1, 1), optimization_value_knn_save.reshape(-1, 1)))
#        filename_knn = save_result_KNN_folder_path + '\\' + dataset_names[dt] + '.csv'
#        
#        open(filename_knn, 'w').close() # make existing file empty
#        
#        with open(filename_knn,'a') as f_handle:
#            f_handle.writelines('Fold, Training time, Testing error, Optimization time, Optimal param\n')
#            np.savetxt(f_handle, data_knn_save, fmt='%s', delimiter=', ')
#            
#        data_svm_save = np.hstack((fold_index.reshape(-1, 1), training_time_svm_save.reshape(-1, 1), testing_error_svm_save.reshape(-1, 1), optimization_time_svm_save.reshape(-1, 1), optimization_value_svm_save.reshape(-1, 1)))
#        filename_svm = save_result_SVM_folder_path + '\\' + dataset_names[dt] + '.csv'
#        
#        open(filename_svm, 'w').close() # make existing file empty
#        
#        with open(filename_svm,'a') as f_handle:
#            f_handle.write('Kernel = RBF, decision = ovo \n')
#            f_handle.writelines('Fold, Training time, Testing error, Optimization time, Optimal param\n')
#            np.savetxt(f_handle, data_svm_save, fmt='%s', delimiter=', ')
#            
#        data_dt_save = np.hstack((fold_index.reshape(-1, 1), training_time_dt_save.reshape(-1, 1), testing_error_dt_save.reshape(-1, 1), optimization_time_dt_save.reshape(-1, 1), optimization_value_dt_save.reshape(-1, 1)))
#        filename_dt = save_result_DT_folder_path + '\\' + dataset_names[dt] + '.csv'
#        
#        open(filename_dt, 'w').close() # make existing file empty
#        
#        with open(filename_dt,'a') as f_handle:
#            f_handle.writelines('Fold, Training time, Testing error, Optimization time, Optimal param\n')
#            np.savetxt(f_handle, data_dt_save, fmt='%s', delimiter=', ')
#            
#        data_gnb_save = np.hstack((fold_index.reshape(-1, 1), training_time_gnb_save.reshape(-1, 1), testing_error_gnb_save.reshape(-1, 1)))
#        filename_gnb = save_result_NB_folder_path + '\\' + dataset_names[dt] + '.csv'
#        
#        open(filename_gnb, 'w').close() # make existing file empty
#        
#        with open(filename_gnb,'a') as f_handle:
#            f_handle.writelines('Fold, Training time, Testing error\n')
#            np.savetxt(f_handle, data_gnb_save, fmt='%s', delimiter=', ')
#       
#        except:
#            pass
        
    print('---Finish---')