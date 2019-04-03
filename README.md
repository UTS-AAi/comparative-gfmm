# comparative-gfmm
This repository contains the implementations of fuzzy min-max neural networks

Before running the python files in this project, we need to install packages first as follows:

Open command line in Windows and navigate to the Hyperbox-classifier folder, and then type the following command:

    python setup.py install

<i>Instruction of executing the online version of GFMM (file: onlinegfmm.py)</i>:

    python onlinegfmm.py arg1 arg2 arg3 arg4 arg5 arg6 arg7 arg8 arg9 arg10

INPUT parameters from command line:

    arg1:  + 1 - training and testing datasets are located in separated files
           + 2 - training and testing datasets are located in the same files
    arg2:  path to file containing the training dataset (arg1 = 1) or both training and testing datasets (arg1 = 2)
    arg3:  + path to file containing the testing dataset (arg1 = 1)
           + percentage of the training dataset in the input file
    arg4:  + True: drawing hyperboxes during the training process
           + False: no drawing
    arg5:  Maximum size of hyperboxes (teta, default: 1)
    arg6:  The minimum value of maximum size of hyperboxes (teta_min: default = teta)
    arg7:  gamma value (default: 1)
    arg8:  Operation used to compute membership value: 'min' or 'prod' (default: 'min')
    arg9:  Do normalization of datasets or not? True: Normilize, False: No normalize (default: True)
    arg10: range of input values after normalization (default: [0, 1])

Note: parameters with string datatype should be put between quotation marks (" ") </br>

Example:

    python onlinegfmm.py 1 synthetic_train.dat synthetic_test.dat True 0.6 0.5 1 min True "[0, 1]"
    
If using Spyder to run the source code, let's configure Spyder as follows:

From <b>Run/Configuration</b> per file or press <b>Ctr+F6</b>, on the open window, select <i>onlinegfmm.py</i> in the field <b>Select a run configuration</b> check on <b>Command line options</b> and input the input parameters such as: <i>1 synthetic_train.dat synthetic_test.dat True 0.6 0.5 1 min True "[0, 1]"</i>.

To the drawing display on the separated window (not inline mode as default), from <b>Tools</b> choose <b>Preferences</b>, and then select <b>IPython console</b>, in tab <b>Graphics</b> let change the value of the field <b>backends</b> to <b>Qt5</b> or <b>Qt4</b>, choose <b>OK</b>. Finally, restart Spyder to update the changes.

<i>Instruction of executing the batch learning versions of GFMM (files: accelbatchgfmm.py, batchgfmm_v1.py, batchgfmm_v2.py)</i>: </br>
<b> Full batch learning</b>: </br>

    python batchgfmm_v2.py arg1 arg2 arg3 arg4 arg5 arg6 arg7 arg8 arg9 arg10 arg11 arg12
    
<b> Faster version of full batch learning</b>: </br>

    python batchgfmm_v1.py arg1 arg2 arg3 arg4 arg5 arg6 arg7 arg8 arg9 arg10 arg11 arg12
    
<b> Improved version of batch learning - AGGLO2 </b>: </br>

    python accelbatchgfmm.py arg1 arg2 arg3 arg4 arg5 arg6 arg7 arg8 arg9 arg10 arg11 arg12

INPUT parameters from command line: </br>

    arg1:  + 1 - training and testing datasets are located in separated files
           + 2 - training and testing datasets are located in the same files
    arg2:  path to file containing the training dataset (arg1 = 1) or both training and testing datasets (arg1 = 2)
    arg3:  + path to file containing the testing dataset (arg1 = 1)
           + percentage of the training dataset in the input file
    arg4:  + True: drawing hyperboxes during the training process
           + False: no drawing
    arg5:  Maximum size of hyperboxes (teta, default: 1)
    arg6:  gamma value (default: 1)
    arg7:  Similarity threshod (default: 0.5)
    arg8:  Similarity measure: 'short', 'long' or 'mid' (default: 'mid')
    arg9:  operation used to compute membership value: 'min' or 'prod' (default: 'min')
    arg10: do normalization of datasets or not? True: Normilize, False: No normalize (default: True)
    arg11: range of input values after normalization (default: [0, 1])   
    arg12: Use 'min' or 'max' (default) memberhsip in case of assymetric similarity measure (simil='mid')
    
For instance: </br>

<i> Full batch learning: </i>

    python batchgfmm_v2.py 1 synthetic_train.dat synthetic_test.dat True 0.6 1 0.5 short min True "[0, 1]"
    
<i> Improved version of batch learning - AGGLO2: </i>

    python accelbatchgfmm.py 1 synthetic_train.dat synthetic_test.dat True 0.6 1 0.5 mid min True "[0, 1]"
