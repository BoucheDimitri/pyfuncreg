# Nonlinear functional regressions in Python

## I / Dependencies
This code was developped using Python 3.7

The following packages are needed on top of the built-in Python packages:
- numpy
- scipy
- pywt
- scikit-learn
- pandas
- multiprocessing
- psutil
- pathlib
- slycot
- control

We strongly advise to use an Anaconda distribution of Python 3.7
which contains already most packages mentionned above, leaving to install only the following ones:
- multiprocessing
- psutil
- pathlib
- slycot
- control

**Important**: note that the packages **slycot** and **control** are easily installed
using conda by running the commands, whereas it proved problematic to do so using pip
for some configurations. 
```
conda install -c conda-forge slycot
conda install -c conda-forge control
```

## II / Running instructions
The different experimental scripts are in the folder `./expes/expes_scripts` 
and the data for the experiments are included in the folder `./data` of the repository.
thus the experiments can be run directly in the following way. 

### 1. DTI dataset
Scripts for the DTI datasets are in the folder
`./expes/expes_scripts/dti`. There are two modes of execution. 

#### a. Quick run
For any script **script.py** in this folder, running 
```
python script.py
```
will fit the regressors using the parameters obtained by cross-validation, compute the MSE on test set and print it.

#### b. Full run
For any script **script.py** in this folder, running 
```
python script.py full
```
will perform the full tuning procedure of the method by cross-validation. It will
save the resulting configuration in a pickle file in the folder defined by the 
global variable **REC_PATH** in each script. This pickle file can be loaded in the following way: 

```
with open(path_to_file, "rb") as inp:
    best_dict, best_result, score_test = pickle.load(inp)
```
where 
- **best_dict** is the configuration selected by cross-validation
- **best_result** is the average cross-validation score for this configuration
- **score_test** is the score obtained on test set with this selected configuration
- **path_to_file** is the path to the pickled file

**SOME WARNINGS AND REMARKS**
- Those experiments take a long time to run, however, smaller grids can be used
by modifying the corresponding global variables in the "**Regressor config**" section at the beginning
of the scripts.  
- The paths for recording the experiments as well as the number of processors to 
use for parallel execution can be set in the **Config** section in each script. 

### 2. Speech dataset
Scripts for the speech datasets are in the folder
`./expes/expes_scripts/speech`. There are two modes of execution. 

#### a. Quick run
For any script **script.py** in this folder, running 
```
python script.py VT
```
will fit the regressors using the parameters obtained by cross-validation, compute the MSE on test set and print it.
Where **VT** corresponds to the vocal tract for which to run the experiment. As a reminder there
are 8 of them, thus **VT** should be in the set {"LA", "LP", "TBCL", "VEL", "GLO", "TTCL", "TTCD"}.

#### b. Full run
For any script **script.py** in this folder, running 
```
python script.py VT full
```
will perform the full tuning procedure of the method by cross-validation. It will
save the resulting configuration in a pickle file in the folder defined by the 
global variable **REC_PATH** in each script. Where **VT** corresponds to the vocal tract for which to run the experiment. As a reminder there
are 8 of them, thus **VT** should be in the set {"LA", "LP", "TBCL", "VEL", "GLO", "TTCL", "TTCD"}.
The **VT** will be added to the name of the pickle file to avoid confusion. 

This pickle file can be loaded in the following way: 

```
with open(path_to_file, "rb") as inp:
    best_dict, best_result, score_test = pickle.load(inp)
```
where 
- **best_dict** is the configuration selected by cross-validation
- **best_result** is the average cross-validation score for this configuration
- **score_test** is the corresponding list of scores obtained on test set with this selected configuration
- **path_to_file** is the path to the pickled file

**SOME WARNINGS AND REMARKS**
- Those experiments take a long time to run, however, smaller grids can be used
by modifying the corresponding global variables in the "**Regressor config**" section at the beginning
of the scripts.  
- The paths for recording the experiments as well as the number of processors to 
use for parallel execution can be set in the **Config** section in each script. 

### 1. Toy dataset
Scripts for the toy datasets are in the 
`./expes/expes_scripts/toy`. 

For any script **script.py** in this folder, running 
```
python script.py
```
will run the full experiments used to generate the plots in the main paper. 
This will record the results in a pickle file which can be loaded the following way: 

```
with open(path_to_file, "rb") as inp:
    input_config, score_test = pickle.load(inp)
```
where 
- **input_config** corresponds to the configuration for the experiment. For **output_noise.py**, 
this corresponds to the level of noise, and for **output_missing.py** this corresponds to 
the couple (number_of_samples, percentage_of_missing_data).
- **score_test** is the corresponding list of scores obtained on test set.  

**SOME WARNINGS AND REMARKS**
- Those experiments take a long time to run, however, smaller grids can be used
by modifying the corresponding global variables in the "**Experiment parameters**" section at the beginning
of the scripts.  
- The paths for recording the experiments can be set in the **Config** section in each script. 