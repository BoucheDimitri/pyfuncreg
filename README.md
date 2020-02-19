# Nonlinear functional regressions in Python

## Dependencies
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

## Data
The data for the experiments are included in the folder `./data` of the repository. 

## Quick start
The different experimental scripts are in the folder `./expes/expes_scripts` and
thus the experiments can be run directly. 

### DTI dataset
Scripts for the DTI datasets are in the folder
`./expes/expes_scripts/dti`.
There are two modes of execution. For any script **script.py** in this folder, running 