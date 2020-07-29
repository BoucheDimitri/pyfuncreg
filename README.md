# Nonlinear functional regressions in Python

## I / Dependencies
This code was developped using Python 3.7

The following packages are needed on top of the built-in Python packages:
- numpy
- scipy
- pywt
- scikit-learn
- pandas
- matplotlib
- multiprocessing
- psutil
- pathlib
- slycot
- python_speech_features

We strongly advise to use an Anaconda distribution of Python 3.7
which contains already most packages mentionned above, leaving to install only the following ones:
- multiprocessing
- psutil
- pathlib
- slycot
- python_speech_features

**Important**: note that the packages **slycot** is easily installed
using conda by running the commands, whereas it proved problematic to do so using pip
for some configurations. 
```
conda install -c conda-forge slycot
```
