Herschel Extragalactic Legacy Project internal python module
============================================================

This python module contains the code used internally within HELP to process our
data.

Installation
------------

The easiest way is to install the module in development mode using conda.
Provided you installed [Anaconda](https://www.continuum.io/) or
[miniconda](http://conda.pydata.org/miniconda.html) here are the commands to
run:

```Shell
$ conda create -n herschelhelp_internal python=3
$ source activate herschelhelp_internal
$ conda install  -c openastronomy astropy healpy scipy pymocs seaborn
$ pip install -r requirements.txt
$ python setup.py develop
```

You will need to activate this new environment with `source activate
herschelhelp_internal` when you want to use this module.

Running inside a Jupyter notebook
---------------------------------

If one need to use this code inside a Jupyter notebook, one should install the
notebook stuff:

```Shell
$ conda install notebook
```

and add the environment Python to the list of kernels for Jupyter:

```Shell
$ python -m ipykernel install --user --name help_int --display-name "Python (herschelhelp_internal)"
```

The notebooks must be set to use this kernel.

*Note: maybe it's not mandatory to install the full notebook and only the
ipykernel is required if Jupyter is already installed on the system.*
