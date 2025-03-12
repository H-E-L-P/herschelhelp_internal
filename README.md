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
$ conda create -n herschelhelp_internal
$ conda activate herschelhelp_internal
$ conda install  -c openastronomy astropy healpy scipy seaborn
$ pip install -r requirements.txt
$ pip install -e .
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
$ python -m ipykernel install --user --name helpint --display-name "Python (herschelhelp_internal)"
```

The notebooks must be set to use this kernel.

*Note: maybe it's not mandatory to install the full notebook and only the
ipykernel is required if Jupyter is already installed on the system.*

E(B-V) maps
-----------

To compute the E(B-V) at each source position, we are using the [sfdmap
package](https://github.com/kbarbary/sfdmap) that needs the Schlegel, Finkbeiner
& Davis (1998) dust maps.  The script `get_sfd_maps.sh` will download them and
place them in the directory `herschelhelp_internal/sfd_data/` where they are
expected by our wrapper function.  This must be done only once.
