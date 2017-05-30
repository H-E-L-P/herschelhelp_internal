import logging
from collections import Counter

import matplotlib as mpl
import numpy as np
import seaborn as sns
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Column, hstack, vstack
from matplotlib import pyplot as plt


LOGGER = logging.getLogger(__name__)


def create_starmask(gaia, radius = 10 * u.arcsec):
    """Create a ds9 region file of circles around every GAIA object

    This function loops through every object in GAIA in a given field and writes
    a ds9 circle at that location. The default radius is 10 arcsec. This is the
    simplest possible star mask and will be updated to vary as a function of 
    magnitude as well as include rectangular diffraction spikes

    Parameters
    ----------
    gaia: string
        Location of the fits table of GAIA objects in the target field.
    radius: astropy quantity (distance)
        Radius for considering sources as duplicates.

    Returns
    -------
    string
        Location of produced starmask .reg file.

    """
    print('test')
    #stars = Table.open(gaia)
    starmask = 'final location'
    return starmask
    

def ds9tomoc(ds9file, order = 13):
    """Convert a ds9 region file into a moc
    
    This function takes a ds9 region file and converts it to a MOC of given order. 
    
    The default order is 13.
    """


