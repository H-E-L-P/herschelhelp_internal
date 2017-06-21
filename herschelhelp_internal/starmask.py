import logging
#from collections import Counter

#import matplotlib as mpl
import numpy as np
#import seaborn as sns
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
#from astropy.io import fits
from astropy.wcs import WCS

import pyregion
#import pyregion._region_filter as filter
#import pyregion._region_filter as region_filter


from pymoc import MOC
#from pymoc.util.catalog import catalog_to_cells
import healpy as hp
from scipy.constants import pi


LOGGER = logging.getLogger(__name__)


def create_holes(gaia, 
                 target, 
                 radius = 10 * u.arcsec, 
                 AB = [np.NaN,np.NaN], 
                 mag_lim = 16):
    """Create a ds9 region file of circles around every GAIA object

    This function loops through every object in GAIA in a given field and writes
    a ds9 circle at that location. The default radius is 10 arcsec. This is the
    simplest possible star mask and will be updated to vary as a function of 
    magnitude as well as include rectangular diffraction spikes

    Parameters
    ----------
    gaia: string
        Location of the fits table of GAIA objects in the target field.
    target: string
        Location of output region file.
    radius: astropy quantity (distance)
        Radius for considering sources as duplicates.
    AB: list of two floats
        Parameters defining hole radius as function of magnitude according to 
        10**(A + B * m)

    Returns
    -------
    string
        Location of produced starmask .reg file.

    """
    
    
    
    stars = Table.read(gaia)['field', 'ra','dec', 'phot_g_mean_mag']

    print('There are ' + str(len(stars)) + ' GAIA stars in ' + stars[0]['field'])
    #for star in stars:
    
    #write the ds9regions to file
    f = open(target, 'w+')
    
    for star in stars:
        if star['phot_g_mean_mag'] < 16 and AB ==[np.NaN,np.NaN]:
            f.writelines('circle(' 
                         + str(star['ra']) + ', ' 
                         + str(star['dec']) + ', '
                         + str(radius/u.arcsec) + '")\n')
        elif star['phot_g_mean_mag'] < mag_lim and AB != [np.NaN,np.NaN]:
        	#If AB present then define annulus inner 1 arc sec and out r_50 from AB
            r_50 = (10**(AB[0] + AB[1] * star['phot_g_mean_mag'])) * u.arcsec
            f.writelines('annulus(' 
                         + str(star['ra']) + ', ' 
                         + str(star['dec']) + ', 1.0", '
                         + str(r_50/u.arcsec) + '")\n')
    

    f.close()
    print( 'Starmask written to ' + target)
    #return 'starmask written to data/starmask.reg'
    

def reg2moc(region_file, fieldmoc, target, ra_typ = 0.0, dec_typ = 0.0, order = 12):
    """Convert a ds9 region file into MOC/fits format
    
    This function takes a ds9 region file and converts it to a MOC of given order. 
    
    The default order is 13.
    
    Parameters
    ----------
    region_file: string
        Location of ds9 region file.
    fieldmoc: string
        Location of the MOC describing the unmasked field 
    order: int
        Order of MOC to use

    Returns
    -------
    string
        Location of produced starmask MOC .fits file.
    """
    
    #The code requires an image WCS
    w = WCS("""
WCSAXES =                    2 / Number of coordinate axes
CRPIX1  =                0.5 / Pixel coordinate of reference point
CRPIX2  =                0.5 / Pixel coordinate of reference point
CDELT1  =                0.675 / [deg] Coordinate increment at reference point
CDELT2  =                0.675 / [deg] Coordinate increment at reference point
RADECSYSa= 'ICRS    '           / International Celestial Ref. System
CUNIT1  = 'deg     '                / Units of coordinate increment and value
CUNIT2  = 'deg     '                / Units of coordinate increment and value
CTYPE1  = 'RA---TAN'           / 
CTYPE2  = 'DEC--TAN'           / 
CRVAL1  =                  247.0 / [deg] Coordinate value at reference point
CRVAL2  =                  55.1 / [deg] Coordinate value at reference point
LONPOLE =                  0.0 / [deg] Native longitude of celestial pole
LATPOLE = 90.0 / [deg] Native latitude of celestial pole
""")#.format(ra,dec) )
#CTYPE1  = 'RA---TAN'           / galactic longitude, Hammer-Aitoff projection
#CTYPE2  = 'DEC--TAN'           / galactic latitude, Hammer-Aitoff projection
    
    # The ds9 -> MOC converter
    
    # Use the HELP-coverage moc of the field you want to use
    # available at hedam.lam.fr/HELP/coverages.html
    # example is with the XMM-LSS MOC
    #moc_file = 'XMM-LSS_MOC.fits'
    moc = MOC(filename = fieldmoc)
    
    # The order of detail for the final MOC which  incorporates the region file
    
    NSIDE = hp.order2nside(order)
    
    # calculate the ra and dec of every healpix pixel of order = ORDER
    hp_idx = list(moc.flattened(order))
    theta, phi = hp.pix2ang(NSIDE, hp_idx, nest = True)
    ra,dec = np.degrees(phi), -np.degrees(theta-pi/2)
    x,y = w.wcs_world2pix(ra,dec,0)
    
    # Opens region file, with wcs, needs wcs to convert to pixel coordinates to make the inside function work.
    # If the ds9 region file is in pixel coordinates than the relevent wcs should be used, if not than the suplied
    # wcs will apply the same conversion to the map and the region file.
    
    r = pyregion.open(region_file).as_imagecoord(w)
    # from r a certain region can be selected if not all regions are relevant, for instance r = r[0] or r = [3:9]
    
    r2 = r.get_filter(w)
    
    # bad_region_mask = TRUE if the region file contains the regions which should be removed from the MOC
    # bad_region_mask = FALSE if the region file contains the regions which should be kept.
    bad_region_mask = True
    if bad_region_mask == True:
        in_range = np.ones(np.size(x), dtype=bool)
        mask = r2.inside(x,y)
        in_range[mask] = False
        mask = in_range
    else:
        mask = r2.inside(x,y)
    
    # if there are several different ds9 region files:
    # (in shell) >>> ls *reg > region_files.txt
    # in python:
    # name = np.loadtxt('region_files.txt',usecols = [0],dtype = str)
    # in_range = np.ones(np.size(x), dtype=bool)
    # for j in range(0,np.size(name)):
    #    r = pyregion.open(name[j]).as_imagecoord(w)
    #    r2 = r.get_filter(w2)
    #    in_mask = r2.inside(x,y)
    #    in_range[in_mask] = False
    # mask = in_range
    
    # creates new MOC, with the ds9 region taken out OR only keeping the ds9 region
    hp_idx = np.array(hp_idx)
    MOC(order=order,cells = hp_idx[mask]).write(target)
    
#    
#def cat2moc(starcatfile,fieldmocfile,radius=10*u.arcsec,order=17):
#    """Turn a star catalogue directly into a MOC
#       
#    Parameters
#    ----------
#    starcat: string
#        Location of ds9 region file.
#    fieldmoc: string
#        Location of star catalogue
#    radius: astropy quantity angle
#        size of circle to mask
#    order: int
#        order of MOC cells to use
#
#    Returns
#    -------
#    string
#        Location of produced starmask MOC .fits file.
#    """
#    startable = Table.read(starcatfile)
#    starcat = SkyCoord.guess_from_table(startable)
#    cells = catalog_to_cells(starcat, radius, order)
#    fieldmoc = MOC(fieldmocfile)
#    maskedmoc = fieldmoc.remove

def flag_artefacts(masterlist, starmask):
    """Take a masterlist and flag sources in artefact producing region of star
    
    Parameters
    ----------
    masterlist: astropy Table
        Full list of sources
    starmask: string
        Location of starmask
    """
    print('function not written yet')

if __name__ == '__main__':
    #create_starmask('data/GAIA_CDFS-SWIRE.fits', 'data/starmask_CDFS-SWIRE.reg')
    reg2moc('data/holes_ELAIS-N1.reg',
            'data/ELAIS-N1_MOC.fits',
            'data/holes-ELAIS-N1-O17_MOC.fits',
            order=17)
    
