import pkg_resources

import healpy as hp
import numpy as np
from sfdmap2 import sfdmap
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.table import Column, Table

import yaml

def mag_to_flux(magnitudes, errors_on_magnitudes=None):
    """Convert AB magnitudes and errors to fluxes

    Given AB magnitudes and associated errors, this function returns the
    corresponding fluxes and associated flux errors (in Jy).

    The fluxes are computed with this formula:
        F = 10^((8.9 - Mag)/2.5)
    and the errors on fluxes with this one:
        F_err = ln(10)/2.5 * F * Mag_err


    Parameters
    ----------
    magnitudes: float or array-like of floats
        AB magnitudes of the sources.
    errors_on_magnitudes: float or array-like of floats
        Error on each magnitudes. None if there are no errors.

    Returns
    -------
    fluxes: float or array-like of floats
        The fluxes in Jy.
    errors: float or array-like of floats
        The errors on fluxes in Jy or None.
    """
    magnitudes = np.array(magnitudes)
    fluxes = 10 ** ((8.9 - magnitudes)/2.5)

    if errors_on_magnitudes is not None:
        errors_on_magnitudes = np.array(errors_on_magnitudes)
        errors = np.log(10)/2.5 * fluxes * errors_on_magnitudes
    else:
        errors = None

    return fluxes, errors


def flux_to_mag(fluxes, errors_on_fluxes=None):
    """Convert fluxes and errors to magnitudes

    Given flux densities in Jy with associated errors, this function returns
    the corresponding AB magnitudes en errors.

    The magnitudes are computed with this formula:
        M = 2.5 * (23 - log10(F)) - 48.6
    and the errors on magnitudes with this one
        M_err = 2.5/ln(10) * F_err / F

    Parameters
    ----------
    fluxes: float or array-like of floats
        The fluxes in Jy.
    errors_on_fluxes: float or array-like of floats
        The flux errors in Jy None if there are no errors.

    Returns
    -------
    magnitudes: float or array-like of floats
        The AB magnitudes.
    errors: float or array-like of floats
        The errors on AB magnitudes.
    """
    fluxes = np.array(fluxes)
    magnitudes = 2.5 * (23 - np.log10(fluxes)) - 48.6

    if errors_on_fluxes is not None:
        errors_on_fluxes = np.array(errors_on_fluxes)
        errors = 2.5 / np.log(10) * errors_on_fluxes / fluxes
    else:
        errors = None

    return magnitudes, errors


def aperture_correction(mag, mag_target, stellarity=None, mag_min=None,
                        mag_max=None):
    """Compute aperture correction

    Given some magnitudes and target magnitudes this function computes the
    aperture correction to apply to the first magnitudes to match the targets.
    It may optionnaly use a stellarity index and magnitude limits to make the
    computation on a supset of the provided magnitudes.

    The computation is done by sigma-clipping the difference of magnitudes at
    3 sigma.

    Parameters
    ----------
    mag: array of floats
        The magnitudes to correct.
    mag_target: array of floats
        The target magnitudes. The length of the array must be the same as for
        mag.
    stellarity: array of floats
        The stellarity of each source. If it is provided, only the sources with
        a stellarity above 0.9 will be used in the computation.
    mag_min: float
        If this parameter is provided, only the sources with a magnitude above
        its value will be used in the computation.
    mag_max: float
        If this parameter is provided, only the sources with a magnitude bellow
        its value will be used in the computation.

    Returns
    -------
    mag_diff: float
        The magnitude to add to the magnitudes to match the targets.
    num: integer
        The number of sources used in the computation.
    std: float
        The final standard deviation on the sigma clipping procedure.

    """

    mask = ~np.isnan(mag) & ~np.isnan(mag_target)
    if stellarity is not None:
        mask &= (stellarity > 0.9)
    if mag_min is not None:
        mask &= (mag >= mag_min)
    if mag_max is not None:
        mask &= (mag <= mag_max)

    num = mask.sum()

    if num == 0:
        raise Exception("Not enough sources!")

    # As we are looking for the value to add to the magnitudes to match the
    # target, the difference must be the targets minus the magnitudes.
    mag_diff = mag_target - mag

    _, median, std = sigma_clipped_stats(mag_diff[mask], sigma=3.0)

    # FIXME: Why do we use the median of the sigma clipping? (It's on Eduardo
    # code).

    return median, num, std


def astrometric_correction(coords, ref_coords, max_radius=0.6*u.arcsec,
                           near_ra0=False):
    """Compute the offset between coordinates and reference

    This function compute the RA, Dec offsets between a list of coordinates and
    the list of reference coordinates.  This is used for correcting the
    astrometry of a catalogue with respect to a reference catalogue.

    If the coordinates are around the ra=0 separation, set near_ra0 parameter
    to True.

    Parameters
    ----------
    coords: astropy.coordinates.SkyCoord
        The studied catalogue coordinates.
    ref_coords: astropy.coordinates.SkyCoord
        The coordinates of the reference catalogue.
    max_radius: quantity
        Maximum radius to look for counterparts.
    near_ra0: bool
        Set to True when the coordinates are around the ra=0 limit; the ra will
        be transformed to be between -180 and 180 to avoid large differences
        like 359 - 1 deg. Default to False.

    Returns
    -------
    quantity, quantity
        Tuple delta_RA, delta_Dec to be added to the coordinates to match
        the reference.

    """

    # We cross-match the two catalogues and only keep the matches that are
    # closer that the maximum radius
    idx, d2d, _ = ref_coords.match_to_catalog_sky(coords)
    to_keep = d2d < max_radius

    if near_ra0:
        ra = coords[idx].ra.wrap_at(180 * u.deg)[to_keep]
        ref_ra = ref_coords.ra.wrap_at(180 * u.deg)[to_keep]
    else:
        ra = coords[idx].ra[to_keep]
        ref_ra = ref_coords.ra[to_keep]

    dec = coords[idx].dec[to_keep]
    ref_dec = ref_coords.dec[to_keep]

    # As we want the values to be added to match the reference, the difference
    # must be the reference minus the coordinates.
    ra_diff = ref_ra - ra
    dec_diff = ref_dec - dec

    _, delta_ra, _ = sigma_clipped_stats(ra_diff.arcsec, sigma=3.0)
    _, delta_dec, _ = sigma_clipped_stats(dec_diff.arcsec, sigma=3.0)

    return delta_ra * u.arcsec, delta_dec * u.arcsec


def coords_to_hpidx(ra, dec, order):
    """Convert coordinates to HEALPix indexes

    Given to list of right ascension and declination, this function computes
    the HEALPix index (in nested scheme) at each position, at the given order.

    Parameters
    ----------
    ra: array or list of floats
        The right ascensions of the sources.
    dec: array or list of floats
        The declinations of the sources.
    order: int
        HEALPix order.

    Returns
    -------
    array of int
        The HEALPix index at each position.

    """
    ra, dec = np.array(ra), np.array(dec)

    theta = 0.5 * np.pi - np.radians(dec)
    phi = np.radians(ra)
    healpix_idx = hp.ang2pix(2**order, theta, phi, nest=True)

    return healpix_idx


def inMoc(ra, dec, moc):
    """Find source position in a MOC

    Given a list of positions and a Multi Order Coverage (MOC) map, this
    function return a boolean mask with True for sources that fall inside the
    MOC and False elsewhere.

    Parameters
    ----------
    ra: array or list of floats
        The right ascensions of the sources.
    dec: array or list of floats
        The declinations of the sources.
    moc: pymoc.MOC
        The MOC read by pymoc

    Returns
    -------
    array of booleans
        The boolean mask with True for sources that fall inside the MOC.

    """
    source_healpix_cells = coords_to_hpidx(
        np.array(ra), np.array(dec), moc.order
    )

    # Array of all the HEALpix cell ids of the MOC at its maximum order.
    moc_healpix_cells = np.array(list(moc.flattened()))

    # We look for sources that are in the MOC and return the mask
    return np.in1d(source_healpix_cells, moc_healpix_cells)


def gen_help_id(ra, dec, base_id=b"HELP_J"):
    """Generate the HELP identifiers for a list of sources

    Parameters
    ----------
    ra: array of float
        Right ascensions of the sources in degrees (J2000).
    dec: array of float
        Declinations of the sources in degrees (J2000).
    base_id: bytes
        Begining of the identifiers.

    Returns
    -------
    astropy.table.Column
        A column with HELP identifiers

    """
    ra = np.array(ra) * u.degree
    dec = np.array(dec) * u.degree
    coords = SkyCoord(ra, dec)

    idcol = np.array(coords.to_string(style='hmsdms', precision=3),
                     dtype=np.bytes_)
    idcol = np.char.replace(idcol, b'h', b'')
    idcol = np.char.replace(idcol, b'm', b'')
    idcol = np.char.replace(idcol, b's ', b'')
    idcol = np.char.replace(idcol, b'd', b'')
    idcol = np.char.replace(idcol, b's', b'')
    idcol = np.full(idcol.shape, base_id, dtype=object) + idcol

    return Column(data=idcol.astype(np.bytes_), name="help_id")


def ebv(ra, dec):
    """Computes E(B-V) at the position.

    This function computes the E(B-V) using the sfdmap package.  This package
    uses the E(B-V) values from the Schlegel, Finkbeiner & Davis (1998) dust
    map and a scaling of 0.86 is applied to the values to reflect the
    recalibration by Schlafly & Finkbeiner (2011).

    Parameters
    ----------
    ra: array-like of floats
        The right ascensions of the positions.
    dec: array-like of floats
        The declinations of the positions.

    Returns
    -------
    ebv_col: astropy.table.Column
        An astropy table column named `ebv`.
    """
    dust_maps = sfdmap.SFDMap(
        pkg_resources.resource_filename(__name__, 'sfd_data')
    )

    return Column(dust_maps.ebv(ra, dec), "ebv")



def add_column_meta(catalogue, column_yml_file):
    """Add column descriptions and other meta data
        
    Given a table and a yml file with a list of column names 
    and descriptions and ucd descriptors write all these to 
    the appropriate columns in the table.


    Parameters
    ----------
    catalogue: astropy.table.Table object
        The table to add descriptions to
    column_yml_file: String
        link to yml file describing column definitions.

    Returns
    -------
    catalogue: astropy.table.Table object
        The output catalogue as input but with added column meta.
    
    """
    meta = yaml.load(open(column_yml_file, "r"))
    
    for col in catalogue.itercols():
        try:
            unit = meta[col.name]['unit']
            if unit != "None":
                catalogue[col.name].unit = unit
            catalogue[col.name].description = meta[col.name]['description']
        except KeyError:
            print("Column {} has missing data".format(col.name))

        
    return catalogue
        
    
    


