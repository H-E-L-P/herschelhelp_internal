import numpy as np
from astropy import coordinates as coord
from astropy import units as u
from astropy.table import Column, Table


def gaia_flag_column(cat_coords, cat_epoch, gaia_cat):
    """Create a Gaia flag column

    This function create a `flag_gaia` column to be added to a catalogue.  This
    flag indicates the probability of each object being a Gaia object:

    - 1 if the object is possibly a Gaia object (the nearest Gaia source is
      between 1.5 arcsec and 2 arcsec).
    - 2 if the object is probably a Gaia object (the nearest Gaia source is
      between 0.6 arcsec and 1.5 arcsec).
    - 3 if the object is definitely a Gaia object (the nearest Gaia source is
      nearer than 0.6 arcsec).
    - 0 otherwise (the nearest Gaia source is farer than 2 arcsec).

    Parameters
    ----------
    cat_coords : astropy.coordinates.SkyCoord
        The coordinates of the objects in the catalogue.
    cat_epoch: int
        Year of observation of the catalogue. This value is used to correct the
        position of Gaia object to their position at the data of observation.
    gaia_cat : astropy.table.Table
        Gaia catalogue from HELP database.

    Returns
    -------
    astropy.table.Column
        The flag column to be added to the catalogue.
    """

    flag = np.full(len(cat_coords), 0, dtype=int)

    # Gaia positions
    gaia_ra = np.array(gaia_cat['ra'])
    gaia_dec = np.array(gaia_cat['dec'])
    gaia_pmra = np.array(gaia_cat['pmra'])
    gaia_pmdec = np.array(gaia_cat['pmdec'])

    # The proper motion is not available everywhere. We set it to 0 where it's
    # not available.
    gaia_pmra[np.isnan(gaia_pmra)] = 0.0
    gaia_pmdec[np.isnan(gaia_pmdec)] = 0.0

    # Correct Gaia positions with proper motion. Gaia gives positions at epoch
    # 2015 while the catalogue was extracted from observations at a previous
    # time.
    gaia_ra += gaia_pmra / 1000. / 3600. * (cat_epoch-2015) * \
        np.cos(np.deg2rad(np.array(gaia_dec)))
    gaia_dec += gaia_pmdec / 1000. / 36000. * (cat_epoch-2015)

    # Gaia positions at the catalogue epoch
    gaia_pos = coord.SkyCoord(gaia_ra * u.degree, gaia_dec * u.degree)

    # Get all the catalogue sources within 2 arcsec of a Gaia object.
    idx_galaxy, _, d2d, _ = gaia_pos.search_around_sky(
        cat_coords, 2 * u.arcsec)

    # All these sources are possible Gaia objects.
    flag[idx_galaxy] = 1

    # Those that are nearer the 1.5 arcsec are probable Gaia objects.
    flag[idx_galaxy[d2d <= 1.5 * u.arcsec]] = 2

    # Those that are nearer the 0.6 arcsec are definitely Gaia objects.
    flag[idx_galaxy[d2d <= .6 * u.arcsec]] = 3

    return Column(flag, "flag_gaia")
