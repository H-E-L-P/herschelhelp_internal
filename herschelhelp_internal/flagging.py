import numpy as np
from astropy import coordinates as coord
from astropy import units as u
from astropy.table import Column


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
    
    
def flag_outliers(catalogue, col1, col2, errcol1, errcol2, flagcol1, flagcol2, labels=["x", "y"]):
    """Add flags to any pair of bands that have mags outside a limiting chisq diff

    This function create a `flag_gaia` column to be added to a catalogue.  This
    flag indicates the probability of each object being a Gaia object:

    - 0 if no conflicts
    - 1 if conflict with any other band

    Parameters
    ----------
    catalogue : astropy.table.Table
        The catalogue to be flagged.
    col1: string
        First column name.
    col2: string
        Second column name.
    errcol1: string
        First error column name.
    errcol1: string
        Second error column name.
    flagcol1: string
        First flag column name.
    flagcol2: string
        Second flag column name.

    Returns
    -------
    astropy.table.Table
        Flagged catalogue.
    """
    x = catalogue[col1]
    y = catalogue[col2]
    xerr = catalogue[errcol1]
    yerr = catalogue[errcol2]

    
    # Add flag columns if does not exist
    #print(flagcol1, catalogue.colnames, (flagcol1 not in catalogue.colnames))
    if flagcol1 not in catalogue.colnames:
        catalogue.add_column(Column(data = np.zeros(len(catalogue)), dtype=bool), name=flagcol1)
    if flagcol2 not in catalogue.colnames:
        catalogue.add_column(Column(data = np.zeros(len(catalogue)), dtype=bool), name=flagcol2)
    flagx, flagy = catalogue[flagcol1], catalogue[flagcol2]
        
    ## Find outliers  
    # Use only finite values
    #print(len(flagx),len(flagy))
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(xerr) & np.isfinite(yerr)
    x, y = np.copy(x[mask]), np.copy(y[mask])
    xerr, yerr = np.copy(xerr[mask]), np.copy(yerr[mask])
    flagx, flagy = np.copy(flagx[mask]), np.copy(flagy[mask])
    
    # Set the minimum error to 10^-3
    if (len(xerr)!=0 and len(yerr)!=0):
        np.clip(xerr, 5e-3, np.max(xerr), out=xerr)
        np.clip(yerr, 5e-3, np.max(yerr), out=yerr)

    diff = y - x
    
    x_label, y_label = labels
    
    # If the difference is all NaN there is nothing to compare.
    if np.isnan(diff).all():
        print("No sources have both {} and {} values.".format(
            x_label, y_label))
        print("")
        return
    
    diff_label = "{} - {}".format(y_label, x_label)
    print("{}:".format(diff_label))
    
    # Chi2 (Normalized difference)
    ichi2 = np.power(diff, 2) / (np.power(xerr, 2) + np.power(yerr, 2))
    mask2 = ichi2 != 0.0
    diff, ichi2 = np.copy(diff[mask2]), np.copy(ichi2[mask2])
    x, y, xerr, yerr = np.copy(x[mask2]), np.copy(y[mask2]), np.copy(xerr[mask2]), np.copy(yerr[mask2])
    flagx = flagx[mask2]
    flagy = flagy[mask2]

    log_ichi2_25p, log_ichi2_75p = np.percentile(np.log10(ichi2), [25., 75.])
    out_lim = log_ichi2_75p + 3.2*abs(log_ichi2_25p-log_ichi2_75p)
    
    outliers = np.log10(ichi2) > out_lim 
    nb_outliers = len(x[outliers])

    print("  Number of outliers: {}".format(nb_outliers))
    
    
    ## Flag outliers
    #print(nb_outliers,len(flagx[outliers]),len(flagy[outliers]))
    flagx[outliers], flagy[outliers] = np.ones(nb_outliers, dtype=bool), np.ones(nb_outliers, dtype=bool)
    
    catalogue[mask][mask2][flagcol1] = flagx
    catalogue[mask][mask2][flagcol2] = flagy
    
    return catalogue
