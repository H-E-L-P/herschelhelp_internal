import logging
from collections import Counter
from itertools import product
from glob import glob

import matplotlib as mpl
import numpy as np
import seaborn as sns
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.table import Column, hstack, Table, vstack
from astropy import visualization as vz
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde

from IPython.display import display

from .utils import aperture_correction

LOGGER = logging.getLogger(__name__)


def remove_duplicates(table, ra_col="ra", dec_col="dec",
                      radius=0.4*u.arcsec, sort_col=None, reverse=False,
                      flag_name="flag_cleaned"):
    """Remove duplicates from a catalogue

    This function remove duplicated sources in a catalogue. The duplicated
    sources are remove by crossmatching the table with itself and keeping the
    first source in each match.  The source kept is the first in the table but
    column names can be given to sort the table prior to removing the
    duplicates.

    Note that the duplicate removing percolates.  If A is close enough to B and
    B close enough to C, B and C will be removed, even if A is far enough
    from C.

    A flag column is added to the table containing True for sources that where
    associated to other ones during the cleaning.

    Parameters
    ----------
    table: astropy.table.Table
        The catalogue to remove duplicates from.
    ra_col: string
        Name of the right ascension column. This column must contain decimal
        degrees.
    dec_col: string
        Name of the declination column. This column must contain decimal
        degrees.
    radius: astropy quantity (distance)
        Radius for considering sources as duplicates.
    sort_col: list of strings
        If given, the catalogue will be sorted by these columns (ascending)
        before removing the duplicates. Only the first row will be taken.
    reverse: boolean
        If true, the sorted table will also be reversed.
    flag_name: string
        Name of the column containing the duplication flag to add to the
        catalogue.

    Returns
    -------
    astropy.table.Table
        A new table with the duplicated sources removed and the flag column
        added.

    """
    table = table.copy()

    if sort_col is not None:
        table.sort(sort_col)

    if reverse:
        table.reverse()

    # Position must be given in degrees
    table[ra_col].unit = u.deg
    table[dec_col].unit = u.deg

    coords = SkyCoord(table[ra_col], table[dec_col])
    idx1, idx2, _, _ = coords.search_around_sky(coords, radius)

    # We remove the association of each source to itself
    mask = (idx1 != idx2)
    idx1 = idx1[mask]
    idx2 = idx2[mask]

    # The remaining indexes are those of duplicated sources (note that idx1 ans
    # idx2 contain the same indexes in a different order). We use them to add
    # flag the sources that have duplicates.
    # We set the fill_value of this column to False so that when we stack some
    # table with astropy, the missing data will be filled with False.
    table.add_column(Column(
        name=flag_name,
        data=np.zeros(len(table)),
        dtype=bool
    ))
    table[flag_name].fill_value = False
    table[flag_name][np.unique(idx1)] = True

    # As we sorted the table (if we don't sort, it does not matter) the lower
    # indexes are the most important. We can look at the idx1 list and remove
    # all the sources that are associated to another source with a lower index.
    remove_idx = idx1[idx1 > idx2]
    keep_idx = np.in1d(np.arange(len(table)), remove_idx, invert=True)

    return table[keep_idx]


def remove_duplicates_tiled(table, ra_col="ra", dec_col="dec",
                            radius=0.4*u.arcsec, sort_col=None, reverse=False,
                            flag_name="flag_cleaned", near_ra0=False,
                            tile_side=1):
    """Remove duplicates from a large catalogue

    This function removes the duplicates from a catalogue too large to be
    treated by the remove_duplicates function.  It works dividing the catalogue
    in tiles processed separately and merges back the results.

    There is a near_ra0 parameter to be set to True when the catalogue overlaps
    the ra=0 meridian.

    Parameters
    ----------
    table: astropy.table.Table
        The catalogue to remove duplicates from.
    ra_col: string
        Name of the right ascension column. This column must contain decimal
        degrees.
    dec_col: string
        Name of the declination column. This column must contain decimal
        degrees.
    radius: astropy quantity (distance)
        Radius for considering sources as duplicates.
    sort_col: list of strings
        If given, the catalogue will be sorted by these columns (ascending)
        before removing the duplicates. Only the first row will be taken.
    reverse: boolean
        If true, the sorted table will also be reversed.
    flag_name: string
        Name of the column containing the duplication flag to add to the
        catalogue.
    near_ra0: boolean
        Set to True when the catalogue overlaps the ra=0 meridian.
        Default: False.
    tile_side: float
        Side length of the tile we are dividing the catalogue into for
        processing, in degree. Default: 1.

    Returns
    -------
    astropy.table.Table
        A new table with the duplicated sources removed and the flag column
        added.
    """

    def _get_coordinates(table, ra_col, dec_col, near_ra0):
        """Function to get the coordinates from a table returning right
        ascension between -180 and 180 when around ra=0"""
        coords = SkyCoord(table[ra_col], table[dec_col])
        dec = coords.dec.value
        if near_ra0:
            ra = coords.ra.wrap_at(180 * u.deg).value
        else:
            ra = coords.ra.value
        return ra, dec

    # Position must be given in degrees
    table[ra_col].unit = u.deg
    table[dec_col].unit = u.deg

    cat_ra, cat_dec = _get_coordinates(table, ra_col, dec_col, near_ra0)

    cat_ra_min, cat_ra_max = np.percentile(cat_ra, [0., 100.])
    cat_dec_min, cat_dec_max = np.percentile(cat_dec, [0., 100.])

    cat_ra_width = cat_ra_max - cat_ra_min
    cat_dec_width = cat_dec_max - cat_dec_min

    # Number of tiles in the ra and dec directions
    nb_tiles_ra, nb_tiles_dec = np.ceil(
        [cat_ra_width/tile_side, cat_dec_width/tile_side]).astype(int)
    LOGGER.info("The catalogue is divided in %i x %i (RA, Dec) tiles",
                nb_tiles_ra, nb_tiles_dec)

    result_table = None

    for idx_tile_ra, idx_tile_dec in product(range(nb_tiles_ra),
                                             range(nb_tiles_dec)):
        tile_ra_min = cat_ra_min + idx_tile_ra * tile_side
        tile_ra_max = tile_ra_min + tile_side
        tile_dec_min = cat_dec_min + idx_tile_dec * tile_side
        tile_dec_max = tile_dec_min + tile_side

        LOGGER.info("Processing RA between %f and %f, and Dec between %f "
                    "and %f", tile_ra_min, tile_ra_max, tile_dec_min,
                    tile_dec_max)

        # As we are cleaning objects based on their surrounding, we must add
        # a margin to the tiles we are processing.  We use a margin of
        # 3 arc-second i.e. 0.05°.
        tile_mask = (cat_ra >= tile_ra_min - 0.05) & \
                    (cat_ra <= tile_ra_max + 0.05) & \
                    (cat_dec >= tile_dec_min - 0.05) & \
                    (cat_dec <= tile_dec_max + 0.05)

        tmp_result = remove_duplicates(table[tile_mask], ra_col=ra_col,
                                       dec_col=dec_col, radius=radius,
                                       sort_col=sort_col, reverse=reverse,
                                       flag_name=flag_name)

        # From this result, we must only keep the sources that are strictly
        # inside the tile, we keep only the right/top border of the tile
        # except for the first tiles in the row/column.
        tmp_ra, tmp_dec = _get_coordinates(tmp_result, ra_col, dec_col,
                                           near_ra0)
        if idx_tile_ra == 0:
            tmp_tile_mask = (tmp_ra >= tile_ra_min) & (tmp_ra <= tile_ra_max)
        else:
            tmp_tile_mask = (tmp_ra > tile_ra_min) & (tmp_ra <= tile_ra_max)

        if idx_tile_dec == 0:
            tmp_tile_mask &= (tmp_dec >= tile_dec_min) & \
                             (tmp_dec <= tile_dec_max)
        else:
            tmp_tile_mask &= (tmp_dec > tile_dec_min) & \
                             (tmp_dec <= tile_dec_max)

        if result_table is None:
            result_table = tmp_result[tmp_tile_mask]
        else:
            result_table = vstack([result_table, tmp_result[tmp_tile_mask]])

    return result_table


def merge_catalogues(cat_1, cat_2, racol_2, decol_2, radius=0.4*u.arcsec):
    """Merge two catalogues

    This function merges the second catalogue into the first one using the
    given radius to associate identical sources.  This function takes care to
    associate only one source of one catalogue to the other.  The sources that
    may be associated to various counterparts in the other catalogue are
    flagged as “maybe spurious association” with a true value in the
    flag_merged column.  If this column is present in the first catalogue, it's
    content is “inherited” during the merge.

    Parameters
    ----------
    cat_1: astropy.table.Table
        The table containing the first catalogue.  This is the master catalogue
        used during the merge.  If it has a “flag_merged” column it's content
        will be re-used in the flagging of the spurious merges.  This catalogue
        must contain a ‘ra’ and a ‘dec’ columns with the position in decimal
        degrees.
    cat_2: astropy.table.Table
        The table containing the second catalogue.
    racol_2: string
        Name of the column in the second table containing the right ascension
        in decimal degrees.
    decol_2: string
        Name of the column in the second table containing the declination in
        decimal degrees.
    radius: astropy.units.quantity.Quantity
        The radius to associate identical sources in the two catalogues.

    Returns
    -------
    astropy.table.Table
        The merged catalogue.

    """
    cat_1['ra'].unit = u.deg
    cat_1['dec'].unit = u.deg
    coords_1 = SkyCoord(cat_1['ra'], cat_1['dec'])

    cat_2[racol_2].unit = u.deg
    cat_2[decol_2].unit = u.deg
    coords_2 = SkyCoord(cat_2[racol_2], cat_2[decol_2])

    # Search for sources in second catalogue matching the sources in the first
    # one.
    idx_2, idx_1, d2d, _ = coords_1.search_around_sky(coords_2, radius)

    # We want to flag the possible mis-associations, i.e. the sources in each
    # catalogue that are associated to several sources in the other one, but
    # also all the sources that are associated to a problematic source in the
    # other catalogue (e.g. if two sources in the first catalogue are
    # associated to the same source in the second catalogue, they must be
    # flagged as potentially problematic).
    #
    # Search for duplicate associations
    toflag_idx_1 = np.unique([item for item, count in Counter(idx_1).items()
                              if count > 1])
    toflag_idx_2 = np.unique([item for item, count in Counter(idx_2).items()
                              if count > 1])
    # Flagging the sources associated to duplicates
    dup_associated_in_idx1 = np.in1d(idx_2, toflag_idx_2)
    dup_associated_in_idx2 = np.in1d(idx_1, toflag_idx_1)
    toflag_idx_1 = np.unique(np.concatenate(
        (toflag_idx_1, idx_1[dup_associated_in_idx1])
    ))
    toflag_idx_2 = np.unique(np.concatenate(
        (toflag_idx_2, idx_2[dup_associated_in_idx2])
    ))

    # Adding the flags to the catalogue.  In the second catalogue, the column
    # is named "flag_merged_2" and will be combined to the flag_merged column
    # one the merge is done.
    try:
        cat_1["flag_merged"] |= np.in1d(np.arange(len(cat_1), dtype=int),
                                        toflag_idx_1)
    except KeyError:
        cat_1.add_column(Column(
            data=np.in1d(np.arange(len(cat_1), dtype=int), toflag_idx_1),
            name="flag_merged"
        ))
    cat_2.add_column(Column(
        data=np.in1d(np.arange(len(cat_2), dtype=int), toflag_idx_2),
        name="flag_merged_2"
    ))

    # Now that we have flagged the maybe spurious associations, we want to
    # associate each source of each catalogue to at most one source in the
    # other one.

    # We sort the indices by the distance to take the nearest counterparts in
    # the following steps.
    sort_idx = np.argsort(d2d)
    idx_1 = idx_1[sort_idx]
    idx_2 = idx_2[sort_idx]

    # These array will contain the indexes of the matching sources in both
    # catalogues.
    match_idx_1 = np.array([], dtype=int)
    match_idx_2 = np.array([], dtype=int)

    while len(idx_1) > 0:

        both_first_idx = np.sort(np.intersect1d(
            np.unique(idx_1, return_index=True)[1],
            np.unique(idx_2, return_index=True)[1],
        ))

        new_match_idx_1 = idx_1[both_first_idx]
        new_match_idx_2 = idx_2[both_first_idx]

        match_idx_1 = np.concatenate((match_idx_1, new_match_idx_1))
        match_idx_2 = np.concatenate((match_idx_2, new_match_idx_2))

        # We remove the matching sources in both catalogues.
        to_remove = (np.in1d(idx_1, new_match_idx_1) |
                     np.in1d(idx_2, new_match_idx_2))
        idx_1 = idx_1[~to_remove]
        idx_2 = idx_2[~to_remove]

    # Indices of un-associated object in both catalogues.
    unmatched_idx_1 = np.delete(np.arange(len(cat_1), dtype=int),match_idx_1)
    unmatched_idx_2 = np.delete(np.arange(len(cat_2), dtype=int),match_idx_2)

    # Sources only in cat_1
    only_in_cat_1 = cat_1[unmatched_idx_1]

    # Sources only in cat_2
    only_in_cat_2 = cat_2[unmatched_idx_2]
    # We are using the ra and dec columns from cat_2 for the position.
    only_in_cat_2[racol_2].name = "ra"
    only_in_cat_2[decol_2].name = "dec"

    # Merged table of sources in both catalogues.
    both_in_cat_1_and_cat_2 = hstack([cat_1[match_idx_1], cat_2[match_idx_2]])
    # We don't need the positions from the second catalogue anymore.
    both_in_cat_1_and_cat_2.remove_columns([racol_2, decol_2])

    # Logging the number of rows
    LOGGER.info("There are %s sources only in the first catalogue",
                len(only_in_cat_1))
    LOGGER.info("There are %s sources only in the second catalogue",
                len(only_in_cat_2))
    LOGGER.info("There are %s sources in both catalogues",
                len(both_in_cat_1_and_cat_2))

    merged_catalogue = vstack([only_in_cat_1, both_in_cat_1_and_cat_2,
                               only_in_cat_2])

    # When vertically stacking the catalogues, some values in the flag columns
    # are masked because they did not exist in the catalogue some row originate
    # from. We must set them to the appropriate value.
    for colname in merged_catalogue.colnames:
        if 'flag' in colname:
            merged_catalogue[colname][merged_catalogue[colname].mask] = False

    # We combined the flag_merged flags
    merged_catalogue['flag_merged'] |= merged_catalogue['flag_merged_2']
    merged_catalogue.remove_column('flag_merged_2')

    return merged_catalogue


def merge_catalogues_tiled(cat1,
                           cat2, cat2_ra_col, cat2_dec_col,
                            radius=0.4*u.arcsec,
                            near_ra0=False,
                            tile_side=1):
    """Merge two catalogues on a tile by tile basis to optimise memory usage

    This function merges the second catalogue into the first one using the
    given radius to associate identical sources.  This function takes care to
    associate only one source of one catalogue to the other.  The sources that
    may be associated to various counterparts in the other catalogue are
    flagged as “maybe spurious association” with a true value in the
    flag_merged column.  If this column is present in the first catalogue, it's
    content is “inherited” during the merge.

    Parameters
    ----------
    cat_1: astropy.table.Table
        The table containing the first catalogue.  This is the master catalogue
        used during the merge.  If it has a “flag_merged” column it's content
        will be re-used in the flagging of the spurious merges.  This catalogue
        must contain a ‘ra’ and a ‘dec’ columns with the position in decimal
        degrees.
    cat_2: astropy.table.Table
        The table containing the second catalogue.
    cat2_racol_2: string
        Name of the column in the second table containing the right ascension
        in decimal degrees.
    cat2_decol_2: string
        Name of the column in the second table containing the declination in
        decimal degrees.
    radius: astropy.units.quantity.Quantity
        The radius to associate identical sources in the two catalogues.

    Returns
    -------
    astropy.table.Table
        The merged catalogue.
    """

    def _get_coordinates(table, ra_col, dec_col, near_ra0):
        """Function to get the coordinates from a table returning right
        ascension between -180 and 180 when around ra=0"""
        coords = SkyCoord(table[ra_col], table[dec_col])
        dec = coords.dec.value
        if near_ra0:
            ra = coords.ra.wrap_at(180 * u.deg).value
        else:
            ra = coords.ra.value
        return ra, dec

    # Position must be given in degrees
    cat1["ra"].unit = u.deg
    cat1["dec"].unit = u.deg

    cat1_ra, cat1_dec = _get_coordinates(cat1, "ra", "dec", near_ra0)
    cat2_ra, cat2_dec = _get_coordinates(cat2, cat2_ra_col, cat2_dec_col, near_ra0)

    both_ra_min, both_ra_max = np.percentile(np.append(cat1_ra, cat2_ra), [0., 100.])
    both_dec_min, both_dec_max = np.percentile(np.append(cat1_dec, cat2_dec), [0., 100.])

    both_ra_width = both_ra_max - both_ra_min
    both_dec_width = both_dec_max - both_dec_min

    # Number of tiles in the ra and dec directions
    nb_tiles_ra, nb_tiles_dec = np.ceil(
        [both_ra_width/tile_side, both_dec_width/tile_side]).astype(int)
    LOGGER.info("The catalogue is divided in %i x %i (RA, Dec) tiles",
                nb_tiles_ra, nb_tiles_dec)

    result_table = None

    for idx_tile_ra, idx_tile_dec in product(range(nb_tiles_ra),
                                             range(nb_tiles_dec)):
        tile_ra_min = both_ra_min + idx_tile_ra * tile_side
        tile_ra_max = tile_ra_min + tile_side
        tile_dec_min = both_dec_min + idx_tile_dec * tile_side
        tile_dec_max = tile_dec_min + tile_side

        LOGGER.info("Processing RA between %f and %f, and Dec between %f "
                    "and %f", tile_ra_min, tile_ra_max, tile_dec_min,
                    tile_dec_max)

        # As we are merging objects based on their surrounding, we must add
        # a margin to the tiles we are processing.  We use a margin of
        # 3 arc-second i.e. 0.05°.
        cat1_tile_mask = (cat1_ra >= tile_ra_min - 0.05) & \
                    (cat1_ra <= tile_ra_max + 0.05) & \
                    (cat1_dec >= tile_dec_min - 0.05) & \
                    (cat1_dec <= tile_dec_max + 0.05)

        cat2_tile_mask = (cat2_ra >= tile_ra_min - 0.05) & \
                    (cat2_ra <= tile_ra_max + 0.05) & \
                    (cat2_dec >= tile_dec_min - 0.05) & \
                    (cat2_dec <= tile_dec_max + 0.05)

        tmp_result = merge_catalogues(cat1[cat1_tile_mask],
                                      cat2[cat2_tile_mask],
                                       cat2_ra_col,
                                       cat2_dec_col,
                                       radius=radius)


        # From this result, we must only keep the sources that are strictly
        # inside the tile, we keep only the right/top border of the tile
        # except for the first tiles in the row/column.
        tmp_ra, tmp_dec = _get_coordinates(tmp_result, "ra", "dec",
                                           near_ra0)
        if idx_tile_ra == 0:
            tmp_tile_mask = (tmp_ra >= tile_ra_min) & (tmp_ra <= tile_ra_max)
        else:
            tmp_tile_mask = (tmp_ra > tile_ra_min) & (tmp_ra <= tile_ra_max)

        if idx_tile_dec == 0:
            tmp_tile_mask &= (tmp_dec >= tile_dec_min) & \
                             (tmp_dec <= tile_dec_max)
        else:
            tmp_tile_mask &= (tmp_dec > tile_dec_min) & \
                             (tmp_dec <= tile_dec_max)

        if result_table is None:
            result_table = tmp_result[tmp_tile_mask]
        else:
            result_table = vstack([result_table, tmp_result[tmp_tile_mask]])

    return result_table

def specz_merge(catalogue, specz, radius=0.4*u.arcsec):
    """Create the spec-z columns to be added to a catalogue.

    This function cross-match a catalogue with the HELP spectroscopic redshift
    catalogue and add some spec-z columns to the first catalogue:

    - specz_id: identifier in the HELP spec-z catalogue,
    - zspec: spectroscopic redshift;
    - zspec_qual: quality flag;
    - zspec_flag: boolean flag that is true when there is a possible
        mis-association.

    Parameters
    ----------
    catalogue: astropy.table.Table
        The table containing the catalogue.  It must contain a ‘ra’ and ‘dec’
        columns with the position in decimal degree.
    specz: astropy.table.Table
        The table containing the specz catalogue from HELP.
    radius: astropy.units.quantity.Quantity
        The radius to look for counterparts.  When more than one counterpart is
        found, the corresponding sources will be flagged.

    Return
    ------
    astropy.table.Table
        The catalogue with spectroscopic redshift columns added.

    """
    cat_coords = SkyCoord(catalogue['ra'].data * u.deg,
                          catalogue['dec'].data * u.deg)
    specz_coords = SkyCoord(specz['ra'].data * u.deg,
                            specz['dec'].data * u.deg)

    idx_cat, idx_specz, d2d, _ = specz_coords.search_around_sky(
        cat_coords, radius)

    # We sort the three array by increasing d2d
    sort_idx = np.argsort(d2d)
    idx_cat = idx_cat[sort_idx]
    idx_specz = idx_specz[sort_idx]

    # We want to flag as possible mis-associations the spec-z that may be
    # associated to different sources with the given radius.
    idx_specz_toflag = np.unique(
        [item for item, count in Counter(idx_specz).items() if count > 1]
    )

    # We keep only the first association of a spec-z to a source
    _, unique_idx = np.unique(idx_specz, return_index=True)
    idx_cat = idx_cat[unique_idx]
    idx_specz = idx_specz[unique_idx]

    # We add the spec-z columns to the catalogue.
    catalogue.add_column(
        Column(data=np.full(len(catalogue), '', dtype='<U33'),
               name="specz_id"))
    catalogue['specz_id'][idx_cat] = specz['specz_id'][idx_specz]

    catalogue.add_column(
        Column(data=np.full(len(catalogue), np.nan),
               name="zspec"))
    catalogue['zspec'][idx_cat] = specz['z_spec'][idx_specz]

    catalogue.add_column(
        Column(data=np.full(len(catalogue), -99, dtype=int),
               name="zspec_qual"))
    catalogue['zspec_qual'][idx_cat] = specz['z_qual'][idx_specz]

    catalogue.add_column(
        Column(data=np.full(len(catalogue), False, dtype=bool),
               name="zspec_association_flag"))
    catalogue['zspec_association_flag'][idx_cat] = \
        np.in1d(idx_specz, idx_specz_toflag)

    return catalogue


def nb_astcor_diag_plot(cat_ra, cat_dec, ref_ra, ref_dec, radius=0.6*u.arcsec,
                        near_ra0=False, limit_nb_points=None):
    """Create a diagnostic plot for astrometry.

    Given catalogue coordinates and reference coordinates (e.g. Gaia), this
    function plots two figures summarising the RA and Dec differences:
    - A joint plot a RA-diff and Dec-diff;
    - A RA, Dec scatter plot of the catalogue using the angle of the RA-diff,
      Dec-diff vector pour the colour and its norm for the size of the dots.

    If the coordinates are around the ra=0 separation, set near_ra0 parameter
    to True.

    This function does not output anything and is intended to be used within
    a notebook to display the figures.

    Parameters
    ----------
    cat_ra: array-like of floats
        The right ascensions of the catalogue.
    cat_dec: array-like of floats
        The declinations of the catalogue.
    ref_ra: array-like of floats
        The right ascensions of the reference.
    ref_dec: array-like of floats
        The declination of the reference.
    radius: astropy.units.Quantity
        The maximum radius for source associations (default to 0.6 arcsec).
    near_ra0: bool
        Set to True when the coordinates are around the ra=0 limit; the ra will
        be transformed to be between -180 and 180 to avoid large differences
        like 359° - 1°. Default to False.
    limit_nb_points: int
        If there are too many matches, limit the number of plotted points to
        a random selection. If None, use all the matches.

    """
    cat_coords = SkyCoord(cat_ra, cat_dec)
    ref_coords = SkyCoord(ref_ra, ref_dec)

    idx, d2d, _ = cat_coords.match_to_catalog_sky(ref_coords)
    to_keep = d2d <= radius

    # We may want to limit the number of points used.
    if limit_nb_points is not None and np.sum(to_keep) > limit_nb_points:
        random_mask = np.full(np.sum(to_keep), False, dtype=bool)
        random_mask[np.random.choice(
            np.arange(np.sum(to_keep)), limit_nb_points, replace=False)] = True
        to_keep[to_keep][~random_mask] = False

    # Use ra between -180 and 180 when around ra=0
    if near_ra0:
        cat_ra = cat_coords.ra.wrap_at(180 * u.deg)[to_keep]
        ref_ra = ref_coords[idx].ra.wrap_at(180 * u.deg)[to_keep]
    else:
        cat_ra = cat_coords.ra[to_keep]
        ref_ra = ref_coords[idx].ra[to_keep]

    cat_dec = cat_coords.dec[to_keep]
    ref_dec = ref_coords[idx].dec[to_keep]

    ra_diff = cat_ra - ref_ra
    dec_diff = cat_dec - ref_dec

    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    sns.set_style("dark")

    # Joint plot
    jointplot = sns.jointplot(ra_diff.arcsec, dec_diff.arcsec, kind='hex')
    jointplot.set_axis_labels("RA diff. [arcsec]", "Dec diff. [arcsec]")
    jointplot.ax_joint.axhline(0, color='black', linewidth=.5)
    jointplot.ax_joint.axvline(0, color='black', linewidth=.5)

    # Scatter plot of the sources.
    _, axis = plt.subplots()

    offset_angle = np.angle(ra_diff.arcsec + dec_diff.arcsec * 1j)
    offset_dist = np.absolute(ra_diff.arcsec + dec_diff.arcsec * 1j)
    offset_distnorm = (offset_dist - np.min(offset_dist)) / np.max(offset_dist)

    cmap = mpl.colors.ListedColormap(sns.color_palette("husl", 300))
    colors = cmap(offset_angle)  # The color is the angle
    colors[:, 3] = offset_distnorm  # The transparency is the distance

    axis.scatter(cat_ra, cat_dec, c=colors, s=15)
    axis.set_xlabel("RA")
    axis.set_ylabel("Dec")


def nb_merge_dist_plot(main_coords, second_coords, max_dist=5 * u.arcsec,
                       limit_nb_points=None):
    """Create a plot to estimate the radius for merging catalogues.

    This function create a plot presenting the distribution of the distances of
    the sources represented by second_coords to the sources represented by
    main_coords.  There should be an over-density of short distances because of
    matching sources and then the number of sources should grow linearly with
    the distance.

    This function does not return anything and is intended to be used within
    a notebook to display a plot.

    Parameters
    ----------
    main_coords: astropy.coordinates.SkyCoord
        The coordinates in the main catalogue.
    second_coords: astropy.coordinates.SkyCoord
        The coordinates in the secondary catalogue.
    max_dist: astropy.units.Quantity
        Maximal distance to search for counterparts (default to 10
        arc-seconds).
    limit_nb_points: int
        If there are too many matches, limit the number of plotted points to
        a random selection. If None, use all the matches.

    """
    _, _, d2d, _ = main_coords.search_around_sky(second_coords, max_dist)

    if len(d2d) == 0:
        print("HELP Warning: There weren't any cross matches. The two surveys probably "
              "don't overlap.")
        return

    # We may want to limit the number of points used.
    if limit_nb_points is not None and len(d2d) > limit_nb_points:
        random_mask = np.full(len(d2d), False, dtype=bool)
        random_mask[np.random.choice(
            np.arange(len(d2d)), limit_nb_points, replace=False)] = True
        d2d = d2d[random_mask]

    if isinstance(d2d, Angle):
        sns.distplot(d2d.arcsec)
        plt.xticks(np.arange(max_dist.value))
        plt.xlabel("Distance [{}]".format(max_dist.unit))
    else:
        print("There weren't any cross matches. The two surveys probably "
              "don't overlap.")


def nb_compare_mags(x, y, labels=("x", "y")):
    """Create plots comparing magnitudes

    This function creates two plots to compare two arrays of of associated
    magnitude (like the values in two similar bands): the histogram of the
    differences and a “hexbin” plot of one vs the other.

    This function does not return anything and is intended to be used within
    a notebook to display a plot.

    Parameters
    ----------
    x: array-like of floats
        The array of magnitude.
    y: array-like of floats
        The second array of magnitudes, must be the same length of x.
    labels: tuple of strings
        The labels of the two values.

    """
    x_label, y_label = labels

    # Use only finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.copy(x[mask])
    y = np.copy(y[mask])

    # Difference
    diff = y - x

    # If the difference is all NaN there is nothing to compare.
    if np.isnan(diff).all():
        print("No sources have both {} and {} values.".format(
            x_label, y_label))
        return

    # Median, Median absolute deviation and 1% and 99% percentiles
    diff_median = np.median(diff)
    diff_mad = np.median(np.abs(diff - diff_median))
    diff_1p, diff_99p = np.percentile(diff, [1., 99.])

    diff_label = "{} - {}".format(y_label, x_label)

    print("{}:".format(diff_label))
    print("- Median: {:.2f}".format(diff_median))
    print("- Median Absolute Deviation: {:.2f}".format(diff_mad))
    print("- 1% percentile: {}".format(diff_1p))
    print("- 99% percentile: {}".format(diff_99p))

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 6))

    # Histogram of the difference
    vz.hist(diff, ax=ax1, bins='knuth')
    ax1.set_xlabel(diff_label)
    ax1.axvline(0, color='black', linestyle='--')
    ax1.axvline(diff_1p, color='grey', linestyle=':')
    ax1.axvline(diff_99p, color='grey', linestyle=':')

    # Hexbin
    hb = ax2.hexbin(x, y, cmap='Oranges', bins="log")
    min_val = np.min(np.r_[x, y])
    max_val = np.max(np.r_[x, y])
    ax2.autoscale(False)
    ax2.plot([min_val, max_val], [min_val, max_val], "k:")
    fig.colorbar(hb, ax=ax2, label="log10(count)")
    ax2.set_xlabel(labels[0])
    ax2.set_ylabel(labels[1])

    display(fig)
    plt.close()


def nb_plot_mag_ap_evol(magnitudes, stellarity, stel_threshold=0.9,
                        labels=None):
    """Create a plot showing magnitude evolution with aperture.

    This function creates a plot simulating the curve of growth of the
    magnitude in growing apertures.  Given a bunch of magnitudes in several
    apertures, it creates a figure with two plots:
    - The evolution of the mean magnitude in each aperture;
    - The mean gain (or loss when negative) of magnitude in each aperture
      compared to the previous.

    This plot is used to find the best target aperture to use when aperture
    correcting magnitudes.

    This function does not return anything and is intended to be used within
    a notebook to display a plot.

    Parameters
    ----------
    magnitudes: numpy.array of floats
        The magnitudes in a 2 axis array of floats. The first axis is the
        aperture and the second axis is the objects.
    stellarity: array like of floats
        The stellarity associated to each object, must have the same length as
        the second axis of the magnitudes parameter.
    stel_threshold: float
        Stellarity threshold, we only use point sources with a stellarity index
        above this threshold.
    labels: list of strings
        The label corresponding to each aperture in the aperture axis of the
        magnitudes parameter.
    """

    mags = magnitudes[:, stellarity > stel_threshold].copy()
    mag_diff = mags[1:, :] - mags[:-1, :]

    fig, [ax1, ax2] = plt.subplots(
        2, 1,
        sharex=True,
        gridspec_kw={'hspace': 0},
        figsize=(9, 12)
    )

    ax1.plot(np.nanmean(mags, axis=1))

    ax2.plot(1+np.arange(len(mag_diff)),
             np.nanmean(mag_diff, axis=1))
    ax2.axhline(0, c='black', linewidth=.5)

    if labels is not None:
        ax2.xaxis.set_ticks(np.arange(magnitudes.shape[0]))
        ax2.set_xticklabels(labels)
        ax2.set_xlabel("Aperture")
    else:
        ax2.set_xlabel("Aperture index")

    ax1.set_ylabel("Mean magnitude")
    ax2.set_ylabel("Mean magnitude gain vs prev. ap.")


def nb_plot_mag_vs_apcor(mag, mag_target, stellarity):
    """Creates a plot of the evolution of ap. correction with aperture.

    This function creates a plot showing the evolution of the aperture
    correction to be applied - with the associated RMS given by the aperture
    correction method - for each magnitude bin.  This plot is used to chose the
    magnitude limits for the objects that will be used to compute aperture
    correction.  We should use wide limits (to use more objects) where the
    correction is table with few dispersion.

    This function does not return anything and is intended to be used within
    a notebook to display a plot.

    Parameters
    ----------
    mag: numpy array of floats
        The magnitudes in the aperture we want to use.

    mag_target: numpy array of floats
        The magnitudes in the target aperture.  The length must be the same as
        the mag parameter.
    stellarity: numpy array of floats
        The stellarity associated to each object.  The length must be the same
        as the mag parameter. Only object with stellarity above 0.9 are used
        to compute aperture correction.
    """
    mask = stellarity > .9

    # We exclude the 0.1% brighter and fainter sources
    mag_min, mag_max = np.nanpercentile(mag[mask], [.001, .999])
    mag_min = np.floor(mag_min)
    mag_max = np.ceil(mag_max)

    mag_bins = np.arange(mag_min, mag_max, step=.1)

    mag_cor = []
    mag_std = []

    for mag_bin_min in mag_bins:
        try:
            mag_diff, _, std = aperture_correction(
                mag, mag_target, stellarity, mag_bin_min, mag_bin_min + .1)
        except:
            mag_diff, std = np.nan, np.nan
        mag_cor.append(mag_diff)
        mag_std.append(std)

    mag_cor = np.array(mag_cor)
    mag_std = np.array(mag_std)

    plt.rc('figure', figsize=(9, 4))
    plt.plot(mag_bins, mag_cor, color='black')
    plt.fill_between(mag_bins, mag_cor - mag_std, mag_cor + mag_std, alpha=.3)


def nb_ccplots(x, y, x_label, y_label, stellarity, alpha=0.01, leg_loc=4,
               invert_x=False, invert_y=False, x_limits=None, y_limits=None):
    """Generate color-color or color-magnitude plots

    This function is used to create color-color or color-magnitude plots.  It
    uses the stellarity index to make one hexbin plot for point sources, one
    for extended sources, and a scatter plot combining the two.

    This function does not return anything and is intended to be used within
    a notebook to display a plot.

    Parameters
    ----------
    x: array-like of floats
        The color or magnitude displayed in X.
    y: array-like of floats
        The color or magnitude displayed in Y.
    x_label: string
        The label for X.
    y_label: string
        The label for Y.
    stellarity: array-like of floats
        The stellarity index. Sources are considered point sources when the
        stellarity is over 0.7.
    alpha: float
        The alpha value of the points in the scatter plot.
    leg_loc: int
        The matplotlib position of the legend.
    invert_x, invert_y: boolean
        Set to true if you want to invert an axis (e.g. for a magnitude axis).
    x_limits, y_limits: tuple of floats
        Limits of X and Y axis. If None (default) the plots are zoomed removing
        the 0.1% outliers and adding 10% space in both ways.

    """
    x = np.array(x)
    y = np.array(y)
    stellarity = np.array(stellarity)

    # Mask of the sources for which we have information to plot
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(stellarity)
    print("Number of source used: {} / {} ({:.2f}%)".format(
        np.sum(mask), len(x), 100 * np.sum(mask)/len(x)))

    if np.sum(mask) == 0:
        print('HELP warning: no sources with observations in both bands')
        return
    # We set the plot limits or zoom to remove outliers
    if x_limits is not None:
        x_min, x_max = x_limits
    else:
        x_min, x_max = np.percentile(x[mask], [.1, 99.9])
        x_delta = .1 * (x_max - x_min)
        x_min -= x_delta
        x_max += x_delta
    if y_limits is not None:
        y_min, y_max = y_limits
    else:
        y_min, y_max = np.percentile(y[mask], [.1, 99.9])
        y_delta = .1 * (y_max - y_min)
        y_min -= y_delta
        y_max += y_delta

    point_source = stellarity[mask] > 0.7

    plt.figure(figsize=(10, 10), edgecolor="gray")

    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2, facecolor="w")
    ax1.hexbin(x[mask][~point_source], y[mask][~point_source],
               cmap='Reds', bins="log")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_title("Extended sources")

    ax2 = plt.subplot2grid((2, 4), (0, 2), colspan=2, sharex=ax1, sharey=ax1,
                           facecolor='w')
    ax2.hexbin(x[mask][point_source], y[mask][point_source],
               cmap='Blues', bins="log")
    ax2.set_xlabel(x_label)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_title("Point sources")

    ax3 = plt.subplot2grid((2, 4), (1, 1), colspan=2, sharex=ax1, sharey=ax1)
    ax3.scatter(x[mask][~point_source], y[mask][~point_source],
                color='r', marker='v', alpha=alpha, s=1,
                label="Extended sources")
    ax3.scatter(x[mask][point_source], y[mask][point_source],
                color='b', marker='v', alpha=alpha, s=1,
                label="Point sources")
    ax3.set_xlabel(x_label)
    ax3.set_ylabel(y_label)
    legend = ax3.legend(loc=leg_loc)
    for lh in legend.legendHandles:
        lh.set_alpha(1)

    ax3.set_xlim([x_min, x_max])
    ax3.set_ylim([y_min, y_max])
    if invert_x:
        ax3.set_xlim([x_max, x_min])
    if invert_y:
        ax3.set_ylim([y_max, y_min])


def nb_histograms(table, column_names, labels=None):
    """Plot histograms of table columns

    This function plots histograms of columns in an astropy table on the same
    figure.

    This function does not return anything and is intended to be used within
    a notebook to display a plot.

    Parameters
    ----------
    table: astropy.table.Table
        The astropy table.
    column_names: list of strings
        The name of the columns for which an histogram will be plotted.
    labels: list of strings
        If provided, the label to use for each histogram.

    """

    if labels is None:
        labels = column_names

    fig, ax = plt.subplots()

    for name, label in zip(column_names, labels):
        mask = np.isfinite(table[name])
        if not np.isnan(table[name]).all():
            vz.hist(table[name][mask], bins='scott', label=label, alpha=.5)
        else:
            print("HELP warning: the column {} ({}) is empty.".format(
                name, label))

    ax.legend()
    display(fig)
    plt.close()


def find_last_ml_suffix(directory="./data/"):
    """Find the data prefix of the last masterlist.

    This function returns the data prefix to use to get the last master list
    from a directory.

    """
    suffix_list = [item.split("_")[-1].split(".")[0] for item in
                   glob("{}master_catalogue*.fits".format(directory))]

    if len(suffix_list) > 0:
        return sorted(suffix_list)[-1]
    else:
        raise ValueError("There is no master list in the directory.")


def quick_checks(catalogue):
    """Performs some quick checks on a master catalogue.

    This function performs some quick checks on the flux and magnitude columns
    of a master catalogue:
    - Look for empty (all NaN) columns;
    - Look for zero or negative values.

    """
    
    check_table = Table()
    phot_columns = [_ for _ in catalogue.colnames if _.startswith("f_") or
                    _.startswith("m_") or _.startswith("ferr_") or
                    _.startswith("merr_")]
    #print(phot_columns)
    check_table.add_column(Column(data=phot_columns, name='Column'))
    check_table.add_column(Column(data=np.full(len(phot_columns), False), 
                                  name='All nan', 
                                  dtype=bool))
    check_table.add_column(Column(data=np.full(len(phot_columns), 0), name='#Measurements', dtype=int))
    check_table.add_column(Column(data=np.full(len(phot_columns), 0), name='#Zeros', dtype=int))
    check_table.add_column(Column(data=np.full(len(phot_columns), 0), name='#Negative', dtype=int))
    check_table.add_column(Column(data=np.full(len(phot_columns), 0.0), name='Minimum value', dtype=float))
        
    for colname in [_ for _ in catalogue.colnames if _.startswith("f_") or
                    _.startswith("m_") or _.startswith("ferr_") or
                    _.startswith("merr_")]:

        column = catalogue[colname]
        check_table['#Measurements'][check_table['Column'] == colname] = np.sum(~np.isnan(column))
        # Empty columns
        if np.isnan(column).all():
            #print("The column {} contains only NaN!".format(colname))
            check_table['All nan'][check_table['Column'] == colname] = True
        else:
            # Negative values
            minimum = np.nanmin(column)
            if minimum <= 0:
                nb_neg = np.sum(column[np.isfinite(column)] < 0)
                nb_zero = np.sum(column[np.isfinite(column)] == 0)
                #print("The column {} contains {} zero or negative values!" \
                #      "it's minimum is {}.".format(colname, nb_neg, minimum))
                check_table['#Zeros'][check_table['Column'] == colname] = nb_zero
                check_table['#Negative'][check_table['Column'] == colname] = nb_neg
                check_table['Minimum value'][check_table['Column'] == colname] = minimum
    #check_table.show_in_notebook()
    print('Table shows only problematic columns.')
    return check_table[~((check_table['All nan'] == False) & (check_table['#Zeros'] == 0) & (check_table['#Negative'] == 0))]
    
    
    

class help_dr1:
    def __init__(self):
        self.masterlist_overview = Table.read('dr1_overview.fits')
