import matplotlib.pyplot as plt
import numpy as np


def load_healpix(path2file, hp_show=False, nest=True, partial=False, field=None):
    """
    # Function to load fits datasets
    # Returns:

    """
    import healpy as hp
    if field is None:
        hp_map = hp.read_map(path2file, nest=nest, partial=partial)
    else:
        hp_map = hp.read_map(path2file, nest=nest, partial=partial, field=field)
    if hp_show is True:
        hp_map_show = hp_map
        if field is not None:
            hp_map_show = hp_map[1]
        hp.mollview(
            hp_map_show,
            norm="hist",
            nest=nest
        )
        hp.graticule()
        plt.show()

    return hp_map


def match_skybrite_2_footprint(path2footprint, path2skybrite, hp_show=False, nest_footprint=True, nest_skybrite=True,
                               partial_footprint=False, partial_skybrite=True, field_footprint=None,
                               field_skybrite=None):
    """
    Main function to run_date
    Returns:

    """
    import healpy as hp
    hp_map_footprint = load_healpix(
        path2file=path2footprint,
        hp_show=hp_show,
        nest=nest_footprint,
        partial=partial_footprint,
        field=field_footprint
    )

    hp_map_skybrite = load_healpix(
        path2file=path2skybrite,
        hp_show=hp_show,
        nest=nest_skybrite,
        partial=partial_skybrite,
        field=field_skybrite
    )
    sky_in_footprint = hp_map_skybrite[:, hp_map_footprint != hp.UNSEEN]
    good_indices = sky_in_footprint[0, :].astype(int)
    return np.column_stack((good_indices, sky_in_footprint[1]))