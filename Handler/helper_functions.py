from sklearn.preprocessing import PowerTransformer
import numpy as np
import os
"""import warnings

warnings.filterwarnings("error")"""


def change_mean_std_of_dist(dist1, dist2=None, dist3=None):
    """"""
    if dist3 is None:
        raise "Distribution 3 is None"

    std_dist_1 = np.std(dist1)
    mean_dist_1 = np.mean(dist1)
    soll_std = std_dist_1
    soll_mean = mean_dist_1

    std_dist_3 = np.std(dist3)
    mean_dist_3 = np.mean(dist3)

    if dist2 is not None:
        std_dist_2 = np.std(dist2)
        mean_dist_2 = np.mean(dist2)
        delta_mean = mean_dist_2 - mean_dist_1
        soll_mean = mean_dist_3 + delta_mean
        delta_std = std_dist_2 - std_dist_1
        soll_std = std_dist_3 + delta_std

    a = np.sqrt(soll_std ** 2 / std_dist_3 ** 2)

    # nur den std aendern. mean macht nur bei Magnituden sinn
    b = 0 # soll_mean - a * mean_dist_3

    arr_dist_4 = a * dist3 + b
    return arr_dist_4


def concatenate_lists(data_list):
    conc_inputs = data_list[0]
    for idx, value in enumerate(data_list):
        if idx + 1 < len(data_list):
            conc_inputs = np.concatenate((conc_inputs, data_list[idx + 1]), axis=0)
    return conc_inputs


def luptize(flux, var, s, zp):
    # s: measurement error (variance) of the flux (with zero pt zp) of an object at the limiting magnitude of the survey
    # a: Pogson's ratio
    # b: softening parameter that sets the scale of transition between linear and log behavior of the luptitudes
    a = 2.5 * np.log10(np.exp(1))
    b = a**(1./2) * s
    mu0 = zp -2.5 * np.log10(b)

    # turn into luptitudes and their errors
    lupt = mu0 - a * np.arcsinh(flux / (2 * b))
    lupt_var = a ** 2 * var / ((2 * b) ** 2 + flux ** 2)
    return lupt, lupt_var


def luptize_deep(flux, bins, var=0, zp=22.5):
    """
    The flux must be in the same dimension as the bins.
    The bins must be given as list like ["i", "g", "r", "z", "u", "Y", "J", "H", "K"]
    the ordering of the softening parameter b
    """
    dict_mags = {
        "i": 24.66,
        "g": 25.57,
        "r": 25.27,
        "z": 24.06,
        "u": 24.64,
        "Y": 24.6,  # y band value is copied from array above because Y band is not in the up to date catalog
        "J": 24.02,
        "H": 23.69,
        "K": 23.58
    }
    lst_mags = []
    for b in bins:
        if b in ["I", "G", "R", "Z", "U"]:
            actual_b = b.lower()
        elif b in ["y", "j", "h", "k"]:
            actual_b = b.upper()
        elif b in ["i", "g", "r", "z", "u", "Y", "J", "H", "K"]:
            actual_b = b
        else:
            raise IOError("bin not defined")
        lst_mags.append(dict_mags[actual_b])
    arr_mags = np.array(lst_mags)
    s = (10**((zp-arr_mags)/2.5)) / 10
    return luptize(flux, var, s, zp)

def luptize_inverse(lupt, lupt_var, s, zp):
    """"""
    # s: measurement error (variance) of the flux (with zero pt zp) of an object at the limiting magnitude of the survey
    # a: Pogson's ratio
    # b: softening parameter that sets the scale of transition between linear and log behavior of the luptitudes
    a = 2.5 * np.log10(np.exp(1))
    b = a**(1./2) * s
    mu0 = zp -2.5 * np.log10(b)

    # turn into luptitudes and their errors
    # lupt = mu0 - a * np.arcsinh(flux / (2 * b))
    flux = 2 * b * np.sinh((mu0 - lupt) / a)
    var = (lupt_var * ((2 * b)**2 + flux**2)) / (a**2)
    # lupt_var = a ** 2 * var / ((2 * b) ** 2 + flux ** 2)
    return flux, var


def luptize_inverse_deep(lupt, bins, lupt_var=0, zp=22.5):
    """
        The flux must be in the same dimension as the bins.
        The bins must be given as list like ["i", "g", "r", "z", "u", "Y", "J", "H", "K"]
        the ordering of the softening parameter b
    """
    dict_mags = {
        "i": 24.66,
        "g": 25.57,
        "r": 25.27,
        "z": 24.06,
        "u": 24.64,
        "Y": 24.6,  # y band value is copied from array above because Y band is not in the up to date catalog
        "J": 24.02,
        "H": 23.69,
        "K": 23.58
    }
    lst_mags = []
    for b in bins:
        if b in ["I", "G", "R", "Z", "U"]:
            actual_b = b.lower()
        elif b in ["y", "j", "h", "k"]:
            actual_b = b.upper()
        elif b in ["i", "g", "r", "z", "u", "Y", "J", "H", "K"]:
            actual_b = b
        else:
            raise IOError("bin not defined")
        lst_mags.append(dict_mags[actual_b])
    arr_mags = np.array(lst_mags)
    s = (10 ** ((zp - arr_mags) / 2.5)) / 10
    return luptize_inverse(lupt, lupt_var, s, zp)


def luptize_inverse_fluxes(data_frame, flux_col, lupt_col, bins):
    """

    :param bins: ["I", "R", "Z", "J", "H", "K"]
    :param data_frame:
    :param flux_col: ("BDF_FLUX_DERED_CALIB_I", "BDF_FLUX_ERR_DERED_CALIB_I")
    :param lupt_col: ("BDF_LUPT_DERED_CALIB", "BDF_LUPT_ERR_DERED_CALIB")
    :return:
    """
    lst_lupt = []
    lst_lupt_var = []
    for bin in bins:
        lst_lupt.append(data_frame[f"{lupt_col[0]}_{bin}"])
        lst_lupt_var.append(data_frame[f"{lupt_col[1]}_{bin}"])
    arr_lupt = np.array(lst_lupt).T
    arr_lupt_var = np.array(lst_lupt_var).T
    arr_flux, arr_var = luptize_inverse_deep(lupt=arr_lupt, bins=bins, lupt_var=arr_lupt_var)
    arr_flux = arr_flux.T
    arr_var = arr_var.T
    for idx_bin, bin in enumerate(bins):
        data_frame[f"{flux_col[0]}_{bin}"] = arr_flux[idx_bin]
        data_frame[f"{flux_col[1]}_{bin}"] = arr_var[idx_bin]
    return data_frame


def mag2flux(magnitude, zero_pt=30):
    # convert flux to magnitude
    try:
        flux = 10**((zero_pt-magnitude)/2.5)
        return flux
    except RuntimeWarning:
        print("Warning")


def flux2mag(flux, zero_pt=30, clip=0.001):
    # convert flux to magnitude
    """lst_mag = []
    for f in flux:
        try:
            magnitude = zero_pt - 2.5 * np.log10(f)
            lst_mag.append(magnitude)
        except RuntimeWarning:
            print("Warning")
            # lst_mag.append(-100)"""
    if clip is None:
        return zero_pt - 2.5 * np.log10(flux)
    return zero_pt - 2.5 * np.log10(flux.clip(clip))


def replace_values(data_frame, replace_value):
    for col in replace_value.keys():
        replace_value_index = None if replace_value[col] == "None" else replace_value[col]
        if replace_value_index is not None:
            replace_value_tuple = eval(replace_value_index)
            data_frame[col] = data_frame[col].replace(replace_value_tuple[0], replace_value_tuple[1])
    return data_frame


def replace_and_transform_data(data_frame, columns, replace_value=None):
    """"""
    dict_pt = {}
    for col in columns:
        pt = PowerTransformer(method="yeo-johnson")
        # data_frame = replace_value(data_frame, replace_value)
        pt.fit(np.array(data_frame[col]).reshape(-1, 1))
        data_frame[col] = pt.transform(np.array(data_frame[col]).reshape(-1, 1))
        dict_pt[f"{col} pt"] = pt
    return data_frame, dict_pt


def unreplace_and_untransform_data(data_frame, dict_pt, columns, replace_value=None):
    """"""
    for col in columns:
        pt = dict_pt[f"{col} pt"]
        data_frame[col] = pt.inverse_transform(np.array(data_frame[col]).reshape(-1, 1)).ravel()
        if replace_value[col] is not None:
            data_frame[col] = data_frame[col].replace(replace_value[col][1], replace_value[col][0])
    return data_frame


def get_os():
    if os.name == 'nt':
        return 'Windows'
    elif os.name == 'posix':
        if 'darwin' in os.uname().sysname.lower():
            return 'Mac'
        else:
            return 'Linux'
    else:
        return 'Unknown OS'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
