from sklearn.preprocessing import PowerTransformer
import numpy as np
import os
import re
import pandas as pd
"""import warnings

warnings.filterwarnings("error")"""


def expected_calibration_error(probs, y_true, n_bins=20):
    probs = np.asarray(probs, float).ravel()
    y_true = np.asarray(y_true, float).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(probs, bins, right=True) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = (idx == b)
        if not np.any(m):
            continue
        conf = probs[m].mean()
        acc  = y_true[m].mean()
        ece += (m.mean()) * abs(acc - conf)
    return float(ece)

def mag_rate_mae(probs, y_true, mag, edges):
    """Gewichtete mittlere Absolutabweisung zwischen mean(p) und mean(y) je Mag-Bin."""
    probs = np.asarray(probs, float).ravel()
    y_true = np.asarray(y_true, float).ravel()
    mag = np.asarray(mag, float).ravel()
    mae, wsum = 0.0, 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (mag >= lo) & (mag <= hi)
        if not np.any(m):
            continue
        mae += m.sum() * abs(probs[m].mean() - y_true[m].mean())
        wsum += m.sum()
    return float(mae / wsum) if wsum else 0.0

def neg_log_loss(probs, y_true, eps=1e-8):
    probs = np.clip(np.asarray(probs, float).ravel(), eps, 1.0 - eps)
    y_true = np.asarray(y_true, float).ravel()
    return float(-np.mean(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs)))


def quantile_edges(x: np.ndarray, n_bins: int):
    q = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, q)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-8
    return edges


def calc_color(data_frame, colors, column_name):
    """"""
    for color in string_to_tuple(str(colors)):
        data_frame[f"{color[0]}-{color[1]}"] = \
            np.array(data_frame[f"{column_name}_{color[0]}"]) - np.array(data_frame[f"{column_name}_{color[1]}"])
    return data_frame


def compute_injection_counts(det_catalog, id_col, count_col):
    '''
    Expects det_catalog to be a pandas DF
    '''
    # `true_id` is the DF id
    unique, ucounts = np.unique(det_catalog[id_col], return_counts=True)

    freq = pd.DataFrame()
    freq[id_col] = unique
    freq[count_col] = ucounts

    return det_catalog.merge(freq, on=id_col, how='left')


def sample_columns(df_balrog, df_gandalf, column_name):
    values_in = df_balrog[column_name]
    df_gandalf[column_name] = None
    df_gandalf.loc[:, column_name] = np.random.choice(values_in, size=len(df_gandalf))
    return df_gandalf


# def calculate_kde(x, y, positions, X):
#     from scipy.stats import gaussian_kde
#     xy = np.vstack([x, y])
#     kde = gaussian_kde(xy, bw_method=0.2)
#     Z = np.reshape(kde(positions).T, X.shape)
#     return Z / Z.max()  # Normalize

def calculate_kde(x, y, positions):
    from scipy.stats import gaussian_kde
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, positions.shape[1:])
    return Z


def select_columns(df_balrog, df_gandalf, column_name, bin_width=0.25):
    min_i_mag = min(df_balrog['BDF_MAG_DERED_CALIB_I'].min(), df_gandalf['BDF_MAG_DERED_CALIB_I'].min())
    max_i_mag = max(df_balrog['BDF_MAG_DERED_CALIB_I'].max(), df_gandalf['BDF_MAG_DERED_CALIB_I'].max())
    bins = np.arange(min_i_mag, max_i_mag + bin_width, bin_width)

    df_balrog['i_mag_bin'] = pd.cut(df_balrog['BDF_MAG_DERED_CALIB_I'], bins, labels=False, include_lowest=True)
    df_gandalf['i_mag_bin'] = pd.cut(df_gandalf['BDF_MAG_DERED_CALIB_I'], bins, labels=False, include_lowest=True)

    df_gandalf[column_name] = None
    for bin in df_gandalf['i_mag_bin'].unique():
        bin_df_balrog = df_balrog[df_balrog['i_mag_bin'] == bin]
        bin_df_gandalf = df_gandalf[df_gandalf['i_mag_bin'] == bin]
        for index in bin_df_gandalf.index:
            if not bin_df_balrog[column_name].empty:  # Check if the array is not empty
                df_gandalf.at[index, column_name] = np.random.choice(bin_df_balrog[column_name])
            else:
                # Handle the case when the array is empty
                # Find the neighboring bins
                lower_bin = bin - 1
                upper_bin = bin + 1
                lower_bin_df_balrog = df_balrog[df_balrog['i_mag_bin'] == lower_bin]
                upper_bin_df_balrog = df_balrog[df_balrog['i_mag_bin'] == upper_bin]
                # Concatenate the neighboring bins
                neighbor_bins_df_balrog = pd.concat([lower_bin_df_balrog, upper_bin_df_balrog])
                # Sample from the neighboring bins
                if not neighbor_bins_df_balrog[column_name].empty:
                    df_gandalf.at[index, column_name] = np.random.choice(neighbor_bins_df_balrog[column_name])

    return df_gandalf


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


def luptize_fluxes(data_frame, flux_col, lupt_col, bins):
    """

    :param bins: ["I", "R", "Z", "J", "H", "K"]
    :param data_frame:
    :param flux_col: ("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB")
    :param lupt_col: ("BDF_LUPT_DERED_CALIB", "BDF_LUPT_ERR_DERED_CALIB")
    :return:
    """
    lst_flux = []
    lst_var = []
    for bin in bins:
        lst_flux.append(data_frame[f"{flux_col[0]}_{bin}"])
        lst_var.append(data_frame[f"{flux_col[1]}_{bin}"])
    arr_flux = np.array(lst_flux).T
    arr_var = np.array(lst_var).T
    lupt_mag, lupt_var = luptize_deep(flux=arr_flux, bins=bins, var=arr_var, zp=30)
    lupt_mag = lupt_mag.T
    lupt_var = lupt_var.T
    for idx_bin, bin in enumerate(bins):
        data_frame[f"{lupt_col[0]}_{bin}"] = lupt_mag[idx_bin]
        data_frame[f"{lupt_col[1]}_{bin}"] = lupt_var[idx_bin]
    return data_frame

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


def luptize_inverse_fluxes(cfg, data_frame, flux_col, lupt_col, bins):
    """

    :param bins: ["I", "R", "Z", "J", "H", "K"]
    :param data_frame:
    :param flux_col: ("BDF_FLUX_DERED_CALIB", "BDF_FLUX_ERR_DERED_CALIB")
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
    arr_flux, arr_var = luptize_inverse_deep(lupt=arr_lupt, bins=bins, lupt_var=arr_lupt_var, zp=cfg["ZERO_POINT"])
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


def flux2mag(flux, zero_pt=30, clip=0.001):  # clip=0.001
    """Convert flux to magnitude, with optional clipping."""
    if clip is not None:
        flux = np.clip(flux, clip, None)
    return zero_pt - 2.5 * np.log10(flux)

def fluxerr2magerr(flux, flux_err):
    """Convert flux uncertainty to magnitude uncertainty."""
    return (2.5 / np.log(10)) * (flux_err / flux)


def calc_mag(cfg, data_frame, flux_col, mag_col, bins):
    """"""
    for b in bins:
        if isinstance(mag_col, tuple):
            data_frame[f"{mag_col[0]}_{b}"] = flux2mag(
                flux=data_frame[f"{flux_col[0]}_{b}"],
                zero_pt=cfg["ZERO_POINT"],
                clip=cfg["CLIP"]
            )
            data_frame[f"{mag_col[1]}_{b}"] = fluxerr2magerr(
                flux=data_frame[f"{flux_col[0]}_{b}"],
                flux_err=data_frame[f"{flux_col[1]}_{b}"]
            )
        else:
            data_frame[f"{mag_col}_{b}"] = flux2mag(data_frame[f"{flux_col}_{b}"])
    return data_frame


def replace_values(data_frame, replace_value):
    for col in replace_value.keys():
        replace_value_index = None if replace_value[col] == "None" else replace_value[col]
        if replace_value_index is not None:
            replace_value_tuple = eval(replace_value_index)
            data_frame[col] = data_frame[col].replace(replace_value_tuple[0], replace_value_tuple[1])
    return data_frame


def unreplace_values(data_frame, replace_value):
    for col in replace_value.keys():
        if replace_value[col] is not None:
            data_frame[col] = data_frame[col].replace(replace_value[col][1], replace_value[col][0])
    return data_frame


def yj_transform_data(data_frame, columns):
    """"""
    dict_pt = {}
    for col in columns:
        pt = PowerTransformer(method="yeo-johnson")
        pt.fit(np.array(data_frame[col]).reshape(-1, 1))
        data_frame[col] = pt.transform(np.array(data_frame[col]).reshape(-1, 1))
        dict_pt[f"{col} pt"] = pt
    return data_frame, dict_pt


def yj_inverse_transform_data(data_frame, dict_pt, columns):
    """"""
    for col in columns:
        pt = dict_pt[f"{col} pt"]
        data_frame[col] = pt.inverse_transform(np.array(data_frame[col]).reshape(-1, 1)).ravel()
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


def string_to_tuple(s):
    matches = re.findall(r'\(([^,]+), ([^)]+)\)', s)
    return [tuple(map(str.strip, match)) for match in matches]


def calculate_percentage_of_outliers(data_frame, column_name, lower_bound, upper_bound):
    total_count = len(data_frame)

    lower_outliers = (data_frame[column_name] < lower_bound).sum()
    upper_outliers = (data_frame[column_name] > upper_bound).sum()

    lower_outliers_percentage = (lower_outliers / total_count) * 100
    upper_outliers_percentage = (upper_outliers / total_count) * 100

    print(f"Number of datapoints smaller than {lower_bound}: {lower_outliers} "
          f"({lower_outliers_percentage:.2f}%)")
    print(f"Number of datapoints bigger than {upper_bound}: {upper_outliers} "
          f"({upper_outliers_percentage:.2f}%)")

    return lower_outliers_percentage, upper_outliers_percentage


def compute_weights_from_magnitude(mags, bin_edges):
    counts, _ = np.histogram(mags, bins=bin_edges)
    weights_per_bin = 1.0 / np.log(counts + 1e-6)
    bin_indices = np.digitize(mags, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(weights_per_bin) - 1)
    data_weights = weights_per_bin[bin_indices]

    return data_weights, counts, weights_per_bin


def correct_over_prediction():
    pass

def dataset_to_tensor(dataset):
    import torch
    data_list = [dataset[i] for i in range(len(dataset))]
    input_data, output_data = zip(*data_list)
    tsr_input = torch.stack(input_data)
    tsr_output = torch.stack(output_data)
    return tsr_input, tsr_output


def compute_ece(y_true, y_prob, n_bins=10):
    """"""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_in_bin * abs(accuracy_in_bin - avg_confidence_in_bin)

    return ece


def assign_loggrid(x, y, xmin, xmax, xsteps, ymin, ymax, ysteps):
    x = np.maximum(x, xmin)
    x = np.minimum(x, xmax)
    y = np.maximum(y, ymin)
    y = np.minimum(y, ymax)
    logstepx = np.log10(xmax / xmin) / xsteps
    logstepy = np.log10(ymax / ymin) / ysteps
    indexx = (np.log10(x / xmin) / logstepx).astype(int)
    indexy = (np.log10(y / ymin) / logstepy).astype(int)
    indexx = np.minimum(indexx, xsteps - 1)
    indexy = np.minimum(indexy, ysteps - 1)
    return indexx, indexy


def apply_loggrid(x, y, grid, xmin=10, xmax=300, xsteps=20, ymin=0.5, ymax=5, ysteps=20):
    indexx, indexy = assign_loggrid(x, y, xmin, xmax, xsteps, ymin, ymax, ysteps)
    res = np.zeros(len(x))
    res = grid[indexx, indexy]
    return res


def assign_new_weights(x, y, path_grid):
    w = np.genfromtxt(path_grid)
    return apply_loggrid(x=x, y=y, grid=w)


def inverse_transform_df(ct, df):
    for name, trans, cols in ct.transformers_:
        if trans == 'passthrough':
            continue
        df[cols] = trans.inverse_transform(df[cols])
    return df