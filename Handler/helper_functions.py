import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.preprocessing import PowerTransformer
from chainconsumer import ChainConsumer
import numpy as np
from natsort import natsorted
import imageio
import os
import torch
# import healpy as hp
import pandas as pd
import seaborn as sns
"""import warnings

warnings.filterwarnings("error")"""


def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError


def plot_2d_kde(x, y, manual_levels, limits=None, x_label="", y_label="", title="", color=None):
    """"""

    xmin = -2
    xmax = 2
    ymin = -2
    ymax = 2

    if color is None:
        color = "Blues"

    if limits is not None:
        xmin = limits[0]
        xmax = limits[1]
        ymin = limits[2]
        ymax = limits[3]

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)

    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Contourf plot
    cfset = ax.contourf(xx, yy, f, cmap=color, alpha=0.3, levels=manual_levels)

    # Or kernel density estimate plot instead of the contourf plot
    # ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])

    # Contour plot
    cset = ax.contour(xx, yy, f, colors='k', levels=manual_levels)

    # Label plot

    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(title)


def plot_2d_kde_compare(x1, y1, x2, y2, manual_levels, limits=None, x_label="", y_label="", title="", color=None):
    """"""

    xmin = -2
    xmax = 2
    ymin = -2
    ymax = 2

    if color is None:
        color = "Blues"

    if limits is not None:
        xmin = limits[0]
        xmax = limits[1]
        ymin = limits[2]
        ymax = limits[3]

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values1 = np.vstack([x1, y1])
    kernel1 = gaussian_kde(values1)
    f1 = np.reshape(kernel1(positions).T, xx.shape)

    values2 = np.vstack([x2, y2])
    kernel2 = gaussian_kde(values2)
    f2 = np.reshape(kernel2(positions).T, xx.shape)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # manual_levels = np.array([0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6])

    # Contourf plot
    ax.contourf(xx, yy, f1, cmap=color[0], alpha=0.3, levels=manual_levels)
    ax.contourf(xx, yy, f2, cmap=color[1], alpha=0.3, levels=manual_levels)

    # Or kernel density estimate plot instead of the contourf plot
    # ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])

    # Contour plot
    cset1 = ax.contour(xx, yy, f1, colors='k', levels=manual_levels)
    cset2 = ax.contour(xx, yy, f2, colors='k', levels=manual_levels)

    # Label plot
    ax.clabel(cset1, inline=1, fontsize=10)
    ax.clabel(cset2, inline=1, fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(title)


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

def load_healpix(path2file, hp_show=False, nest=True, partial=False, field=None):
    """
    # Function to load fits datasets
    # Returns:

    """
    """if field is None:
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
        
    return hp_map"""


def match_skybrite_2_footprint(path2footprint, path2skybrite, hp_show=False, nest_footprint=True, nest_skybrite=True,
                               partial_footprint=False, partial_skybrite=True, field_footprint=None,
                               field_skybrite=None):
    """
    Main function to run
    Returns:

    """
    """hp_map_footprint = load_healpix(
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
    return np.column_stack((good_indices, sky_in_footprint[1]))"""


def generate_normal_distribution(size, mu, sigma, num=1, as_tensor=True):
    """
    Generate uniform distributed random data for discriminator.

    Args:
        size: size of the tensor

    Returns:
        random data as torch tensor
    """
    # random_data = torch.randn(size)

    if as_tensor is False:
        return np.random.normal(mu, sigma, size=(size, num))
    return torch.FloatTensor([np.random.normal(mu, sigma, size=(size, num))[0][0]])


def generate_uniform_distribution(size, low, high, num=1, as_tensor=True):
    """
    Generate normal distributed random data for generator.

    Args:
        size: size of the tensor

    Returns:
        random data as torch tensor
    """
    # random_data = torch.rand(size)

    if not as_tensor:
        return np.random.uniform(low, high, size=(num, size))
    return torch.FloatTensor(np.random.uniform(low, high, size=(num, size)))


def calc_ks_critical_value(n1, n2, a=0.10):
    if a == 0.10:
        c_a = 1.22
    elif a == 0.05:
        c_a = 1.36
    elif a == 0.025:
        c_a = 1.48
    elif a == 0.01:
        c_a = 1.63
    elif a == 0.005:
        c_a = 1.73
    elif a == 0.001:
        c_a = 1.95
    else:
        raise Exception("Wrong value for a. a must be one of these values [0.10, 0.05, 0.025, 0.01, 0.005, 0.001]")
    return c_a * np.sqrt((n1+n2)/(n1*n2))


def standard_deviation(arr, mean):
    """
    Calculate the standard deviation

    Args:
        arr: array of values

    Returns:
        the calculated standard deviation as array
    """
    return np.sqrt(np.sum((arr - mean)**2) / len(arr))


def standard_scaler(arr, mean, std_dev):
    """
    Calculate the standard scale

    Args:
        arr: array of values

    Returns:
        calculated standard scale as array
    """
    return (arr - mean) / std_dev


def reverses_standard_scaler(arr, mean, std_dev):
    """
    reverses the standard scale

    Args:
        arr: array of values

    Returns:
        calculated original array
    """

    return std_dev * np.array(arr) + mean


def normalizing(values):
    return values / values.max()


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


def calc_mse_loss(loss_generator, loss_discriminator_true, loss_discriminator_fake):
    mse_loss = np.sqrt(
        (loss_generator[-1] - 0.5) ** 2 +
        (loss_discriminator_true[-1] - 0.5) ** 2 +
        (loss_discriminator_fake[-1] - 0.5) ** 2) / 3
    return mse_loss


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
    # return np.array(lst_mag)


def make_gif(frame_folder, name_save_folder, fps=10):
    filenames = natsorted(os.listdir(frame_folder))
    images_data = []
    for filename in filenames:
        image = imageio.imread(f"{frame_folder}/{filename}")
        images_data.append(image)
    try:
        imageio.mimwrite(
            uri=f"{name_save_folder}",
            ims=images_data,
            format='.gif',
            duration=fps
        )
    except:
        pass


def unsheared_mag_cut(data_frame):
    """"""
    print("Apply unsheared mag cuts")
    mag_cuts = (
            (18 < data_frame["unsheared/mag_i"]) &
            (data_frame["unsheared/mag_i"] < 23.5) &
            (15 < data_frame["unsheared/mag_r"]) &
            (data_frame["unsheared/mag_r"] < 26) &
            (15< data_frame["unsheared/mag_z"]) &
            (data_frame["unsheared/mag_z"] < 26) &
            (-1.5 < data_frame["unsheared/mag_r"] - data_frame["unsheared/mag_i"]) &
            (data_frame["unsheared/mag_r"] - data_frame["unsheared/mag_i"] < 4) &
            (-4 < data_frame["unsheared/mag_z"] - data_frame["unsheared/mag_i"]) &
            (data_frame["unsheared/mag_z"] - data_frame["unsheared/mag_i"] < 1.5)
    )
    data_frame = data_frame[mag_cuts]
    shear_cuts = (
            (10 < data_frame["unsheared/snr"]) &
            (data_frame["unsheared/snr"] < 1000) &
            (0.5 < data_frame["unsheared/size_ratio"]) &
            (data_frame["unsheared/T"] < 10)
    )
    data_frame = data_frame[shear_cuts]
    data_frame = data_frame[~((2 < data_frame["unsheared/T"]) & (data_frame["unsheared/snr"] < 30))]
    print('Length of catalog after applying unsheared mag cuts: {}'.format(len(data_frame)))
    return data_frame


def replace_and_transform_data(data_frame, columns, replace_value=None):
    """"""
    dict_pt = {}
    # df_test = pd.DataFrame({})
    for idx_col, col in enumerate(columns):
        pt = PowerTransformer(method="yeo-johnson")
        replace_value_index = None if replace_value[idx_col] == "None" else replace_value[idx_col]
        if replace_value_index is not None:
            replace_value_tuple = eval(replace_value_index)
            data_frame[col] = data_frame[col].replace(replace_value_tuple[0], replace_value_tuple[1])
        pt.fit(np.array(data_frame[col]).reshape(-1, 1))
        data_frame[col] = pt.transform(np.array(data_frame[col]).reshape(-1, 1))
        # df_test[col] = pt.inverse_transform(np.array(data_frame[col]).reshape(-1, 1)).ravel()
        # transformed_data = pt.transform(np.array(data_frame[col]).reshape(-1, 1))
        # df_test[col] = pt.inverse_transform(transformed_data).ravel()
        dict_pt[f"{col} pt"] = pt
    # print(df_test.isna().sum().sum())
    # nan_indices = df_test[df_test.isna().any(axis=1)].index
    # if nan_indices is not None:
    #     df_nan = df_test[df_test.isna().any(axis=1)]
    #     df_nonan = data_frame.loc[nan_indices]
    #     for col in df_nan.columns:
    #         if df_nan[col].isna().sum() > 0:
    #             print("nan", df_nan[col])
    #             print("nonan", df_nonan[col])
    # exit()
    return data_frame, dict_pt


def unreplace_and_untransform_data(data_frame, dict_pt, columns, replace_value=None):
    """"""
    for idx_col, col in enumerate(columns):
        pt = dict_pt[f"{col} pt"]
        data_frame[col] = pt.inverse_transform(np.array(data_frame[col]).reshape(-1, 1)).ravel()
        if replace_value[idx_col] is not None:
            data_frame[col] = data_frame[col].replace(replace_value[idx_col][1], replace_value[idx_col][0])
    # print(data_frame.isna().sum().sum())
    return data_frame


def plot_chain(data_frame, plot_name, max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12, columns=None,
               parameter=None, extends=None):
    """

    :param extends: extents={
                "mag r": (17.5, 26),
                "mag i": (17.5, 26),
                "mag z": (17.5, 26),
                "snr": (-11, 55),
                "size ratio": (-1.5, 4),
                "T": (-1, 2.5)
            }
    :param label_font_size:
    :param tick_font_size:
    :param shade_alpha:
    :param max_ticks:
    :param plot_name: "generated observed properties: chat*"
    :param data_frame:
    :param columns: Mutable list, default values are columns = [
            "unsheared/mag_r",
            "unsheared/mag_i",
            "unsheared/mag_z",
            "unsheared/snr",
            "unsheared/size_ratio",
            "unsheared/T"
        ]
    :param parameter: Mutable list, default values are parameter = [
                "mag r",
                "mag i",
                "mag z",
                "snr",              # signal-noise      Range: min=0.3795, max=38924.4662
                "size ratio",       # T/psfrec_T        Range: min=-0.8636, max=4346136.5645
                "T"                 # T=<x^2>+<y^2>     Range: min=-0.6693, max=1430981.5103
            ]
    :return:
    """
    df_plot = pd.DataFrame({})

    if columns is None:
        columns = [
            "unsheared/mag_r",
            "unsheared/mag_i",
            "unsheared/mag_z",
            "unsheared/snr",
            "unsheared/size_ratio",
            "unsheared/T"
        ]

    if parameter is None:
        parameter = [
                "mag r",
                "mag i",
                "mag z",
                "snr",              # signal-noise      Range: min=0.3795, max=38924.4662
                "size ratio",       # T/psfrec_T        Range: min=-0.8636, max=4346136.5645
                "T"                 # T=<x^2>+<y^2>     Range: min=-0.6693, max=1430981.5103
            ]

    for col in columns:
        df_plot[col] = np.array(data_frame[col])

    chain = ChainConsumer()
    chain.add_chain(df_plot.to_numpy(), parameters=parameter, name=plot_name)
    chain.configure(
        max_ticks=max_ticks,
        shade_alpha=shade_alpha,
        tick_font_size=tick_font_size,
        label_font_size=label_font_size
    )
    # if extends is not None:
    chain.plotter.plot(
        figsize="page",
        extents=extends
    )
    plt.show()
    plt.clf()



# def bdf_mag_cuts(data_frame):
#     """"""
#     bdf_cuts = (
#         (data_frame["BDF_MAG_ERR_DERED_CALIB_J"] < 37.5) &
#         (10 < data_frame["BDF_MAG_ERR_DERED_CALIB_J"]) &
#         (data_frame["BDF_MAG_ERR_DERED_CALIB_H"] < 37.5) &
#         (10 < data_frame["BDF_MAG_ERR_DERED_CALIB_H"]) &
#         (data_frame["BDF_MAG_ERR_DERED_CALIB_K"] < 37.5) &
#         (10 < data_frame["BDF_MAG_ERR_DERED_CALIB_K"])
#     )
#     data_frame = data_frame[bdf_cuts]
#     return data_frame


def metacal_cuts(data_frame):
    """"""
    print("Apply mcal cuts")
    mcal_cuts = (data_frame["unsheared/extended_class_sof"] >= 0) & (data_frame["unsheared/flags_gold"] < 2)
    data_frame = data_frame[mcal_cuts]
    print('Length of catalog after applying mcal cuts: {}'.format(len(data_frame)))
    return data_frame


def detection_cuts(data_frame):
    """"""
    print("Apply detection cuts")
    detect_cuts = (data_frame["match_flag_1.5_asec"] < 2) & \
                  (data_frame["flags_foreground"] == 0) & \
                  (data_frame["flags_badregions"] < 2) & \
                  (data_frame["flags_footprint"] == 1)
    data_frame = data_frame[detect_cuts]
    print('Length of catalog after applying detection cuts: {}'.format(len(data_frame)))
    return data_frame


def airmass_cut(data_frame):
    """"""
    print('Cut mcal_detect_df_survey catalog so that AIRMASS_WMEAN_R is not null')
    data_frame = data_frame[pd.notnull(data_frame["AIRMASS_WMEAN_R"])]
    print('Length of catalog after applying AIRMASS_WMEAN_R cuts: {}'.format(len(data_frame)))
    return data_frame


def loss_plot(
        epoch,
        lst_train_loss_per_batch,
        lst_train_loss_per_epoch,
        lst_valid_loss_per_batch,
        lst_valid_loss_per_epoch,
        show_plot,
        save_plot,
        save_name
):
    statistical_figure, ((stat_ax1, stat_ax2), (stat_ax3, stat_ax4)) = plt.subplots(nrows=2, ncols=2)
    statistical_figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)
    statistical_figure.suptitle(f"Epoch: {epoch}", fontsize=16)

    # Create dataframe of progress list
    df_training_loss_per_batch = pd.DataFrame({
        "training loss": lst_train_loss_per_batch
    })
    df_training_loss_per_epoch = pd.DataFrame({
        "training loss": lst_train_loss_per_epoch
    })
    df_valid_loss_per_batch = pd.DataFrame({
        "validation loss": lst_valid_loss_per_batch
    })
    df_valid_loss_per_epoch = pd.DataFrame({
        "validation loss": lst_valid_loss_per_epoch
    })

    # Create plot
    df_training_loss_per_batch.plot(
        figsize=(16, 9),
        alpha=0.5,
        marker=".",
        grid=True,
        # yticks=(0, 0.25, 0.5, 0.69, 1.0, 5.0),
        ax=stat_ax1)

    stat_ax1.set_xlabel("batch", fontsize=10, loc='right')
    stat_ax1.set_ylabel("loss", fontsize=12, loc='top')
    stat_ax1.set_title(f"Loss per batch")

    df_training_loss_per_epoch.plot(
        figsize=(16, 9),
        alpha=0.5,
        marker=".",
        grid=True,
        # yticks=(0, 0.25, 0.5, 0.69, 1.0, 5.0),
        ax=stat_ax2)

    stat_ax2.set_xlabel("epoch", fontsize=10, loc='right')
    stat_ax2.set_ylabel("loss", fontsize=12, loc='top')
    stat_ax2.set_title(f"Loss per epoch")

    # Create plot
    df_valid_loss_per_batch.plot(
        figsize=(16, 9),
        alpha=0.5,
        marker=".",
        grid=True,
        # yticks=(0, 0.25, 0.5, 0.69, 1.0, 5.0),
        ax=stat_ax3)

    stat_ax3.set_xlabel("batch", fontsize=10, loc='right')
    stat_ax3.set_ylabel("loss", fontsize=12, loc='top')
    stat_ax3.set_title(f"Loss per batch")

    df_valid_loss_per_epoch.plot(
        figsize=(16, 9),
        alpha=0.5,
        marker=".",
        grid=True,
        # yticks=(0, 0.25, 0.5, 0.69, 1.0, 5.0),
        ax=stat_ax4)

    stat_ax4.set_xlabel("epoch", fontsize=10, loc='right')
    stat_ax4.set_ylabel("loss", fontsize=12, loc='top')
    stat_ax4.set_title(f"Loss per epoch")

    if show_plot is True:
        statistical_figure.show()
    if save_plot is True:
        statistical_figure.savefig(f"{save_name}", dpi=200)

    # Clear and close open figure to avoid memory overload
    statistical_figure.clf()
    plt.close(statistical_figure)
    plt.clf()


def color_color_plot(data_frame_generated, data_frame_true, colors, show_plot, save_name, extents=None):
    """"""
    df_generated_measured = pd.DataFrame({})
    df_true_measured = pd.DataFrame({})
    for color in colors:
        df_generated_measured[f"{color[0]}-{color[1]}"] = \
            np.array(data_frame_generated[f"unsheared/lupt_{color[0]}"]) - np.array(
                data_frame_generated[f"unsheared/lupt_{color[1]}"])
        df_true_measured[f"{color[0]}-{color[1]}"] = \
            np.array(data_frame_true[f"unsheared/lupt_{color[0]}"]) - np.array(
                data_frame_true[f"unsheared/lupt_{color[1]}"])

    arr_true = df_true_measured.to_numpy()
    arr_generated = df_generated_measured.to_numpy()
    parameter = [
        "unsheared/lupt r-i",
        "unsheared/lupt i-z"
    ]
    chainchat = ChainConsumer()
    chainchat.add_chain(arr_true, parameters=parameter, name="true observed properties: chat")
    chainchat.add_chain(arr_generated, parameters=parameter,
                        name="generated observed properties: chat*")
    chainchat.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
    chainchat.plotter.plot(
        filename=save_name,
        figsize="page",
        extents=extents
    )
    if show_plot is True:
        plt.show()
    plt.clf()
    plt.close()


def residual_plot(data_frame_generated, data_frame_true, bands, plot_title, show_plot, save_plot, save_name):
    """"""
    hist_figure, ((stat_ax1), (stat_ax2), (stat_ax3)) = \
        plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
    hist_figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    hist_figure.suptitle(plot_title, fontsize=16)

    lst_axis_res = [
        stat_ax1,
        stat_ax2,
        stat_ax3
    ]

    lst_xlim_res = [
        (-2.5, 2.5),
        (-2.5, 2.5),
        (-2.5, 2.5)
    ]

    df_hist_balrog = pd.DataFrame({
        "dataset": ["balrog" for _ in range(len(data_frame_true[f"unsheared/lupt_r"]))]
    })
    df_hist_generated = pd.DataFrame({
        "dataset": ["generated" for _ in range(len(data_frame_generated[f"unsheared/lupt_r"]))]
    })
    for band in bands:
        df_hist_balrog[f"BDF_LUPT_DERED_CALIB - unsheared/lupt {band}"] = data_frame_true[
                                                                              f"BDF_LUPT_DERED_CALIB_{band.upper()}"] - \
                                                                          data_frame_true[f"unsheared/lupt_{band}"]
        df_hist_generated[f"BDF_LUPT_DERED_CALIB - unsheared/lupt {band}"] = data_frame_generated[
                                                                                 f"BDF_LUPT_DERED_CALIB_{band.upper()}"] - \
                                                                             data_frame_generated[
                                                                                 f"unsheared/lupt_{band}"]

    for idx, band in enumerate(bands):
        sns.histplot(
            data=df_hist_balrog,
            x=f"BDF_LUPT_DERED_CALIB - unsheared/lupt {band}",
            ax=lst_axis_res[idx],
            element="step",
            stat="density",
            color="dodgerblue",
            bins=50,
            label="balrog"
        )
        sns.histplot(
            data=df_hist_generated,
            x=f"BDF_LUPT_DERED_CALIB - unsheared/lupt {band}",
            ax=lst_axis_res[idx],
            element="step",
            stat="density",
            color="darkorange",
            fill=False,
            bins=50,
            label="generated"
        )
        lst_axis_res[idx].axvline(
            x=df_hist_balrog[f"BDF_LUPT_DERED_CALIB - unsheared/lupt {band}"].median(),
            color='dodgerblue',
            ls='--',
            lw=1.5,
            label="Mean balrog"
        )
        lst_axis_res[idx].axvline(
            x=df_hist_generated[f"BDF_LUPT_DERED_CALIB - unsheared/lupt {band}"].median(),
            color='darkorange',
            ls='--',
            lw=1.5,
            label="Mean generated"
        )
        lst_axis_res[idx].set_xlim(lst_xlim_res[idx][0], lst_xlim_res[idx][1])
        if idx == 0:
            lst_axis_res[idx].legend()
        else:
            lst_axis_res[idx].legend([], [], frameon=False)
    hist_figure.tight_layout()
    if show_plot is True:
        plt.show()

    if save_plot is True:
        plt.savefig(save_name)
    plt.clf()
    plt.close()


def plot_chain_compare(data_frame_generated, data_frame_true, epoch, show_plot, save_name, columns=None, parameter=None,
                       extends=None, max_ticks = 5, shade_alpha = 0.8, tick_font_size = 12, label_font_size = 12):
    """"""
    df_plot_generated = pd.DataFrame({})
    df_plot_true = pd.DataFrame({})

    if columns is None:
        columns = [
            "unsheared/mag_r",
            "unsheared/mag_i",
            "unsheared/mag_z",
            "unsheared/snr",
            "unsheared/size_ratio",
            "unsheared/T"
        ]

    if parameter is None:
        parameter = [
            "mag r",
            "mag i",
            "mag z",
            "snr",  # signal-noise      Range: min=0.3795, max=38924.4662
            "size ratio",  # T/psfrec_T        Range: min=-0.8636, max=4346136.5645
            "T"  # T=<x^2>+<y^2>     Range: min=-0.6693, max=1430981.5103
        ]

    for col in columns:
        df_plot_generated[col] = np.array(data_frame_generated[col])
        df_plot_true[col] = np.array(data_frame_true[col])

    chainchat = ChainConsumer()
    chainchat.add_chain(df_plot_true.to_numpy(), parameters=parameter, name="balrog observed properties: chat")
    chainchat.add_chain(df_plot_generated.to_numpy(), parameters=parameter, name="generated observed properties: chat*")
    chainchat.configure(
        max_ticks=max_ticks,
        shade_alpha=shade_alpha,
        tick_font_size=tick_font_size,
        label_font_size=label_font_size
    )
    try:
        chainchat.plotter.plot(
            filename=save_name,
            figsize="page",
            extents=extends
        )
    except:
        print("chain error at epoch", epoch + 1)
    if show_plot is True:
        plt.show()
    plt.clf()


def plot_mean_or_std(data_frame_generated, data_frame_true, lists_to_plot, list_epochs, columns, lst_labels, lst_marker,
                     lst_color, plot_title, show_plot, save_plot, save_name, statistic_type="mean"):
    """"""
    y_label = ""
    for idx_col, col in enumerate(columns):
        if statistic_type.upper() == "MEAN":
            lists_to_plot[idx_col].append(data_frame_generated[col].mean() / data_frame_true[col].mean())
            y_label = "mean(chat*) / mean(chat)"
        elif statistic_type.upper() == "STD":
            lists_to_plot[idx_col].append(data_frame_generated[col].std() / data_frame_true[col].std())
            y_label = "std(chat*) / std(chat)"

    for idx_col, col in enumerate(columns):
        plt.plot(
            list_epochs,
            lists_to_plot[idx_col],
            marker=lst_marker[idx_col],
            linestyle='-',
            color=lst_color[idx_col],
            label=lst_labels[idx_col]
        )
    plt.legend()
    plt.title(plot_title)
    plt.xlabel("epoch")
    plt.ylabel(y_label)

    if show_plot is True:
        plt.show()
    if save_plot is True:
        plt.savefig(save_name, dpi=200)
    plt.clf()
    plt.close()

    return lists_to_plot
