import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from natsort import natsorted
import imageio
import os
import torch
# import healpy as hp
import pandas as pd
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

def luptize_deep_kids(flux, bins, var=0, zp=30):
    """
    The flux must be in the same dimension as the bins.
    The bins must be given as list like ["i", "g", "r", "z", "u", "Y", "J", "H", "K"]
    the ordering of the softening parameter b
    """
    # Todo Use kids depth
    dict_mags = {
        "u": 24.8,
        "g": 25.4,
        "r": 25.2,
        "i": 24.2,
        "Z": 23.1,
        "Y": 22.3,  # y band value is copied from array above because Y band is not in the up to date catalog
        "J": 22.1,
        "H": 21.5,
        "Ks": 21.2
    }
    lst_mags = []
    for b in bins:
        # if b in ["I", "G", "R", "Z", "U"]:
        #     actual_b = b.lower()
        # elif b in ["y", "j", "h", "k"]:
        #     actual_b = b.upper()
        # elif b in ["i", "g", "r", "z", "u", "Y", "J", "H", "K"]:
        #     actual_b = b
        # else:
        #     raise IOError("bin not defined")
        lst_mags.append(dict_mags[b])
    arr_mags = np.array(lst_mags)
    s = (10**((zp-arr_mags)/2.5)) / 10
    return luptize(flux, var, s, zp)


"""def luptize(flux, var, s, zp):
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


def luptize_deep(flux, var=0, zp=22.5):
    # The flux must be 8 dimensional and must be given in the order [f_i, f_g, f_r, f_z, f_u, f_Y, f_J, f_H, f_K] to match the ordering of the softening parameter b
    #lim_mags_des = np.array([22.9, 23.7, 23.5, 22.2, 25]) # old
    #lim_mags_vista = np.array([24.6, 24.5, 24.0, 23.5]) # old
    lim_mags_des = np.array([24.66, 25.57, 25.27, 24.06, 24.64])
    lim_mags_vista = np.array([24.6, 24.02, 23.69, 23.58]) # y band value is copied from array above because Y band is not in the up to date catalog
    s_des = (10**((zp-lim_mags_des)/2.5)) / 10  # des limiting mag is 10 sigma
    s_vista = (10**((zp-lim_mags_vista)/2.5)) / 10  # vista limiting mag is 10 sigma

    s = np.concatenate([s_des, s_vista])

    return luptize(flux, var, s, zp)"""


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
    imageio.mimwrite(
        uri=f"{name_save_folder}",
        ims=images_data,
        format='.gif',
        fps=fps
    )