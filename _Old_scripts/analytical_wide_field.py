import pickle

from Handler.helper_functions import flux2mag, mag2flux
from Handler.plot_functions import plot_2d_kde
from scipy.optimize import curve_fit
from scipy.stats import kstest, gaussian_kde
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.io import fits
from astropy.table import Table
import scipy as sp
# import corner
# import h5py
from chainconsumer import ChainConsumer


def set_plot_settings():
    # Set figsize and figure layout
    plt.rcParams["figure.figsize"] = [16, 9]
    plt.rcParams["figure.autolayout"] = True
    sns.set_theme()
    sns.set_context("paper")
    sns.set(font_scale=2)
    # plt.rc('font', size=10)  # controls default text size
    plt.rc('axes', titlesize=18)  # fontsize of the title
    plt.rc('axes', labelsize=14)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=10)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=10)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=12)  # fontsize of the legend


def exp_func(x, a, b):
    """
    Exponential fit function: f(x)=exp[a+b*x]
    Args:
        x (np.array): array of x values
        a (int): intersection with y-axis
        b (int): slope

    Returns:
        y (numpy array): array with calculated f(x)
    """
    y = np.exp(a) * np.exp(b*x)
    return y


def load_data(filename, data_type="pkl"):
    """
    Load pickle data as pandas data frame
    Args:
        filename (str): path of pickle file
        data_type (str): postfix of filename, has to be pkl or csv
    Returns:
        df (pandas Dataframe): DataFrame of the catalogs
    """
    if data_type == "pkl":
        infile = open(filename, 'rb')
        # load pickle as pandas dataframe
        df = pd.read_pickle(infile)
        # close file
        infile.close()
        return df
    elif data_type == "csv":
        infile = open(filename, 'rb')
        # load pickle as pandas dataframe
        df = pd.read_csv(infile)
        # close file
        infile.close()
        return df
    elif data_type == "fits":
        with fits.open(filename) as hdu:
            asn_table = Table(hdu[1].data)
        return asn_table
    elif data_type == "h5":
        # h5 = h5py.File(filename, 'r')
        return None # h5
    else:
        print(f"Data type {data_type} not defined!")
        exit()


def calc_fit_parameters(distribution, lower_bound, upper_bound, bin_size, function, column=None, plot_data=None,
                        save_plot=None, save_path_plots=""):
    """"""
    # Create bins for fit range
    bins = np.linspace(lower_bound, upper_bound, bin_size)

    # Calculate bin width for hist plot
    bin_width = bins[1] - bins[0]

    # Calculate the center of the bins
    bins_centers = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)])

    if plot_data is True:
        sns.set(font_scale=2)
        sns.histplot(
            x=distribution,
            element="step",
            fill=False,
            color="darkred",
            binwidth=bin_width,
            log_scale=(False, True),
            stat="probability",
            label=f"probability of DES deep field bin: {column}"
        )

        # Set some plot parameters
        plt.title(f"probability histogram of DES deep field '{column}'")
        plt.xlabel("magnitude deep field i band")
        plt.legend()
        if save_plot is True:
            plt.savefig(f"{save_path_plots}/Power_law_DES-{column}_log_wo_bounds.png")
        plt.show()

    distribution = distribution[distribution[:] <= upper_bound]
    distribution = distribution[lower_bound <= distribution[:]]

    # Calculate the histogram fit values
    data_entries = np.histogram(distribution, bins=bins)[0]

    # Get the probability
    data_probability_entries = data_entries / np.sum(data_entries)

    if plot_data is True:
        sns.histplot(
            x=distribution,
            element="step",
            fill=False,
            color="darkred",
            binwidth=bin_width,
            log_scale=(False, True),
            stat="probability",
            label=f"probability of DES deep field bin: {column}"
        )

        plt.plot(bins_centers, data_probability_entries, '.', color="darkred")

        # Set some plot parameters
        plt.xlim((lower_bound, upper_bound))
        plt.title(f"probability histogram of DES deep field '{column}'")
        plt.xlabel("magnitude deep field i band")
        plt.legend()
        if save_plot is True:
            plt.savefig(f"{save_path_plots}/Probability_DES-{column}.png")
        plt.show()

    # fit exponential function to calculated histogram values from before
    popt, pcov = curve_fit(function, xdata=bins_centers, ydata=data_probability_entries)

    if plot_data is True:
        sns.histplot(
            x=distribution,
            element="step",
            fill=False,
            color="darkred",
            binwidth=bin_width,
            log_scale=(False, True),
            stat="probability",
            label=f"probability of DES deep field bin: {column}"
        )

        plt.plot(bins_centers, data_probability_entries, '.', color="darkred")

        # Plot the fit function
        plt.plot(
            bins_centers,
            function(bins_centers, *popt),
            color='darkblue',
            linewidth=2.5,
            label=f'power law - f(x)=Exp[a+b*x]\n bin width {bin_width:2.4f} ; a={popt[0]:2.4f} ; b={popt[1]:2.4f}'
        )

        # Set some plot parameters
        plt.xlim((lower_bound, upper_bound))
        plt.title(f"probability histogram and power law of DES deep field '{column}'")
        plt.xlabel("magnitude deep field i band")
        plt.legend()
        if save_plot is True:
            plt.savefig(f"{save_path_plots}/Curve_fit_DES-{column}.png")
        plt.show()

    dictionary_fit_parameter = {
        f"{column} parameter optimal": popt,
        f"{column} parameter covariance": pcov,
        f"{column}": np.array(distribution),
        f"{column} probability": data_probability_entries,
        f"{column} counts": data_entries,
        "bin width": bin_width,
        "bin centers": bins_centers,
        "bins": bins
    }

    return dictionary_fit_parameter


def generate_distribution(dict_fit_params, column, lower_bound, upper_bound, plot_data, save_plot, save_path_plots,
                          size):
    """
    Generate a distribution with given fit parameters
    Args:
        dict_fit_params (dict): dictionary with given fit parameters and column names (same names as get_imag_distribution column names)
        column (list): column names (same names as get_imag_distribution column names)
        lower_bound (int): lower bound of distribution
        upper_bound (int): upper bound of distribution
        bin_size (int): number of bins
        size (int): How many values do you want to generate?

    Returns:
        dictionary_generated_data (dict): dictionary with generated bands with calculated distribution
    """

    # Create numpy array for x-axis
    xspace = np.linspace(lower_bound, upper_bound, num=size, endpoint=True)

    # Init new dictionary
    dictionary_generated_data = {}

    # Calculating the probability with defined fit function and calculated fit parameters
    prob = exp_func(
        xspace,
        *dict_fit_params[f"{column} parameter optimal"]
    )

    # Normalize the probability
    norm_prob = prob / np.sum(prob)

    # Use probability to select magnitudes
    arr_generated_data = np.random.choice(xspace, size=size, p=norm_prob)

    if plot_data is True:
        sns.histplot(
            x=arr_generated_data,
            element="step",
            fill=False,
            color="darkgreen",
            binwidth=dict_fit_params["bin width"],
            log_scale=(False, True),
            stat="probability",
            label=f"generated probability"
        )

        # plt.xlim((lower_bound, upper_bound))
        plt.title(f"generated probability histogram")
        plt.xlabel("magnitude deep field i band")
        plt.legend()
        if save_plot is True:
            plt.savefig(f"{save_path_plots}/Generated_distribution.png")
        plt.show()

    # Write generated Data to dictionary
    dictionary_generated_data[f"generated {column}"] = arr_generated_data

    # Calculate the histogram fit values
    data_entries = np.histogram(arr_generated_data, bins=dict_fit_params["bins"])[0]

    # fit exponential function to calculated histogram values from before
    # Get the probability
    data_probability_entries = data_entries / np.sum(data_entries)
    popt, pcov = curve_fit(exp_func, xdata=dict_fit_params["bin centers"], ydata=data_probability_entries)

    print(f"generated {column} slope", popt[1])
    print(f"generated {column} offset", popt[0])
    print(f"{column} slope", dict_fit_params[f"{column} parameter optimal"][1])
    print(f"{column} offet", dict_fit_params[f"{column} parameter optimal"][0])

    if plot_data is True:
        sns.histplot(
            x=dict_fit_params[f"{column}"],
            element="step",
            fill=False,
            color="darkred",
            binwidth=dict_fit_params["bin width"],
            log_scale=(False, True),
            stat="probability",
            label=f"probability of DES deep field mag in {column}"
        )

        plt.plot(dict_fit_params["bin centers"], dict_fit_params[f"{column} probability"], '.', color="darkred")

        sns.histplot(
            x=arr_generated_data,
            element="step",
            fill=False,
            color="darkgreen",
            binwidth=dict_fit_params["bin width"],
            log_scale=(False, True),
            stat="probability",
            label=f"probability of generated mag in I"
        )

        plt.plot(dict_fit_params["bin centers"], data_probability_entries, '.', color="darkgreen")

        # Plot the fit function
        plt.plot(
            dict_fit_params["bin centers"],
            exp_func(dict_fit_params["bin centers"], *dict_fit_params[f"{column} parameter optimal"]),
            color='darkblue',
            linewidth=2.5,
            label=f'power law of BDF_FLUX_DERED_CALIB_I - f(x)=Exp[a+b*x]\nbin width {dict_fit_params["bin width"]:2.4f} ; a={dict_fit_params[f"{column} parameter optimal"][0]:2.4f} ; b={dict_fit_params[f"{column} parameter optimal"][1]:2.4f}'
        )

        # Plot the fit function
        plt.plot(
            xspace,
            exp_func(xspace, *popt),
            color='darkorange',
            linewidth=2.5,
            label=f'power law generated magnitude - f(x)=Exp[a+b*x]\nbin width {dict_fit_params["bin width"]:2.4f} ; a={popt[0]:2.4f} ; b={popt[1]:2.4f}'
        )

        # Set some plot parameters
        plt.xlim((lower_bound, upper_bound))
        plt.title(f"Calculate {column} fit from DES deep field catalog")
        plt.xlabel("magnitude deep field i band")
        plt.legend()
        if save_plot is True:
            plt.savefig(f"{save_path_plots}/Curve_fit_DES-{column}_and_generated_data.png")
        plt.show()

    return dictionary_generated_data


def make_some_cuts(data_frame, columns_riz):
    print("length of df_balrog before drop", len(data_frame))
    data_frame = data_frame.drop_duplicates()
    print("length of df_balrog droped", len(data_frame))

    bdf_r_mag_cut = (flux2mag(data_frame[columns_riz[0]]) > 17) & (flux2mag(data_frame[columns_riz[0]]) < 26)
    bdf_i_mag_cut = (flux2mag(data_frame[columns_riz[1]]) > 18) & (flux2mag(data_frame[columns_riz[1]]) < 24)
    bdf_z_mag_cut = (flux2mag(data_frame[columns_riz[2]]) > 17) & (flux2mag(data_frame[columns_riz[2]]) < 26)
    data_frame = data_frame[bdf_i_mag_cut & bdf_r_mag_cut & bdf_z_mag_cut]
    print("length of df_balrog cut", len(data_frame))
    # Todo check these cuts
    without_negative = ((data_frame["AIRMASS_WMEAN_R"] >= 0) &
                        (data_frame["AIRMASS_WMEAN_I"] >= 0) &
                        (data_frame["AIRMASS_WMEAN_Z"] >= 0) &
                        (data_frame["unsheared/snr"] <= 50) &
                        (data_frame["unsheared/snr"] >= -50) &
                        (data_frame["unsheared/size_ratio"] <= 5) &
                        (data_frame["unsheared/size_ratio"] >= -10)
                        )
    data_frame = data_frame[without_negative]
    print("length of df_balrog cut", len(data_frame))
    return data_frame


def get_covariance_matrix(data_frame, columns_riz, columns_ugjhks, plot_data, save_plot, save_path_plots):
    """
    Calculate the covariance matrix of given pandas DataFrame. The columns must be a list of the column name of the
    different bins in i,r, and z. The order of the column names has to be i-band, r-band, z-band.
    e.g ["BDF_MAG_DERED_CALIB_I", "BDF_MAG_DERED_CALIB_R", "BDF_MAG_DERED_CALIB_Z"].

    Args:
        data_frame (pandas DataFrame): DataFrame with magnitudes in different bands (i,r and z)
        columns (list): List of column names in the order i,r and z. e.g.:
                                        ["BDF_MAG_DERED_CALIB_I", "BDF_MAG_DERED_CALIB_R", "BDF_MAG_DERED_CALIB_Z"]

    Returns:
        dict_cov_matrix (dict): dictionary of the calculated covariance matrix and all usefull informations (covariance
        matrix, original data in all three bans, magnitude difference in r-i and i-z, magnitude difference shifted to
        zero as mean in r-i and i-z, mean value of r-i and i-z, standard deviation in r-i and i-z)
    """

    arr_u_mag_df = np.array(flux2mag(data_frame[columns_ugjhks[0]]))
    arr_g_mag_df = np.array(flux2mag(data_frame[columns_ugjhks[1]]))
    arr_r_mag_df = np.array(flux2mag(data_frame[columns_riz[0]]))
    arr_i_mag_df = np.array(flux2mag(data_frame[columns_riz[1]]))
    arr_z_mag_df = np.array(flux2mag(data_frame[columns_riz[2]]))
    arr_j_mag_df = np.array(flux2mag(data_frame[columns_ugjhks[2]]))
    arr_h_mag_df = np.array(flux2mag(data_frame[columns_ugjhks[3]]))
    arr_ks_mag_df = np.array(flux2mag(data_frame[columns_ugjhks[4]]))

    # Calculating the magnitude differences r-i and i-z
    arr_u_g_mag_df = arr_u_mag_df - arr_g_mag_df
    arr_g_r_mag_df = arr_g_mag_df - arr_r_mag_df
    arr_r_i_mag_df = arr_r_mag_df - arr_i_mag_df
    arr_i_z_mag_df = arr_i_mag_df - arr_z_mag_df
    arr_z_j_mag_df = arr_z_mag_df - arr_j_mag_df
    arr_j_h_mag_df = arr_j_mag_df - arr_h_mag_df
    arr_h_ks_mag_df = arr_h_mag_df - arr_ks_mag_df

    # Create a matrix r-i and i-z and calculating the covariance matrix
    arr_mag_ug_gr_df = np.array([arr_u_g_mag_df, arr_g_r_mag_df])
    covariance_matrix_ug_gr_df = np.cov(arr_mag_ug_gr_df)

    arr_mag_gr_ri_df = np.array([arr_g_r_mag_df, arr_r_i_mag_df])
    covariance_matrix_gr_ri_df = np.cov(arr_mag_gr_ri_df)

    arr_mag_ri_iz_df = np.array([arr_r_i_mag_df, arr_i_z_mag_df])
    covariance_matrix_ri_iz_df = np.cov(arr_mag_ri_iz_df)

    arr_mag_iz_zj_df = np.array([arr_i_z_mag_df, arr_z_j_mag_df])
    covariance_matrix_iz_zj_df = np.cov(arr_mag_iz_zj_df)

    arr_mag_zj_jh_df = np.array([arr_z_j_mag_df, arr_j_h_mag_df])
    covariance_matrix_zj_jh_df = np.cov(arr_mag_zj_jh_df)

    arr_mag_jh_hks_df = np.array([arr_j_h_mag_df, arr_h_ks_mag_df])
    covariance_matrix_jh_hks_df = np.cov(arr_mag_jh_hks_df)

    print(f"covariance matrix deep distribution u-g, g-r: {covariance_matrix_ug_gr_df}")
    print(f"covariance matrix deep distribution g-r, r-i: {covariance_matrix_gr_ri_df}")
    print(f"covariance matrix deep distribution r-i, i-z: {covariance_matrix_ri_iz_df}")
    print(f"covariance matrix deep distribution i-z, z-j: {covariance_matrix_iz_zj_df}")
    print(f"covariance matrix deep distribution z-j, j-h: {covariance_matrix_zj_jh_df}")
    print(f"covariance matrix deep distribution j-h, h-ks: {covariance_matrix_jh_hks_df}")

    # Creating the covariance matrix dictionary
    dictionary_cov_matrix = {
        "covariance matrix deep u-g, g-r": covariance_matrix_ug_gr_df,
        "covariance matrix deep g-r, r-i": covariance_matrix_gr_ri_df,
        "covariance matrix deep r-i, i-z": covariance_matrix_ri_iz_df,
        "covariance matrix deep i-z, z-j": covariance_matrix_iz_zj_df,
        "covariance matrix deep z-j, j-h": covariance_matrix_zj_jh_df,
        "covariance matrix deep j-h, h-ks": covariance_matrix_jh_hks_df,
        "mean deep u-g": arr_u_g_mag_df.mean(),
        "mean deep g-r": arr_g_r_mag_df.mean(),
        "mean deep r-i": arr_r_i_mag_df.mean(),
        "mean deep i-z": arr_i_z_mag_df.mean(),
        "mean deep z-j": arr_z_j_mag_df.mean(),
        "mean deep j-h": arr_j_h_mag_df.mean(),
        "mean deep h-ks": arr_h_ks_mag_df.mean(),
        "array deep u": arr_u_mag_df,
        "array deep g": arr_g_mag_df,
        "array deep r": arr_r_mag_df,
        "array deep i": arr_i_mag_df,
        "array deep z": arr_z_mag_df,
        "array deep j": arr_j_mag_df,
        "array deep h": arr_h_mag_df,
        "array deep ks": arr_ks_mag_df,
        "array deep u-g": arr_u_g_mag_df,
        "array deep g-r": arr_g_r_mag_df,
        "array deep r-i": arr_r_i_mag_df,
        "array deep i-z": arr_i_z_mag_df,
        "array deep z-j": arr_z_j_mag_df,
        "array deep j-h": arr_j_h_mag_df,
        "array deep h-ks": arr_h_ks_mag_df
    }

    if plot_data is True:
        plot_2d_kde(
            x=arr_u_g_mag_df,
            y=arr_g_r_mag_df,
            x_label="deep field u-g mag",
            y_label="deep field g-r mag",
            color="Greens",
            title="color plot (kde) U-G, G-R",
            manual_levels=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        )
        if save_plot is True:
            plt.savefig(f"{save_path_plots}/Colorplot_ug_gr.png")
        plt.show()
        plt.clf()

        plot_2d_kde(
            x=arr_g_r_mag_df,
            y=arr_r_i_mag_df,
            x_label="deep field g-r mag",
            y_label="deep field r-i mag",
            color="Greens",
            title="color plot (kde) G-R, R-I",
            manual_levels=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        )
        if save_plot is True:
            plt.savefig(f"{save_path_plots}/Colorplot_gr_ri.png")
        plt.show()
        plt.clf()

        plot_2d_kde(
            x=arr_r_i_mag_df,
            y=arr_i_z_mag_df,
            x_label="deep field r-i mag",
            y_label="deep field i-z mag",
            color="Greens",
            title="color plot (kde) R-I, I-Z",
            manual_levels=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        )
        if save_plot is True:
            plt.savefig(f"{save_path_plots}/Colorplot_ri_iz.png")
        plt.show()
        plt.clf()

        plot_2d_kde(
            x=arr_i_z_mag_df,
            y=arr_z_j_mag_df,
            x_label="deep field i-z mag",
            y_label="deep field z-j mag",
            color="Greens",
            title="color plot (kde) I-Z, Z-J",
            manual_levels=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        )
        if save_plot is True:
            plt.savefig(f"{save_path_plots}/Colorplot_iz_zj.png")
        plt.show()
        plt.clf()

        plot_2d_kde(
            x=arr_z_j_mag_df,
            y=arr_j_h_mag_df,
            x_label="deep field z-j mag",
            y_label="deep field j-h mag",
            color="Greens",
            title="color plot (kde) Z-J, J-H",
            manual_levels=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        )
        if save_plot is True:
            plt.savefig(f"{save_path_plots}/Colorplot_zj_jh.png")
        plt.show()
        plt.clf()

        plot_2d_kde(
            x=arr_j_h_mag_df,
            y=arr_h_ks_mag_df,
            x_label="deep field j-h mag",
            y_label="deep field h-ks mag",
            color="Greens",
            title="color plot (kde) J-H, H-KS",
            manual_levels=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        )
        if save_plot is True:
            plt.savefig(f"{save_path_plots}/Colorplot_jh_hks.png")
        plt.show()
        plt.clf()
    return dictionary_cov_matrix


def generate_mag_distribution(dictionary_cov_matrix, arr_i_mag, size, plot_data, save_plot, save_path_plots):
    """
    Generate the values in the r and z band depend on the covariance matrix and the distribution of generated i-band
    values (e.g. from generate_distribution in i-band)

    Args:
        dictionary_cov_matrix (dict): dictionary of the covariance matrix. Calculated with the get_covariance_matrix
                                        function
        arr_mag (numpy array): array of generated magnitudes in one band e.g. i-band

    Returns:
        dict_generated_data (dict): dictionary of generated data in all three bands (i,r and z)
    """

    arr_mean_ug_gr_df = np.array([dictionary_cov_matrix["mean deep u-g"], dictionary_cov_matrix["mean deep g-r"]])
    arr_multi_normal_ug_gr_df = np.random.multivariate_normal(
        arr_mean_ug_gr_df, dictionary_cov_matrix["covariance matrix deep u-g, g-r"], size)

    arr_mean_gr_ri_df = np.array([dictionary_cov_matrix["mean deep g-r"], dictionary_cov_matrix["mean deep r-i"]])
    arr_multi_normal_gr_ri_df = np.random.multivariate_normal(
        arr_mean_gr_ri_df, dictionary_cov_matrix["covariance matrix deep g-r, r-i"], size)

    arr_mean_ri_iz_df = np.array([dictionary_cov_matrix["mean deep r-i"], dictionary_cov_matrix["mean deep i-z"]])
    arr_multi_normal_ri_iz_df = np.random.multivariate_normal(
        arr_mean_ri_iz_df, dictionary_cov_matrix["covariance matrix deep r-i, i-z"], size)

    arr_mean_iz_zj_df = np.array([dictionary_cov_matrix["mean deep i-z"], dictionary_cov_matrix["mean deep z-j"]])
    arr_multi_normal_iz_zj_df = np.random.multivariate_normal(
        arr_mean_iz_zj_df, dictionary_cov_matrix["covariance matrix deep i-z, z-j"], size)

    arr_mean_zj_jh_df = np.array([dictionary_cov_matrix["mean deep z-j"], dictionary_cov_matrix["mean deep j-h"]])
    arr_multi_normal_zj_jh_df = np.random.multivariate_normal(
        arr_mean_zj_jh_df, dictionary_cov_matrix["covariance matrix deep z-j, j-h"], size)

    arr_mean_jh_hk_df = np.array([dictionary_cov_matrix["mean deep j-h"], dictionary_cov_matrix["mean deep h-ks"]])
    arr_multi_normal_jh_hks_df = np.random.multivariate_normal(
        arr_mean_jh_hk_df, dictionary_cov_matrix["covariance matrix deep j-h, h-ks"], size)

    lst_u_df = []
    lst_g_df = []
    lst_r_df = []
    lst_z_df = []
    lst_j_df = []
    lst_h_df = []
    lst_k_df = []
    for idx, value in enumerate(arr_multi_normal_ri_iz_df):
        lst_r_df.append(value[0] + arr_i_mag[idx])
        lst_z_df.append(arr_i_mag[idx] - value[1])
    arr_r_mag_df = np.array(lst_r_df)
    arr_z_mag_df = np.array(lst_z_df)

    for idx, value in enumerate(arr_multi_normal_gr_ri_df):
        lst_g_df.append(value[0] + arr_r_mag_df[idx])
    arr_g_mag_df = np.array(lst_g_df)

    for idx, value in enumerate(arr_multi_normal_ug_gr_df):
        lst_u_df.append(value[0] + arr_g_mag_df[idx])
    arr_u_mag_df = np.array(lst_u_df)

    for idx, value in enumerate(arr_multi_normal_iz_zj_df):
        lst_j_df.append(arr_z_mag_df[idx] - value[1])
    arr_j_mag_df = np.array(lst_j_df)

    for idx, value in enumerate(arr_multi_normal_zj_jh_df):
        lst_h_df.append(arr_j_mag_df[idx] - value[1])
    arr_h_mag_df = np.array(lst_h_df)

    for idx, value in enumerate(arr_multi_normal_jh_hks_df):
        lst_k_df.append(arr_h_mag_df[idx] - value[1])
    arr_ks_mag_df = np.array(lst_k_df)

    arr_u_g_mag_df = arr_u_mag_df - arr_g_mag_df
    arr_g_r_mag_df = arr_g_mag_df - arr_r_mag_df
    arr_r_i_mag_df = arr_r_mag_df - arr_i_mag
    arr_i_z_mag_df = arr_i_mag - arr_z_mag_df
    arr_z_j_mag_df = arr_z_mag_df - arr_j_mag_df
    arr_j_h_mag_df = arr_j_mag_df - arr_h_mag_df
    arr_h_ks_mag_df = arr_h_mag_df - arr_ks_mag_df

    cov_matrix_gen_ug_gr_df = np.cov(np.array([arr_u_g_mag_df, arr_g_r_mag_df]))
    cov_matrix_gen_gr_ri_df = np.cov(np.array([arr_g_r_mag_df, arr_r_i_mag_df]))
    cov_matrix_gen_ri_iz_df = np.cov(np.array([arr_r_i_mag_df, arr_i_z_mag_df]))
    cov_matrix_gen_iz_zj_df = np.cov(np.array([arr_i_z_mag_df, arr_z_j_mag_df]))
    cov_matrix_gen_zj_jh_df = np.cov(np.array([arr_z_j_mag_df, arr_j_h_mag_df]))
    cov_matrix_gen_jh_hks_df = np.cov(np.array([arr_j_h_mag_df, arr_h_ks_mag_df]))
    dictionary_generated_data = {
        "generated deep u mag": arr_u_mag_df,
        "generated deep g mag": arr_g_mag_df,
        "generated deep r mag": arr_r_mag_df,
        "generated deep i mag": arr_i_mag,
        "generated deep z mag": arr_z_mag_df,
        "generated deep j mag": arr_j_mag_df,
        "generated deep h mag": arr_h_mag_df,
        "generated deep ks mag": arr_ks_mag_df,
        "generated deep i flux": mag2flux(arr_i_mag),
        "generated deep r flux": mag2flux(arr_r_mag_df),
        "generated deep z flux": mag2flux(arr_z_mag_df),
        "generated deep u-g mag": arr_u_g_mag_df,
        "generated deep g-r mag": arr_g_r_mag_df,
        "generated deep r-i mag": arr_r_i_mag_df,
        "generated deep i-z mag": arr_i_z_mag_df,
        "generated deep z-j mag": arr_z_j_mag_df,
        "generated deep j-h mag": arr_j_h_mag_df,
        "generated deep h-ks mag": arr_h_ks_mag_df,
        "cov matrix generated deep u-g, g-r": cov_matrix_gen_ug_gr_df,
        "cov matrix generated deep g-r, r-i": cov_matrix_gen_gr_ri_df,
        "cov matrix generated deep r-i, i-z": cov_matrix_gen_ri_iz_df,
        "cov matrix generated deep i-z, z-j": cov_matrix_gen_iz_zj_df,
        "cov matrix generated deep z-j, j-h": cov_matrix_gen_zj_jh_df,
        "cov matrix generated deep j-h, h-ks": cov_matrix_gen_jh_hks_df
    }

    print(f"compare cov matrix deep u-g, g-r:"
          f"\t generated {cov_matrix_gen_ug_gr_df}"
          f"\t original {dictionary_cov_matrix['covariance matrix deep u-g, g-r']}")
    print(f"compare cov matrix deep g-r, r-i:"
          f"\t generated {cov_matrix_gen_gr_ri_df}"
          f"\t original {dictionary_cov_matrix['covariance matrix deep g-r, r-i']}")
    print(f"compare cov matrix deep r-i, i-z:"
          f"\t generated {cov_matrix_gen_ri_iz_df}"
          f"\t original {dictionary_cov_matrix['covariance matrix deep r-i, i-z']}")
    print(f"compare cov matrix deep i-z, z-j:"
          f"\t generated {cov_matrix_gen_iz_zj_df}"
          f"\t original {dictionary_cov_matrix['covariance matrix deep i-z, z-j']}")
    print(f"compare cov matrix deep z-j, j-h:"
          f"\t generated {cov_matrix_gen_zj_jh_df}"
          f"\t original {dictionary_cov_matrix['covariance matrix deep z-j, j-h']}")
    print(f"compare cov matrix deep j-h, h-ks:"
          f"\t generated {cov_matrix_gen_jh_hks_df}"
          f"\t original {dictionary_cov_matrix['covariance matrix deep j-h, h-ks']}")

    if plot_data is True:
        chaincon = ChainConsumer()
        df_deep_field = pd.DataFrame({
            "array deep r-i": np.array(dictionary_cov_matrix["array deep r-i"]),
            "array deep i-z": np.array(dictionary_cov_matrix["array deep i-z"])
        })
        arr_deep_field = df_deep_field.to_numpy()
        df_generated = pd.DataFrame({
            "generated deep r-i mag": np.array(dictionary_generated_data["generated deep r-i mag"]),
            "generated deep i-z mag": np.array(dictionary_generated_data["generated deep i-z mag"])
        })
        arr_generated = df_generated.to_numpy()
        parameter = [
            "r-i",
            "i-z"
        ]
        chaincon.add_chain(arr_deep_field, parameters=parameter, name="BDF_MAG_DERED_CALIB_{R, I, Z}")
        chaincon.add_chain(arr_generated, parameters=parameter, name="generated deep field mag {R, I, Z}")
        chaincon.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
        chaincon.plotter.plot(
            # filename=f"{dictionary_plot_paths['chain plot']}/chain_plot_epoch_{epoch}.png",
            figsize="page",
            display=True,
            # truth=[
            #     df_test_data["unsheared/mag_r"].mean(),
            #     df_test_data["unsheared/mag_i"].mean(),
            #     df_test_data["unsheared/mag_z"].mean(),
            #     df_test_data["unsheared/snr"].mean(),
            #     df_test_data["unsheared/size_ratio"].mean(),
            #     df_test_data["unsheared/T"].mean(),
            # ]
        )
        plt.clf()

        chaincon = ChainConsumer()
        df_deep_field = pd.DataFrame({
            "array deep r": np.array(dictionary_cov_matrix["array deep r"]),
            "array deep i": np.array(dictionary_cov_matrix["array deep i"]),
            "array deep z": np.array(dictionary_cov_matrix["array deep z"]),
        })
        arr_deep_field = df_deep_field.to_numpy()
        df_generated = pd.DataFrame({
            "generated deep r mag": np.array(dictionary_generated_data["generated deep r mag"]),
            "generated deep i mag": np.array(dictionary_generated_data["generated deep i mag"]),
            "generated deep z mag": np.array(dictionary_generated_data["generated deep z mag"])
        })
        arr_generated = df_generated.to_numpy()
        parameter = [
            "r",
            "i",
            "z"
        ]
        chaincon.add_chain(arr_deep_field, parameters=parameter, name="BDF_MAG_DERED_CALIB_{R, I, Z}")
        chaincon.add_chain(arr_generated, parameters=parameter, name="generated deep field mag {R, I, Z}")
        chaincon.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
        chaincon.plotter.plot(
            # filename=f"{dictionary_plot_paths['chain plot']}/chain_plot_epoch_{epoch}.png",
            figsize="page",
            display=True,
            # truth=[
            #     df_test_data["unsheared/mag_r"].mean(),
            #     df_test_data["unsheared/mag_i"].mean(),
            #     df_test_data["unsheared/mag_z"].mean(),
            #     df_test_data["unsheared/snr"].mean(),
            #     df_test_data["unsheared/size_ratio"].mean(),
            #     df_test_data["unsheared/T"].mean(),
            # ]
        )
        plt.clf()

        # plot_2d_kde(
        #     x=arr_u_g_mag_df,
        #     y=arr_g_r_mag_df,
        #     x_label="deep field u-g mag generated",
        #     y_label="deep field g-r mag generated",
        #     color="Greens",
        #     title="color plot (kde) U-G, G-R generated",
        #     manual_levels=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        # )
        # if save_plot is True:
        #     plt.savefig(f"{save_path_plots}/Colorplot_ug_gr_generated.png")
        # plt.show()
        # plt.clf()
        #
        # plot_2d_kde(
        #     x=arr_g_r_mag_df,
        #     y=arr_r_i_mag_df,
        #     x_label="deep field g-r mag generated",
        #     y_label="deep field r-i mag generated",
        #     color="Greens",
        #     title="color plot (kde) G-R, R-I generated",
        #     manual_levels=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        # )
        # if save_plot is True:
        #     plt.savefig(f"{save_path_plots}/Colorplot_gr_ri_generated.png")
        # plt.show()
        # plt.clf()
        #
        # plot_2d_kde(
        #     x=arr_r_i_mag_df,
        #     y=arr_i_z_mag_df,
        #     x_label="deep field r-i mag generated",
        #     y_label="deep field i-z mag generated",
        #     color="Greens",
        #     title="color plot (kde) R-I, I-Z generated",
        #     manual_levels=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        # )
        # if save_plot is True:
        #     plt.savefig(f"{save_path_plots}/Colorplot_ri_iz_generated.png")
        # plt.show()
        # plt.clf()
        #
        # plot_2d_kde(
        #     x=arr_i_z_mag_df,
        #     y=arr_z_j_mag_df,
        #     x_label="deep field i-z mag generated",
        #     y_label="deep field z-j mag generated",
        #     color="Greens",
        #     title="color plot (kde) I-Z, Z-J generated",
        #     manual_levels=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        # )
        # if save_plot is True:
        #     plt.savefig(f"{save_path_plots}/Colorplot_iz_zj_generated.png")
        # plt.show()
        # plt.clf()
        #
        # plot_2d_kde(
        #     x=arr_z_j_mag_df,
        #     y=arr_j_h_mag_df,
        #     x_label="deep field z-j mag generated",
        #     y_label="deep field j-h mag generated",
        #     color="Greens",
        #     title="color plot (kde) Z-J, J-H generated",
        #     manual_levels=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        # )
        # if save_plot is True:
        #     plt.savefig(f"{save_path_plots}/Colorplot_zj_jh_generated.png")
        # plt.show()
        # plt.clf()
        #
        # plot_2d_kde(
        #     x=arr_j_h_mag_df,
        #     y=arr_h_ks_mag_df,
        #     x_label="deep field j-h mag generated",
        #     y_label="deep field h-ks mag generated",
        #     color="Greens",
        #     title="color plot (kde) J-H, H-KS generated",
        #     manual_levels=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        # )
        # if save_plot is True:
        #     plt.savefig(f"{save_path_plots}/Colorplot_jh_hks_generated.png")
        # plt.show()
        # plt.clf()

    return dictionary_generated_data


def generate_wide_field_data(dictionary_generated_data, data_frame, columns, bin_width, save_path_plots,
                             plot_data=False, save_plot=False):
    """"""

    # Create numpy array of all three bands
    arr_mcal_df_r_flux = np.array(data_frame[columns[0]])
    arr_mcal_df_i_flux = np.array(data_frame[columns[1]])
    arr_mcal_df_z_flux = np.array(data_frame[columns[2]])
    arr_mcal_df_r_mag = flux2mag(arr_mcal_df_r_flux)
    arr_mcal_df_i_mag = flux2mag(arr_mcal_df_i_flux)
    arr_mcal_df_z_mag = flux2mag(arr_mcal_df_z_flux)
    arr_mcal_wf_r_flux = np.array(data_frame[columns[3]])
    arr_mcal_wf_i_flux = np.array(data_frame[columns[4]])
    arr_mcal_wf_z_flux = np.array(data_frame[columns[5]])
    arr_mcal_wf_r_mag = flux2mag(arr_mcal_wf_r_flux)
    arr_mcal_wf_i_mag = flux2mag(arr_mcal_wf_i_flux)
    arr_mcal_wf_z_mag = flux2mag(arr_mcal_wf_z_flux)

    # Calculating the magnitude differences r-i and i-z
    arr_mcal_df_ri_mag = arr_mcal_df_r_mag - arr_mcal_df_i_mag
    arr_mcal_df_iz_mag = arr_mcal_df_i_mag - arr_mcal_df_z_mag
    arr_mcal_wf_ri_mag = arr_mcal_wf_r_mag - arr_mcal_wf_i_mag
    arr_mcal_wf_iz_mag = arr_mcal_wf_i_mag - arr_mcal_wf_z_mag
    
    dict_wide_field = {
        "metacal flux r deep field": arr_mcal_df_r_flux,
        "metacal flux i deep field": arr_mcal_df_i_flux,
        "metacal flux z deep field": arr_mcal_df_z_flux,
        "metacal flux r wide field": arr_mcal_wf_r_flux,
        "metacal flux i wide field": arr_mcal_wf_i_flux,
        "metacal flux z wide field": arr_mcal_wf_z_flux,
        "metacal mag r deep field": arr_mcal_df_r_mag,
        "metacal mag i deep field": arr_mcal_df_i_mag,
        "metacal mag z deep field": arr_mcal_df_z_mag,
        "metacal mag r wide field": arr_mcal_wf_r_mag,
        "metacal mag i wide field": arr_mcal_wf_i_mag,
        "metacal mag z wide field": arr_mcal_wf_z_mag,
        "generated mag u deep field": dictionary_generated_data["generated deep u mag"],
        "generated mag g deep field": dictionary_generated_data["generated deep g mag"],
        "generated mag r deep field": dictionary_generated_data["generated deep r mag"],
        "generated mag i deep field": dictionary_generated_data["generated deep i mag"],
        "generated mag z deep field": dictionary_generated_data["generated deep z mag"],
        "generated mag j deep field": dictionary_generated_data["generated deep j mag"],
        "generated mag h deep field": dictionary_generated_data["generated deep h mag"],
        "generated mag ks deep field": dictionary_generated_data["generated deep ks mag"],
        "generated flux r deep field": dictionary_generated_data["generated deep r flux"],
        "generated flux i deep field": dictionary_generated_data["generated deep i flux"],
        "generated flux z deep field": dictionary_generated_data["generated deep z flux"],
    }

    cov_r_flux = np.cov(np.array(
        [dict_wide_field["metacal flux r deep field"],
         dict_wide_field["metacal flux r wide field"]]))

    cov_i_flux = np.cov(np.array(
        [dict_wide_field["metacal flux i deep field"],
         dict_wide_field["metacal flux i wide field"]]))

    cov_z_flux = np.cov(np.array(
        [dict_wide_field["metacal flux z deep field"],
         dict_wide_field["metacal flux z wide field"]]))

    print(f"covariance matrix r-band between deep and wide field flux: {cov_r_flux}")
    print(f"covariance matrix i-band between deep and wide field flux: {cov_i_flux}")
    print(f"covariance matrix z-band between deep and wide field flux: {cov_z_flux}")

    print(f"var r: sqrt({cov_r_flux[0, 1]})={np.sqrt(cov_r_flux[0, 1])}")
    print(f"var i: sqrt({cov_i_flux[0, 1]})={np.sqrt(cov_i_flux[0, 1])}")
    print(f"var z: sqrt({cov_z_flux[0, 1]})={np.sqrt(cov_z_flux[0, 1])}")

    _alpha = 1.15
    _beta = 100
    arr_normal_r_flux = np.random.normal(
        loc=0,
        scale=np.sqrt(cov_r_flux[0, 1]),
        size=len(dict_wide_field["generated flux r deep field"]))
    dict_wide_field[f"generated flux r wide field"] = \
        _alpha * dict_wide_field[f"generated flux r deep field"] + (_beta * (1 - arr_normal_r_flux) / arr_normal_r_flux)

    arr_normal_i_flux = np.random.normal(
        loc=0,
        scale=np.sqrt(cov_i_flux[0, 1]),
        size=len(dict_wide_field["generated flux i deep field"]))
    dict_wide_field[f"generated flux i wide field"] = \
        _alpha * dict_wide_field[f"generated flux i deep field"] + (_beta * (1 - arr_normal_i_flux) / arr_normal_i_flux)

    arr_normal_z_flux = np.random.normal(
        loc=0,
        scale=np.sqrt(cov_z_flux[0, 1]),
        size=len(dict_wide_field["generated flux z deep field"]))
    dict_wide_field[f"generated flux z wide field"] = \
        _alpha * dict_wide_field[f"generated flux z deep field"] + (_beta * (1 - arr_normal_z_flux) / arr_normal_z_flux)

    cov_r_flux_generated = np.cov(np.array(
        [dict_wide_field["generated flux r deep field"],
         dict_wide_field["generated flux r wide field"]]))

    cov_i_flux_generated = np.cov(np.array(
        [dict_wide_field["generated flux i deep field"],
         dict_wide_field["generated flux i wide field"]]))

    cov_z_flux_generated = np.cov(np.array(
        [dict_wide_field["generated flux z deep field"],
         dict_wide_field["generated flux z wide field"]]))

    print(f"covariance matrix r-band between deep and wide field flux generated: {cov_r_flux_generated}")
    print(f"covariance matrix i-band between deep and wide field flux generated: {cov_i_flux_generated}")
    print(f"covariance matrix z-band between deep and wide field flux generated: {cov_z_flux_generated}")

    print(f"var r generated: sqrt({cov_r_flux_generated[0, 1]})={np.sqrt(cov_r_flux_generated[0, 1])}")
    print(f"var i generated: sqrt({cov_i_flux_generated[0, 1]})={np.sqrt(cov_i_flux_generated[0, 1])}")
    print(f"var z generated: sqrt({cov_z_flux_generated[0, 1]})={np.sqrt(cov_z_flux_generated[0, 1])}")

    dict_wide_field[f"generated mag r wide field"] = flux2mag(dict_wide_field[f"generated flux r wide field"])
    dict_wide_field[f"generated mag i wide field"] = flux2mag(dict_wide_field[f"generated flux i wide field"])
    dict_wide_field[f"generated mag z wide field"] = flux2mag(dict_wide_field[f"generated flux z wide field"])

    arr_gen_ri_df = dict_wide_field[f"generated mag r deep field"] - dict_wide_field[f"generated mag i deep field"]
    arr_gen_iz_df = dict_wide_field[f"generated mag i deep field"] - dict_wide_field[f"generated mag z deep field"]
    arr_gen_ri_wf = dict_wide_field[f"generated mag r wide field"] - dict_wide_field[f"generated mag i wide field"]
    arr_gen_iz_wf = dict_wide_field[f"generated mag i wide field"] - dict_wide_field[f"generated mag z wide field"]

    cov_matrix_gen_df = np.cov(np.array([arr_gen_ri_df, arr_gen_iz_df]))
    cov_matrix_gen_wf = np.cov(np.array([arr_gen_ri_wf, arr_gen_iz_wf]))
    cov_matrix_mcal_df = np.cov(np.array([arr_mcal_df_ri_mag, arr_mcal_df_iz_mag]))
    cov_matrix_mcal_wf = np.cov(np.array([arr_mcal_wf_ri_mag, arr_mcal_wf_iz_mag]))

    print("generated covariance matrix r-i, i-z deep field", cov_matrix_gen_df)
    print("generated covariance matrix r-i, i-z wide field", cov_matrix_gen_wf)
    print("metacal covariance matrix r-i, i-z deep field", cov_matrix_mcal_df)
    print("metacal covariance matrix r-i, i-z wide field", cov_matrix_mcal_wf)

    if plot_data is True:
        chaincon = ChainConsumer()
        df_gen_deep_field = pd.DataFrame({
            "generated mag r deep field": np.array(dict_wide_field["generated mag r deep field"]),
            "generated mag i deep field": np.array(dict_wide_field["generated mag i deep field"]),
            "generated mag z deep field": np.array(dict_wide_field["generated mag z deep field"]),
        })
        arr_gen_deep_field = df_gen_deep_field.to_numpy()
        df_deep_field = pd.DataFrame({
            "deep r mag": np.array(dict_wide_field["metacal mag r deep field"]),
            "deep i mag": np.array(dict_wide_field["metacal mag i deep field"]),
            "deep z mag": np.array(dict_wide_field["metacal mag z deep field"])
        })
        arr_deep_field = df_deep_field.to_numpy()
        df_gen_wide_field = pd.DataFrame({
            "generated mag r wide field": np.array(dict_wide_field["generated mag r wide field"]),
            "generated mag i wide field": np.array(dict_wide_field["generated mag i wide field"]),
            "generated mag z wide field": np.array(dict_wide_field["generated mag z wide field"]),
        })
        arr_gen_wide_field = df_gen_wide_field.to_numpy()
        df_wide_field = pd.DataFrame({
            "wide r mag": np.array(dict_wide_field["metacal mag r wide field"]),
            "wide i mag": np.array(dict_wide_field["metacal mag i wide field"]),
            "wide z mag": np.array(dict_wide_field["metacal mag z wide field"])
        })
        arr_wide_field = df_wide_field.to_numpy()
        parameter = [
            "r",
            "i",
            "z"
        ]
        chaincon.add_chain(arr_deep_field, parameters=parameter, name="BDF_MAG_DERED_CALIB_{R, I, Z}")
        chaincon.add_chain(arr_gen_deep_field, parameters=parameter, name="generated deep field mag {R, I, Z}")
        chaincon.add_chain(arr_wide_field, parameters=parameter, name="unsheared/flux_{r, i, z}")
        chaincon.add_chain(arr_gen_wide_field, parameters=parameter, name="generated deep field mag {r, i, z}")
        chaincon.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
        chaincon.plotter.plot(
            # filename=f"{dictionary_plot_paths['chain plot']}/chain_plot_epoch_{epoch}.png",
            figsize="page",
            display=True,
            # truth=[
            #     df_test_data["unsheared/mag_r"].mean(),
            #     df_test_data["unsheared/mag_i"].mean(),
            #     df_test_data["unsheared/mag_z"].mean(),
            #     df_test_data["unsheared/snr"].mean(),
            #     df_test_data["unsheared/size_ratio"].mean(),
            #     df_test_data["unsheared/T"].mean(),
            # ]
        )
        plt.clf()
        # sns.histplot(
        #     x=flux2mag(dict_wide_field[f"generated flux r wide field"]),
        #     kde=True,
        #     color="darkred",
        #     stat="density",
        #     label=f"generated wide field mag in r"
        # )
        #
        # sns.histplot(
        #     x=flux2mag(dict_wide_field[f"generated flux r deep field"]),
        #     kde=True,
        #     color="darkblue",
        #     stat="density",
        #     label=f"generated deep field mag in r"
        # )
        #
        # sns.histplot(
        #     x=dict_wide_field[f"metacal mag r wide field"],
        #     kde=True,
        #     color="darkgreen",
        #     stat="density",
        #     label=f"metacal wide field mag in r"
        # )
        #
        # sns.histplot(
        #     x=dict_wide_field["metacal mag r deep field"],
        #     kde=True,
        #     color="darkorange",
        #     stat="density",
        #     label=f"metacal deep field mag in r"
        # )
        # plt.title(f"Compare generated density with metacal in r-band")
        # plt.xlabel("r-mag")
        # plt.legend()
        # if save_plot is True:
        #     plt.savefig(f"{save_path_plots}/hist_compare_r_df_wf.png")
        # plt.show()
        #
        # sns.histplot(
        #     x=flux2mag(dict_wide_field[f"generated flux i wide field"]),
        #     kde=True,
        #     color="darkred",
        #     stat="density",
        #     label=f"generated wide field mag in i"
        # )
        #
        # sns.histplot(
        #     x=flux2mag(dict_wide_field[f"generated flux i deep field"]),
        #     kde=True,
        #     color="darkblue",
        #     stat="density",
        #     label=f"generated deep field mag in i"
        # )
        #
        # sns.histplot(
        #     x=dict_wide_field[f"metacal mag i wide field"],
        #     kde=True,
        #     color="darkgreen",
        #     stat="density",
        #     label=f"metacal wide field mag in i"
        # )
        #
        # sns.histplot(
        #     x=dict_wide_field["metacal mag i deep field"],
        #     kde=True,
        #     color="darkorange",
        #     stat="density",
        #     label=f"metacal deep field mag in i"
        # )
        #
        # plt.title(f"Compare generated density with metacal in i-band")
        # plt.xlabel("i-mag")
        # plt.legend()
        # if save_plot is True:
        #     plt.savefig(f"{save_path_plots}/hist_compare_i_df_wf.png")
        # plt.show()
        #
        # sns.histplot(
        #     x=flux2mag(dict_wide_field[f"generated flux z wide field"]),
        #     kde=True,
        #     color="darkred",
        #     stat="density",
        #     label=f"generated wide field mag in z"
        # )
        #
        # sns.histplot(
        #     x=flux2mag(dict_wide_field[f"generated flux z deep field"]),
        #     kde=True,
        #     color="darkblue",
        #     stat="density",
        #     label=f"generated deep field mag in z"
        # )
        #
        # sns.histplot(
        #     x=dict_wide_field[f"metacal mag z wide field"],
        #     kde=True,
        #     color="darkgreen",
        #     stat="density",
        #     label=f"metacal wide field mag in z"
        # )
        #
        # sns.histplot(
        #     x=dict_wide_field["metacal mag z deep field"],
        #     kde=True,
        #     color="darkorange",
        #     stat="density",
        #     label=f"metacal deep field mag in z"
        # )
        #
        # plt.title(f"Compare generated density with metacal in z-band")
        # plt.xlabel("z-mag")
        # plt.legend()
        # if save_plot is True:
        #     plt.savefig(f"{save_path_plots}/hist_compare_z_df_wf.png")
        # plt.show()
        #
        # plot_2d_kde_compare(
        #     x1=arr_gen_ri_wf,
        #     y1=arr_gen_iz_wf,
        #     x2=arr_gen_ri_df,
        #     y2=arr_gen_iz_df,
        #     color=["hsv", "hsv"],
        #     x_label="r-i mag",
        #     y_label="i-z mag",
        #     title="color plot (kde) generated deep field and wide field",
        #     manual_levels=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        # )
        # if save_plot is True:
        #     plt.savefig(f"{save_path_plots}/color_ri_iz_gen.png")
        # plt.show()
        #
        # plot_2d_kde_compare(
        #     x1=arr_mcal_wf_ri_mag,
        #     y1=arr_mcal_wf_iz_mag,
        #     x2=arr_mcal_df_ri_mag,
        #     y2=arr_mcal_df_iz_mag,
        #     color=["hsv", "hsv"],
        #     x_label="r-i mag",
        #     y_label="i-z mag",
        #     title="color plot (kde) metacal deep field and wide field",
        #     manual_levels=np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        # )
        # if save_plot is True:
        #     plt.savefig(f"{save_path_plots}/color_ri_iz_mcal.png")
        # plt.show()
    return dict_wide_field


def survey_conditions(dict_wide_field_data, data_frame, detected_length, columns_survey_conditions, plot_data,
                      save_plot, save_path_plots, only_detected_objects):
    """"""

    dict_data = {
        f"BDF_MAG_DERED_CALIB_U": dict_wide_field_data[f"generated mag u deep field"],
        f"BDF_MAG_DERED_CALIB_G": dict_wide_field_data[f"generated mag g deep field"],
        f"BDF_MAG_DERED_CALIB_R": dict_wide_field_data[f"generated mag r deep field"],
        f"BDF_MAG_DERED_CALIB_I": dict_wide_field_data[f"generated mag i deep field"],
        f"BDF_MAG_DERED_CALIB_Z": dict_wide_field_data[f"generated mag z deep field"],
        f"BDF_MAG_DERED_CALIB_J": dict_wide_field_data[f"generated mag j deep field"],
        f"BDF_MAG_DERED_CALIB_H": dict_wide_field_data[f"generated mag h deep field"],
        f"BDF_MAG_DERED_CALIB_KS": dict_wide_field_data[f"generated mag ks deep field"],
        f"BDF_MAG_ERR_DERED_CALIB_R": [1 for _ in range(len(dict_wide_field_data[f"generated mag u deep field"]))],
        f"BDF_MAG_ERR_DERED_CALIB_I": [1 for _ in range(len(dict_wide_field_data[f"generated mag u deep field"]))],
        f"BDF_MAG_ERR_DERED_CALIB_Z": [1 for _ in range(len(dict_wide_field_data[f"generated mag u deep field"]))],
        f"BDF_FLUX_DERED_CALIB_R": dict_wide_field_data[f"generated flux r deep field"],
        f"BDF_FLUX_DERED_CALIB_I": dict_wide_field_data[f"generated flux i deep field"],
        f"BDF_FLUX_DERED_CALIB_Z": dict_wide_field_data[f"generated flux z deep field"],
        f"BDF_FLUX_ERR_DERED_CALIB_R": [1 for _ in range(len(dict_wide_field_data[f"generated mag u deep field"]))],
        f"BDF_FLUX_ERR_DERED_CALIB_I": [1 for _ in range(len(dict_wide_field_data[f"generated mag u deep field"]))],
        f"BDF_FLUX_ERR_DERED_CALIB_Z": [1 for _ in range(len(dict_wide_field_data[f"generated mag u deep field"]))],
        f"unsheared/flux_r": dict_wide_field_data[f"generated flux r wide field"],
        f"unsheared/flux_i": dict_wide_field_data[f"generated flux i wide field"],
        f"unsheared/flux_z": dict_wide_field_data[f"generated flux z wide field"],
        f"unsheared/flux_err_r": [1 for _ in range(len(dict_wide_field_data[f"generated mag u deep field"]))],
        f"unsheared/flux_err_i": [1 for _ in range(len(dict_wide_field_data[f"generated mag u deep field"]))],
        f"unsheared/flux_err_z": [1 for _ in range(len(dict_wide_field_data[f"generated mag u deep field"]))],
        f"unsheared/mag_r": dict_wide_field_data[f"generated mag r wide field"],
        f"unsheared/mag_i": dict_wide_field_data[f"generated mag i wide field"],
        f"unsheared/mag_z": dict_wide_field_data[f"generated mag z wide field"],
        f"unsheared/mag_err_r": [1 for _ in range(len(dict_wide_field_data[f"generated mag u deep field"]))],
        f"unsheared/mag_err_i": [1 for _ in range(len(dict_wide_field_data[f"generated mag u deep field"]))],
        f"unsheared/mag_err_z": [1 for _ in range(len(dict_wide_field_data[f"generated mag u deep field"]))]
    }

    for col in columns_survey_conditions:
        if col == "true_detected":
            if only_detected_objects is True:
                dict_data[f"{col}"] = np.ones(len(dict_wide_field_data[f"generated flux z deep field"]))
            else:
                ones = np.ones(detected_length[0])
                zeros = np.zeros(detected_length[1])
                dict_data[f"{col}"] = np.concatenate((ones, zeros))
        elif col == "unsheared/flags":
            dict_data[f"{col}"] = np.zeros(len(dict_wide_field_data[f"generated flux z deep field"]))
        elif col == "BDF_G":
            data_frame[f"{col}"] = np.sqrt(data_frame["BDF_G_0"]**2 + data_frame["BDF_G_1"]**2)
            print(col, np.array(data_frame[col]))
            print("mean", np.mean(np.array(data_frame[col])))
            print("std", np.std(np.array(data_frame[col])))
            gen = np.random.normal(
                loc=np.mean(np.array(data_frame[col])),
                scale=np.std(np.array(data_frame[col])),
                size=len(dict_wide_field_data[f"generated flux z deep field"]))
            print(f"gen {col}", gen)
            dict_data[f"{col}"] = gen
        elif col == "BDF_T":
            # df_balrog[f"{col}"] = np.sqrt(df_balrog[col])
            print(col, np.array(data_frame[col]))
            print("mean", np.mean(np.array(data_frame[col])))
            print("std", np.std(np.array(data_frame[col])))
            gen = np.random.normal(
                loc=np.mean(np.array(data_frame[col])),
                scale=np.std(np.array(data_frame[col])),
                size=len(dict_wide_field_data[f"generated flux z deep field"]))
            print(f"gen {col}", gen)
            dict_data[f"{col}"] = gen
        else:
            print(col, np.array(data_frame[col]))
            print("mean", np.mean(np.array(data_frame[col])))
            print("std", np.std(np.array(data_frame[col])))
            gen = np.random.normal(
                loc=np.mean(np.array(data_frame[col])),
                scale=np.std(np.array(data_frame[col])),
                size=len(dict_wide_field_data[f"generated flux z deep field"]))
            print(f"gen {col}", gen)

            dict_data[f"{col}"] = gen

        # if plot_data is True:

        # if col not in ["true_detected", "unsheared/flags"]:
        #     sns.histplot(
        #         x=df_balrog[col],
        #         kde=True,
        #         color="red",
        #         stat="density",
        #         label=f"{col}"
        #     )
        #
        #     sns.histplot(
        #         x=dict_data[f"{col}"],
        #         kde=True,
        #         color="blue",
        #         stat="density",
        #         label=f"Generated {col}"
        #     )
        #     plt.title(f"{col}")
        #     plt.legend()
        #     if save_plot is True:
        #         plt.savefig(f"{save_path_plots}/survey_condition_{col.replace('/', '_')}.png")
        #     plt.show()

    if plot_data is True:
        df_true = pd.DataFrame({
            f"BDF_T": data_frame[f"BDF_T"],
            f"BDF_G": data_frame[f"BDF_G"],
            f"EBV_SFD98": data_frame[f"EBV_SFD98"]
        })
        df_generated = pd.DataFrame({
            f"BDF_T": dict_data[f"BDF_T"],
            f"BDF_G": dict_data[f"BDF_G"],
            f"EBV_SFD98": dict_data[f"EBV_SFD98"]
        })
        parameter = [
            "BDF_T",
            "BDF_G",
            "EBV_SFD98"
        ]

        chaincon = ChainConsumer()
        arr_true = df_true.to_numpy()
        arr_generated = df_generated.to_numpy()

        chaincon.add_chain(arr_true, parameters=parameter, name="true survey conditions")
        chaincon.add_chain(arr_generated, parameters=parameter, name="generated survey conditions")
        chaincon.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
        chaincon.plotter.plot(
            figsize="page",
            display=True,

        )
        plt.clf()

        df_true = pd.DataFrame({
            f"unsheared/snr": data_frame[f"unsheared/snr"],
            f"unsheared/size_ratio": data_frame[f"unsheared/size_ratio"],
            f"unsheared/T": data_frame[f"unsheared/T"]
        })
        df_generated = pd.DataFrame({
            f"unsheared/snr": dict_data[f"unsheared/snr"],
            f"unsheared/size_ratio": dict_data[f"unsheared/size_ratio"],
            f"unsheared/T": dict_data[f"unsheared/T"]
        })
        parameter = [
            f"unsheared/snr",
            f"unsheared/size_ratio",
            f"unsheared/T"
        ]

        chaincon = ChainConsumer()
        arr_true = df_true.to_numpy()
        arr_generated = df_generated.to_numpy()

        chaincon.add_chain(arr_true, parameters=parameter, name="true survey conditions")
        chaincon.add_chain(arr_generated, parameters=parameter, name="generated survey conditions")
        chaincon.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
        chaincon.plotter.plot(
            figsize="page",
            display=True,

        )
        plt.clf()

        df_true = pd.DataFrame({
            f"AIRMASS_WMEAN_R": data_frame[f"AIRMASS_WMEAN_R"],
            f"AIRMASS_WMEAN_I": data_frame[f"AIRMASS_WMEAN_I"],
            f"AIRMASS_WMEAN_Z": data_frame[f"AIRMASS_WMEAN_Z"]

        })
        df_generated = pd.DataFrame({
            f"AIRMASS_WMEAN_R": dict_data[f"AIRMASS_WMEAN_R"],
            f"AIRMASS_WMEAN_I": dict_data[f"AIRMASS_WMEAN_I"],
            f"AIRMASS_WMEAN_Z": dict_data[f"AIRMASS_WMEAN_Z"]
        })
        parameter = [
            f"AIRMASS_WMEAN_R",
            f"AIRMASS_WMEAN_I",
            f"AIRMASS_WMEAN_Z"
        ]

        chaincon = ChainConsumer()
        arr_true = df_true.to_numpy()
        arr_generated = df_generated.to_numpy()

        chaincon.add_chain(arr_true, parameters=parameter, name="true survey conditions")
        chaincon.add_chain(arr_generated, parameters=parameter, name="generated survey conditions")
        chaincon.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
        chaincon.plotter.plot(
            figsize="page",
            display=True,

        )
        plt.clf()

        df_true = pd.DataFrame({
            f"FWHM_WMEAN_R": data_frame[f"FWHM_WMEAN_R"],
            f"FWHM_WMEAN_I": data_frame[f"FWHM_WMEAN_I"],
            f"FWHM_WMEAN_Z": data_frame[f"FWHM_WMEAN_Z"]

        })
        df_generated = pd.DataFrame({
            f"FWHM_WMEAN_R": dict_data[f"FWHM_WMEAN_R"],
            f"FWHM_WMEAN_I": dict_data[f"FWHM_WMEAN_I"],
            f"FWHM_WMEAN_Z": dict_data[f"FWHM_WMEAN_Z"]
        })
        parameter = [
            f"FWHM_WMEAN_R",
            f"FWHM_WMEAN_I",
            f"FWHM_WMEAN_Z"
        ]

        chaincon = ChainConsumer()
        arr_true = df_true.to_numpy()
        arr_generated = df_generated.to_numpy()

        chaincon.add_chain(arr_true, parameters=parameter, name="true survey conditions")
        chaincon.add_chain(arr_generated, parameters=parameter, name="generated survey conditions")
        chaincon.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
        chaincon.plotter.plot(
            figsize="page",
            display=True,

        )
        plt.clf()

        df_true = pd.DataFrame({
            f"MAGLIM_R": data_frame[f"MAGLIM_R"],
            f"MAGLIM_I": data_frame[f"MAGLIM_I"],
            f"MAGLIM_Z": data_frame[f"MAGLIM_Z"]

        })
        df_generated = pd.DataFrame({
            f"MAGLIM_R": dict_data[f"MAGLIM_R"],
            f"MAGLIM_I": dict_data[f"MAGLIM_I"],
            f"MAGLIM_Z": dict_data[f"MAGLIM_Z"]
        })
        parameter = [
            f"MAGLIM_R",
            f"MAGLIM_I",
            f"MAGLIM_Z"
        ]

        chaincon = ChainConsumer()
        arr_true = df_true.to_numpy()
        arr_generated = df_generated.to_numpy()

        chaincon.add_chain(arr_true, parameters=parameter, name="true survey conditions")
        chaincon.add_chain(arr_generated, parameters=parameter, name="generated survey conditions")
        chaincon.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
        chaincon.plotter.plot(
            figsize="page",
            display=True,

        )
        plt.clf()
    return dict_data


def not_detected_objects(data_frame):
    data_frame.loc[(data_frame.detected == 0), f"unsheared/flux_r"] = -999
    data_frame.loc[(data_frame.detected == 0), f"unsheared/flux_i"] = -999
    data_frame.loc[(data_frame.detected == 0), f"unsheared/flux_z"] = -999
    data_frame.loc[(data_frame.detected == 0), f"unsheared/snr"] = -999
    data_frame.loc[(data_frame.detected == 0), f"unsheared/size_ratio"] = -999
    data_frame.loc[(data_frame.detected == 0), f"unsheared/T"] = -999
    data_frame.loc[(data_frame.detected == 0), f"unsheared/flags"] = -999
    return data_frame


def write_data_2_file(df_generated_data, df_data, save_path, number_of_sources, pickle_2, columns_riz, only_balrog):
    """"""
    if only_balrog[0] is True:

        df_data["BDF_MAG_DERED_CALIB_U"] = flux2mag(df_data["BDF_FLUX_DERED_CALIB_U"])
        df_data["BDF_MAG_DERED_CALIB_G"] = flux2mag(df_data["BDF_FLUX_DERED_CALIB_G"])
        df_data["BDF_MAG_DERED_CALIB_R"] = flux2mag(df_data["BDF_FLUX_DERED_CALIB_R"])
        df_data["BDF_MAG_DERED_CALIB_I"] = flux2mag(df_data["BDF_FLUX_DERED_CALIB_I"])
        df_data["BDF_MAG_DERED_CALIB_Z"] = flux2mag(df_data["BDF_FLUX_DERED_CALIB_Z"])
        df_data["BDF_MAG_DERED_CALIB_J"] = flux2mag(df_data["BDF_FLUX_DERED_CALIB_J"])
        df_data["BDF_MAG_DERED_CALIB_H"] = flux2mag(df_data["BDF_FLUX_DERED_CALIB_H"])
        df_data["BDF_MAG_DERED_CALIB_KS"] = flux2mag(df_data["BDF_FLUX_DERED_CALIB_KS"])
        df_data["BDF_MAG_ERR_DERED_CALIB_R"] = flux2mag(df_data["BDF_FLUX_ERR_DERED_CALIB_R"])
        df_data["BDF_MAG_ERR_DERED_CALIB_I"] = flux2mag(df_data["BDF_FLUX_ERR_DERED_CALIB_I"])
        df_data["BDF_MAG_ERR_DERED_CALIB_Z"] = flux2mag(df_data["BDF_FLUX_ERR_DERED_CALIB_Z"])

        df_data["unsheared/mag_r"] = flux2mag(df_data["unsheared/flux_r"])
        df_data["unsheared/mag_i"] = flux2mag(df_data["unsheared/flux_i"])
        df_data["unsheared/mag_z"] = flux2mag(df_data["unsheared/flux_z"])
        df_data["unsheared/mag_err_r"] = flux2mag(df_data["unsheared/flux_err_r"])
        df_data["unsheared/mag_err_i"] = flux2mag(df_data["unsheared/flux_err_i"])
        df_data["unsheared/mag_err_z"] = flux2mag(df_data["unsheared/flux_err_z"])

        df_data['BDF_G'] = np.sqrt(df_data["BDF_G_0"]**2 + df_data["BDF_G_1"]**2)

        df_data = df_data[[
            'BDF_MAG_DERED_CALIB_U',
            'BDF_MAG_DERED_CALIB_G',
            'BDF_MAG_DERED_CALIB_R',
            'BDF_MAG_DERED_CALIB_I',
            'BDF_MAG_DERED_CALIB_Z',
            'BDF_MAG_DERED_CALIB_J',
            'BDF_MAG_DERED_CALIB_H',
            'BDF_MAG_DERED_CALIB_KS',
            'BDF_MAG_ERR_DERED_CALIB_R',
            'BDF_MAG_ERR_DERED_CALIB_I',
            'BDF_MAG_ERR_DERED_CALIB_Z',
            'BDF_FLUX_DERED_CALIB_R',
            'BDF_FLUX_DERED_CALIB_I',
            'BDF_FLUX_DERED_CALIB_Z',
            'BDF_FLUX_ERR_DERED_CALIB_R',
            'BDF_FLUX_ERR_DERED_CALIB_I',
            'BDF_FLUX_ERR_DERED_CALIB_Z',
            'unsheared/flux_r',
            'unsheared/flux_i',
            'unsheared/flux_z',
            'unsheared/flux_err_r',
            'unsheared/flux_err_i',
            'unsheared/flux_err_z',
            'unsheared/mag_r',
            'unsheared/mag_i',
            'unsheared/mag_z',
            'unsheared/mag_err_r',
            'unsheared/mag_err_i',
            'unsheared/mag_err_z',
            'unsheared/snr',
            'unsheared/size_ratio',
            'unsheared/flags',
            'unsheared/T',
            'BDF_T',
            'BDF_G',
            'AIRMASS_WMEAN_R',
            'AIRMASS_WMEAN_I',
            'AIRMASS_WMEAN_Z',
            'FWHM_WMEAN_R',
            'FWHM_WMEAN_I',
            'FWHM_WMEAN_Z',
            'MAGLIM_R',
            'MAGLIM_I',
            'MAGLIM_Z',
            'EBV_SFD98',
            'true_detected']]
        xspace = np.arange(len(df_data))
        random_choice_idx = np.random.choice(xspace, size=int(only_balrog[1]))
        arr_training_data = df_data.to_numpy()
        arr_training_data = arr_training_data[random_choice_idx]
        if df_generated_data is not None:
            columns = df_generated_data.keys()
            df_generated_data = pd.DataFrame(data=arr_training_data, columns=columns)
        else:
            df_generated_data = pd.DataFrame(data=arr_training_data, columns=df_data.columns)

    if pickle_2 is True:
        only_detected = (df_generated_data["true_detected"] == 1)
        df_generated_data = df_generated_data[only_detected]

        df_generated_data["BDF_FLUX_DERED_CALIB_U"] = mag2flux(df_generated_data["BDF_MAG_DERED_CALIB_U"])
        df_generated_data["BDF_FLUX_DERED_CALIB_G"] = mag2flux(df_generated_data["BDF_MAG_DERED_CALIB_G"])
        df_generated_data["BDF_FLUX_DERED_CALIB_J"] = mag2flux(df_generated_data["BDF_MAG_DERED_CALIB_J"])
        df_generated_data["BDF_FLUX_DERED_CALIB_H"] = mag2flux(df_generated_data["BDF_MAG_DERED_CALIB_H"])
        df_generated_data["BDF_FLUX_DERED_CALIB_K"] = mag2flux(df_generated_data["BDF_MAG_DERED_CALIB_KS"])

        df_generated_data["BDF_FLUX_ERR_DERED_CALIB_U"] = 1/df_generated_data["BDF_FLUX_DERED_CALIB_U"]
        df_generated_data["BDF_FLUX_ERR_DERED_CALIB_G"] = 1/df_generated_data["BDF_FLUX_DERED_CALIB_G"]
        df_generated_data["BDF_FLUX_ERR_DERED_CALIB_R"] = 1/df_generated_data["BDF_FLUX_DERED_CALIB_R"]
        df_generated_data["BDF_FLUX_ERR_DERED_CALIB_I"] = 1/df_generated_data["BDF_FLUX_DERED_CALIB_I"]
        df_generated_data["BDF_FLUX_ERR_DERED_CALIB_Z"] = 1/df_generated_data["BDF_FLUX_DERED_CALIB_Z"]
        df_generated_data["BDF_FLUX_ERR_DERED_CALIB_J"] = 1/df_generated_data["BDF_FLUX_DERED_CALIB_J"]
        df_generated_data["BDF_FLUX_ERR_DERED_CALIB_H"] = 1/df_generated_data["BDF_FLUX_DERED_CALIB_H"]
        df_generated_data["BDF_FLUX_ERR_DERED_CALIB_K"] = 1/df_generated_data["BDF_FLUX_DERED_CALIB_K"]
        with open(f"{save_path}{number_of_sources}.pkl", "wb") as f:
            pickle.dump(df_generated_data.to_dict(), f, protocol=2)
    else:
        df_generated_data.to_pickle(f"{save_path}{number_of_sources}.pkl")


def main(path_to_data, size, save_path, save_plot, plot_fit_parameter, plot_generated_i_band, plot_covariance,
         plot_generated_df, plot_generated_wf, plot_survey, write_data, lower_bound, upper_bound, bin_size,
         column_flux_i_band, columns_flux_riz_band, columns_flux_ugjhks_band, columns_df_wf, columns_survey_conditions,
         only_detected_objects, only_balrog, save_path_plots="", pickle_2=False):

    """"""
    # Load deep field catalog (a pkl file)
    df_data = load_data(path_to_data, "pkl")
    # df_data = cut_i_band_mag(df_data, column_flux_i_band)
    data_detected = len(df_data[df_data["true_detected"] == 1])
    data_undetected = len(df_data[df_data["true_detected"] == 0])
    print(f"length true_detected = {data_detected} \t length not true_detected = {data_undetected}")

    length_gen_detected = int(size * (data_detected / (data_detected + data_undetected)))
    length_gen_undetected = int(size * (data_undetected / (data_detected + data_undetected)))
    while length_gen_detected + length_gen_undetected != size:
        if length_gen_detected + length_gen_undetected < size:
            length_gen_detected += 1
        elif length_gen_detected + length_gen_undetected > size:
            length_gen_detected -= 1
    print(f"length generated true_detected = {length_gen_detected} \t length generated not true_detected = {length_gen_undetected}")
    tpl_detected_length = (length_gen_detected, length_gen_undetected)

    print(f"Length of data set is {len(df_data)}")
    df_data = df_data[df_data["true_detected"] == 1]
    print(f"Length of data set for use true_detected objects only: {len(df_data)}")
    set_plot_settings()

    df_data = make_some_cuts(data_frame=df_data, columns_riz=columns_flux_riz_band)

    if size == -1:
        size = len(df_data[column_flux_i_band])

    if save_plot is True:
        if not os.path.exists(save_path_plots):
            os.mkdir(save_path_plots)
    if only_balrog[0] is False:
        dict_fit_dist = calc_fit_parameters(
            distribution=flux2mag(df_data[column_flux_i_band]),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            bin_size=bin_size,
            function=exp_func,
            column=column_flux_i_band,
            plot_data=plot_fit_parameter,
            save_plot=save_plot,
            save_path_plots=save_path_plots
        )

        bin_width = dict_fit_dist["bin width"]

        # Generate distribution
        dict_generated_dist = generate_distribution(
            dict_fit_params=dict_fit_dist,
            column=column_flux_i_band,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            plot_data=plot_generated_i_band,
            save_plot=save_plot,
            save_path_plots=save_path_plots,
            size=size)

        # Call get_covariance_matrix to get the covariance matrix
        dict_cov_matrix = get_covariance_matrix(
            data_frame=df_data,
            columns_riz=columns_flux_riz_band,
            columns_ugjhks=columns_flux_ugjhks_band,
            plot_data=plot_covariance,
            save_plot=save_plot,
            save_path_plots=save_path_plots
        )

        # Call generate_mag_distribution to generate 267229 data points in r and z-band with the generated i-band and the
        # calculated covariance matrix
        dict_generated_data = generate_mag_distribution(
            dictionary_cov_matrix=dict_cov_matrix,
            arr_i_mag=dict_generated_dist[f"generated {column_flux_i_band}"],
            size=size,
            plot_data=plot_generated_df,
            save_plot=save_plot,
            save_path_plots=save_path_plots
        )

        dict_wf_data = generate_wide_field_data(
            dictionary_generated_data=dict_generated_data,
            data_frame=df_data,
            columns=columns_df_wf,
            bin_width=bin_width,
            save_path_plots=save_path_plots,
            plot_data=plot_generated_wf,
            save_plot=save_plot
        )

        dict_gen_data = survey_conditions(
            dict_wide_field_data=dict_wf_data,
            data_frame=df_data,
            detected_length=tpl_detected_length,
            columns_survey_conditions=columns_survey_conditions,
            plot_data=plot_survey,
            save_plot=save_plot,
            save_path_plots=save_path_plots,
            only_detected_objects=only_detected_objects
        )

        df_generated_data = pd.DataFrame(dict_gen_data)

        if only_detected_objects is not True:
            df_generated_data = not_detected_objects(df_generated_data)

        df_generated_data = df_generated_data.sample(frac=1)
    else:
        df_generated_data = None

    if write_data is True:
        write_data_2_file(
            df_generated_data=df_generated_data,
            df_data=df_data,
            save_path=save_path,
            number_of_sources=size,
            pickle_2=pickle_2,
            columns_riz=columns_flux_riz_band,
            only_balrog=only_balrog
        )


if __name__ == "__main__":

    # Set working path
    path = os.path.abspath(sys.path[0])

    # Path to the data
    path_data = path + r"/../Data/mcal_detect_df_survey_21558485.pkl"

    # Path to output folder
    s_path = f"{path}/../Data/balrog_training_data_"

    # Path to output folder
    s_path_plot = f"{path}/../Data/Plots"

    main(
        path_to_data=path_data,
        size=int(2500000),
        save_path=s_path,
        save_plot=False,
        save_path_plots=s_path_plot,
        plot_fit_parameter=True,
        plot_generated_i_band=True,
        plot_covariance=True,
        plot_generated_df=True,
        plot_generated_wf=True,
        plot_survey=True,
        write_data=True,
        lower_bound=18.5,
        upper_bound=23,
        bin_size=40,
        pickle_2=False,
        only_detected_objects=True,
        only_balrog=(True, 2500000),
        column_flux_i_band="BDF_FLUX_DERED_CALIB_I",
        columns_flux_riz_band=[
            "BDF_FLUX_DERED_CALIB_R",
            "BDF_FLUX_DERED_CALIB_I",
            "BDF_FLUX_DERED_CALIB_Z",
            "BDF_FLUX_ERR_DERED_CALIB_R",
            "BDF_FLUX_ERR_DERED_CALIB_I",
            "BDF_FLUX_ERR_DERED_CALIB_Z"],
        columns_flux_ugjhks_band=[
            "BDF_FLUX_DERED_CALIB_U",
            "BDF_FLUX_DERED_CALIB_G",
            "BDF_FLUX_DERED_CALIB_J",
            "BDF_FLUX_DERED_CALIB_H",
            "BDF_FLUX_DERED_CALIB_KS"],
        columns_df_wf=[
            "BDF_FLUX_DERED_CALIB_R",
            "BDF_FLUX_DERED_CALIB_I",
            "BDF_FLUX_DERED_CALIB_Z",
            "BDF_FLUX_ERR_DERED_CALIB_R",
            "BDF_FLUX_ERR_DERED_CALIB_I",
            "BDF_FLUX_ERR_DERED_CALIB_Z",
            "unsheared/flux_err_r",
            "unsheared/flux_err_i",
            "unsheared/flux_err_z"],
        columns_survey_conditions=[
            f"unsheared/snr",
            f"unsheared/size_ratio",
            f"unsheared/flags",
            f"unsheared/T",
            "BDF_T",
            "BDF_G",
            "AIRMASS_WMEAN_R",
            "AIRMASS_WMEAN_I",
            "AIRMASS_WMEAN_Z",
            "FWHM_WMEAN_R",
            "FWHM_WMEAN_I",
            "FWHM_WMEAN_Z",
            "MAGLIM_R",
            "MAGLIM_I",
            "MAGLIM_Z",
            "EBV_SFD98",
            "true_detected"
        ]
    )
