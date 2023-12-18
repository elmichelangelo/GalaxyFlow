from Handler.helper_functions import flux2mag, plot_2d_kde
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib import colors
import os
import sys
import pandas as pd
import pickle as pkl
import h5py
import fitsio
import healpy as hp
from argparse import ArgumentParser
from astropy.table import Table, vstack, join
import seaborn as sns


def set_plot_settings():
    """"""
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['figure.figsize'] = (16.,8.)
    dpi = 150
    plt.rcParams.update({
    'lines.linewidth':1.0,
    'lines.linestyle':'-',
    'lines.color':'black',
    'font.family':'serif',
    'font.weight':'bold', #normal
    'font.size':16.0, #10.0
    'text.color':'black',
    'text.usetex':False,
    'axes.edgecolor':'black',
    'axes.linewidth':1.0,
    'axes.grid':False,
    'axes.titlesize':'x-large',
    'axes.labelsize':'x-large',
    'axes.labelweight':'bold', #normal
    'axes.labelcolor':'black',
    'axes.formatter.limits':[-4,4],
    'xtick.major.size':7,
    'xtick.minor.size':4,
    'xtick.major.pad':8,
    'xtick.minor.pad':8,
    'xtick.labelsize':'x-large',
    'xtick.minor.width':1.0,
    'xtick.major.width':1.0,
    'ytick.major.size':7,
    'ytick.minor.size':4,
    'ytick.major.pad':8,
    'ytick.minor.pad':8,
    'ytick.labelsize':'x-large',
    'ytick.minor.width':1.0,
    'ytick.major.width':1.0,
    'legend.numpoints':1,
    'legend.fontsize':'x-large',
    'legend.shadow':False,
    'legend.frameon':False})
    plt.style.use('seaborn')
    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("whitegrid")


def read_catalogs(path_metacal, path_detection, path_deep_field, path_master_cat, metacal_cols, detection_cols,
                  deep_field_cols):
    """"""
    metacal_data = h5py.File(path_metacal, 'r')

    df_mcal = pd.DataFrame()
    for i, col in enumerate(metacal_cols):
        print(i, col)
        df_mcal[col] = np.array(metacal_data['catalog/' + col]).byteswap().newbyteorder("<")
    df_mcal = df_mcal.rename(columns={'unsheared/bal_id': 'bal_id'})
    print('Length of mcal catalog: {}'.format(len(df_mcal)))

    detection_data = Table(fitsio.read(path_detection, columns=detection_cols).byteswap().newbyteorder())
    df_detect = pd.DataFrame()
    for i, col in enumerate(detection_cols):
        print(i, col)
        df_detect[col] = detection_data[col]
    print('Length of detection catalog: {}'.format(len(df_detect)))

    deep_field_data = pd.read_pickle(path_deep_field)
    df_deep_field = pd.DataFrame()
    for i, col in enumerate(deep_field_cols):
        print(i, col)
        df_deep_field[col] = deep_field_data[col]
    df_deep_field = df_deep_field.rename(columns={'ID': 'true_id'})
    print('Length of deep field catalog: {}'.format(len(df_deep_field)))

    """master_cat = h5py.File(path_master_cat)
    print('Length of master catalog: {}'.format(len(master_cat)))"""
    master_cat = None
    return df_mcal, df_detect, df_deep_field, master_cat


def merge_catalogs(metacal, detect, deep_field):
    """"""
    print('Merging catalogs...')
    df_merged = pd.merge(metacal, detect, on='bal_id')
    print('Length of merged  metacal+detection catalog: {}'.format(len(df_merged)))
    print(len(df_merged))
    exit()

    df_merged = pd.merge(deep_field, df_merged, on='true_id')
    print('Length of merged (metacal+detection)+deep field catalog: {}'.format(len(df_merged)))

    return df_merged


def cut_data(merged, master, cols_of_interrest):
    """"""

    print("define mask")
    theta = (np.pi / 180.) * (90. - merged['unsheared/dec'].to_numpy())
    phi = (np.pi / 180.) * merged['unsheared/ra'].to_numpy()
    gpix = hp.ang2pix(16384, theta, phi, nest=True)
    mask_cut = np.in1d(gpix // (hp.nside2npix(16384) // hp.nside2npix(4096)), master['index/mask/hpix'][:],
                       assume_unique=False)
    npass = np.sum(mask_cut)
    print('pass: ', npass)
    print('fail: ', len(mask_cut) - npass)

    print('define shape cuts')
    unsheared_snr_cut = (merged['unsheared/snr'] > 10) & (merged['unsheared/snr'] < 1000)
    unsheared_size_ratio_cut = merged['unsheared/size_ratio'] > 0.5
    unsheared_flags_cut = merged['unsheared/flags'] == 0
    unsheared_size_cut = (merged['unsheared/T'] < 10)
    unsheared_shape_cuts = unsheared_snr_cut & unsheared_size_ratio_cut & unsheared_flags_cut & unsheared_size_cut

    print('define flag cuts')
    flags_foreground_cut = merged['flags_foreground'] == 0
    flags_badregions_cut = merged['flags_badregions'] < 2
    flags_gold_cut = merged['meas_FLAGS_GOLD_SOF_ONLY'] < 2
    flags_footprint_cut = merged['flags_footprint'] == 1
    gold_flags_cut = flags_foreground_cut & flags_badregions_cut & flags_gold_cut & flags_footprint_cut

    print("perform shape and flag cuts and masking")
    merged = merged[gold_flags_cut & unsheared_shape_cuts & mask_cut]
    print('len w/ flags, shape and mask cut', len(merged))

    print("define binaries cut")
    highe_cut = np.greater(np.sqrt(np.power(merged['unsheared/e_1'], 2.) + np.power(merged['unsheared/e_2'], 2)), 0.8)
    c = 22.5
    m = 3.5
    magT_cut = np.log10(merged['unsheared/T']) < (c - flux2mag(merged['unsheared/flux_r'])) / m
    binaries = highe_cut * magT_cut

    print("perform binaries cut")
    merged = merged[~binaries]
    print('len w/ binaries', len(merged))

    print('define additional cuts')
    unsheared_imag_cut = (flux2mag(merged['unsheared/flux_i']) > 18) & (flux2mag(merged['unsheared/flux_i']) < 24)
    unsheared_rmag_cut = (flux2mag(merged['unsheared/flux_r']) > 15) & (flux2mag(merged['unsheared/flux_r']) < 26)
    unsheared_zmag_cut = (flux2mag(merged['unsheared/flux_z']) > 15) & (flux2mag(merged['unsheared/flux_z']) < 26)
    unsheared_zi_cut = ((flux2mag(merged['unsheared/flux_z']) - flux2mag(merged['unsheared/flux_i'])) < 1.5) & \
                       ((flux2mag(merged['unsheared/flux_z']) - flux2mag(merged['unsheared/flux_i'])) > -4)
    unsheared_ri_cut = ((flux2mag(merged['unsheared/flux_r']) - flux2mag(merged['unsheared/flux_i'])) < 4) & \
                       ((flux2mag(merged['unsheared/flux_r']) - flux2mag(merged['unsheared/flux_i'])) > -1.5)
    merged = merged[unsheared_imag_cut & unsheared_rmag_cut & unsheared_zmag_cut & unsheared_zi_cut & unsheared_ri_cut]

    unsheared_new_cut = (merged['unsheared/snr'] < 30) & (merged['unsheared/T'] > 2)
    merged = merged[~unsheared_new_cut]

    merged = merged[merged["match_flag_1.5_asec"] < 2]

    print('len w/ additional cuts', len(merged))

    df_final = pd.DataFrame()
    for i, col in enumerate(cols_of_interrest):
        print(i, col)
        df_final[col] = merged[col]
    print('Length of merged catalog: {}'.format(len(df_final)))

    return df_final


def plot_data(df_final):
    """"""
    # Create bins for fit range
    bins = np.linspace(18, 24, 40)

    # Calculate bin width for hist plot
    bin_width = bins[1] - bins[0]

    sns.histplot(
        x=df_final["BDF_MAG_DERED_CALIB_I"],
        element="step",
        fill=False,
        color="darkgreen",
        binwidth=bin_width,
        log_scale=(False, True),
        stat="probability",
        label=f"Generated distribution"
    )
    plt.xlim((18, 24))
    plt.title(f"Generated distribution")
    plt.xlabel("magnitude i")
    plt.legend()
    plt.show()

    sns.histplot(
        x=df_final["BDF_MAG_DERED_CALIB_R"],
        element="step",
        fill=False,
        color="darkgreen",
        binwidth=bin_width,
        log_scale=(False, True),
        stat="probability",
        label=f"Generated distribution"
    )
    plt.xlim((18, 24))
    plt.title(f"Generated distribution")
    plt.xlabel("magnitude r")
    plt.legend()
    plt.show()

    sns.histplot(
        x=df_final["BDF_MAG_DERED_CALIB_Z"],
        element="step",
        fill=False,
        color="darkgreen",
        binwidth=bin_width,
        log_scale=(False, True),
        stat="probability",
        label=f"Generated distribution"
    )
    plt.xlim((18, 24))
    plt.title(f"Generated distribution")
    plt.xlabel("magnitude y")
    plt.legend()
    plt.show()

    flux2mag(df_final['unsheared/flux_i'])

    sns.histplot(
        x=flux2mag(df_final['unsheared/flux_i']),
        element="step",
        fill=False,
        color="darkgreen",
        binwidth=bin_width,
        log_scale=(False, True),
        stat="probability",
        label=f"Generated distribution"
    )
    plt.xlim((18, 24))
    plt.title(f"Generated distribution")
    plt.xlabel("magnitude i")
    plt.legend()
    plt.show()

    sns.histplot(
        x=flux2mag(df_final['unsheared/flux_r']),
        element="step",
        fill=False,
        color="darkgreen",
        binwidth=bin_width,
        log_scale=(False, True),
        stat="probability",
        label=f"Generated distribution"
    )
    plt.xlim((18, 24))
    plt.title(f"Generated distribution")
    plt.xlabel("magnitude i")
    plt.legend()
    plt.show()

    sns.histplot(
        x=flux2mag(df_final['unsheared/flux_z']),
        element="step",
        fill=False,
        color="darkgreen",
        binwidth=bin_width,
        log_scale=(False, True),
        stat="probability",
        label=f"Generated distribution"
    )
    plt.xlim((18, 24))
    plt.title(f"Generated distribution")
    plt.xlabel("magnitude i")
    plt.legend()
    plt.show()

    arr_r_i_mag = np.array(df_final["BDF_MAG_DERED_CALIB_R"]) - np.array(df_final["BDF_MAG_DERED_CALIB_I"])
    arr_i_z_mag = np.array(df_final["BDF_MAG_DERED_CALIB_I"]) - np.array(df_final["BDF_MAG_DERED_CALIB_Z"])

    plot_2d_kde(
        x=arr_r_i_mag,
        y=arr_i_z_mag,
        limits=(-1.5, 1.5, -1.5, 1.5),
        x_label="r-i mag",
        y_label="i-z mag",
        color="Greens",
        title="color plot (kde) DES deep field deep field"
    )
    plt.show()

    arr_r_i_mag_metacal = np.array(flux2mag(df_final["unsheared/flux_r"])) - flux2mag(np.array(df_final["unsheared/flux_i"]))
    arr_i_z_mag_metacal = np.array(flux2mag(df_final["unsheared/flux_i"])) - flux2mag(np.array(df_final["unsheared/flux_z"]))

    plot_2d_kde(
        x=arr_r_i_mag_metacal,
        y=arr_i_z_mag_metacal,
        # limits=(-1.5, 1.5, -1.5, 1.5),
        x_label="r-i mag",
        y_label="i-z mag",
        color="Greens",
        title="color plot (kde) DES deep field metacal"
    )
    plt.show()


def write_cat(df_final, path_output_dir, file_name):
    """"""
    of_pkl = os.path.join(path_output_dir, file_name)

    print('write ' + of_pkl)
    df_final.to_pickle(of_pkl)


def main(path_metacal, path_detection, path_deep_field, path_master_cat, path_save_data, name_save_file, metacal_cols,
         detection_cols, deep_field_cols, cols_of_interrest):
    """"""
    set_plot_settings()
    df_mcal, df_detect, df_deep_field, mastercat = read_catalogs(
        path_metacal=path_metacal,
        path_detection=path_detection,
        path_deep_field=path_deep_field,
        path_master_cat=path_master_cat,
        metacal_cols=metacal_cols,
        detection_cols=detection_cols,
        deep_field_cols=deep_field_cols
    )

    df_merged = merge_catalogs(df_mcal, df_detect, df_deep_field)

    df_final = cut_data(df_merged, mastercat, cols_of_interrest)

    plot_data(df_final)

    # write_cat(df_final, path_save_data, name_save_file)


if __name__ == "__main__":
    # Set working path
    path_data_dir = os.path.abspath(sys.path[0])
    path_output_dir = f"{path_data_dir}/Output/"
    des_run = "y3-merged"
    des_version = 1.2

    other_metacal_cols = [
        'unsheared/ra',
        'unsheared/dec',
        'unsheared/coadd_object_id',
        'unsheared/snr',
        'unsheared/size_ratio',
        'unsheared/flags',
        'unsheared/bal_id',
        'unsheared/T',
        'unsheared/T_err',
        'unsheared/e_1',
        'unsheared/e_2',
        'unsheared/R11',
        'unsheared/R22',
        'unsheared/weight']

    main(
        path_metacal=f"{path_data_dir}/../../Data/balrog_mcal_stack-y3v02-0-riz-noNB-mcal_{des_run}_v{des_version}.h5",
        path_detection=f"{path_data_dir}/../../Data/balrog_detection_catalog_sof_{des_run}_v{1.2}.fits",
        path_deep_field=f"{path_data_dir}/../../Data/deep_ugriz.mof02_sn.jhk.ff04_c.jhk.ff02_052020_realerrors_May20calib.pkl",
        path_master_cat=f"{path_data_dir}/../../Data/Y3_mastercat_03_31_20.h5",
        path_save_data=path_output_dir,
        name_save_file=f"{des_run}_deep_field_metacal_cuts_v{des_version}.pkl",
        metacal_cols=other_metacal_cols + ['unsheared/flux_{}'.format(i) for i in 'irz'],
        detection_cols=[
            'bal_id',
            'true_id',
            'true_detected',
            'flags_footprint',
            'flags_foreground',
            'flags_badregions',
            'meas_FLAGS_GOLD_SOF_ONLY',
            'match_flag_1.5_asec'],
        deep_field_cols=["ID",
                         "BDF_FLUX_DERED_CALIB_I",
                         "BDF_FLUX_DERED_CALIB_R",
                         "BDF_FLUX_DERED_CALIB_Z",
                         "BDF_MAG_DERED_CALIB_I",
                         "BDF_MAG_DERED_CALIB_R",
                         "BDF_MAG_DERED_CALIB_Z"],
        cols_of_interrest=[
            'bal_id',
            'true_id',
            'true_detected',
            "BDF_FLUX_DERED_CALIB_I",
            "BDF_FLUX_DERED_CALIB_R",
            "BDF_FLUX_DERED_CALIB_Z",
            "BDF_MAG_DERED_CALIB_I",
            "BDF_MAG_DERED_CALIB_R",
            "BDF_MAG_DERED_CALIB_Z"] + ['unsheared/flux_{}'.format(i) for i in 'irz']
    )

