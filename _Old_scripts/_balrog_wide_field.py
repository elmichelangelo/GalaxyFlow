import pickle
import pandas as pd
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
from Handler.helper_functions import flux2mag
from chainconsumer import ChainConsumer
import seaborn as sns


def open_all_balrog_dataset(path_all_balrog_data):
    """"""
    infile = open(path_all_balrog_data, 'rb')
    # load pickle as pandas dataframe
    df_balrog = pd.DataFrame(pickle.load(infile, encoding='latin1'))
    # close file
    infile.close()
    return df_balrog


def rename_cols(data_frame):
    """"""
    data_frame["BDF_FLUX_DERED_CALIB_K"] = data_frame["BDF_FLUX_DERED_CALIB_KS"]
    data_frame["BDF_FLUX_ERR_DERED_CALIB_K"] = data_frame["BDF_FLUX_ERR_DERED_CALIB_KS"]
    data_frame["BDF_MAG_DERED_CALIB_U"] = flux2mag(data_frame["BDF_FLUX_DERED_CALIB_U"])
    data_frame["BDF_MAG_DERED_CALIB_G"] = flux2mag(data_frame["BDF_FLUX_DERED_CALIB_G"])
    data_frame["BDF_MAG_DERED_CALIB_R"] = flux2mag(data_frame["BDF_FLUX_DERED_CALIB_R"])
    data_frame["BDF_MAG_DERED_CALIB_I"] = flux2mag(data_frame["BDF_FLUX_DERED_CALIB_I"])
    data_frame["BDF_MAG_DERED_CALIB_Z"] = flux2mag(data_frame["BDF_FLUX_DERED_CALIB_Z"])
    data_frame["BDF_MAG_DERED_CALIB_J"] = flux2mag(data_frame["BDF_FLUX_DERED_CALIB_J"])
    data_frame["BDF_MAG_DERED_CALIB_H"] = flux2mag(data_frame["BDF_FLUX_DERED_CALIB_H"])
    data_frame["BDF_MAG_DERED_CALIB_KS"] = flux2mag(data_frame["BDF_FLUX_DERED_CALIB_KS"])
    data_frame["BDF_MAG_DERED_CALIB_K"] = flux2mag(data_frame["BDF_FLUX_DERED_CALIB_KS"])
    data_frame["BDF_MAG_ERR_DERED_CALIB_U"] = flux2mag(data_frame["BDF_FLUX_ERR_DERED_CALIB_U"])
    data_frame["BDF_MAG_ERR_DERED_CALIB_G"] = flux2mag(data_frame["BDF_FLUX_ERR_DERED_CALIB_G"])
    data_frame["BDF_MAG_ERR_DERED_CALIB_R"] = flux2mag(data_frame["BDF_FLUX_ERR_DERED_CALIB_R"])
    data_frame["BDF_MAG_ERR_DERED_CALIB_I"] = flux2mag(data_frame["BDF_FLUX_ERR_DERED_CALIB_I"])
    data_frame["BDF_MAG_ERR_DERED_CALIB_Z"] = flux2mag(data_frame["BDF_FLUX_ERR_DERED_CALIB_Z"])
    data_frame["BDF_MAG_ERR_DERED_CALIB_J"] = flux2mag(data_frame["BDF_FLUX_ERR_DERED_CALIB_J"])
    data_frame["BDF_MAG_ERR_DERED_CALIB_H"] = flux2mag(data_frame["BDF_FLUX_ERR_DERED_CALIB_H"])
    data_frame["BDF_MAG_ERR_DERED_CALIB_KS"] = flux2mag(data_frame["BDF_FLUX_ERR_DERED_CALIB_KS"])
    data_frame["BDF_MAG_ERR_DERED_CALIB_K"] = flux2mag(data_frame["BDF_FLUX_ERR_DERED_CALIB_KS"])

    data_frame["Color Mag U-G"] = data_frame["BDF_MAG_DERED_CALIB_U"] - data_frame["BDF_MAG_DERED_CALIB_G"]
    data_frame["Color Mag G-R"] = data_frame["BDF_MAG_DERED_CALIB_G"] - data_frame["BDF_MAG_DERED_CALIB_R"]
    data_frame["Color Mag R-I"] = data_frame["BDF_MAG_DERED_CALIB_R"] - data_frame["BDF_MAG_DERED_CALIB_I"]
    data_frame["Color Mag I-Z"] = data_frame["BDF_MAG_DERED_CALIB_I"] - data_frame["BDF_MAG_DERED_CALIB_Z"]
    data_frame["Color Mag Z-J"] = data_frame["BDF_MAG_DERED_CALIB_Z"] - data_frame["BDF_MAG_DERED_CALIB_J"]
    data_frame["Color Mag J-H"] = data_frame["BDF_MAG_DERED_CALIB_J"] - data_frame["BDF_MAG_DERED_CALIB_H"]
    data_frame["Color Mag H-K"] = data_frame["BDF_MAG_DERED_CALIB_H"] - data_frame["BDF_MAG_DERED_CALIB_K"]
    data_frame["BDF_G"] = np.sqrt(data_frame["BDF_G_0"]**2 + data_frame["BDF_G_1"]**2)

    data_frame["unsheared/mag_r"] = flux2mag(data_frame["unsheared/flux_r"])
    data_frame["unsheared/mag_i"] = flux2mag(data_frame["unsheared/flux_i"])
    data_frame["unsheared/mag_z"] = flux2mag(data_frame["unsheared/flux_z"])
    data_frame["unsheared/mag_err_r"] = flux2mag(data_frame["unsheared/flux_err_r"])
    data_frame["unsheared/mag_err_i"] = flux2mag(data_frame["unsheared/flux_err_i"])
    data_frame["unsheared/mag_err_z"] = flux2mag(data_frame["unsheared/flux_err_z"])
    return data_frame


def save_balrog_subset(data_frame, path_balrog_subset, protocol):
    """"""
    if protocol == 2:
        with open(path_balrog_subset, "wb") as f:
            pickle.dump(data_frame.to_dict(), f, protocol=2)
    else:
        data_frame.to_pickle(path_balrog_subset)


def unsheared_mag_cut(data_frame):
    """"""
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
    return data_frame


def plot_chain(data_frame):
    """"""
    df_generated_measured = pd.DataFrame({
        "unsheared/mag_r": np.array(data_frame["unsheared/mag_r"]),
        "unsheared/mag_i": np.array(data_frame["unsheared/mag_i"]),
        "unsheared/mag_z": np.array(data_frame["unsheared/mag_z"]),
        "unsheared/snr": np.array(data_frame["unsheared/snr"]),
        "unsheared/size_ratio": np.array(data_frame["unsheared/size_ratio"]),
        "unsheared/T": np.array(data_frame["unsheared/T"])
    })
    arr_generated = df_generated_measured.to_numpy()
    parameter = [
        "mag r",
        "mag i",
        "mag z",
        "snr",              # signal-noise      Range: min=0.3795, max=38924.4662
        "size ratio",       # T/psfrec_T        Range: min=-0.8636, max=4346136.5645
        "T"                 # T=<x^2>+<y^2>     Range: min=-0.6693, max=1430981.5103
    ]
    chainchat = ChainConsumer()
    chainchat.add_chain(arr_generated, parameters=parameter, name="generated observed properties: chat*")
    # chainchat.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
    chainchat.plotter.plot(
        figsize="page",
        # extents={
        #     "mag r": (17.5, 26),
        #     "mag i": (17.5, 26),
        #     "mag z": (17.5, 26),
        #     "snr": (-11, 55),
        #     "size ratio": (-1.5, 4),
        #     "T": (-1, 2.5)
        # }
    )
    plt.show()
    plt.clf()


def plot_histo(data_frame):
    sns.histplot(data_frame["unsheared/snr"], stat="density", bins=1000, kde=True)
    plt.show()

    sns.histplot(data_frame["unsheared/size_ratio"], stat="density", bins=1000, kde=True)
    plt.show()

    sns.histplot(data_frame["unsheared/T"], stat="density", bins=1000, kde=True)
    plt.show()


def create_balrog_subset(path_all_balrog_data, path_save, name_save_file, number_of_samples, only_detected, protocol=None):
    """"""
    df_balrog = open_all_balrog_dataset(path_all_balrog_data)

    if only_detected is True:
        df_balrog = df_balrog[df_balrog["true_detected"] == 1]
        print(f"length of only true_detected balrog objects {len(df_balrog)}")
    df_balrog = rename_cols(data_frame=df_balrog)
    df_balrog = unsheared_mag_cut(data_frame=df_balrog)
    print(f"length of catalog after applying cuts {len(df_balrog)}")
    if number_of_samples is None:
        number_of_samples = len(df_balrog)
    df_balrog_subset = df_balrog.sample(number_of_samples, ignore_index=True, replace=False)

    plot_chain(data_frame=df_balrog_subset)
    # plot_histo(df_balrog=df_balrog_subset)

    path_balrog_subset = f"{path_save}/{name_save_file}_{number_of_samples}.pkl"

    save_balrog_subset(
        data_frame=df_balrog_subset,
        path_balrog_subset=path_balrog_subset,
        protocol=protocol
    )


if __name__ == '__main__':
    path = os.path.abspath(sys.path[0])
    no_samples = int(250000)  # int(250000) # 20000 # int(8E6)
    create_balrog_subset(
        path_all_balrog_data=f"{path}/../../Data/balrog_catalog_mcal_detect_deepfield_21558485.pkl",
        path_save = f"{path}/../../Data",
        name_save_file="balrog_subset",
        number_of_samples=no_samples,
        only_detected=True,
        protocol=2
    )