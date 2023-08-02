import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from Handler.helper_functions import replace_and_transform_data
from Handler.plot_functions import plot_chain
from Handler.cut_functions import *
import numpy as np
import pandas as pd
import pickle
import os
import sys

sys.path.append(os.path.dirname(__file__))


def load_test_data(path_test_data):
    """"""
    # open file
    infile = open(path_test_data, 'rb')  # filename

    # load pickle as pandas dataframe
    data = pickle.load(infile, encoding='latin1')

    # close file
    infile.close()
    return data


def load_data(
        path_training_data,
        path_output,
        luminosity_type,
        size_training_dataset,
        size_validation_dataset,
        size_test_dataset,
        apply_object_cut,
        apply_flag_cut,
        apply_airmass_cut,
        apply_unsheared_mag_cut,
        apply_unsheared_shear_cut,
        selected_scaler,
        lst_replace_transform_cols=None,
        lst_replace_values=None,
        reproducible=True,
        run=None
):
    """"""

    if size_training_dataset + size_validation_dataset + size_test_dataset > 1.0:
        raise f"{size_training_dataset}+{size_validation_dataset}+{size_test_dataset} > 1"

    # open file
    infile = open(path_training_data, 'rb')  # filename

    # load pickle as pandas dataframe
    df_training_data = pd.DataFrame(pickle.load(infile, encoding='latin1'))

    # close file
    infile.close()

    if apply_object_cut is True:
        df_training_data = unsheared_object_cuts(df_training_data)

    if apply_flag_cut is True:
        df_training_data = flag_cuts(df_training_data)

    if apply_unsheared_mag_cut is True:
        df_training_data = unsheared_mag_cut(df_training_data)

    if apply_unsheared_shear_cut is True:
        df_training_data = unsheared_shear_cuts(df_training_data)

    if apply_airmass_cut is True:
        df_training_data = airmass_cut(df_training_data)

    df_training_data, dict_pt = replace_and_transform_data(
        data_frame=df_training_data,
        columns=lst_replace_transform_cols,
        replace_value=lst_replace_values
    )

    if reproducible is True:
        df_training_data = df_training_data.sample(frac=1, random_state=42)

    else:
        df_training_data = df_training_data.sample(frac=1)

    plot_chain(
        data_frame=df_training_data,
        plot_name=f"unsheared_{luminosity_type.lower()}",
        columns=[
            f"unsheared/{luminosity_type.lower()}_r",
            f"unsheared/{luminosity_type.lower()}_i",
            f"unsheared/{luminosity_type.lower()}_z",
            "unsheared/snr",
            "unsheared/size_ratio",
            "unsheared/T"
        ],
        parameter=[
                f"{luminosity_type.lower()} r",
                f"{luminosity_type.lower()} i",
                f"{luminosity_type.lower()} z",
                "snr",              # signal-noise      Range: min=0.3795, max=38924.4662
                "size ratio",       # T/psfrec_T        Range: min=-0.8636, max=4346136.5645
                "T"                 # T=<x^2>+<y^2>     Range: min=-0.6693, max=1430981.5103
            ],
        extends=None
    )

    plot_chain(
        data_frame=df_training_data,
        plot_name=f"unsheared_{luminosity_type.lower()}",
        columns=[
            f"unsheared/{luminosity_type.lower()}_err_r",
            f"unsheared/{luminosity_type.lower()}_err_i",
            f"unsheared/{luminosity_type.lower()}_err_z",
            "unsheared/snr",
            "unsheared/size_ratio",
            "unsheared/T"
        ],
        parameter=[
            f"{luminosity_type.lower()} err r",
            f"{luminosity_type.lower()} err i",
            f"{luminosity_type.lower()} err z",
            "snr",  # signal-noise      Range: min=0.3795, max=38924.4662
            "size ratio",  # T/psfrec_T        Range: min=-0.8636, max=4346136.5645
            "T"  # T=<x^2>+<y^2>     Range: min=-0.6693, max=1430981.5103
        ],
        extends=None
    )

    plot_chain(
        data_frame=df_training_data,
        plot_name=f"BDF_{luminosity_type.upper()}",
        columns=[
            f"BDF_{luminosity_type.upper()}_DERED_CALIB_U",
            f"BDF_{luminosity_type.upper()}_DERED_CALIB_G",
            f"BDF_{luminosity_type.upper()}_DERED_CALIB_R",
            f"BDF_{luminosity_type.upper()}_DERED_CALIB_I",
            f"BDF_{luminosity_type.upper()}_DERED_CALIB_Z",
            f"BDF_{luminosity_type.upper()}_DERED_CALIB_J",
            f"BDF_{luminosity_type.upper()}_DERED_CALIB_H",
            f"BDF_{luminosity_type.upper()}_DERED_CALIB_K",
        ],
        parameter=[
            f"BDF {luminosity_type.upper()} U",
            f"BDF {luminosity_type.upper()} G",
            f"BDF {luminosity_type.upper()} R",
            f"BDF {luminosity_type.upper()} I",
            f"BDF {luminosity_type.upper()} Z",
            f"BDF {luminosity_type.upper()} J",
            f"BDF {luminosity_type.upper()} H",
            f"BDF {luminosity_type.upper()} K",
        ],
        extends=None
    )

    plot_chain(
        data_frame=df_training_data,
        plot_name=f"BDF_{luminosity_type.upper()}",
        columns=[
            f"BDF_{luminosity_type.upper()}_ERR_DERED_CALIB_U",
            f"BDF_{luminosity_type.upper()}_ERR_DERED_CALIB_G",
            f"BDF_{luminosity_type.upper()}_ERR_DERED_CALIB_R",
            f"BDF_{luminosity_type.upper()}_ERR_DERED_CALIB_I",
            f"BDF_{luminosity_type.upper()}_ERR_DERED_CALIB_Z",
            f"BDF_{luminosity_type.upper()}_ERR_DERED_CALIB_J",
            f"BDF_{luminosity_type.upper()}_ERR_DERED_CALIB_H",
            f"BDF_{luminosity_type.upper()}_ERR_DERED_CALIB_K",
        ],
        parameter=[
            f"BDF {luminosity_type.upper()} ERR U",
            f"BDF {luminosity_type.upper()} ERR G",
            f"BDF {luminosity_type.upper()} ERR R",
            f"BDF {luminosity_type.upper()} ERR I",
            f"BDF {luminosity_type.upper()} ERR Z",
            f"BDF {luminosity_type.upper()} ERR J",
            f"BDF {luminosity_type.upper()} ERR H",
            f"BDF {luminosity_type.upper()} ERR K",
        ],
        extends=None
    )

    plot_chain(
        data_frame=df_training_data,
        plot_name=f"BDF_COLOR_{luminosity_type.upper()}",
        columns=[
            f"Color BDF {luminosity_type.upper()} U-G",
            f"Color BDF {luminosity_type.upper()} G-R",
            f"Color BDF {luminosity_type.upper()} R-I",
            f"Color BDF {luminosity_type.upper()} I-Z",
            f"Color BDF {luminosity_type.upper()} Z-J",
            f"Color BDF {luminosity_type.upper()} J-H",
            f"Color BDF {luminosity_type.upper()} H-K",
        ],
        parameter=[
            f"{luminosity_type.upper()} U-G",
            f"{luminosity_type.upper()} G-R",
            f"{luminosity_type.upper()} R-I",
            f"{luminosity_type.upper()} I-Z",
            f"{luminosity_type.upper()} Z-J",
            f"{luminosity_type.upper()} J-H",
            f"{luminosity_type.upper()} H-K",
        ],
        extends=None
    )

    plot_chain(
        data_frame=df_training_data,
        plot_name="AIRMASS",
        columns=[
            "AIRMASS_WMEAN_R",
            "AIRMASS_WMEAN_I",
            "AIRMASS_WMEAN_Z"
        ],
        parameter=[
            "AIRMASS R",
            "AIRMASS I",
            "AIRMASS Z"
        ],
        extends=None
    )

    plot_chain(
        data_frame=df_training_data,
        plot_name="FWHM",
        columns=[
            "FWHM_WMEAN_R",
            "FWHM_WMEAN_I",
            "FWHM_WMEAN_Z"
        ],
        parameter=[
            "FWHM R",
            "FWHM I",
            "FWHM Z"
        ],
        extends=None
    )

    plot_chain(
        data_frame=df_training_data,
        plot_name="MAGLIM",
        columns=[
            "MAGLIM_R",
            "MAGLIM_I",
            "MAGLIM_Z"
        ],
        parameter=[
            "MAGLIM R",
            "MAGLIM I",
            "MAGLIM Z"
        ],
        extends=None
    )

    plot_chain(
        data_frame=df_training_data,
        plot_name="OBS",
        columns=[
            "BDF_T",
            "BDF_G",
            "EBV_SFD98"
        ],
        parameter=[
            "BDF T",
            "BDF G",
            "EBV_SFD98"
        ],
        extends=None
    )

    scaler = None
    if selected_scaler == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif selected_scaler == "MaxAbsScaler":
        scaler = MaxAbsScaler()
    elif selected_scaler == "StandardScaler":
        scaler = StandardScaler()
    elif selected_scaler is None:
        pass
    else:
        raise TypeError(f"{selected_scaler} is no valid scaler")

    if scaler is not None:
        scaler.fit(df_training_data)
        scaled = scaler.transform(df_training_data)
        df_training_data_scaled = pd.DataFrame(scaled, columns=df_training_data.columns)
    else:
        df_training_data_scaled = df_training_data

    arr_data = np.array(df_training_data)
    arr_data_scaled = np.array(df_training_data_scaled)

    train_end = int(len(arr_data_scaled) * size_training_dataset)

    dict_training_data = {
        f"data frame training data": pd.DataFrame(data=arr_data_scaled[:train_end], columns=df_training_data.columns),
        f"columns": df_training_data.columns,
        "scaler": scaler
    }

    val_start = train_end
    val_end = train_end + int(len(arr_data_scaled) * size_validation_dataset)

    dict_validation_data = {
        f"data frame validation data": pd.DataFrame(
            data=arr_data_scaled[val_start:val_end],
            columns=df_training_data.columns
        ),
        f"columns": df_training_data.columns,
        "scaler": scaler
    }

    test_start = val_end
    test_end = val_end + int(len(arr_data_scaled) * size_test_dataset)

    dict_test_data = {
        f"data frame test data": pd.DataFrame(
            data=arr_data_scaled[test_start:test_end],
            columns=df_training_data.columns
        ),
        f"data frame test data unscaled": pd.DataFrame(
            data=arr_data[test_start:test_end],
            columns=df_training_data.columns
        ),
        f"columns": df_training_data.columns,
        "scaler": scaler,
        "power transformer": dict_pt
    }

    with open(
            f"{path_output}/df_train_data_{len(dict_training_data[f'data frame training data'])}_run_{run}.pkl",
            "wb") as f:
        pickle.dump(dict_training_data, f, protocol=2)
    with open(
            f"{path_output}/df_validation_data_{len(dict_validation_data[f'data frame validation data'])}_run_{run}.pkl",
            "wb") as f:
        pickle.dump(dict_validation_data, f, protocol=2)
    with open(f"{path_output}/df_test_data_{len(dict_test_data[f'data frame test data'])}_run_{run}.pkl",
              "wb") as f:
        pickle.dump(dict_test_data, f, protocol=2)

    return dict_training_data, dict_validation_data, dict_test_data


if __name__ == "__main__":
    path = os.path.abspath(sys.path[0])
    filepath = path + r"/../Data"
