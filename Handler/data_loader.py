import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from Handler.helper_functions import yj_transform_data
from Handler.plot_functions import plot_chain, plot_corner, plot_compare_corner
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
    dict_data = pickle.load(infile, encoding='latin1')

    # close file
    infile.close()

    data = dict_data["data frame complete data"]

    return dict_data, data


def load_data(
        cfg,
        luminosity_type,
        apply_object_cut,
        apply_flag_cut,
        apply_airmass_cut,
        apply_unsheared_mag_cut,
        apply_unsheared_shear_cut,
        writer,
        plot_data=False
):
    """"""

    if cfg['SIZE_TRAINING_DATA'] + cfg['SIZE_VALIDATION_DATA'] + cfg['SIZE_TEST_DATA'] > 1.0:
        raise f"{cfg['SIZE_TRAINING_DATA']}+{cfg['SIZE_VALIDATION_DATA']}+{cfg['SIZE_TEST_DATA']} > 1"

    # open file
    infile = open(cfg['PATH_DATA']+cfg['DATA_FILE_NAME'], 'rb')

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

    df_training_data, dict_pt = yj_transform_data(
        data_frame=df_training_data,
        columns=cfg['TRANSFORM_COLS'],
    )

    if cfg['REPRODUCIBLE'] is True:
        df_training_data = df_training_data.sample(frac=1, random_state=42)

    else:
        df_training_data = df_training_data.sample(frac=1)

    if cfg['PLOT_LOAD_DATA'] is True:
        img_grid = plot_corner(
            data_frame=df_training_data,
            labels=[
                f"{luminosity_type.lower()}_r",
                f"{luminosity_type.lower()}_i",
                f"{luminosity_type.lower()}_z",
                "snr",
                "size_ratio",
                "T"
            ],
            columns=[
                f"unsheared/{luminosity_type.lower()}_r",
                f"unsheared/{luminosity_type.lower()}_i",
                f"unsheared/{luminosity_type.lower()}_z",
                "unsheared/snr",
                "unsheared/size_ratio",
                "unsheared/T"
            ],
            ranges=[(16, 32), (16, 32), (16, 32), (-3, 3), (-3, 3), (-3, 3)],
            show_plot=cfg["SHOW_LOAD_DATA"]
        )
        writer.add_image("unsheared chain plot", img_grid)
        img_grid = plot_corner(
            data_frame=df_training_data,
            labels=[
                f"{luminosity_type.lower()}_err_r",
                f"{luminosity_type.lower()}_err_i",
                f"{luminosity_type.lower()}_err_z",
                "snr",
                "size_ratio",
                "T"
            ],
            columns=[
                f"unsheared/{luminosity_type.lower()}_err_r",
                f"unsheared/{luminosity_type.lower()}_err_i",
                f"unsheared/{luminosity_type.lower()}_err_z",
                "unsheared/snr",
                "unsheared/size_ratio",
                "unsheared/T"
            ],
            ranges=[(16, 32), (16, 32), (16, 32), (-3, 3), (-3, 3), (-3, 3)],
            show_plot=cfg["SHOW_LOAD_DATA"]
        )
        writer.add_image("unsheared error chain plot", img_grid)
        img_grid = plot_corner(
            data_frame=df_training_data,
            labels=[
                f"BDF {luminosity_type.upper()} U",
                f"BDF {luminosity_type.upper()} G",
                f"BDF {luminosity_type.upper()} R",
                f"BDF {luminosity_type.upper()} I",
                f"BDF {luminosity_type.upper()} Z",
                f"BDF {luminosity_type.upper()} J",
                f"BDF {luminosity_type.upper()} H",
                f"BDF {luminosity_type.upper()} K",
            ],
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
            # ranges=[(16, 32), (16, 32), (16, 32), (-3, 3), (-3, 3), (-3, 3)],
            show_plot=cfg["SHOW_LOAD_DATA"]
        )
        writer.add_image("BDF chain plot", img_grid)
        img_grid = plot_corner(
            data_frame=df_training_data,
            labels=[
                f"BDF {luminosity_type.upper()} ERR U",
                f"BDF {luminosity_type.upper()} ERR G",
                f"BDF {luminosity_type.upper()} ERR R",
                f"BDF {luminosity_type.upper()} ERR I",
                f"BDF {luminosity_type.upper()} ERR Z",
                f"BDF {luminosity_type.upper()} ERR J",
                f"BDF {luminosity_type.upper()} ERR H",
                f"BDF {luminosity_type.upper()} ERR K",
            ],
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
            # ranges=[(16, 32), (16, 32), (16, 32), (-3, 3), (-3, 3), (-3, 3)],
            show_plot=cfg["SHOW_LOAD_DATA"]
        )
        writer.add_image("BDF error chain plot", img_grid)
        img_grid = plot_corner(
            data_frame=df_training_data,
            labels=[
                f"{luminosity_type.upper()} U-G",
                f"{luminosity_type.upper()} G-R",
                f"{luminosity_type.upper()} R-I",
                f"{luminosity_type.upper()} I-Z",
                f"{luminosity_type.upper()} Z-J",
                f"{luminosity_type.upper()} J-H",
                f"{luminosity_type.upper()} H-K",
            ],
            columns=[
                f"Color BDF {luminosity_type.upper()} U-G",
                f"Color BDF {luminosity_type.upper()} G-R",
                f"Color BDF {luminosity_type.upper()} R-I",
                f"Color BDF {luminosity_type.upper()} I-Z",
                f"Color BDF {luminosity_type.upper()} Z-J",
                f"Color BDF {luminosity_type.upper()} J-H",
                f"Color BDF {luminosity_type.upper()} H-K",
            ],
            # ranges=[(16, 32), (16, 32), (16, 32), (-3, 3), (-3, 3), (-3, 3)],
            show_plot=cfg["SHOW_LOAD_DATA"]
        )
        writer.add_image("COLOR BDF chain plot", img_grid)
        img_grid = plot_corner(
            data_frame=df_training_data,
            labels=[
                "AIRMASS R",
                "AIRMASS I",
                "AIRMASS Z"
            ],
            columns=[
                "AIRMASS_WMEAN_R",
                "AIRMASS_WMEAN_I",
                "AIRMASS_WMEAN_Z"
            ],
            # ranges=[(16, 32), (16, 32), (16, 32), (-3, 3), (-3, 3), (-3, 3)],
            show_plot=cfg["SHOW_LOAD_DATA"]
        )
        writer.add_image("AIRMASS chain plot", img_grid)

        img_grid = plot_corner(
            data_frame=df_training_data,
            labels=[
                "FWHM R",
                "FWHM I",
                "FWHM Z"
            ],
            columns=[
                "FWHM_WMEAN_R",
                "FWHM_WMEAN_I",
                "FWHM_WMEAN_Z"
            ],
            # ranges=[(16, 32), (16, 32), (16, 32), (-3, 3), (-3, 3), (-3, 3)],
            show_plot=cfg["SHOW_LOAD_DATA"]
        )
        writer.add_image("FWHM chain plot", img_grid)

        img_grid = plot_corner(
            data_frame=df_training_data,
            labels=[
                "MAGLIM R",
                "MAGLIM I",
                "MAGLIM Z"
            ],
            columns=[
                "MAGLIM_R",
                "MAGLIM_I",
                "MAGLIM_Z"
            ],
            # ranges=[(16, 32), (16, 32), (16, 32), (-3, 3), (-3, 3), (-3, 3)],
            show_plot=cfg["SHOW_LOAD_DATA"]
        )
        writer.add_image("MAGLIM chain plot", img_grid)

        img_grid = plot_corner(
            data_frame=df_training_data,
            labels=[
                "BDF T",
                "BDF G",
                "EBV_SFD98"
            ],
            columns=[
                "BDF_T",
                "BDF_G",
                "EBV_SFD98"
            ],
            # ranges=[(16, 32), (16, 32), (16, 32), (-3, 3), (-3, 3), (-3, 3)],
            show_plot=cfg["SHOW_LOAD_DATA"]
        )
        writer.add_image("GALAXY chain plot", img_grid)
    exit()
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
    df_training_data = df_training_data.drop(columns=["FIELD"])
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
    dict_complete_data = {
        f"data frame complete data": pd.DataFrame(
            data=arr_data_scaled,
            columns=df_training_data.columns
        ),
        f"data frame complete data unscaled": pd.DataFrame(
            data=arr_data,
            columns=df_training_data.columns
        ),
        f"columns": df_training_data.columns,
        "scaler": scaler,
        "power transformer": dict_pt
    }

    
    with open(
            f"{path_output}/df_train_data_{len(dict_training_data[f'data frame training data'])}_run_{run}.pkl",
            "wb") as f:
        if cfg["PROTOCOL"] == 2:
            pickle.dump(dict_training_data, f, protocol=2)
        else:
            pickle.dump(dict_training_data, f)
    with open(
            f"{path_output}/df_validation_data_{len(dict_validation_data[f'data frame validation data'])}_run_{run}.pkl",
            "wb") as f:
        if cfg["PROTOCOL"] == 2:
            pickle.dump(dict_validation_data, f, protocol=2)
        else:
            pickle.dump(dict_validation_data, f)
    with open(f"{path_output}/df_test_data_{len(dict_test_data[f'data frame test data'])}_run_{run}.pkl",
              "wb") as f:
        if cfg["PROTOCOL"] == 2:
            pickle.dump(dict_test_data, f, protocol=2)
        else:
            pickle.dump(dict_test_data, f)
    with open(
            f"{path_output}/df_complete_data_{len(dict_complete_data[f'data frame complete data'])}.pkl",
            "wb") as f:
        if cfg["PROTOCOL"] == 2:
            pickle.dump(dict_complete_data, f, protocol=2)
        else:
            pickle.dump(dict_complete_data, f)

    return dict_training_data, dict_validation_data, dict_test_data


if __name__ == "__main__":
    path = os.path.abspath(sys.path[0])
    filepath = path + r"/../Data"
