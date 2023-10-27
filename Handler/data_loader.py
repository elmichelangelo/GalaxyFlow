import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from Handler.helper_functions import yj_transform_data
from Handler.plot_functions import plot_corner
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
        writer
):
    """"""

    if cfg['SIZE_TRAINING_DATA'] + cfg['SIZE_VALIDATION_DATA'] + cfg['SIZE_TEST_DATA'] > 1.0:
        raise f"{cfg['SIZE_TRAINING_DATA']}+{cfg['SIZE_VALIDATION_DATA']}+{cfg['SIZE_TEST_DATA']} > 1"

    # open file
    infile = open(f"{cfg['PATH_DATA']}/{cfg['DATA_FILE_NAME']}", 'rb')

    # load pickle as pandas dataframe
    df_training_data = pd.DataFrame(pickle.load(infile, encoding='latin1'))

    # close file
    infile.close()

    if cfg['APPLY_OBJECT_CUT'] is True:
        df_training_data = unsheared_object_cuts(df_training_data)

    if cfg['APPLY_FLAG_CUT'] is True:
        df_training_data = flag_cuts(df_training_data)

    if cfg['APPLY_UNSHEARED_MAG_CUT'] is True:
        df_training_data = unsheared_mag_cut(df_training_data)

    if cfg['APPLY_UNSHEARED_SHEAR_CUT'] is True:
        df_training_data = unsheared_shear_cuts(df_training_data)

    if cfg['APPLY_AIRMASS_CUT'] is True:
        df_training_data = airmass_cut(df_training_data)

    dict_pt = None

    if cfg['APPLY_YJ_TRANSFORM'] is True:
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
                f"{cfg['LUM_TYPE'].lower()}_r",
                f"{cfg['LUM_TYPE'].lower()}_i",
                f"{cfg['LUM_TYPE'].lower()}_z",
                "snr",
                "size_ratio",
                "T"
            ],
            columns=[
                f"unsheared/{cfg['LUM_TYPE'].lower()}_r",
                f"unsheared/{cfg['LUM_TYPE'].lower()}_i",
                f"unsheared/{cfg['LUM_TYPE'].lower()}_z",
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
                f"{cfg['LUM_TYPE'].lower()}_err_r",
                f"{cfg['LUM_TYPE'].lower()}_err_i",
                f"{cfg['LUM_TYPE'].lower()}_err_z",
                "snr",
                "size_ratio",
                "T"
            ],
            columns=[
                f"unsheared/{cfg['LUM_TYPE'].lower()}_err_r",
                f"unsheared/{cfg['LUM_TYPE'].lower()}_err_i",
                f"unsheared/{cfg['LUM_TYPE'].lower()}_err_z",
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
                f"BDF {cfg['LUM_TYPE'].upper()} U",
                f"BDF {cfg['LUM_TYPE'].upper()} G",
                f"BDF {cfg['LUM_TYPE'].upper()} R",
                f"BDF {cfg['LUM_TYPE'].upper()} I",
                f"BDF {cfg['LUM_TYPE'].upper()} Z",
                f"BDF {cfg['LUM_TYPE'].upper()} J",
                f"BDF {cfg['LUM_TYPE'].upper()} H",
                f"BDF {cfg['LUM_TYPE'].upper()} K",
            ],
            columns=[
                f"BDF_{cfg['LUM_TYPE'].upper()}_DERED_CALIB_U",
                f"BDF_{cfg['LUM_TYPE'].upper()}_DERED_CALIB_G",
                f"BDF_{cfg['LUM_TYPE'].upper()}_DERED_CALIB_R",
                f"BDF_{cfg['LUM_TYPE'].upper()}_DERED_CALIB_I",
                f"BDF_{cfg['LUM_TYPE'].upper()}_DERED_CALIB_Z",
                f"BDF_{cfg['LUM_TYPE'].upper()}_DERED_CALIB_J",
                f"BDF_{cfg['LUM_TYPE'].upper()}_DERED_CALIB_H",
                f"BDF_{cfg['LUM_TYPE'].upper()}_DERED_CALIB_K",
            ],
            # ranges=[(16, 32), (16, 32), (16, 32), (-3, 3), (-3, 3), (-3, 3)],
            show_plot=cfg["SHOW_LOAD_DATA"]
        )
        writer.add_image("BDF chain plot", img_grid)
        img_grid = plot_corner(
            data_frame=df_training_data,
            labels=[
                f"BDF {cfg['LUM_TYPE'].upper()} ERR U",
                f"BDF {cfg['LUM_TYPE'].upper()} ERR G",
                f"BDF {cfg['LUM_TYPE'].upper()} ERR R",
                f"BDF {cfg['LUM_TYPE'].upper()} ERR I",
                f"BDF {cfg['LUM_TYPE'].upper()} ERR Z",
                f"BDF {cfg['LUM_TYPE'].upper()} ERR J",
                f"BDF {cfg['LUM_TYPE'].upper()} ERR H",
                f"BDF {cfg['LUM_TYPE'].upper()} ERR K",
            ],
            columns=[
                f"BDF_{cfg['LUM_TYPE'].upper()}_ERR_DERED_CALIB_U",
                f"BDF_{cfg['LUM_TYPE'].upper()}_ERR_DERED_CALIB_G",
                f"BDF_{cfg['LUM_TYPE'].upper()}_ERR_DERED_CALIB_R",
                f"BDF_{cfg['LUM_TYPE'].upper()}_ERR_DERED_CALIB_I",
                f"BDF_{cfg['LUM_TYPE'].upper()}_ERR_DERED_CALIB_Z",
                f"BDF_{cfg['LUM_TYPE'].upper()}_ERR_DERED_CALIB_J",
                f"BDF_{cfg['LUM_TYPE'].upper()}_ERR_DERED_CALIB_H",
                f"BDF_{cfg['LUM_TYPE'].upper()}_ERR_DERED_CALIB_K",
            ],
            # ranges=[(16, 32), (16, 32), (16, 32), (-3, 3), (-3, 3), (-3, 3)],
            show_plot=cfg["SHOW_LOAD_DATA"]
        )
        writer.add_image("BDF error chain plot", img_grid)
        img_grid = plot_corner(
            data_frame=df_training_data,
            labels=[
                f"{cfg['LUM_TYPE'].upper()} U-G",
                f"{cfg['LUM_TYPE'].upper()} G-R",
                f"{cfg['LUM_TYPE'].upper()} R-I",
                f"{cfg['LUM_TYPE'].upper()} I-Z",
                f"{cfg['LUM_TYPE'].upper()} Z-J",
                f"{cfg['LUM_TYPE'].upper()} J-H",
                f"{cfg['LUM_TYPE'].upper()} H-K",
            ],
            columns=[
                f"Color BDF {cfg['LUM_TYPE'].upper()} U-G",
                f"Color BDF {cfg['LUM_TYPE'].upper()} G-R",
                f"Color BDF {cfg['LUM_TYPE'].upper()} R-I",
                f"Color BDF {cfg['LUM_TYPE'].upper()} I-Z",
                f"Color BDF {cfg['LUM_TYPE'].upper()} Z-J",
                f"Color BDF {cfg['LUM_TYPE'].upper()} J-H",
                f"Color BDF {cfg['LUM_TYPE'].upper()} H-K",
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
    scaler = None
    if cfg['SCALER'] == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif cfg['SCALER'] == "MaxAbsScaler":
        scaler = MaxAbsScaler()
    elif cfg['SCALER'] == "StandardScaler":
        scaler = StandardScaler()
    elif cfg['SCALER'] is None:
        pass
    else:
        raise TypeError(f"{cfg['SCALER']} is no valid scaler")
    df_training_data = df_training_data.drop(columns=["FIELD"])
    if scaler is not None:
        scaler.fit(df_training_data)
        scaled = scaler.transform(df_training_data)
        df_training_data_scaled = pd.DataFrame(scaled, columns=df_training_data.columns)
    else:
        df_training_data_scaled = df_training_data

    arr_data = np.array(df_training_data)
    arr_data_scaled = np.array(df_training_data_scaled)

    train_end = int(len(arr_data_scaled) * cfg['SIZE_TRAINING_DATA'])

    dict_training_data = {
        f"data frame training data": pd.DataFrame(data=arr_data_scaled[:train_end], columns=df_training_data.columns),
        f"columns": df_training_data.columns,
        "scaler": scaler
    }

    val_start = train_end
    val_end = train_end + int(len(arr_data_scaled) * cfg['SIZE_VALIDATION_DATA'])

    dict_validation_data = {
        f"data frame validation data": pd.DataFrame(
            data=arr_data_scaled[val_start:val_end],
            columns=df_training_data.columns
        ),
        f"columns": df_training_data.columns,
        "scaler": scaler
    }

    test_start = val_end
    test_end = val_end + int(len(arr_data_scaled) * cfg['SIZE_TEST_DATA'])

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
    save_training_catalogs(
        dict_data=dict_training_data,
        cfg=cfg,
        save_name=f"df_train_data_{len(dict_training_data[f'data frame training data'])}_{cfg['RUN_DATE']}.pkl"
    )
    save_training_catalogs(
        dict_data=dict_validation_data,
        cfg=cfg,
        save_name=f"df_validation_data_{len(dict_validation_data[f'data frame validation data'])}_{cfg['RUN_DATE']}.pkl"
    )
    save_training_catalogs(
        dict_data=dict_test_data,
        cfg=cfg,
        save_name=f"df_test_data_{len(dict_test_data[f'data frame test data'])}_{cfg['RUN_DATE']}.pkl"
    )
    save_training_catalogs(
        dict_data=dict_complete_data,
        cfg=cfg,
        save_name=f"df_complete_data_{len(dict_complete_data[f'data frame complete data'])}_{cfg['RUN_DATE']}.pkl"
    )

    return dict_training_data, dict_validation_data, dict_test_data


def save_training_catalogs(dict_data, cfg, save_name):
    """"""
    with open(f"{cfg['PATH_OUTPUT_SUBFOLDER_CATALOGS']}/{save_name}", "wb") as f:
        if cfg["PROTOCOL"] == 2:
            pickle.dump(dict_data, f, protocol=2)
        else:
            pickle.dump(dict_data, f)


if __name__ == "__main__":
    path = os.path.abspath(sys.path[0])
    filepath = path + r"/../Data"
