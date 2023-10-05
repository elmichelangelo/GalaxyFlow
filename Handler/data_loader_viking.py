import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from Handler.helper_functions import yj_transform_data
from Handler.plot_functions import plot_chain
from Handler.cut_functions import *
from astropy.table import Table
import fitsio
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
        writer,
        lst_yj_transform_cols=None,
        reproducible=True,
        run=None,
        plot_data=False
):
    """"""

    if size_training_dataset + size_validation_dataset + size_test_dataset > 1.0:
        raise f"{size_training_dataset}+{size_validation_dataset}+{size_test_dataset} > 1"

    galaxy_columns = [
        "Re_input",
        "e1_input",
        "e2_input",
        "g1_in",
        "g2_in",
        "sersic_n_input",
        "u_input",
        "g_input",
        "r_input",
        "i_input",
        "Z_input",
        "Y_input",
        "J_input",
        "H_input",
        "Ks_input",
        "MAG_GAAP_u",
        "MAG_GAAP_g",
        "MAG_GAAP_r",
        "MAG_GAAP_i",
        "MAG_GAAP_Z",
        "MAG_GAAP_Y",
        "MAG_GAAP_J",
        "MAG_GAAP_H",
        "MAG_GAAP_Ks",
        "InputSeeing_u",
        "InputSeeing_g",
        "InputSeeing_r",
        "InputSeeing_i",
        "InputSeeing_Z",
        "InputSeeing_Y",
        "InputSeeing_J",
        "InputSeeing_H",
        "InputSeeing_Ks",
        "InputBeta_u",
        "InputBeta_g",
        "InputBeta_r",
        "InputBeta_i",
        "InputBeta_Z",
        "InputBeta_Y",
        "InputBeta_J",
        "InputBeta_H",
        "InputBeta_Ks"
    ]

    df_training_data = pd.read_pickle(path_training_data)[galaxy_columns]
    dict_pt = None
    # df_training_data, dict_pt = yj_transform_data(
    #     data_frame=df_training_data,
    #     columns=lst_yj_transform_cols
    # )

    if reproducible is True:
        df_training_data = df_training_data.sample(frac=1, random_state=42)

    else:
        df_training_data = df_training_data.sample(frac=1)

    plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))  # 20., 6.
    ax[0].hist(df_training_data['u_input'], 100, (16, 38), color='red', label='u', histtype='step')
    ax[0].hist(df_training_data['g_input'], 100, (16, 38), color='green', label='g', histtype='step')
    ax[0].hist(df_training_data['r_input'], 100, (16, 38), color='blue', label='r', histtype='step')
    ax[0].hist(df_training_data['i_input'], 100, (16, 38), color='grey', label='i', histtype='step')
    ax[0].hist(df_training_data['Z_input'], 100, (16, 38), color='purple', label='z', histtype='step')
    ax[0].hist(df_training_data['Y_input'], 100, (16, 38), color='orange', label='Y', histtype='step')
    ax[0].hist(df_training_data['J_input'], 100, (16, 38), color='yellow', label='J', histtype='step')
    ax[0].hist(df_training_data['H_input'], 100, (16, 38), color='black', label='H', histtype='step')
    ax[0].hist(df_training_data['Ks_input'], 100, (16, 38), color='cyan', label='K', histtype='step')
    ax[0].set_xlabel('input mag')
    ax[0].legend()

    ax[1].hist(df_training_data['MAG_GAAP_u'], 100, (16, 38), color='red', label='u', histtype='step')
    ax[1].hist(df_training_data['MAG_GAAP_g'], 100, (16, 38), color='green', label='g', histtype='step')
    ax[1].hist(df_training_data['MAG_GAAP_r'], 100, (16, 38), color='blue', label='r', histtype='step')
    ax[1].hist(df_training_data['MAG_GAAP_i'], 100, (16, 38), color='grey', label='i', histtype='step')
    ax[1].hist(df_training_data['MAG_GAAP_Z'], 100, (16, 38), color='purple', label='z', histtype='step')
    ax[1].hist(df_training_data['MAG_GAAP_Y'], 100, (16, 38), color='orange', label='Y', histtype='step')
    ax[1].hist(df_training_data['MAG_GAAP_J'], 100, (16, 38), color='yellow', label='J', histtype='step')
    ax[1].hist(df_training_data['MAG_GAAP_H'], 100, (16, 38), color='black', label='H', histtype='step')
    ax[1].hist(df_training_data['MAG_GAAP_Ks'], 100, (16, 38), color='cyan', label='K', histtype='step')
    ax[1].set_xlabel('output mag')
    ax[1].legend()
    plt.show()


    plt.figure()
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))  # 20., 6.
    ax[0].hist(df_training_data['InputSeeing_u'], 100, color='red', label='u', histtype='step')
    ax[0].hist(df_training_data['InputSeeing_g'], 100, color='green', label='g', histtype='step')
    ax[0].hist(df_training_data['InputSeeing_r'], 100, color='blue', label='r', histtype='step')
    ax[0].hist(df_training_data['InputSeeing_i'], 100, color='grey', label='i', histtype='step')
    ax[0].hist(df_training_data['InputSeeing_Z'], 100, color='purple', label='z', histtype='step')
    ax[0].hist(df_training_data['InputSeeing_Y'], 100, color='orange', label='Y', histtype='step')
    ax[0].hist(df_training_data['InputSeeing_J'], 100, color='yellow', label='J', histtype='step')
    ax[0].hist(df_training_data['InputSeeing_H'], 100, color='black', label='H', histtype='step')
    ax[0].hist(df_training_data['InputSeeing_Ks'], 100, color='cyan', label='K', histtype='step')
    ax[0].set_xlabel('Seeing')
    ax[0].legend()

    ax[1].hist(df_training_data['InputBeta_u'], 100, color='red', label='u', histtype='step')
    ax[1].hist(df_training_data['InputBeta_g'], 100, color='green', label='g', histtype='step')
    ax[1].hist(df_training_data['InputBeta_r'], 100, color='blue', label='r', histtype='step')
    ax[1].hist(df_training_data['InputBeta_i'], 100, color='grey', label='i', histtype='step')
    ax[1].hist(df_training_data['InputBeta_Z'], 100, color='purple', label='z', histtype='step')
    ax[1].hist(df_training_data['InputBeta_Y'], 100, color='orange', label='Y', histtype='step')
    ax[1].hist(df_training_data['InputBeta_J'], 100, color='yellow', label='J', histtype='step')
    ax[1].hist(df_training_data['InputBeta_H'], 100, color='black', label='H', histtype='step')
    ax[1].hist(df_training_data['InputBeta_Ks'], 100, color='cyan', label='K', histtype='step')
    ax[1].set_xlabel('Beta')
    ax[1].legend()
    plt.show()

    plt.hist(df_training_data[f"Re_input"], 100, label='Re', histtype='step')
    plt.show()
    plt.hist(df_training_data[f"e1_input"], 100, label='e1', histtype='step')
    plt.show()
    plt.hist(df_training_data[f"e2_input"], 100, label='e2', histtype='step')
    plt.show()
    plt.hist(df_training_data[f"g1_in"], 100, label='g1', histtype='step')
    plt.show()
    plt.hist(df_training_data[f"g2_in"], 100, label='g2', histtype='step')
    plt.show()
    plt.hist(df_training_data[f"sersic_n_input"], 100, label='sersic', histtype='step')
    plt.show()
    plot_data = False
    if plot_data is True:
        img_grid = plot_chain(
            data_frame=df_training_data,
            plot_name=f"viking color",
            columns=[
                f"u_input",
                f"g_input",
                f"r_input",
                f"i_input",
                f"Z_input",
                f"Y_input",
                f"J_input",
                f"H_input",
                f"Ks_input"
            ],
            parameter=[
                    f"u",
                    f"g",
                    f"r",
                    f"i",
                    f"Z",
                    f"Y",
                    f"J",
                    f"H",
                    f"Ks"
                ],
            extends=None,
            show_plot=True
        )
        writer.add_image("unsheared chain plot", img_grid)

        img_grid = plot_chain(
            data_frame=df_training_data,
            plot_name=f"viking color",
            columns=[
                f"MAG_GAAP_u",
                f"MAG_GAAP_g",
                f"MAG_GAAP_r",
                f"MAG_GAAP_i",
                f"MAG_GAAP_Z",
                f"MAG_GAAP_Y",
                f"MAG_GAAP_J",
                f"MAG_GAAP_H",
                f"MAG_GAAP_Ks"
            ],
            parameter=[
                f"u",
                f"g",
                f"r",
                f"i",
                f"Z",
                f"Y",
                f"J",
                f"H",
                f"Ks"
            ],
            extends=None,
            show_plot=True
        )
        writer.add_image("unsheared chain plot", img_grid)
        try:
            img_grid = plot_chain(
                data_frame=df_training_data,
                plot_name=f"obs",
                columns=[
                    f"Re_input",
                    f"e1_input",
                    f"e2_input",
                    f"g1_in",
                    f"g2_in",
                    f"sersic_n_input"
                ],
                parameter=[
                    f"Re",
                    f"e1",
                    f"e2",
                    f"g1",
                    f"g2",
                    f"sersic"
                ],
                extends=None,
                show_plot=False
            )
            writer.add_image("obs", img_grid)
        except IndexError:
            print("error plot obs")
        img_grid = plot_chain(
            data_frame=df_training_data,
            plot_name=f"obs",
            columns=[
                f"InputSeeing_u",
                f"InputSeeing_g",
                f"InputSeeing_r",
                f"InputSeeing_i",
                f"InputSeeing_Z",
                f"InputSeeing_Y",
                f"InputSeeing_J",
                f"InputSeeing_H",
                f"InputSeeing_Ks"
            ],
            parameter=[
                f"u",
                f"g",
                f"r",
                f"i",
                f"Z",
                f"Y",
                f"J",
                f"H",
                f"Ks"
            ],
            extends=None,
            show_plot=False
        )
        writer.add_image("Seeing", img_grid)
        img_grid = plot_chain(
            data_frame=df_training_data,
            plot_name=f"obs",
            columns=[
                f"InputBeta_u",
                f"InputBeta_g",
                f"InputBeta_r",
                f"InputBeta_i",
                f"InputBeta_Z",
                f"InputBeta_Y",
                f"InputBeta_J",
                f"InputBeta_H",
                f"InputBeta_Ks"
            ],
            parameter=[
                f"u",
                f"g",
                f"r",
                f"i",
                f"Z",
                f"Y",
                f"J",
                f"H",
                f"Ks"
            ],
            extends=None,
            show_plot=False
        )
        writer.add_image("Seeing", img_grid)

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
