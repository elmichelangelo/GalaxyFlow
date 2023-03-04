import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import sys
import os
import glob
import pandas as pd
import time
import shutil
import pickle
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from chainconsumer import ChainConsumer

from src.Handler.helper_functions import concatenate_lists, mag2flux


def load_data(path_data, selected_scaler, all_column):
    print("Load data")
    # open file
    infile = open(path_data, 'rb')

    # load pickle as pandas dataframe
    df_data = pd.DataFrame(pickle.load(infile, encoding='latin1'))

    # close file
    infile.close()

    df_data["Color Mag U-G"] = df_data[f"BDF_MAG_DERED_CALIB_U"].to_numpy() - \
                               df_data[f"BDF_MAG_DERED_CALIB_G"].to_numpy()
    df_data["Color Mag G-R"] = df_data[f"BDF_MAG_DERED_CALIB_G"].to_numpy() - \
                               df_data[f"BDF_MAG_DERED_CALIB_R"].to_numpy()
    df_data["Color Mag R-I"] = df_data[f"BDF_MAG_DERED_CALIB_R"].to_numpy() - \
                               df_data[f"BDF_MAG_DERED_CALIB_I"].to_numpy()
    df_data["Color Mag I-Z"] = df_data[f"BDF_MAG_DERED_CALIB_I"].to_numpy() - \
                               df_data[f"BDF_MAG_DERED_CALIB_Z"].to_numpy()
    df_data["Color Mag Z-J"] = df_data[f"BDF_MAG_DERED_CALIB_Z"].to_numpy() - \
                               df_data[f"BDF_MAG_DERED_CALIB_J"].to_numpy()
    df_data["Color Mag J-H"] = df_data[f"BDF_MAG_DERED_CALIB_J"].to_numpy() - \
                               df_data[f"BDF_MAG_DERED_CALIB_H"].to_numpy()
    df_data["Color Mag H-KS"] = df_data[f"BDF_MAG_DERED_CALIB_H"].to_numpy() - \
                                df_data[f"BDF_MAG_DERED_CALIB_KS"].to_numpy()

    df_data["BDF_FLUX_DERED_CALIB_U"] = mag2flux(df_data["BDF_MAG_DERED_CALIB_U"])
    df_data["BDF_FLUX_DERED_CALIB_G"] = mag2flux(df_data["BDF_MAG_DERED_CALIB_G"])
    df_data["BDF_FLUX_DERED_CALIB_J"] = mag2flux(df_data["BDF_MAG_DERED_CALIB_J"])
    df_data["BDF_FLUX_DERED_CALIB_H"] = mag2flux(df_data["BDF_MAG_DERED_CALIB_H"])
    df_data["BDF_FLUX_DERED_CALIB_K"] = mag2flux(df_data["BDF_MAG_DERED_CALIB_KS"])
    df_data["BDF_FLUX_ERR_DERED_CALIB_U"] = 1 / df_data["BDF_FLUX_DERED_CALIB_U"]
    df_data["BDF_FLUX_ERR_DERED_CALIB_G"] = 1 / df_data["BDF_FLUX_DERED_CALIB_G"]
    df_data["BDF_FLUX_ERR_DERED_CALIB_R"] = 1 / df_data["BDF_FLUX_DERED_CALIB_R"]
    df_data["BDF_FLUX_ERR_DERED_CALIB_I"] = 1 / df_data["BDF_FLUX_DERED_CALIB_I"]
    df_data["BDF_FLUX_ERR_DERED_CALIB_Z"] = 1 / df_data["BDF_FLUX_DERED_CALIB_Z"]
    df_data["BDF_FLUX_ERR_DERED_CALIB_J"] = 1 / df_data["BDF_FLUX_DERED_CALIB_J"]
    df_data["BDF_FLUX_ERR_DERED_CALIB_H"] = 1 / df_data["BDF_FLUX_DERED_CALIB_H"]
    df_data["BDF_FLUX_ERR_DERED_CALIB_K"] = 1 / df_data["BDF_FLUX_DERED_CALIB_K"]

    df_all_data = df_data[all_column]
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
        raise f"{selected_scaler} is no valid scaler"

    if scaler is not None:
        scaler.fit(df_all_data)
        scaled = scaler.transform(df_all_data)
        df_all_data_scaled = pd.DataFrame(scaled, columns=df_all_data.columns)
    else:
        df_all_data_scaled = df_all_data

    return df_all_data_scaled, df_all_data, scaler


def load_gandalf(path_nn):
    print("Load gandalf...")
    nn_gandalf = torch.load(path_nn)
    return nn_gandalf


def emulate_data(nn, all_nn_data, input_column, sompz_columns, latent_size, size, scaler):
    print(f"emulate galaxies...")
    start = time.time()
    lst_data = []
    input_nn_data = all_nn_data[input_column]
    # nn_data_idx = np.arange(0, len(input_nn_data))
    # nn_random_idx = np.random.choice(nn_data_idx, size=size)
    arr_nn_input = input_nn_data.to_numpy()
    arr_input = arr_nn_input  # [nn_random_idx]

    sompz_nn = all_nn_data[input_column + sompz_columns]
    arr_sompz_nn = sompz_nn.to_numpy()
    arr_sompz = arr_sompz_nn  # [nn_random_idx]

    latent_space = concatenate_lists([np.random.randn(size, latent_size)])
    stacked_test_input_fake = np.column_stack((latent_space, arr_input))
    arr_input_nn = arr_input

    for idx, test_item in enumerate(stacked_test_input_fake):
        gandalf_data = nn.forward(torch.FloatTensor(test_item)).detach().numpy()
        lst_data.append(list(arr_sompz[idx]) + list(gandalf_data))

    end = time.time()
    print(f"Time: {end-start}s")

    df_gandalf_data = pd.DataFrame(data=lst_data, columns=list(all_nn_data.keys()))
    generator_rescaled = scaler.inverse_transform(df_gandalf_data)
    df_generator_rescaled = pd.DataFrame(data=generator_rescaled, columns=list(all_nn_data.keys()))

    return df_generator_rescaled


def compare_plots(path_save_plot, df_data_true, df_data_generated, plot_parameter, boxplot_parameter, nn_name,
                  show_plot, save_plot):
    print("Plotting data ...")
    chaincon = ChainConsumer()
    df_data_true_output = df_data_true[plot_parameter]
    df_data_generated_output = df_data_generated[plot_parameter]

    df_data_true_boxplot = df_data_true[boxplot_parameter]
    df_data_generated_boxplot = df_data_generated[boxplot_parameter]

    PLT_COMPARE_BANDS = f"{path_save_plot}/compare_bands"
    PLT_CHAIN = f"{path_save_plot}/chain_plot"
    PLT_COLORS = f"{path_save_plot}/colors"
    PLT_SNR = f"{path_save_plot}/snr"
    PLT_SIZE_RATIO = f"{path_save_plot}/size_ratio"
    PLT_COMPARE_T = f"{path_save_plot}/compare_t"
    PLT_DETECTED = f"{path_save_plot}/detected"
    PLT_FLAGS = f"{path_save_plot}/flags"

    if not os.path.exists(PLT_COMPARE_BANDS):
        os.mkdir(PLT_COMPARE_BANDS)
    if not os.path.exists(PLT_CHAIN):
        os.mkdir(PLT_CHAIN)
    if not os.path.exists(PLT_COLORS):
        os.mkdir(PLT_COLORS)
    if not os.path.exists(PLT_SNR):
        os.mkdir(PLT_SNR)
    if not os.path.exists(PLT_SIZE_RATIO):
        os.mkdir(PLT_SIZE_RATIO)
    if not os.path.exists(PLT_COMPARE_T):
        os.mkdir(PLT_COMPARE_T)
    if not os.path.exists(PLT_DETECTED):
        os.mkdir(PLT_DETECTED)
    if not os.path.exists(PLT_FLAGS):
        os.mkdir(PLT_FLAGS)

    # plot_parameter_short = [
    #     "mag_r",
    #     "mag_i",
    #     "mag_z",
    #     "snr",
    #     "size_ratio",
    #     "T",
    # ]
    # chaincon.add_chain(df_data_true_output.to_numpy(), parameters=plot_parameter_short, name="true")
    # chaincon.add_chain(df_data_generated_output.to_numpy(), parameters=plot_parameter_short, name="generated")
    # if show_plot is True and save_plot is False:
    #     chaincon.plotter.plot(
    #         figsize="page",
    #         truth=[
    #             df_data_true_output["unsheared/mag_r"].mean(),
    #             df_data_true_output["unsheared/mag_i"].mean(),
    #             df_data_true_output["unsheared/mag_z"].mean(),
    #             df_data_true_output["unsheared/snr"].mean(),
    #             df_data_true_output["unsheared/size_ratio"].mean(),
    #             df_data_true_output["unsheared/T"].mean(),
    #         ]
    #     )
    # elif save_plot is True:
    #     chaincon.plotter.plot(
    #         filename=f"{PLT_CHAIN}/chain_plot_{nn_name}.png",
    #         figsize="page",
    #         truth=[
    #             df_data_true_output["unsheared/mag_r"].mean(),
    #             df_data_true_output["unsheared/mag_i"].mean(),
    #             df_data_true_output["unsheared/mag_z"].mean(),
    #             df_data_true_output["unsheared/snr"].mean(),
    #             df_data_true_output["unsheared/size_ratio"].mean(),
    #             df_data_true_output["unsheared/T"].mean(),
    #         ]
    #     )
    # plt.clf()
    #
    # len_true = len(df_data_true_boxplot["unsheared/mag_r"])
    # len_generated = len(df_data_generated_boxplot["unsheared/mag_r"])
    # lst_bands_true = list(df_data_true_boxplot["unsheared/mag_r"]) +\
    #            list(df_data_true_boxplot["unsheared/mag_i"]) +\
    #            list(df_data_true_boxplot["unsheared/mag_z"])
    # lst_bands_generated = list(df_data_generated_boxplot["unsheared/mag_r"]) + \
    #                 list(df_data_generated_boxplot["unsheared/mag_i"]) + \
    #                 list(df_data_generated_boxplot["unsheared/mag_z"])
    # df_compare_bands = pd.DataFrame({
    #     f"mag": lst_bands_true + lst_bands_generated,
    #     f"color": ["r" for i in range(len_true)] +
    #               ["i" for i in range(len_true)] +
    #               ["z" for i in range(len_true)] +
    #               ["r" for i in range(len_generated)] +
    #               ["i" for i in range(len_generated)] +
    #               ["z" for i in range(len_generated)],
    #     f"data": ["true" for i in range(len(lst_bands_true))] + ["generated" for i in range(len(lst_bands_generated))]
    # })
    #
    # lst_colors_true = list(df_data_true_boxplot["unsheared/mag_r"] - df_data_true_boxplot["unsheared/mag_i"]) + \
    #                   list(df_data_true_boxplot["unsheared/mag_i"] - df_data_true_boxplot["unsheared/mag_z"])
    # lst_colors_generated = list(
    #     df_data_generated_boxplot["unsheared/mag_r"] - df_data_generated_boxplot["unsheared/mag_i"]) + \
    #                        list(df_data_generated_boxplot["unsheared/mag_i"] - df_data_generated_boxplot[
    #                            "unsheared/mag_z"])
    # df_compare_colors = pd.DataFrame({
    #     f"mag": lst_colors_true + lst_colors_generated,
    #     f"color": ["r-i" for i in range(len_true)] +
    #               ["i-z" for i in range(len_true)] +
    #               ["r-i" for i in range(len_generated)] +
    #               ["i-z" for i in range(len_generated)],
    #     f"data": ["true" for i in range(len(lst_colors_true))] + ["generated" for i in range(len(lst_colors_generated))]
    # })
    #
    # lst_snr_true = list(df_data_true_boxplot["unsheared/snr"])
    # lst_snr_generated = list(df_data_generated_boxplot["unsheared/snr"])
    # df_compare_snr = pd.DataFrame({
    #     f"snr": lst_snr_true + lst_snr_generated,
    #     f"data": ["true" for i in range(len(lst_snr_true))] + ["generated" for i in range(len(lst_snr_generated))]
    # })
    #
    # lst_size_ratio_true = list(df_data_true_boxplot["unsheared/size_ratio"])
    # lst_size_ratio_generated = list(df_data_generated_boxplot["unsheared/size_ratio"])
    # df_compare_size_ratio = pd.DataFrame({
    #     f"size_ratio": lst_size_ratio_true + lst_size_ratio_generated,
    #     f"data": ["true" for i in range(len(lst_size_ratio_true))] +
    #              ["generated" for i in range(len(lst_size_ratio_generated))]
    # })
    #
    # lst_T_true = list(df_data_true_boxplot["unsheared/T"])
    # lst_T_generated = list(df_data_generated_boxplot["unsheared/T"])
    # df_compare_T = pd.DataFrame({
    #     f"T": lst_T_true + lst_T_generated,
    #     f"data": ["true" for i in range(len(lst_T_true))] +
    #              ["generated" for i in range(len(lst_T_generated))]
    # })
    #
    # lst_flags_true = list(df_data_true_boxplot["unsheared/flags"])
    # lst_flags_generated = list(df_data_generated_boxplot["unsheared/flags"])
    # df_compare_flags = pd.DataFrame({
    #     f"flags": lst_flags_true + lst_flags_generated,
    #     f"data": ["true" for i in range(len(lst_flags_true))] +
    #              ["generated" for i in range(len(lst_flags_generated))]
    # })
    #
    # lst_detected_true = list(df_data_true_boxplot["detected"])
    # lst_detected_generated = list(df_data_generated_boxplot["detected"])
    # df_compare_detected = pd.DataFrame({
    #     f"detected": lst_detected_true + lst_detected_generated,
    #     f"data": ["true" for i in range(len(lst_detected_true))] +
    #              ["generated" for i in range(len(lst_detected_generated))]
    # })
    #
    # sns.boxplot(
    #     data=df_compare_bands,
    #     x="mag",
    #     y="color",
    #     hue="data",
    #     dodge=True,
    #     notch=False,
    #     showcaps=True,
    #     showfliers=False
    # )
    # plt.title("Compare bands")
    # if save_plot is True:
    #     plt.savefig(f"{PLT_COMPARE_BANDS}/compare_bands_{nn_name}.png")
    #
    # if show_plot is True:
    #     plt.show()
    # plt.clf()
    #
    # sns.boxplot(
    #     data=df_compare_colors,
    #     x="mag",
    #     y="color",
    #     hue="data",
    #     dodge=True,
    #     notch=False,
    #     showcaps=True,
    #     showfliers=False
    # )
    # plt.title("Compare colors")
    #
    # if save_plot is True:
    #     plt.savefig(f"{PLT_COLORS}/colors_{nn_name}.png")
    #
    # if show_plot is True:
    #     plt.show()
    # plt.clf()
    #
    # sns.boxplot(
    #     data=df_compare_snr,
    #     x="snr",
    #     y="data",
    #     hue="data",
    #     dodge=True,
    #     notch=False,
    #     showcaps=True,
    #     showfliers=False
    # )
    # plt.title("Compare snr")
    #
    # if save_plot is True:
    #     plt.savefig(f"{PLT_SNR}/snr_{nn_name}.png")
    #
    # if show_plot is True:
    #     plt.show()
    # plt.clf()
    #
    # sns.boxplot(
    #     data=df_compare_size_ratio,
    #     x="size_ratio",
    #     y="data",
    #     hue="data",
    #     dodge=True,
    #     notch=False,
    #     showcaps=True,
    #     showfliers=False
    # )
    # plt.title("Compare size ratio")
    #
    # if save_plot is True:
    #     plt.savefig(f"{PLT_SIZE_RATIO}/size_ratio_{nn_name}.png")
    #
    # if show_plot is True:
    #     plt.show()
    # plt.clf()
    #
    # sns.boxplot(
    #     data=df_compare_T,
    #     x="T",
    #     y="data",
    #     hue="data",
    #     dodge=True,
    #     notch=False,
    #     showcaps=True,
    #     showfliers=False
    # )
    # plt.title("Compare T")
    #
    # if save_plot is True:
    #     plt.savefig(f"{PLT_COMPARE_T}/T_{nn_name}.png")
    #
    # if show_plot is True:
    #     plt.show()
    #
    # plt.clf()
    #
    # sns.boxplot(
    #     data=df_compare_detected,
    #     x="detected",
    #     y="data",
    #     hue="data",
    #     dodge=True,
    #     notch=False,
    #     showcaps=True,
    #     showfliers=False
    # )
    # plt.title("Compare detected")
    #
    # if save_plot is True:
    #     plt.savefig(f"{PLT_DETECTED}/detected_{nn_name}.png")
    #
    # if show_plot is True:
    #     plt.show()
    # plt.clf()
    #
    # sns.boxplot(
    #     data=df_compare_flags,
    #     x="flags",
    #     y="data",
    #     hue="data",
    #     dodge=True,
    #     notch=False,
    #     showcaps=True,
    #     showfliers=False
    # )
    # plt.title("Compare flags")
    #
    # if save_plot is True:
    #     plt.savefig(f"{PLT_FLAGS}/flags_{nn_name}.png")
    #
    # if show_plot is True:
    #     plt.show()
    # plt.clf()

    df_compare_balrog = pd.DataFrame({
        'BDF_MAG_DERED_CALIB_I': df_data_true['BDF_MAG_DERED_CALIB_I'],
        'Measured - BDF_MAG_DERED_CALIB_I': df_data_true['unsheared/mag_i'] - df_data_true['BDF_MAG_DERED_CALIB_I'],
    })

    df_compare_generated = pd.DataFrame({
        'BDF_MAG_DERED_CALIB_I': df_data_generated['BDF_MAG_DERED_CALIB_I'],
        'Measured - BDF_MAG_DERED_CALIB_I': df_data_generated['unsheared/mag_i'] - df_data_generated['BDF_MAG_DERED_CALIB_I'],
    })

    plot_parameter = ["BDF_MAG_DERED_CALIB_I", 'unsheared/mag_i - BDF_MAG_DERED_CALIB_I']

    chaincon.add_chain(df_compare_balrog.to_numpy(), parameters=plot_parameter, name="Balrog data")
    chaincon.add_chain(df_compare_generated.to_numpy(), parameters=plot_parameter, name="generated data")

    chaincon.plotter.plot(
        figsize="page",
        extents={
            "unsheared/mag_i - BDF_MAG_DERED_CALIB_I": (-1, 1),
            "BDF_MAG_DERED_CALIB_I": (18, 24)
        },
        truth=[
            0,
            0,
            # df_data_true_output["unsheared/mag_i"].mean(),
            # df_data_true_output["unsheared/mag_z"].mean(),
            # df_data_true_output["unsheared/snr"].mean(),
            # df_data_true_output["unsheared/size_ratio"].mean(),
            # df_data_true_output["unsheared/T"].mean(),
        ]
    )

    # sns.jointplot(x=df_data_true['BDF_MAG_DERED_CALIB_I'], y=df_data_true['unsheared/mag_i'] - df_data_true['BDF_MAG_DERED_CALIB_I'],
    #               kind="kde")
    # sns.jointplot(x=df_data_generated['BDF_MAG_DERED_CALIB_I'],
    #               y=df_data_generated['unsheared/mag_i'] - df_data_generated['BDF_MAG_DERED_CALIB_I'],
    #               kind="kde")
    plt.show()


def save_emulated_cat(path_save_catalog, df_data):
    with open(f"{path_save_catalog}", "wb") as f:
        pickle.dump(df_data.to_dict(), f, protocol=2)
    # df_data.to_pickle(path_save_catalog)


def main(path_data, selected_scaler, input_column, output_column, path_nn, latent_size, path_save_plot, plot_parameter,
         boxplot_parameter, nn_name, path_save_catalog, size, show_plot, save_plot, save_data, sompz_columns):

    df_all_data_scaled, df_true_data, scaler = load_data(
        path_data=path_data,
        selected_scaler=selected_scaler,
        all_column=input_column + sompz_columns + output_column
    )

    gandalf = load_gandalf(path_nn=path_nn)

    df_emulated_data = emulate_data(
        nn=gandalf,
        all_nn_data=df_all_data_scaled,
        input_column=input_column,
        sompz_columns=sompz_columns,
        latent_size=latent_size,
        scaler=scaler,
        size=size
    )

    compare_plots(
        path_save_plot=path_save_plot,
        df_data_true=df_true_data,
        df_data_generated=df_emulated_data,
        plot_parameter=plot_parameter,
        boxplot_parameter=boxplot_parameter,
        nn_name=nn_name,
        show_plot=show_plot,
        save_plot=save_plot
    )

    if save_data is True:
        save_emulated_cat(
            path_save_catalog=path_save_catalog,
            df_data=df_emulated_data
        )

    print("All done!")


def move_nn_2_folder(folder_nn, new_folder, new_name):
    """"""
    lst_nn_paths = glob.glob(f"{folder_nn}/generator*.pt")
    lst_nn_names = []
    for nn_path in lst_nn_paths:
        nn_number = nn_path.split("/")[-1]
        nn_number = nn_number.split(".")[0]
        nn_number = nn_number.split("_")[-1]
        nn_name = f"{new_name}{nn_number}"
        lst_nn_names.append(nn_name)
        nn_new_path = f"{new_folder}/{nn_name}.pt"
        shutil.copyfile(src=nn_path, dst=nn_new_path)
    print(lst_nn_names)


if __name__ == '__main__':
    ROOT_PATH = os.path.abspath(sys.path[1])
    # FOLDER_GAN_OUT = "GAN_test_input_tensor_10_batch_size_32_scaler_MaxAbsScaler_adjust_generator_False_adjust_discriminator_False"
    # FOLDER_RUN_OUT = "run_1_lr_generator_1e-05_lr_discriminator_0.0001_LeakyReLU_0.01"
    # PATH_2_TRAINING_NN = f"{ROOT_PATH}/Output/{FOLDER_GAN_OUT}/{FOLDER_RUN_OUT}/save_nn/"
    PATH_NN_FOLDER = f"{ROOT_PATH}/Pretrained_NN"
    LST_NN_NAME = f"{PATH_NN_FOLDER}/generator_epoch_199.pt"
    PATH_DATA = f"{ROOT_PATH}/Running/GANdalf_Data/Balrog_2_data_MAG_250000.pkl"
    PATH_SAVE_DICT = f"{ROOT_PATH}/Running/GANdalf_Output"
    PATH_SAVE_PLOT = f"{PATH_SAVE_DICT}/Plots"
    PATH_SAVE_DATA = f"{PATH_SAVE_DICT}/Emulated_Data"
    if not os.path.exists(PATH_SAVE_DICT):
        os.mkdir(PATH_SAVE_DICT)
    if not os.path.exists(PATH_SAVE_PLOT):
        os.mkdir(PATH_SAVE_PLOT)
    if not os.path.exists(PATH_SAVE_DATA):
        os.mkdir(PATH_SAVE_DATA)

    # move_nn_2_folder(folder_nn=PATH_2_TRAINING_NN, new_folder=PATH_NN_FOLDER, new_name="T10_B32_MAS_A00_L54_E")

    lst_input = [
        "BDF_MAG_DERED_CALIB_R",
        "BDF_MAG_DERED_CALIB_I",
        "BDF_MAG_DERED_CALIB_Z",
        "Color Mag U-G",
        "Color Mag G-R",
        "Color Mag R-I",
        "Color Mag I-Z",
        "Color Mag Z-J",
        "Color Mag J-H",
        "Color Mag H-KS",
        "BDF_T",
        "BDF_G",
        "FWHM_WMEAN_R",
        "FWHM_WMEAN_I",
        "FWHM_WMEAN_Z",
        "AIRMASS_WMEAN_R",
        "AIRMASS_WMEAN_I",
        "AIRMASS_WMEAN_Z",
        "MAGLIM_R",
        "MAGLIM_I",
        "MAGLIM_Z",
        "EBV_SFD98"
    ]

    lst_output = [
        "unsheared/mag_r",
        "unsheared/mag_i",
        "unsheared/mag_z",
        "unsheared/snr",
        "unsheared/size_ratio",
        "unsheared/flags",
        "unsheared/T",
        "detected"
    ]

    lst_plot_parameter = [
        "unsheared/mag_r",
        "unsheared/mag_i",
        "unsheared/mag_z",
        "unsheared/snr",
        "unsheared/size_ratio",
        "unsheared/T",
    ]

    lst_boxplot_parameter = [
        "unsheared/mag_r",
        "unsheared/mag_i",
        "unsheared/mag_z",
        "unsheared/snr",
        "unsheared/size_ratio",
        "unsheared/flags",
        "unsheared/T",
        "detected"
    ]

    lst_sompz_columns = [
        "BDF_FLUX_DERED_CALIB_U",
        "BDF_FLUX_DERED_CALIB_G",
        "BDF_FLUX_DERED_CALIB_R",
        "BDF_FLUX_DERED_CALIB_I",
        "BDF_FLUX_DERED_CALIB_Z",
        "BDF_FLUX_DERED_CALIB_J",
        "BDF_FLUX_DERED_CALIB_H",
        "BDF_FLUX_DERED_CALIB_K",
        "BDF_FLUX_ERR_DERED_CALIB_U",
        "BDF_FLUX_ERR_DERED_CALIB_G",
        "BDF_FLUX_ERR_DERED_CALIB_R",
        "BDF_FLUX_ERR_DERED_CALIB_I",
        "BDF_FLUX_ERR_DERED_CALIB_Z",
        "BDF_FLUX_ERR_DERED_CALIB_J",
        "BDF_FLUX_ERR_DERED_CALIB_H",
        "BDF_FLUX_ERR_DERED_CALIB_K"
    ]
    NN_NAME_WO_NO = "generator"
    # for PATH_NN in LST_NN_NAME:
    # NN_NAME = LST_NN_NAME
    # NN_NAME = NN_NAME.split(".")[0]
    # splited_nn_name = NN_NAME.split('_')
    NN_NUMBER = 199

    NN_NAME = f"{NN_NAME_WO_NO}_{NN_NUMBER}"
    # NN_NAME_WO_NO = f"{splited_nn_name[0]}_{splited_nn_name[1]}_{splited_nn_name[2]}_{splited_nn_name[3]}_" \
    #                 f"{splited_nn_name[4]}"
    PATH_SAVE_DATA_FOLDER = f"{PATH_SAVE_DATA}/{NN_NAME_WO_NO}"
    PATH_SAVE_PLOT_FOLDER = f"{PATH_SAVE_PLOT}/{NN_NAME_WO_NO}"
    if not os.path.exists(PATH_SAVE_DATA_FOLDER):
        os.mkdir(PATH_SAVE_DATA_FOLDER)
    if not os.path.exists(PATH_SAVE_PLOT_FOLDER):
        os.mkdir(PATH_SAVE_PLOT_FOLDER)
    PATH_SAVE_CAT = f"{PATH_SAVE_DATA_FOLDER}/{NN_NAME}.pkl"

    main(
        path_data=PATH_DATA,
        selected_scaler="MaxAbsScaler",
        input_column=lst_input,
        path_nn=LST_NN_NAME,
        latent_size=10,
        output_column=lst_output,
        path_save_plot=PATH_SAVE_PLOT_FOLDER,
        nn_name=NN_NAME,
        plot_parameter=lst_plot_parameter,
        boxplot_parameter=lst_boxplot_parameter,
        sompz_columns=lst_sompz_columns,
        path_save_catalog=PATH_SAVE_CAT,
        size=int(250000),
        save_data=True,
        show_plot=False,
        save_plot=True
    )

