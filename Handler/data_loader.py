from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from src.Handler.helper_functions import flux2mag
import numpy as np
import pandas as pd
import pickle
import os
import sys

sys.path.append(os.path.dirname(__file__))


def load_data(path_training_data, input_discriminator, input_generator, output_generator, analytical_data, apply_cuts,
              only_detected, selected_scaler):
    """
    Load the Data from *.pkl-file

    Args:
        selected_scaler: MaxAbsScaler, MinMaxScaler or StandardScaler
        path_training_data: path of the *.pkl-file as string

    Returns: dictionary with list of training data, list of test data, used deep field data and used flux data
    """

    # open file
    infile = open(path_training_data, 'rb')  # filename

    # load pickle as pandas dataframe
    df_data = pd.DataFrame(pickle.load(infile, encoding='latin1'))

    # close file
    infile.close()

    if apply_cuts is True:
        df_data = data_cuts(df_data, analytical_data, only_detected)

    if analytical_data is not True:
        df_data[f"BDF_MAG_DERED_CALIB_U"] = flux2mag(df_data[f"BDF_FLUX_DERED_CALIB_U"])
        df_data[f"BDF_MAG_DERED_CALIB_G"] = flux2mag(df_data[f"BDF_FLUX_DERED_CALIB_G"])
        df_data[f"BDF_MAG_DERED_CALIB_R"] = flux2mag(df_data[f"BDF_FLUX_DERED_CALIB_R"])
        df_data[f"BDF_MAG_DERED_CALIB_I"] = flux2mag(df_data[f"BDF_FLUX_DERED_CALIB_I"])
        df_data[f"BDF_MAG_DERED_CALIB_Z"] = flux2mag(df_data[f"BDF_FLUX_DERED_CALIB_Z"])
        df_data[f"BDF_MAG_DERED_CALIB_J"] = flux2mag(df_data[f"BDF_FLUX_DERED_CALIB_J"])
        df_data[f"BDF_MAG_DERED_CALIB_H"] = flux2mag(df_data[f"BDF_FLUX_DERED_CALIB_H"])
        df_data[f"BDF_MAG_DERED_CALIB_KS"] = flux2mag(df_data[f"BDF_FLUX_DERED_CALIB_KS"])
        df_data[f"BDF_G"] = np.sqrt(df_data["BDF_G_0"] ** 2 + df_data["BDF_G_1"] ** 2)
        df_data["unsheared/mag_r"] = flux2mag(df_data["unsheared/flux_r"])
        df_data["unsheared/mag_i"] = flux2mag(df_data["unsheared/flux_i"])
        df_data["unsheared/mag_z"] = flux2mag(df_data["unsheared/flux_i"])

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

    df_training_data = df_data[input_discriminator]

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

    # convert to float 32
    df_training_data = pd.DataFrame(np.float32(df_training_data), columns=df_training_data.columns)

    if scaler is not None:
        scaler.fit(df_training_data)
        scaled = scaler.transform(df_training_data)
        df_training_data_scaled = pd.DataFrame(scaled, columns=df_training_data.columns)
    else:
        df_training_data_scaled = df_training_data

    df_training_data_scaled = df_training_data_scaled.sample(frac=1, random_state=1)

    arr_label_generator = np.array(df_training_data_scaled[input_generator])
    arr_output_generator = np.array(df_training_data_scaled[output_generator])
    arr_input_discriminator = np.array(df_training_data_scaled[input_discriminator])

    dict_training_data = {
        f"label generator in order {input_generator}": arr_label_generator[:int(len(arr_label_generator) * .8)],
        f"output generator in order {output_generator}": arr_output_generator[:int(len(arr_output_generator) * .8)],
        f"input discriminator in order {input_discriminator}": arr_input_discriminator[
                                                               :int(len(arr_input_discriminator) * .8)],
        "columns label generator": input_generator,
        "columns output generator": output_generator,
        "columns input discriminator": input_discriminator,
        "scaler": scaler
    }

    dict_test_data = {
        f"label generator in order {input_generator}": arr_label_generator[int(len(arr_label_generator) * .8):],
        f"output generator in order {output_generator}": arr_output_generator[int(len(arr_output_generator) * .8):],
        f"input discriminator in order {input_discriminator}": arr_input_discriminator[
                                                               int(len(arr_input_discriminator) * .8):],
        "columns label generator": input_generator,
        "columns output generator": output_generator,
        "columns input discriminator": input_discriminator,
        "scaler": scaler
    }

    return dict_training_data, dict_test_data


def data_cuts(data_frame, analytical_data, only_detected):
    """"""
    if only_detected is True:
        only_detected = (data_frame["detected"] == 1)
        data_frame = data_frame[only_detected]

    if analytical_data is not True:
        bdf_r_mag_cut = (flux2mag(data_frame["BDF_FLUX_DERED_CALIB_R"]) > 17) & (flux2mag(data_frame["BDF_FLUX_DERED_CALIB_R"]) < 26)
        bdf_i_mag_cut = (flux2mag(data_frame["BDF_FLUX_DERED_CALIB_I"]) > 18) & (flux2mag(data_frame["BDF_FLUX_DERED_CALIB_I"]) < 24)
        bdf_z_mag_cut = (flux2mag(data_frame["BDF_FLUX_DERED_CALIB_Z"]) > 17) & (flux2mag(data_frame["BDF_FLUX_DERED_CALIB_Z"]) < 26)
        data_frame = data_frame[bdf_i_mag_cut & bdf_r_mag_cut & bdf_z_mag_cut]
        print("length of data_frame cut", len(data_frame))
        without_negative = ((data_frame["AIRMASS_WMEAN_R"] >= 0) &
                            (data_frame["AIRMASS_WMEAN_I"] >= 0) &
                            (data_frame["AIRMASS_WMEAN_Z"] >= 0) &
                            (data_frame["unsheared/snr"] <= 50) &
                            (data_frame["unsheared/snr"] >= -50) &
                            (data_frame["unsheared/size_ratio"] <= 5) &
                            (data_frame["unsheared/size_ratio"] >= -10)
                            )
        data_frame = data_frame[without_negative]
    return data_frame


def main(filename):
    """"""
    lst_label_generator = [
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

    lst_output_generator = [
        "unsheared/mag_r",
        "unsheared/mag_i",
        "unsheared/mag_z",
        "unsheared/snr",
        "unsheared/size_ratio",
        "unsheared/flags",
        "unsheared/T",
        "detection"
    ]

    lst_input_discriminator = [
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
        "EBV_SFD98",
        "unsheared/mag_r",
        "unsheared/mag_i",
        "unsheared/mag_z",
        "unsheared/snr",
        "unsheared/size_ratio",
        "unsheared/flags",
        "unsheared/T",
        "detection"
    ]
    train_data, test_data = load_data(
        path_training_data=filename,
        input_discriminator=lst_input_discriminator,
        input_generator=lst_label_generator,
        output_generator=lst_output_generator
    )
    print(train_data)


if __name__ == "__main__":
    path = os.path.abspath(sys.path[0])
    filepath = path + r"/../../Data"
    main(filename=filepath + r"/analytical_conditional_balrog_data_10000.pkl")
