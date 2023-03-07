from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from Handler.helper_functions import flux2mag
from chainconsumer import ChainConsumer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
import sys

sys.path.append(os.path.dirname(__file__))


def load_data(path_training_data, input_flow, output_flow, selected_scaler):
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

    df_training_data = df_data[input_flow+output_flow]

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

    arr_label_flow = np.array(df_training_data_scaled[input_flow])
    arr_output_flow = np.array(df_training_data_scaled[output_flow])

    dict_training_data = {
        f"label flow in order {input_flow}": arr_label_flow[:int(len(arr_label_flow) * .6)],
        f"output flow in order {output_flow}": arr_output_flow[:int(len(arr_output_flow) * .6)],
        "columns label flow": input_flow,
        "columns output flow": output_flow,
        "scaler": scaler
    }

    dict_validation_data = {
        f"label flow in order {input_flow}": arr_label_flow[int(len(arr_label_flow) * .6):int(len(arr_label_flow) * .8)],
        f"output flow in order {output_flow}": arr_output_flow[int(len(arr_label_flow) * .6):int(len(arr_output_flow) * .8)],
        "columns label flow": input_flow,
        "columns output flow": output_flow,
        "scaler": scaler
    }

    dict_test_data = {
        f"label flow in order {input_flow}": arr_label_flow[int(len(arr_label_flow) * .8):],
        f"output flow in order {output_flow}": arr_output_flow[int(len(arr_output_flow) * .8):],
        "columns label flow": input_flow,
        "columns output flow": output_flow,
        "scaler": scaler
    }

    return dict_training_data, dict_validation_data, dict_test_data


def load_data_kidz(path_training_data, input_flow, output_flow, selected_scaler, apply_cuts):
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

    print(f'Length of catalog: {len(df_data)}')

    if apply_cuts is True:
        print(f"Apply some star cuts")
        df_data = data_cuts_kids(df_data)
        print(f'Length of catalog now: {len(df_data)}')

    # plot_true_dataset(df_data)

    # Use only necessary columns
    df_training_data = df_data[input_flow+output_flow]

    # Define the scaler
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

    # convert to float 32 because of the normalizing flow network
    df_training_data = pd.DataFrame(np.float32(df_training_data), columns=df_training_data.columns)

    # Use scaler to scale the date for the network
    if scaler is not None:
        scaler.fit(df_training_data)
        scaled = scaler.transform(df_training_data)
        df_training_data_scaled = pd.DataFrame(scaled, columns=df_training_data.columns)
    else:
        df_training_data_scaled = df_training_data
    df_training_data_scaled = df_training_data_scaled.sample(frac=1, random_state=1)

    # Define array that can easy be used as the input Tensor
    arr_label_flow = np.array(df_training_data_scaled[input_flow])
    arr_output_flow = np.array(df_training_data_scaled[output_flow])

    # Define dictionaries for the training
    dict_training_data = {
        f"label flow in order {input_flow}": arr_label_flow[:int(len(arr_label_flow) * .6)],
        f"output flow in order {output_flow}": arr_output_flow[:int(len(arr_output_flow) * .6)],
        "columns label flow": input_flow,
        "columns output flow": output_flow,
        "scaler": scaler
    }

    dict_validation_data = {
        f"label flow in order {input_flow}": arr_label_flow[int(len(arr_label_flow) * .6):int(len(arr_label_flow) * .8)],
        f"output flow in order {output_flow}": arr_output_flow[int(len(arr_label_flow) * .6):int(len(arr_output_flow) * .8)],
        "columns label flow": input_flow,
        "columns output flow": output_flow,
        "scaler": scaler
    }

    dict_test_data = {
        f"label flow in order {input_flow}": arr_label_flow[int(len(arr_label_flow) * .8):],
        f"output flow in order {output_flow}": arr_output_flow[int(len(arr_output_flow) * .8):],
        "columns label flow": input_flow,
        "columns output flow": output_flow,
        "scaler": scaler
    }

    return dict_training_data, dict_validation_data, dict_test_data


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


def data_cuts_kids(data_frame):
    """"""
    input_cut = (
        (data_frame["axis_ratio_input"] != -999) #&
        # (data_frame["luptize_u"] < 25) &
        # (data_frame["luptize_g"] < 25) &
        # (data_frame["luptize_r"] < 25) &
        # (data_frame["luptize_i"] < 25) &
        # (data_frame["luptize_Z"] < 25) &
        # (data_frame["luptize_Y"] < 25) &
        # (data_frame["luptize_J"] < 25) &
        # (data_frame["luptize_H"] < 25) &
        # (data_frame["luptize_Ks"] < 25)
    )
    data_frame = data_frame[input_cut]
    return data_frame


def plot_true_dataset(data_frame):
    df_true_measured = pd.DataFrame({
        "luptize_u": np.array(data_frame["luptize_u"]),
        "luptize_g": np.array(data_frame["luptize_g"]),
        "luptize_r": np.array(data_frame["luptize_r"]),
        "luptize_i": np.array(data_frame["luptize_i"]),
        "luptize_Z": np.array(data_frame["luptize_Z"]),
        "luptize_Y": np.array(data_frame["luptize_Y"]),
        "luptize_J": np.array(data_frame["luptize_J"]),
        "luptize_H": np.array(data_frame["luptize_H"]),
        "luptize_Ks": np.array(data_frame["luptize_Ks"])
    })

    parameter = [
        "luptize u",
        "luptize g",
        "luptize r",
        "luptize i",
        "luptize Z",
        "luptize Y",
        "luptize J",
        "luptize H",
        "luptize Ks",
    ]
    arr_true = df_true_measured.to_numpy()
    chainchat = ChainConsumer()
    chainchat.add_chain(arr_true, parameters=parameter, name="true observed properties: chat")
    chainchat.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
    chainchat.plotter.plot(
        # filename=f'{self.path_chain_plot}/Loaded_data.png',
        figsize="page",
        extents={
            "luptize u": (17, 30),
            "luptize g": (17, 30),
            "luptize r": (17, 30),
            "luptize i": (17, 30),
            "luptize Z": (17, 30),
            "luptize Y": (17, 30),
            "luptize J": (17, 30),
            "luptize H": (17, 30),
            "luptize Ks": (17, 30)
        }
    )
    plt.show()
    plt.clf()


def main(filename):
    """"""
    lst_input = [
        "axis_ratio_input",
        "Re_input",
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
        "InputBeta_Ks",
        "rmsAW_u",
        "rmsAW_g",
        "rmsAW_r",
        "rmsAW_i",
        "rms_Z",
        "rms_Y",
        "rms_J",
        "rms_H",
        "rms_Ks"
    ]

    lst_output = [
        "FLUX_GAAP_u",
        "FLUX_GAAP_g",
        "FLUX_GAAP_r",
        "FLUX_GAAP_i",
        "FLUX_GAAP_Z",
        "FLUX_GAAP_Y",
        "FLUX_GAAP_J",
        "FLUX_GAAP_H",
        "FLUX_GAAP_Ks",
        "FLUXERR_GAAP_u",
        "FLUXERR_GAAP_g",
        "FLUXERR_GAAP_r",
        "FLUXERR_GAAP_i",
        "FLUXERR_GAAP_Z",
        "FLUXERR_GAAP_Y",
        "FLUXERR_GAAP_J",
        "FLUXERR_GAAP_H",
        "FLUXERR_GAAP_Ks",
        "FLUX_AUTO",
        "FLUXERR_AUTO",
    ]

    train_data, valid_data, test_data = load_data_kidz(
        path_training_data=filename,
        input_flow=lst_input,
        output_flow=lst_output,
        selected_scaler="MaxAbsScaler",
        apply_cuts=True
    )


if __name__ == "__main__":
    path = os.path.abspath(sys.path[0])
    filepath = path + r"/../Data"
    main(filename=filepath + r"/kids_training_catalog.pkl")
