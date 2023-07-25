from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from Handler.helper_functions import replace_and_transform_data
from Handler.plot_functions import plot_chain
from Handler.cut_functions import unsheared_object_cuts, flag_cuts
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
        size_training_dataset,
        size_validation_dataset,
        size_test_dataset,
        apply_cuts,
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

    if apply_cuts is True:
        df_training_data = unsheared_object_cuts(df_training_data)
        df_training_data = flag_cuts(df_training_data)

    df_training_data, dict_pt = replace_and_transform_data(
        data_frame=df_training_data,
        columns=lst_replace_transform_cols,
        replace_value=lst_replace_values
    )

    if reproducible is True:
        df_training_data = df_training_data.sample(frac=1, random_state=42)

    else:
        df_training_data = df_training_data.sample(frac=1)

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
            f"{path_output}/df_validation_data_{len(dict_validation_data[f'data frame valid data'])}_run_{run}.pkl",
            "wb") as f:
        pickle.dump(dict_validation_data, f, protocol=2)
    with open(f"{path_output}/df_test_data_{len(dict_test_data[f'data frame test data'])}_run_{run}.pkl",
              "wb") as f:
        pickle.dump(dict_test_data, f, protocol=2)

    return dict_training_data, dict_validation_data, dict_test_data


if __name__ == "__main__":
    path = os.path.abspath(sys.path[0])
    filepath = path + r"/../Data"
