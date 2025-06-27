from torch.utils.data import Dataset, TensorDataset, Subset, DataLoader
from sklearn.preprocessing import PowerTransformer, MaxAbsScaler, MinMaxScaler, StandardScaler
from itertools import accumulate
import numpy as np
import pandas as pd
import sys
import torch
import gc
import joblib
from Handler.helper_functions import compute_weights_from_magnitude


class GalaxyDataset(Dataset):
    def __init__(self, cfg, dataset_logger):
        self.dataset_logger = dataset_logger
        self.dataset_logger.log_info_stream(f"Init GalaxyDataset")
        self.cfg = cfg

        data_frames = self._load_data_frames()

        # cut_result = self._split_special_columns(data_frames)

        self._build_datasets(data_frames)

        gc.collect()

    def _load_data_frames(self):
        if self.cfg["TRAINING"] is True:
            df_train = self._load_data(filename=self.cfg["FILENAME_TRAIN_DATA"])
            df_valid = self._load_data(filename=self.cfg["FILENAME_VALIDATION_DATA"])
            df_test = self._load_data(filename=self.cfg["FILENAME_TEST_DATA"])
            return (df_train, df_valid, df_test)
        else:
            df_data = self._load_data(filename=self.cfg["FILENAME_TEST_DATA"])
            self.dataset_logger.log_info_stream(f"Sample {self.cfg['NUMBER_SAMPLES']} random data from test data set")
            df_run = df_data.sample(n=self.cfg['NUMBER_SAMPLES'], replace=True).reset_index(drop=True)
            del df_data
            return df_run

    def _load_data(self, filename):
        self.dataset_logger.log_info_stream(f"Load {filename} data set")
        with open(f"{self.cfg['PATH_DATA']}/{filename}", 'rb') as file:
            data_frame = pd.read_pickle(file)
        self.dataset_logger.log_info_stream(f"shape dataset: {data_frame.shape}")

        return data_frame

    # def _split_special_columns(self, dataframes):
    #     if self.cfg["TRAINING"] is True:
    #         df_train, df_valid, df_test = dataframes
    #         if self.postfix == "_CLASSF":
    #             arr_classf_train_output_cols = df_train[self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]].values
    #             df_train = df_train[
    #                 self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] + self.cfg[f"OUTPUT_COLS_{self.lum_type}_FLOW"]
    #                 ]
    #             arr_classf_valid_output_cols = df_valid[self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]].values
    #             df_valid = df_valid[
    #                 self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] + self.cfg[f"OUTPUT_COLS_{self.lum_type}_FLOW"]
    #                 ]
    #             arr_classf_test_output_cols = df_test[self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]].values
    #             df_test = df_test[
    #                 self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] + self.cfg[f"OUTPUT_COLS_{self.lum_type}_FLOW"]
    #                 ]
    #             return dict(
    #                 df_train=df_train,
    #                 df_valid=df_valid,
    #                 df_test=df_test,
    #                 arr_classf_train_output_cols=arr_classf_train_output_cols,
    #                 arr_classf_valid_output_cols=arr_classf_valid_output_cols,
    #                 arr_classf_test_output_cols=arr_classf_test_output_cols,
    #                 arr_run_all_output_cols=None,
    #             )
    #         self.dataset_logger.log_info_stream(f"Training cut cols: {self.cfg[f'CUT_COLS{self.postfix}']}")
    #         self.df_train_cut_cols = df_train[self.cfg[f'CUT_COLS{self.postfix}']]
    #         self.df_valid_cut_cols = df_valid[self.cfg[f'CUT_COLS{self.postfix}']]
    #         self.df_test_cut_cols = df_test[self.cfg[f'CUT_COLS{self.postfix}']]
    #
    #         self.dataset_logger.log_info_stream(f"Training input cols: {self.cfg[f'INPUT_COLS_{self.lum_type}{self.postfix}']}")
    #         self.dataset_logger.log_info_stream(f"Training output cols: {self.cfg[f'OUTPUT_COLS_{self.lum_type}{self.postfix}']}")
    #         df_train = df_train[
    #             self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] +
    #             self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]
    #             ]
    #         df_valid = df_valid[
    #             self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] +
    #             self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]
    #             ]
    #         df_test = df_test[
    #             self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] +
    #             self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]
    #             ]
    #         return dict(
    #             df_train=df_train,
    #             df_valid=df_valid,
    #             df_test=df_test,
    #             arr_classf_train_output_cols=None,
    #             arr_classf_valid_output_cols=None,
    #             arr_classf_test_output_cols=None,
    #             arr_run_all_output_cols=None,
    #         )
    #     else:
    #         self.dataset_logger.log_info_stream(f"Run cut cols: {self.cfg[f'CUT_COLS{self.postfix}']}")
    #         self.df_run_cut_cols = dataframes[self.cfg[f'CUT_COLS{self.postfix}']]
    #
    #         self.dataset_logger.log_info_stream(f"Run output cols classiefier: {self.cfg[f'OUTPUT_COLS_CLASSF{self.postfix}']}")
    #         arr_run_all_output_cols = dataframes[self.cfg[f"OUTPUT_COLS_CLASSF{self.postfix}"]].values
    #
    #         self.dataset_logger.log_info_stream(f"Run input cols: {self.cfg[f'INPUT_COLS_{self.lum_type}{self.postfix}']}")
    #         self.dataset_logger.log_info_stream(f"Run output cols: {self.cfg[f'OUTPUT_COLS_{self.lum_type}{self.postfix}']}")
    #         dataframes = dataframes[
    #             self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] +
    #             self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]
    #             ]
    #         return dict(
    #             df_run=dataframes,
    #             arr_classf_train_output_cols=None,
    #             arr_classf_valid_output_cols=None,
    #             arr_classf_test_output_cols=None,
    #             arr_run_all_output_cols=arr_run_all_output_cols,
    #         )

    def _build_datasets(self, data_frames):
        if self.cfg["TRAINING"] is True:
            df_train = data_frames[0]
            df_valid = data_frames[1]
            df_test = data_frames[2]
            if self.cfg["TRAINING_TYPE"] == "classifier":
                self.train_dataset = df_train
                self.valid_dataset = df_valid
                self.test_dataset = df_test
                # self.train_dataset = TensorDataset(
                #     torch.tensor(df_train[self.cfg[f"INPUT_COLS"]].values, dtype=torch.float32),
                #     torch.tensor(df_train[self.cfg[f"OUTPUT_COLS"]].values, dtype=torch.float32)
                # )
                # self.valid_dataset = TensorDataset(
                #     torch.tensor(df_valid[self.cfg[f"INPUT_COLS"]].values, dtype=torch.float32),
                #     torch.tensor(df_valid[self.cfg[f"OUTPUT_COLS"]].values)
                # )
                # self.test_dataset = TensorDataset(
                #     torch.tensor(df_test[self.cfg[f"INPUT_COLS"]].values, dtype=torch.float32),
                #     torch.tensor(df_test[self.cfg[f"OUTPUT_COLS"]].values, dtype=torch.float32)
                # )
            else:
                self.train_dataset = TensorDataset(
                    torch.tensor(df_train[self.cfg[f"INPUT_COLS"]].values, dtype=torch.float32),
                    torch.tensor(df_train[self.cfg[f"OUTPUT_COLS"]].values, dtype=torch.float32)
                )
                self.valid_dataset = TensorDataset(
                    torch.tensor(df_valid[self.cfg[f"INPUT_COLS"]].values, dtype=torch.float32),
                    torch.tensor(df_valid[self.cfg[f"OUTPUT_COLS"]].values, dtype=torch.float32)
                )
                self.test_dataset = TensorDataset(
                    torch.tensor(df_test[self.cfg[f"INPUT_COLS"]].values, dtype=torch.float32),
                    torch.tensor(df_test[self.cfg[f"OUTPUT_COLS"]].values, dtype=torch.float32)
                )
                # self.test_dataset = df_test[
                #     self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"]+self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]
                # ]
                # self.test_dataset = TensorDataset(
                #     torch.tensor(df_test[self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"]].values, dtype=torch.float32),
                #     torch.tensor(df_test[self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]].values, dtype=torch.float32)
                # )
            del df_train
            del df_valid
            del df_test
        else:
            self.test_dataset = data_frames

    def __len__(self):
        return len(self.train_dataset + self.valid_dataset + self.test_dataset)

    def __getitem__(self, idx):
        sample = self.test_dataset[idx]
        return sample

    # def sample_random_data_from_dataframe(self, dataframe):
    #     num_samples = self.cfg['NUMBER_SAMPLES']
    #     sampled_df = dataframe.sample(n=num_samples, replace=True)
    #     sampled_df.reset_index(drop=True, inplace=True)
    #     return sampled_df

    # @staticmethod
    # def custom_random_split(dataset, lst_split, df_data, cfg, indices=None):
    #     """
    #     Splits the dataset into non-overlapping new datasets of given proportions.
    #     Returns both the datasets and the indices.
    #
    #     Arguments:
    #     - dataset: The entire dataset to split.
    #     - lst_split: List of proportions of the dataset to split.
    #     - indices: Indices to be used for the split. If None, they will be created.
    #
    #     Returns:
    #     - datasets: Tuple of split datasets in order (train, val, test).
    #     - indices: Indices used for the splits.
    #     """
    #     # Wandle die Prozentsätze in absolute Längen um
    #     total_length = len(dataset)
    #     lengths = [int(total_length * proportion) for proportion in lst_split]
    #
    #     lengths[0] = total_length - sum(lengths[1:])
    #
    #     if sum(lengths) != total_length:
    #         raise ValueError("Sum of input proportions does not equal 1!")
    #
    #     if indices is None:
    #         indices = torch.randperm(total_length)
    #     accumulated_lengths = list(accumulate(lengths))
    #     train_indices = indices[:accumulated_lengths[0]]
    #     val_indices = indices[accumulated_lengths[0]:accumulated_lengths[1]]
    #     test_indices = indices[accumulated_lengths[1]:]
    #
    #     train_dataset = Subset(dataset, train_indices)
    #     val_dataset = Subset(dataset, val_indices)
    #     test_dataset = Subset(dataset, test_indices)
    #
    #     dict_indices = {
    #         "train": train_indices,
    #         "val": val_indices,
    #         "test": test_indices
    #     }
    #
    #     return train_dataset, val_dataset, test_dataset, dict_indices

    # def scale_data(self, data_frame, scaler=None):
    #     """"""
    #     if scaler is None:
    #         self.name_scaler = self.cfg[f'FILENAME_SCALER_{self.data_set_type}_{self.lum_type}{self.applied_yj_transform}']
    #         scaler = joblib.load(
    #             filename=f"{self.cfg['PATH_TRANSFORMERS']}/{self.name_scaler}"
    #         )
    #     self.dataset_logger.log_info(f"Use {self.name_scaler} to scale data")
    #     self.dataset_logger.log_stream(f"Use {self.name_scaler} to scale data")
    #     data_frame_scaled = None
    #     if scaler is not None:
    #         scaled = scaler.transform(data_frame)
    #         data_frame_scaled = pd.DataFrame(scaled, columns=data_frame.columns)
    #     return data_frame_scaled, scaler
    #
    # def inverse_scale_data(self, data_frame):
    #     """"""
    #     print(f"Use {self.name_scaler} to inverse scale data")
    #     data_frame = pd.DataFrame(self.scaler.inverse_transform(data_frame), columns=data_frame.keys())
    #     return data_frame

    # def scale_data_on_fly(self, data_frame, scaler):
    #     """"""
    #     print(f"Use {self.name_scaler} to scale data")
    #     data_frame_scaled = None
    #     if scaler is not None:
    #         scaled = scaler.transform(data_frame)
    #         data_frame_scaled = pd.DataFrame(scaled, columns=data_frame.columns)
    #     return data_frame_scaled

    # def yj_inverse_transform_data(self, data_frame, columns):
    #     """"""
    #     self.dataset_logger.log_info_stream(f"Use {self.name_yj_transformer} to inverse transform data")
    #     for col in columns:
    #         pt = self.dict_pt[f"{col} pt"]
    #         self.dataset_logger.log_debug(f"Lambda for {col} is {pt.lambdas_[0]} ")
    #         self.dataset_logger.log_debug(f"Mean for {col} is {pt._scaler.mean_[0]} ")
    #         self.dataset_logger.log_debug(f"std for {col} is {pt._scaler.scale_[0]} ")
    #         value = data_frame[col].values
    #         # clipped_value = np.clip(value, a_min=None, a_max=abs(1 / pt.lambdas_[0]))
    #         data_frame.loc[:, col] = pt.inverse_transform(np.array(value).reshape(-1, 1)).ravel()
    #     return data_frame
    #
    # def yj_transform_data(self, data_frame, columns, dict_pt=None):
    #     """"""
    #     if dict_pt is None:
    #         dict_pt = joblib.load(
    #             filename=f"{self.cfg['PATH_TRANSFORMERS']}/{self.cfg[f'FILENAME_YJ_TRANSFORMER_{self.data_set_type}']}"
    #         )
    #     self.name_yj_transformer = self.cfg[f'FILENAME_YJ_TRANSFORMER_{self.data_set_type}']
    #     self.dataset_logger.log_info_stream(f"Use {self.name_yj_transformer} to transform data")
    #
    #     data_frame = data_frame.copy()
    #     for col in columns:
    #         pt = dict_pt[f"{col} pt"]
    #         # value = data_frame[col].values.astype(np.float64)
    #         # transformed = pt.transform(np.array(data_frame[col]).reshape(-1, 1)).ravel()
    #         data_frame.loc[:, col] = pt.transform(np.array(data_frame[col]).reshape(-1, 1)).ravel()
    #     return data_frame, dict_pt

    # def yj_transform_data_on_fly(self, data_frame, columns, dict_pt):
    #     """"""
    #     self.dataset_logger.log_info(f"Use {self.name_yj_transformer} to transform data")
    #     self.dataset_logger.log_stream(f"Use {self.name_yj_transformer} to transform data")
    #     data_frame = data_frame.copy()
    #     for col in columns:
    #         pt = dict_pt[f"{col} pt"]
    #         transformed = pt.transform(np.array(data_frame[col]).reshape(-1, 1)).ravel()
    #         data_frame.loc[:, col] = transformed.astype(np.float32)
    #         # data_frame.loc[:, col] = pt.transform(np.array(data_frame[col]).reshape(-1, 1))
    #     return data_frame

    # @staticmethod
    # def unsheared_object_cuts(data_frame):
    #     """"""
    #     print("Apply unsheared object cuts")
    #     cuts = (data_frame["unsheared/extended_class_sof"] >= 0) & (data_frame["unsheared/flags_gold"] < 2)
    #     data_frame = data_frame[cuts]
    #     print('Length of catalog after applying unsheared object cuts: {}'.format(len(data_frame)))
    #     return data_frame
    #
    # @staticmethod
    # def flag_cuts(data_frame):
    #     """"""
    #     print("Apply flag cuts")
    #     cuts = (data_frame["match_flag_1.5_asec"] < 2) & \
    #            (data_frame["flags_foreground"] == 0) & \
    #            (data_frame["flags_badregions"] < 2) & \
    #            (data_frame["flags_footprint"] == 1)
    #     data_frame = data_frame[cuts]
    #     print('Length of catalog after applying flag cuts: {}'.format(len(data_frame)))
    #     return data_frame
    #
    # @staticmethod
    # def airmass_cut(data_frame):
    #     """"""
    #     print('Cut mcal_detect_df_survey catalog so that AIRMASS_WMEAN_R is not null')
    #     data_frame = data_frame[pd.notnull(data_frame["AIRMASS_WMEAN_R"])]
    #     print('Length of catalog after applying AIRMASS_WMEAN_R cuts: {}'.format(len(data_frame)))
    #     return data_frame
    #
    # @staticmethod
    # def unsheared_mag_cut(data_frame):
    #     """"""
    #     print("Apply unsheared mag cuts")
    #     cuts = (
    #             (18 < data_frame["unsheared/mag_i"]) &
    #             (data_frame["unsheared/mag_i"] < 23.5) &
    #             (15 < data_frame["unsheared/mag_r"]) &
    #             (data_frame["unsheared/mag_r"] < 26) &
    #             (15 < data_frame["unsheared/mag_z"]) &
    #             (data_frame["unsheared/mag_z"] < 26) &
    #             (-1.5 < data_frame["unsheared/mag_r"] - data_frame["unsheared/mag_i"]) &
    #             (data_frame["unsheared/mag_r"] - data_frame["unsheared/mag_i"] < 4) &
    #             (-4 < data_frame["unsheared/mag_z"] - data_frame["unsheared/mag_i"]) &
    #             (data_frame["unsheared/mag_z"] - data_frame["unsheared/mag_i"] < 1.5)
    #     )
    #     data_frame = data_frame[cuts]
    #     print('Length of catalog after applying unsheared mag cuts: {}'.format(len(data_frame)))
    #     return data_frame
    #
    # @staticmethod
    # def unsheared_shear_cuts(data_frame):
    #     """"""
    #     print("Apply unsheared shear cuts")
    #     cuts = (
    #             (10 < data_frame["unsheared/snr"]) &
    #             (data_frame["unsheared/snr"] < 1000) &
    #             (0.5 < data_frame["unsheared/size_ratio"]) &
    #             (data_frame["unsheared/T"] < 10)
    #     )
    #     data_frame = data_frame[cuts]
    #     data_frame = data_frame[~((2 < data_frame["unsheared/T"]) & (data_frame["unsheared/snr"] < 30))]
    #     print('Length of catalog after applying unsheared shear cuts: {}'.format(len(data_frame)))
    #     return data_frame
    #
    # @staticmethod
    # def remove_outliers(data_frame):
    #     print("Remove outliers...")
    #     not_outliers = (
    #         ~((data_frame["unsheared/mag_r"] <= 7.812946519176684) |
    #           (data_frame["unsheared/mag_r"] >= 37.5) |
    #           (data_frame["unsheared/mag_i"] <= 7.342380502886332) |
    #           (data_frame["unsheared/mag_i"] >= 37.5) |
    #           (data_frame["unsheared/mag_z"] <= 7.69900077218967) |
    #           (data_frame["unsheared/mag_z"] >= 37.5) |
    #           (data_frame["unsheared/snr"] <= 0) |
    #           (data_frame["unsheared/snr"] >= 200) |
    #           (data_frame["unsheared/size_ratio"] <= -1) |
    #           (data_frame["unsheared/size_ratio"] >= 4) |
    #           (data_frame["unsheared/T"] <= -1) |
    #           (data_frame["unsheared/T"] >= 4) |
    #           (data_frame["unsheared/weight"] <= 10.0) |
    #           (data_frame["unsheared/weight"] >= 77.58102207403836))
    #     )
    #     data_frame = data_frame[not_outliers]
    #     print('Length of catalog after removing outliers: {}'.format(len(data_frame)))
    #     return data_frame

    # def get_dict_pt(self):
    #     return self.dict_pt


if __name__ == '__main__':
    pass
