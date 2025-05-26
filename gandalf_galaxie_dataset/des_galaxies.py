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
    def __init__(self, logg, cfg, kind, lst_split: list = None):
        self.logg = logg
        self.logg.log_info_stream(f"Init GalaxyDataset")
        self.name_yj_transformer = ""
        # self.name_scaler = ""
        self.cfg = cfg
        self.applied_yj_transform = ""
        self.postfix = f"_{kind.upper()}"

        self._set_config_dependent_parameters(kind)

        dataframes = self._load_dataframes()

        cut_result = self._split_special_columns(dataframes)

        transformed_result = self._apply_transformations(cut_result)

        self._build_datasets(transformed_result)

        gc.collect()

    def _set_config_dependent_parameters(self, kind):
        if kind.upper() == "FLOW":
            self.data_set_type = "ODET"
        elif kind.upper() == "CLASSF":
            self.data_set_type = "ALL"
        elif kind.upper() == "RUN":
            if self.cfg["CLASSF_GALAXIES"] is True:
                self.data_set_type = "ALL"
            else:
                self.data_set_type = "ODET"
        else:
            self.logg.log_error(f"{kind} is no valid kind")
            self.logg.log_stream(f"{kind} is no valid kind")
            raise TypeError(f"{kind} is no valid kind")
        self.lum_type = self.cfg[f'LUM_TYPE{self.postfix}']

    def _load_dataframes(self):
        cfg = self.cfg
        if self.postfix == "_RUN":
            file_key = f'FILENAME_{cfg["DATASET_TYPE"].upper()}_DATA_{self.data_set_type}'
            filename = f"{cfg['PATH_DATA']}/{cfg[file_key]}"
            self.logg.log_info_stream(f"Load {filename}  data set")
            with open(filename, 'rb') as file_run:
                df_data = pd.read_pickle(file_run)
            self.logg.log_info_stream(f"shape run dataset: {df_data.shape}")
            fraction_detected = len(df_data[df_data["detected"]==1])/len(df_data[df_data["detected"]==0])
            self.logg.log_info_stream(f"Sample {cfg['NUMBER_SAMPLES']} random data from run data set")
            df_run = df_data.sample(n=cfg['NUMBER_SAMPLES'], replace=True).reset_index(drop=True)
            del df_data
            return df_run
        else:
            df_train = self._load_training_dataframes(kind="train")
            df_valid = self._load_training_dataframes(kind="validation")
            df_test = self._load_training_dataframes(kind="test")
            return (df_train, df_valid, df_test)

    def _load_training_dataframes(self, kind):
        filename = self.cfg[f'FILENAME_{kind.upper()}_DATA_{self.data_set_type}']
        self.logg.log_info_stream(f"Load {filename} {kind.lower()} data set")
        with open(f"{self.cfg['PATH_DATA']}/{filename}", 'rb') as file:
            data_frame = pd.read_pickle(file)
        self.logg.log_info_stream(f"shape {kind.lower()} dataset: {data_frame.shape}")

        return data_frame

    def _split_special_columns(self, dataframes):
        if self.postfix == "_RUN":
            self.logg.log_info_stream(f"Run cut cols: {self.cfg[f'CUT_COLS{self.postfix}']}")
            self.df_run_cut_cols = dataframes[self.cfg[f'CUT_COLS{self.postfix}']]

            self.logg.log_info_stream(f"Run output cols classiefier: {self.cfg[f'OUTPUT_COLS_CLASSF{self.postfix}']}")
            arr_run_all_output_cols = dataframes[self.cfg[f"OUTPUT_COLS_CLASSF{self.postfix}"]].values

            self.logg.log_info_stream(f"Run input cols: {self.cfg[f'INPUT_COLS_{self.lum_type}{self.postfix}']}")
            self.logg.log_info_stream(f"Run output cols: {self.cfg[f'OUTPUT_COLS_{self.lum_type}{self.postfix}']}")
            dataframes = dataframes[
                self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] +
                self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]
                ]
            return dict(
                df_run=dataframes,
                arr_classf_train_output_cols=None,
                arr_classf_valid_output_cols=None,
                arr_classf_test_output_cols=None,
                arr_run_all_output_cols=arr_run_all_output_cols,
            )

        else:
            df_train, df_valid, df_test = dataframes
            if self.postfix == "_CLASSF":
                arr_classf_train_output_cols = df_train[self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]].values
                df_train = df_train[
                    self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] + self.cfg[f"OUTPUT_COLS_{self.lum_type}_FLOW"]
                    ]
                arr_classf_valid_output_cols = df_valid[self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]].values
                df_valid = df_valid[
                    self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] + self.cfg[f"OUTPUT_COLS_{self.lum_type}_FLOW"]
                    ]
                arr_classf_test_output_cols = df_test[self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]].values
                df_test = df_test[
                    self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] + self.cfg[f"OUTPUT_COLS_{self.lum_type}_FLOW"]
                    ]
                return dict(
                    df_train=df_train,
                    df_valid=df_valid,
                    df_test=df_test,
                    arr_classf_train_output_cols=arr_classf_train_output_cols,
                    arr_classf_valid_output_cols=arr_classf_valid_output_cols,
                    arr_classf_test_output_cols=arr_classf_test_output_cols,
                    arr_run_all_output_cols=None,
                )
            self.logg.log_info_stream(f"Training cut cols: {self.cfg[f'CUT_COLS{self.postfix}']}")
            self.df_train_cut_cols = df_train[self.cfg[f'CUT_COLS{self.postfix}']]
            self.df_valid_cut_cols = df_valid[self.cfg[f'CUT_COLS{self.postfix}']]
            self.df_test_cut_cols = df_test[self.cfg[f'CUT_COLS{self.postfix}']]

            self.logg.log_info_stream(f"Training input cols: {self.cfg[f'INPUT_COLS_{self.lum_type}{self.postfix}']}")
            self.logg.log_info_stream(f"Training output cols: {self.cfg[f'OUTPUT_COLS_{self.lum_type}{self.postfix}']}")
            df_train = df_train[
                self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] +
                self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]
                ]
            df_valid = df_valid[
                self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] +
                self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]
                ]
            df_test = df_test[
                self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] +
                self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]
                ]
            return dict(
                df_train=df_train,
                df_valid=df_valid,
                df_test=df_test,
                arr_classf_train_output_cols=None,
                arr_classf_valid_output_cols=None,
                arr_classf_test_output_cols=None,
                arr_run_all_output_cols=None,
            )


    def _apply_transformations(self, cut_result):
        self.applied_scaler = False
        if self.postfix == "_RUN":
            df_run = cut_result["df_run"]
            if self.cfg.get(f"APPLY_YJ_TRANSFORM_CLASSF{self.postfix}", False) is True:
                self.logg.log_info(f"Apply Yeo-Johnson transformation on classifier data")
                self.logg.log_stream(f"Apply Yeo-Johnson transformation on classifier data")
                if self.cfg.get(f"TRANSFORM_COLS{self.postfix}") is None:
                    self.logg.log_info(f"Yeo-Johnson transformation columns: {df_run.keys()}")
                    self.logg.log_stream(f"Yeo-Johnson transformation columns: {df_run.keys()}")
                    df_run, self.dict_pt = self.yj_transform_data(
                        data_frame=df_run,
                        columns=df_run.keys()
                    )
                else:
                    self.logg.log_info(f"Yeo-Johnson transformation columns: {self.cfg[f'TRANSFORM_COLS{self.postfix}']}")
                    self.logg.log_stream(f"Yeo-Johnson transformation columns: {self.cfg[f'TRANSFORM_COLS{self.postfix}']}")
                    df_run, self.dict_pt = self.yj_transform_data(
                        data_frame=df_run,
                        columns=self.cfg[f"TRANSFORM_COLS{self.postfix}"]
                    )
                self.applied_yj_transform = "_YJ"
            # Return all relevant for _build_datasets
            return dict(
                df_run=df_run,
                arr_run_all_output_cols=cut_result["arr_run_all_output_cols"],
                arr_classf_train_output_cols=None,
                arr_classf_valid_output_cols=None,
                arr_classf_test_output_cols=None,
                arr_train_weights=None,
            )
        else:
            df_train = cut_result["df_train"]
            df_valid = cut_result["df_valid"]
            df_test = cut_result["df_test"]
            # YJ transform
            if self.cfg.get(f"APPLY_YJ_TRANSFORM{self.postfix}", False) is True:
                self.logg.log_info_stream(f"Apply YJ transformation")
                if self.cfg.get(f"TRANSFORM_COLS{self.postfix}") is None:
                    self.logg.log_info_stream(f"Transformation columns: {df_train.keys()}")
                    df_train, self.dict_pt = self.yj_transform_data(
                        data_frame=df_train,
                        columns=df_train.keys()
                    )
                    df_valid, self.dict_pt = self.yj_transform_data(
                        data_frame=df_valid,
                        columns=df_valid.keys()
                    )
                    df_test, self.dict_pt = self.yj_transform_data(
                        data_frame=df_test,
                        columns=df_test.keys()
                    )
                else:
                    self.logg.log_info_stream(f"Transformation columns: {self.cfg[f'TRANSFORM_COLS{self.postfix}']}")
                    df_train, self.dict_pt = self.yj_transform_data(
                        data_frame=df_train,
                        columns=self.cfg[f"TRANSFORM_COLS{self.postfix}"]
                    )
                    df_valid, self.dict_pt = self.yj_transform_data(
                        data_frame=df_valid,
                        columns=self.cfg[f"TRANSFORM_COLS{self.postfix}"]
                    )
                    df_test, self.dict_pt = self.yj_transform_data(
                        data_frame=df_test,
                        columns=self.cfg[f"TRANSFORM_COLS{self.postfix}"]
                    )
                self.applied_yj_transform = "_YJ"
            # Only keep input columns for training if CLASSF
            if self.postfix == "_CLASSF":
                df_train = df_train[self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"]]
                df_valid = df_valid[self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"]]
                df_test = df_test[self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"]]
            # Restore output columns for classifier
            arr_classf_train_output_cols = cut_result.get("arr_classf_train_output_cols")
            arr_classf_valid_output_cols = cut_result.get("arr_classf_valid_output_cols")
            arr_classf_test_output_cols = cut_result.get("arr_classf_test_output_cols")
            if arr_classf_train_output_cols is not None:
                df_train[self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]] = arr_classf_train_output_cols
            if arr_classf_valid_output_cols is not None:
                df_valid[self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]] = arr_classf_valid_output_cols
            if arr_classf_test_output_cols is not None:
                df_test[self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]] = arr_classf_test_output_cols
            return dict(
                df_train=df_train,
                df_valid=df_valid,
                df_test=df_test,
                arr_classf_train_output_cols=arr_classf_train_output_cols,
                arr_classf_valid_output_cols=arr_classf_valid_output_cols,
                arr_classf_test_output_cols=arr_classf_test_output_cols,
                arr_run_all_output_cols=None,
            )

    def _build_datasets(self, transformed_result):

        if self.postfix == "_RUN":
            df_run = transformed_result["df_run"]
            arr_run_all_output_cols = transformed_result.get("arr_run_all_output_cols")

            self.logg.log_info_stream(f"Concatenate {self.cfg[f'OUTPUT_COLS_CLASSF{self.postfix}']} and {self.cfg[f'CUT_COLS{self.postfix}']} together again")
            if arr_run_all_output_cols is not None:
                df_run[self.cfg[f"OUTPUT_COLS_CLASSF{self.postfix}"]] = arr_run_all_output_cols
                df_run[self.cfg[f"CUT_COLS{self.postfix}"]] = self.df_run_cut_cols
            self.run_dataset = df_run
            del df_run
        else:
            df_train = transformed_result["df_train"]
            df_valid = transformed_result["df_valid"]
            df_test = transformed_result["df_test"]
            arr_classf_train_output_cols = transformed_result.get("arr_classf_train_output_cols")
            # If classifier, use output cols, else set to zeros
            if arr_classf_train_output_cols is not None:
                self.train_dataset = TensorDataset(
                    torch.tensor(df_train[self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"]].values, dtype=torch.float64),
                    torch.tensor(arr_classf_train_output_cols, dtype=torch.float64)
                )
                self.valid_dataset = TensorDataset(
                    torch.tensor(df_valid[self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"]].values, dtype=torch.float64),
                    torch.tensor(df_valid[self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]].values)
                )
                self.test_dataset = TensorDataset(
                    torch.tensor(df_test[self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"]].values, dtype=torch.float64),
                    torch.tensor(df_test[self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]].values, dtype=torch.float64)
                )
            else:
                self.train_dataset = TensorDataset(
                    torch.tensor(df_train[self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"]].values, dtype=torch.float64),
                    torch.tensor(df_train[self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]].values, dtype=torch.float64)
                )
                self.valid_dataset = TensorDataset(
                    torch.tensor(df_valid[self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"]].values, dtype=torch.float64),
                    torch.tensor(df_valid[self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]].values, dtype=torch.float64)
                )
                self.test_dataset = df_test[
                    self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"]+self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]
                ]
                # self.test_dataset = TensorDataset(
                #     torch.tensor(df_test[self.cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"]].values, dtype=torch.float32),
                #     torch.tensor(df_test[self.cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]].values, dtype=torch.float32)
                # )
            del df_train
            del df_valid
            del df_test

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
    #     self.logg.log_info(f"Use {self.name_scaler} to scale data")
    #     self.logg.log_stream(f"Use {self.name_scaler} to scale data")
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

    def yj_inverse_transform_data(self, data_frame, columns):
        """"""
        self.logg.log_info_stream(f"Use {self.name_yj_transformer} to inverse transform data")
        for col in columns:
            pt = self.dict_pt[f"{col} pt"]
            self.logg.log_info_stream(f"Lambda for {col} is {pt.lambdas_[0]} ")
            self.logg.log_info_stream(f"Mean for {col} is {pt._scaler.mean_[0]} ")
            self.logg.log_info_stream(f"std for {col} is {pt._scaler.scale_[0]} ")
            # value = data_frame[col].values.astype(np.float64)
            # inverse_transformed = pt.inverse_transform(np.array(data_frame[col]).reshape(-1, 1)).ravel()
            data_frame.loc[:, col] = pt.inverse_transform(np.array(data_frame[col]).reshape(-1, 1)).ravel() # inverse_transformed  .astype(np.float32)
        return data_frame

    def yj_transform_data(self, data_frame, columns, dict_pt=None):
        """"""
        if dict_pt is None:
            dict_pt = joblib.load(
                filename=f"{self.cfg['PATH_TRANSFORMERS']}/{self.cfg[f'FILENAME_YJ_TRANSFORMER_{self.data_set_type}']}"
            )
        self.name_yj_transformer = self.cfg[f'FILENAME_YJ_TRANSFORMER_{self.data_set_type}']
        self.logg.log_info_stream(f"Use {self.name_yj_transformer} to transform data")

        data_frame = data_frame.copy()
        for col in columns:
            pt = dict_pt[f"{col} pt"]
            # value = data_frame[col].values.astype(np.float64)
            # transformed = pt.transform(np.array(data_frame[col]).reshape(-1, 1)).ravel()
            data_frame.loc[:, col] = pt.transform(np.array(data_frame[col]).reshape(-1, 1)).ravel()
        return data_frame, dict_pt

    # def yj_transform_data_on_fly(self, data_frame, columns, dict_pt):
    #     """"""
    #     self.logg.log_info(f"Use {self.name_yj_transformer} to transform data")
    #     self.logg.log_stream(f"Use {self.name_yj_transformer} to transform data")
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

    def get_dict_pt(self):
        return self.dict_pt


if __name__ == '__main__':
    pass
