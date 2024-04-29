from torch.utils.data import Dataset, TensorDataset, Subset, DataLoader
from sklearn.preprocessing import PowerTransformer, MaxAbsScaler, MinMaxScaler, StandardScaler
from itertools import accumulate
import numpy as np
import pandas as pd
import torch
import gc
import joblib
from Handler.helper_functions import calculate_percentage_of_outliers


class GalaxyDataset(Dataset):
    def __init__(self, cfg, kind, lst_split: list = None):
        arr_classf_train_output_cols = None
        arr_classf_valid_output_cols = None
        arr_classf_test_output_cols = None
        arr_run_all_output_cols = None
        self.name_yj_transformer = ""
        self.name_scaler = ""
        self.cfg = cfg
        if kind == "flow_training":
            self.postfix = "_FLOW"
            self.data_set_type = "ODET"
        elif kind == "classifier_training":
            self.postfix = "_CLASSF"
            self.data_set_type = "ALL"
        elif kind == "run_gandalf":
            self.postfix = "_RUN"
            if self.cfg["CLASSF_GALAXIES"] is True:
                self.data_set_type = "ALL"
            else:
                self.data_set_type = "ODET"
        else:
            raise TypeError(f"{kind} is no valid kind")
        self.lum_type = self.cfg[f'LUM_TYPE{self.postfix}']

        if self.postfix == "_RUN":
            if self.cfg["DATASET_TYPE"] == "All":
                filename = f"{cfg[f'PATH_DATA']}/{cfg[f'FILENAME_DATA_{self.data_set_type}']}"
            elif self.cfg["DATASET_TYPE"] == "Train":
                filename = f"{cfg[f'PATH_DATA']}/{cfg[f'FILENAME_TRAIN_DATA_{self.data_set_type}']}"
            elif self.cfg["DATASET_TYPE"] == "Valid":
                filename = f"{cfg[f'PATH_DATA']}/{cfg[f'FILENAME_VALIDATION_DATA_{self.data_set_type}']}"
            else:
                filename = f"{cfg[f'PATH_DATA']}/{cfg[f'FILENAME_TEST_DATA_{self.data_set_type}']}"
            print(f"Load {filename}  data set")
            with open(f"{filename}", 'rb') as file_run:
                df_data = pd.read_pickle(file_run)

                if cfg['SPATIAL_TEST'] is True:
                    df_spatial = pd.read_pickle(f"{cfg['PATH_DATA']}/{cfg['FILENAME_SPATIAL']}")
                    selected_row = df_spatial.iloc[cfg["SPATIAL_NUMBER"]]

                    print(f"Selected  spatial row: {selected_row}")

                    for col in cfg['SPATIAL_COLS']:
                        df_data[col] = selected_row[col]

                    print(df_data[cfg['SPATIAL_COLS']])

            file_run.close()
            print(f"shape run dataset: {df_data.shape}")
            print(f"Sample {cfg['NUMBER_SAMPLES']} random data from run data set")
            if cfg['NUMBER_SAMPLES'] == -1:
                cfg['NUMBER_SAMPLES'] = len(df_data)
            #     df_run = df_data.copy()
            # else:
            df_run = self.sample_random_data_from_dataframe(df_data)
            del df_data
        else:
            print(f"Load {cfg[f'FILENAME_TRAIN_DATA_{self.data_set_type}']} train data set")
            with open(f"{cfg[f'PATH_DATA']}/{cfg[f'FILENAME_TRAIN_DATA_{self.data_set_type}']}", 'rb') as file_train:
                df_train = pd.read_pickle(file_train)
            file_train.close()
            print(f"shape train dataset: {df_train.shape}")
            print(f"Load {cfg[f'FILENAME_VALIDATION_DATA_{self.data_set_type}']} validation data set")
            with open(f"{cfg[f'PATH_DATA']}/{cfg[f'FILENAME_VALIDATION_DATA_{self.data_set_type}']}", 'rb') as file_valid:
                df_valid = pd.read_pickle(file_valid)
            file_valid.close()
            print(f"shape valid dataset: {df_valid.shape}")
            print(f"Load {cfg[f'FILENAME_TEST_DATA_{self.data_set_type}']} test data set")
            with open(f"{cfg[f'PATH_DATA']}/{cfg[f'FILENAME_TEST_DATA_{self.data_set_type}']}", 'rb') as file_test:
                df_test = pd.read_pickle(file_test)
            file_test.close()
            print(f"shape test dataset: {df_test.shape}")
        if self.postfix == "_RUN":
            self.df_run_cut_cols = df_run[cfg[f'CUT_COLS{self.postfix}']]
        else:
            self.df_train_cut_cols = df_train[cfg[f'CUT_COLS{self.postfix}']]
            self.df_valid_cut_cols = df_valid[cfg[f'CUT_COLS{self.postfix}']]
            self.df_test_cut_cols = df_test[cfg[f'CUT_COLS{self.postfix}']]

        if self.postfix == "_CLASSF":
            arr_classf_train_output_cols = df_train[cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]].values
            df_train = df_train[
                cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] + cfg[f"OUTPUT_COLS_{self.lum_type}_FLOW"]
            ]

            arr_classf_valid_output_cols = df_valid[cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]].values
            df_valid = df_valid[
                cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] + cfg[f"OUTPUT_COLS_{self.lum_type}_FLOW"]
            ]

            arr_classf_test_output_cols = df_test[cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]].values
            df_test = df_test[
                cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] + cfg[f"OUTPUT_COLS_{self.lum_type}_FLOW"]
            ]
        elif self.postfix == "_RUN":
            arr_run_all_output_cols = df_run[cfg[f"OUTPUT_COLS_CLASSF{self.postfix}"]].values
            df_run = df_run[cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] +
                            cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]
                            ]
        else:
            df_train = df_train[
                cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] +
                cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]
                ]
            df_valid = df_valid[
                cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] +
                cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]
                ]
            df_test = df_test[
                cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"] +
                cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]
                ]

        if self.postfix == "_RUN":
            if cfg[f"APPLY_YJ_TRANSFORM_CLASSF{self.postfix}"] is True:
                if cfg[f"TRANSFORM_COLS{self.postfix}"] is None:
                    df_run, self.dict_pt = self.yj_transform_data(
                        data_frame=df_run,
                        columns=df_run.keys()
                    )
                else:
                    df_run, self.dict_pt = self.yj_transform_data(
                        data_frame=df_run,
                        columns=cfg[f"TRANSFORM_COLS{self.postfix}"]
                    )
                self.applied_yj_transform = "_YJ"
        else:
            if cfg[f"APPLY_YJ_TRANSFORM{self.postfix}"] is True:
                if cfg[f"TRANSFORM_COLS{self.postfix}"] is None:
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
                    df_train, self.dict_pt = self.yj_transform_data(
                        data_frame=df_train,
                        columns=cfg[f"TRANSFORM_COLS{self.postfix}"]
                    )
                    df_valid, self.dict_pt = self.yj_transform_data(
                        data_frame=df_valid,
                        columns=cfg[f"TRANSFORM_COLS{self.postfix}"]
                    )
                    df_test, self.dict_pt = self.yj_transform_data(
                        data_frame=df_test,
                        columns=cfg[f"TRANSFORM_COLS{self.postfix}"]
                    )
                self.applied_yj_transform = "_YJ"

        self.applied_scaler = False
        if self.postfix == "_RUN":
            if cfg[f"APPLY_SCALER_CLASSF{self.postfix}"] is True:
                df_run, self.scaler = self.scale_data(data_frame=df_run)
                self.applied_scaler = True
        else:
            if cfg[f"APPLY_SCALER{self.postfix}"] is True:
                df_train, self.scaler = self.scale_data(data_frame=df_train)
                df_valid, self.scaler = self.scale_data(data_frame=df_valid)
                df_test, self.scaler = self.scale_data(data_frame=df_test)
                self.applied_scaler = True

        # Test #########################################################################################################
        # if cfg[f"APPLY_SCALER_FLOW{self.postfix}"] is True:
        #     df_run_is = self.inverse_scale_data(df_balrog=df_run)
        #     print(df_run_is)
        # if cfg[f"APPLY_YJ_TRANSFORM_FLOW{self.postfix}"] is True:
        #     df_run_yj = self.yj_inverse_transform_data(
        #         df_balrog=df_run_is,
        #         columns=df_run_is.keys()
        #     )
        #     print(df_run_yj)
        # Test #########################################################################################################
        if self.postfix == "_CLASSF":
            df_train = df_train[cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"]]
            df_valid = df_valid[cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"]]
            df_test = df_test[cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"]]

        if arr_classf_train_output_cols is not None:
            df_train[cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]] = arr_classf_train_output_cols
        if arr_classf_valid_output_cols is not None:
            df_valid[cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]] = arr_classf_valid_output_cols
        if arr_classf_test_output_cols is not None:
            df_test[cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]] = arr_classf_test_output_cols

        if arr_run_all_output_cols is not None:
            df_run[cfg[f"OUTPUT_COLS_CLASSF{self.postfix}"]] = arr_run_all_output_cols
            df_run[cfg[f"CUT_COLS{self.postfix}"]] = self.df_run_cut_cols
        if self.postfix == "_RUN":
            self.run_dataset = df_run

            # TensorDataset(
            #     torch.tensor(df_run[cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"]].values),
            #     torch.tensor(df_run[cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]].values),
            #     torch.tensor(df_run[cfg[f"OUTPUT_COLS_CLASSF{self.postfix}"]].values),
            #     torch.tensor(df_run[cfg[f"CUT_COLS{self.postfix}"]].values)
            #
            # )
        else:
            self.train_dataset = TensorDataset(
                torch.tensor(df_train[cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"]].values),
                torch.tensor(df_train[cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]].values)
            )
            self.valid_dataset = TensorDataset(
                torch.tensor(df_valid[cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"]].values),
                torch.tensor(df_valid[cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]].values)
            )
            self.test_dataset = TensorDataset(
                torch.tensor(df_test[cfg[f"INPUT_COLS_{self.lum_type}{self.postfix}"]].values),
                torch.tensor(df_test[cfg[f"OUTPUT_COLS_{self.lum_type}{self.postfix}"]].values)
            )

        if self.postfix == "_RUN":
            del df_run
        else:
            del df_train
            del df_valid
            del df_test
        gc.collect()

    def __len__(self):
        return len(self.train_dataset + self.valid_dataset + self.test_dataset)

    def __getitem__(self, idx):
        exit()
        sample = self.test_dataset[idx]
        return sample

    def sample_random_data_from_dataframe(self, dataframe):
        num_samples = self.cfg['NUMBER_SAMPLES']
        sampled_df = dataframe.sample(n=num_samples, replace=True)
        sampled_df.reset_index(drop=True, inplace=True)
        return sampled_df

    @staticmethod
    def custom_random_split(dataset, lst_split, df_data, cfg, indices=None):
        """
        Splits the dataset into non-overlapping new datasets of given proportions.
        Returns both the datasets and the indices.

        Arguments:
        - dataset: The entire dataset to split.
        - lst_split: List of proportions of the dataset to split.
        - indices: Indices to be used for the split. If None, they will be created.

        Returns:
        - datasets: Tuple of split datasets in order (train, val, test).
        - indices: Indices used for the splits.
        """
        # Wandle die Prozentsätze in absolute Längen um
        total_length = len(dataset)
        lengths = [int(total_length * proportion) for proportion in lst_split]

        lengths[0] = total_length - sum(lengths[1:])

        if sum(lengths) != total_length:
            raise ValueError("Sum of input proportions does not equal 1!")

        if indices is None:
            indices = torch.randperm(total_length)
        accumulated_lengths = list(accumulate(lengths))
        train_indices = indices[:accumulated_lengths[0]]
        val_indices = indices[accumulated_lengths[0]:accumulated_lengths[1]]
        test_indices = indices[accumulated_lengths[1]:]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        dict_indices = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices
        }

        return train_dataset, val_dataset, test_dataset, dict_indices

    def scale_data(self, data_frame):
        """"""
        self.name_scaler = self.cfg[f'FILENAME_SCALER_{self.data_set_type}_{self.lum_type}{self.applied_yj_transform}']

        scaler = joblib.load(
            filename=f"{self.cfg['PATH_TRANSFORMERS']}/{self.name_scaler}"
        )

        print(f"Use {self.name_scaler} to scale data")
        data_frame_scaled = None
        if scaler is not None:
            scaled = scaler.transform(data_frame)
            data_frame_scaled = pd.DataFrame(scaled, columns=data_frame.columns)
        return data_frame_scaled, scaler

    def inverse_scale_data(self, data_frame):
        """"""
        print(f"Use {self.name_scaler} to inverse scale data")
        data_frame = pd.DataFrame(self.scaler.inverse_transform(data_frame), columns=data_frame.keys())
        return data_frame

    def scale_data_on_fly(self, data_frame, scaler):
        """"""
        print(f"Use {self.name_scaler} to scale data")
        data_frame_scaled = None
        if scaler is not None:
            scaled = scaler.transform(data_frame)
            data_frame_scaled = pd.DataFrame(scaled, columns=data_frame.columns)
        return data_frame_scaled

    def yj_inverse_transform_data(self, data_frame, columns):
        """"""
        print(f"Use {self.name_yj_transformer} to inverse transform data")
        for col in columns:
            pt = self.dict_pt[f"{col} pt"]
            data_frame.loc[:, col] = pt.inverse_transform(np.array(data_frame[col]).reshape(-1, 1)).ravel()
        return data_frame

    def yj_transform_data(self, data_frame, columns):
        """"""
        dict_pt = joblib.load(
            filename=f"{self.cfg['PATH_TRANSFORMERS']}/{self.cfg[f'FILENAME_YJ_TRANSFORMER_{self.data_set_type}']}"
        )
        self.name_yj_transformer = self.cfg[f'FILENAME_YJ_TRANSFORMER_{self.data_set_type}']
        print(f"Use {self.name_yj_transformer} to transform data")
        for col in columns:
            pt = dict_pt[f"{col} pt"]
            data_frame.loc[:, col] = pt.transform(np.array(data_frame[col]).reshape(-1, 1))
        return data_frame, dict_pt

    def yj_transform_data_on_fly(self, data_frame, columns, dict_pt):
        """"""
        print(f"Use {self.name_yj_transformer} to transform data")
        for col in columns:
            pt = dict_pt[f"{col} pt"]
            data_frame.loc[:, col] = pt.transform(np.array(data_frame[col]).reshape(-1, 1))
        return data_frame

    @staticmethod
    def unsheared_object_cuts(data_frame):
        """"""
        print("Apply unsheared object cuts")
        cuts = (data_frame["unsheared/extended_class_sof"] >= 0) & (data_frame["unsheared/flags_gold"] < 2)
        data_frame = data_frame[cuts]
        print('Length of catalog after applying unsheared object cuts: {}'.format(len(data_frame)))
        return data_frame

    @staticmethod
    def flag_cuts(data_frame):
        """"""
        print("Apply flag cuts")
        cuts = (data_frame["match_flag_1.5_asec"] < 2) & \
               (data_frame["flags_foreground"] == 0) & \
               (data_frame["flags_badregions"] < 2) & \
               (data_frame["flags_footprint"] == 1)
        data_frame = data_frame[cuts]
        print('Length of catalog after applying flag cuts: {}'.format(len(data_frame)))
        return data_frame

    @staticmethod
    def airmass_cut(data_frame):
        """"""
        print('Cut mcal_detect_df_survey catalog so that AIRMASS_WMEAN_R is not null')
        data_frame = data_frame[pd.notnull(data_frame["AIRMASS_WMEAN_R"])]
        print('Length of catalog after applying AIRMASS_WMEAN_R cuts: {}'.format(len(data_frame)))
        return data_frame

    @staticmethod
    def unsheared_mag_cut(data_frame):
        """"""
        print("Apply unsheared mag cuts")
        cuts = (
                (18 < data_frame["unsheared/mag_i"]) &
                (data_frame["unsheared/mag_i"] < 23.5) &
                (15 < data_frame["unsheared/mag_r"]) &
                (data_frame["unsheared/mag_r"] < 26) &
                (15 < data_frame["unsheared/mag_z"]) &
                (data_frame["unsheared/mag_z"] < 26) &
                (-1.5 < data_frame["unsheared/mag_r"] - data_frame["unsheared/mag_i"]) &
                (data_frame["unsheared/mag_r"] - data_frame["unsheared/mag_i"] < 4) &
                (-4 < data_frame["unsheared/mag_z"] - data_frame["unsheared/mag_i"]) &
                (data_frame["unsheared/mag_z"] - data_frame["unsheared/mag_i"] < 1.5)
        )
        data_frame = data_frame[cuts]
        print('Length of catalog after applying unsheared mag cuts: {}'.format(len(data_frame)))
        return data_frame

    @staticmethod
    def unsheared_shear_cuts(data_frame):
        """"""
        print("Apply unsheared shear cuts")
        cuts = (
                (10 < data_frame["unsheared/snr"]) &
                (data_frame["unsheared/snr"] < 1000) &
                (0.5 < data_frame["unsheared/size_ratio"]) &
                (data_frame["unsheared/T"] < 10)
        )
        data_frame = data_frame[cuts]
        data_frame = data_frame[~((2 < data_frame["unsheared/T"]) & (data_frame["unsheared/snr"] < 30))]
        print('Length of catalog after applying unsheared shear cuts: {}'.format(len(data_frame)))
        return data_frame

    @staticmethod
    def remove_outliers(data_frame):
        print("Remove outliers...")
        not_outliers = (
            ~((data_frame["unsheared/mag_r"] <= 7.812946519176684) |
              (data_frame["unsheared/mag_r"] >= 37.5) |
              (data_frame["unsheared/mag_i"] <= 7.342380502886332) |
              (data_frame["unsheared/mag_i"] >= 37.5) |
              (data_frame["unsheared/mag_z"] <= 7.69900077218967) |
              (data_frame["unsheared/mag_z"] >= 37.5) |
              (data_frame["unsheared/snr"] <= 0) |
              (data_frame["unsheared/snr"] >= 200) |
              (data_frame["unsheared/size_ratio"] <= -1) |
              (data_frame["unsheared/size_ratio"] >= 4) |
              (data_frame["unsheared/T"] <= -1) |
              (data_frame["unsheared/T"] >= 4) |
              (data_frame["unsheared/weight"] <= 10.0) |
              (data_frame["unsheared/weight"] >= 77.58102207403836))
        )
        data_frame = data_frame[not_outliers]
        print('Length of catalog after removing outliers: {}'.format(len(data_frame)))
        return data_frame

    def get_dict_pt(self):
        return self.dict_pt


if __name__ == '__main__':
    pass
