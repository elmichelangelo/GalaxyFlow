from torch.utils.data import Dataset, TensorDataset, Subset, DataLoader
from sklearn.preprocessing import PowerTransformer, MaxAbsScaler, MinMaxScaler, StandardScaler
from itertools import accumulate
import numpy as np
import pandas as pd
import sys
import torch
import gc
import joblib

from Handler import apply_quality_cuts
from Handler.helper_functions import compute_weights_from_magnitude


class GalaxyDataset(Dataset):
    def __init__(self, cfg, dataset_logger, data_type="classifier", data_frame: pd.DataFrame | None = None):
        self.dataset_logger = dataset_logger
        self.dataset_logger.log_info_stream(f"Init GalaxyDataset {data_type}")
        self.cfg = cfg
        self.name_scaler = None
        self.scalers = None
        self.data_type = data_type
        self.run_dataset = None

        if self.data_type == "classifier":
            cfg_key_filename_sample_data = "FILENAME_SAMPLE_DATA_CF"
            data_frames = self._load_data_frames(
                cfg_key_filename_sample_data=cfg_key_filename_sample_data
            )
        elif self.data_type == "flow":
            cfg_key_filename_sample_data = "FILENAME_SAMPLE_DATA_NF"
            if data_frame is None:
                data_frames = self._load_data_frames(
                    cfg_key_filename_sample_data=cfg_key_filename_sample_data
                )
            else:
                data_frames = data_frame
        else:
            self.dataset_logger.log_error(f"Invalid data type: {self.data_type}")
            raise NotImplementedError

        self._build_datasets(data_frames)

        gc.collect()

    def _load_data_frames(self,
                          cfg_key_filename_sample_data: str | None = None
                          ):
        if self.cfg["TRAINING"] is True:
            df_train = self._load_data(filename=self.cfg["FILENAME_TRAIN_DATA"])
            df_valid = self._load_data(filename=self.cfg["FILENAME_VALIDATION_DATA"])
            df_test = self._load_data(filename=self.cfg["FILENAME_TEST_DATA"])

            frac = self.cfg.get("SAMPLE_FRACTION", 1.0)
            seed = self.cfg.get("SAMPLE_SEED", 42)

            if frac < 1.0:
                self.dataset_logger.log_info_stream(
                    f"Subsampling {frac * 100:.0f}% of train/valid/test with seed={seed}"
                )
                df_train = df_train.sample(frac=frac, random_state=seed).reset_index(drop=True)
                df_valid = df_valid.sample(frac=frac, random_state=seed).reset_index(drop=True)
                df_test = df_test.sample(frac=frac, random_state=seed).reset_index(drop=True)

            return df_train, df_valid, df_test
        else:
            df_data = self._load_data(filename=self.cfg[cfg_key_filename_sample_data])
            self.dataset_logger.log_info_stream(f"Sample {self.cfg['NUMBER_SAMPLES']} random data for {self.data_type}")
            df_run = df_data.sample(n=self.cfg['NUMBER_SAMPLES'], replace=True).reset_index(drop=True)
            del df_data
            return df_run

    def _load_data(self, filename):
        self.dataset_logger.log_info_stream(f"Load {filename} data set")
        with open(f"{self.cfg['PATH_DATA']}/{filename}", 'rb') as file:
            data_frame = pd.read_pickle(file)
        self.dataset_logger.log_info_stream(f"shape dataset: {data_frame.shape}")

        return data_frame

    def _build_datasets(self, data_frames):
        if self.cfg["TRAINING"] is True:
            df_train = data_frames[0]
            df_valid = data_frames[1]
            df_test = data_frames[2]
            if self.cfg["TRAINING_TYPE"] == "classifier":
                self.train_dataset = df_train.copy()
                self.valid_dataset = df_valid.copy()
                self.test_dataset = df_test.copy()
            elif self.cfg["TRAINING_TYPE"].lower() == "flow":
                df_train = df_train[self.cfg["COLUMNS"]]
                df_valid = df_valid[self.cfg["COLUMNS"]]
                df_test = df_test[self.cfg["COLUMNS"]]
                self.train_dataset = df_train.copy()
                self.valid_dataset = df_valid.copy()
                self.test_dataset = df_test.copy()
            del df_train
            del df_valid
            del df_test
        else:
            self.run_dataset = data_frames

    def __len__(self):
        if self.cfg["TRAINING"]:
            return len(self.train_dataset)
        else:
            return len(self.run_dataset)

    def __getitem__(self, idx):
        if self.cfg["TRAINING"]:
            return self.train_dataset.iloc[idx]
        else:
            return self.run_dataset.iloc[idx]

    def apply_log10(self,
                    data_frame: pd.DataFrame | None = None,
                    cfg_key_log10_cols: str = "LOG10_COLS"
                    ):
        if data_frame is not None:
            self.run_dataset = data_frame

        if self.cfg["TRAINING"] is True:
            for col in self.cfg[cfg_key_log10_cols]:
                self.train_dataset[col] = np.log10(self.train_dataset[col])
                self.valid_dataset[col] = np.log10(self.valid_dataset[col])
                self.test_dataset[col] = np.log10(self.test_dataset[col])
            return self.train_dataset, self.valid_dataset, self.test_dataset
        else:
            for col in self.cfg[cfg_key_log10_cols]:
                self.run_dataset[col] = np.log10(self.run_dataset[col])
            return self.run_dataset


    def scale_data(self,
                   data_frame: pd.DataFrame | None = None,
                   cfg_key_cols_interest: str = "COLUMNS_OF_INTEREST",
                   cfg_key_filename_scaler: str = "FILENAME_STANDARD_SCALER"
                   ):
        """"""
        if data_frame is not None:
                self.run_dataset = data_frame

        if self.scalers is None:
            self.name_scaler = self.cfg[cfg_key_filename_scaler]
            self.scalers = joblib.load(
                filename=f"{self.cfg['PATH_TRANSFORMERS']}/{self.name_scaler}"
            )

            self.dataset_logger.log_info_stream(f"Use {self.name_scaler} to scale {self.data_type} data")

        if self.cfg["TRAINING"] is True:
            for col in self.cfg[cfg_key_cols_interest]:
                scaler = self.scalers[col]
                mean = scaler.mean_[0]
                scale = scaler.scale_[0]
                self.train_dataset[col] = (self.train_dataset[col] - mean) / scale
                self.valid_dataset[col] = (self.valid_dataset[col] - mean) / scale
                self.test_dataset[col] = (self.test_dataset[col] - mean) / scale
            return self.train_dataset, self.valid_dataset, self.test_dataset
        else:
            for col in self.cfg[cfg_key_cols_interest]:
                scaler = self.scalers[col]
                mean = scaler.mean_[0]
                scale = scaler.scale_[0]
                self.run_dataset[col] = (self.run_dataset[col] - mean) / scale
            return self.run_dataset

    def inverse_scale_data(self,
                           data_frame: pd.DataFrame | None = None,
                           cfg_key_filename_scaler: str | None = None,
                           ):
        """"""
        if self.scalers is None:
            self.name_scaler = self.cfg[cfg_key_filename_scaler]
            self.scalers = joblib.load(
                filename=f"{self.cfg['PATH_TRANSFORMERS']}/{self.name_scaler}"
            )

        self.dataset_logger.log_info_stream(f"Use {self.name_scaler} to inverse scale {self.data_type} data")

        for col in data_frame.columns:
            if col in self.scalers.keys():
                scaler = self.scalers[col]
                mean = scaler.mean_[0]
                scale = scaler.scale_[0]
                data_frame[col] = (data_frame[col] * scale) + mean

        return data_frame

if __name__ == '__main__':
    pass
