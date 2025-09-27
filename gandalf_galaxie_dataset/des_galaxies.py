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
    def __init__(self, cfg, dataset_logger):
        self.dataset_logger = dataset_logger
        self.dataset_logger.log_info_stream(f"Init GalaxyDataset")
        self.cfg = cfg
        self.name_scaler = None
        self.scalers = None

        data_frames = self._load_data_frames()

        self._build_datasets(data_frames)

        gc.collect()

    def _load_data_frames(self):
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
                self.train_dataset = df_train[df_train["detected"]==1].copy()
                self.valid_dataset = df_valid[df_valid["detected"]==1].copy()
                self.test_dataset = df_test[df_test["detected"]==1].copy()
            del df_train
            del df_valid
            del df_test
        else:
            self.run_dataset = data_frames

    def __len__(self):
        return len(self.train_dataset + self.valid_dataset + self.test_dataset)

    def __getitem__(self, idx):
        sample = self.test_dataset[idx]
        return sample

    def apply_log10(self):
        for col in self.cfg["LOG10_COLS"]:
            if self.cfg["TRAINING"] is True:
                self.train_dataset[col] = np.log10(self.train_dataset[col])
                self.valid_dataset[col] = np.log10(self.valid_dataset[col])
                self.test_dataset[col] = np.log10(self.test_dataset[col])
            else:
                self.run_dataset[col] = np.log10(self.run_dataset[col])

    def scale_data(self):
        """"""
        if self.scalers is None:
            self.name_scaler = self.cfg[f'FILENAME_STANDARD_SCALER']
            self.scalers = joblib.load(
                filename=f"{self.cfg['PATH_TRANSFORMERS']}/{self.name_scaler}"
            )

        self.dataset_logger.log_info_stream(f"Use {self.name_scaler} to scale data")

        for col in self.cfg["COLUMNS_OF_INTEREST"]:
            scaler = self.scalers[col]
            mean = scaler.mean_[0]
            scale = scaler.scale_[0]
            if self.cfg["TRAINING"] is True:
                self.train_dataset[col] = (self.train_dataset[col] - mean) / scale
                self.valid_dataset[col] = (self.valid_dataset[col] - mean) / scale
                self.test_dataset[col] = (self.test_dataset[col] - mean) / scale
            else:
                self.run_dataset[col] = (self.run_dataset[col] - mean) / scale

    def inverse_scale_data(self, data_frame):
        """"""
        if self.scalers is None:
            self.name_scaler = self.cfg[f'FILENAME_STANDARD_SCALER']
            self.scalers = joblib.load(
                filename=f"{self.cfg['PATH_TRANSFORMERS']}/{self.name_scaler}"
            )

        self.dataset_logger.log_info_stream(f"Use {self.name_scaler} to inverse scale data")

        for col in data_frame.columns:
            if col in self.scalers.keys():
                scaler = self.scalers[col]
                mean = scaler.mean_[0]
                scale = scaler.scale_[0]
                data_frame[col] = (data_frame[col] * scale) + mean

        return data_frame

if __name__ == '__main__':
    pass
