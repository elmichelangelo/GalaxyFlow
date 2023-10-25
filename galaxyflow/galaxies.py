from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, Subset
# from Handler.helper_functions import yj_transform_data
from sklearn.preprocessing import PowerTransformer, MaxAbsScaler, MinMaxScaler, StandardScaler
from itertools import accumulate
import numpy as np
import pandas as pd
import torch
import gc


class GalaxyDataset(Dataset):
    def __init__(self, cfg, lst_split: list = None):
        with open(f"{cfg['PATH_DATA']}/{cfg['DATA_FILE_NAME']}", 'rb') as f:
            df_data = pd.read_pickle(f)
            self.df_cut_cols = df_data[cfg['CUT_COLS']]
            df_data = df_data[
                cfg[f"INPUT_COLS_{cfg['LUM_TYPE']}"] + cfg[f"OUTPUT_COLS_{cfg['LUM_TYPE']}"] + cfg['CUT_COLS']]

            self.applied_object_cut = False
            if cfg['APPLY_OBJECT_CUT'] is True:
                df_data = self.unsheared_object_cuts(df_data)
                self.applied_object_cut = True

            self.applied_flag_cut = False
            if cfg['APPLY_FLAG_CUT'] is True:
                df_data = self.flag_cuts(df_data)
                self.applied_flag_cut = True

            self.applied_mag_cut = False
            if cfg['APPLY_UNSHEARED_MAG_CUT'] is True:
                df_data = self.unsheared_mag_cut(df_data)
                self.applied_mag_cut = True

            self.applied_shear_cut = False
            if cfg['APPLY_UNSHEARED_SHEAR_CUT'] is True:
                df_data = self.unsheared_shear_cuts(df_data)
                self.applied_shear_cut = True

            self.applied_airmass_cut = False
            if cfg['APPLY_AIRMASS_CUT'] is True:
                df_data = self.airmass_cut(df_data)
                self.applied_airmass_cut = True

            df_data = df_data[cfg[f"INPUT_COLS_{cfg['LUM_TYPE']}"] + cfg[f"OUTPUT_COLS_{cfg['LUM_TYPE']}"]]

            self.applied_yj_transform = False
            if cfg['APPLY_YJ_TRANSFORM'] is True:
                if cfg['TRANSFORM_COLS'] is None:
                    df_data, self.dict_pt = self.yj_transform_data(data_frame=df_data, columns=df_data.keys())
                else:
                    df_data, self.dict_pt = self.yj_transform_data(data_frame=df_data, columns=cfg['TRANSFORM_COLS'])
                self.applied_yj_transform = True

            self.applied_scaler = False
            if cfg['APPLY_SCALER'] is True:
                df_data, self.scaler = self.scale_data(data_frame=df_data, cfg=cfg)
                self.applied_scaler = True

            self.tsr_data = TensorDataset(
                torch.tensor(df_data[cfg['INPUT_COLS_MAG']].values),
                torch.tensor(df_data[cfg['OUTPUT_COLS_MAG']].values))

            self.train_dataset, self.val_dataset, self.test_dataset, self.dict_indices = self.custom_random_split(
                dataset=self.tsr_data,
                lst_split=lst_split,
                df_data=df_data,
                cfg=cfg
            )

            del df_data
            gc.collect()

    def __len__(self):
        return len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset)

    def __getitem__(self, idx):
        sample = self.tsr_data[idx]
        return sample

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

    @staticmethod
    def scale_data(data_frame, cfg):
        """"""
        if cfg['SCALER'] == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif cfg['SCALER'] == "MaxAbsScaler":
            scaler = MaxAbsScaler()
        elif cfg['SCALER'] == "StandardScaler":
            scaler = StandardScaler()
        elif cfg['SCALER'] is None:
            scaler = None
        else:
            raise TypeError(f"{cfg['SCALER']} is no valid scaler")
        data_frame_scaled = None
        if scaler is not None:
            scaler.fit(data_frame)
            scaled = scaler.transform(data_frame)
            data_frame_scaled = pd.DataFrame(scaled, columns=data_frame.columns)
        return data_frame_scaled, scaler

    def yj_inverse_transform_data(self, data_frame, columns):
        """"""
        for col in columns:
            pt = self.dict_pt[f"{col} pt"]
            data_frame[col] = pt.inverse_transform(np.array(data_frame[col]).reshape(-1, 1)).ravel()
        return data_frame

    @staticmethod
    def yj_transform_data(data_frame, columns):
        """"""
        dict_pt = {}
        for col in columns:
            pt = PowerTransformer(method="yeo-johnson")
            pt.fit(np.array(data_frame[col]).reshape(-1, 1))
            data_frame[col] = pt.transform(np.array(data_frame[col]).reshape(-1, 1))
            dict_pt[f"{col} pt"] = pt
        return data_frame, dict_pt

    @staticmethod
    def unsheared_object_cuts(data_frame):
        """"""
        # print("Apply unsheared object cuts")
        cuts = (data_frame["unsheared/extended_class_sof"] >= 0) & (data_frame["unsheared/flags_gold"] < 2)
        data_frame = data_frame[cuts]
        # print('Length of catalog after applying unsheared object cuts: {}'.format(len(data_frame)))
        return data_frame

    @staticmethod
    def flag_cuts(data_frame):
        """"""
        # print("Apply flag cuts")
        cuts = (data_frame["match_flag_1.5_asec"] < 2) & \
               (data_frame["flags_foreground"] == 0) & \
               (data_frame["flags_badregions"] < 2) & \
               (data_frame["flags_footprint"] == 1)
        data_frame = data_frame[cuts]
        # print('Length of catalog after applying flag cuts: {}'.format(len(data_frame)))
        return data_frame

    @staticmethod
    def airmass_cut(data_frame):
        """"""
        # print('Cut mcal_detect_df_survey catalog so that AIRMASS_WMEAN_R is not null')
        data_frame = data_frame[pd.notnull(data_frame["AIRMASS_WMEAN_R"])]
        # print('Length of catalog after applying AIRMASS_WMEAN_R cuts: {}'.format(len(data_frame)))
        return data_frame

    @staticmethod
    def unsheared_mag_cut(data_frame):
        """"""
        # print("Apply unsheared mag cuts")
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
        # print('Length of catalog after applying unsheared mag cuts: {}'.format(len(data_frame)))
        return data_frame

    @staticmethod
    def unsheared_shear_cuts(data_frame):
        """"""
        # print("Apply unsheared shear cuts")
        cuts = (
                (10 < data_frame["unsheared/snr"]) &
                (data_frame["unsheared/snr"] < 1000) &
                (0.5 < data_frame["unsheared/size_ratio"]) &
                (data_frame["unsheared/T"] < 10)
        )
        data_frame = data_frame[cuts]
        data_frame = data_frame[~((2 < data_frame["unsheared/T"]) & (data_frame["unsheared/snr"] < 30))]
        # print('Length of catalog after applying unsheared shear cuts: {}'.format(len(data_frame)))
        return data_frame

    def get_dict_pt(self):
        return self.dict_pt


if __name__ == '__main__':
    pass

