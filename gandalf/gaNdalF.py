from gandalf_galaxie_dataset import DESGalaxies
from torch.utils.data import DataLoader, RandomSampler
from scipy.stats import binned_statistic, median_abs_deviation
from Handler import *
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import h5py
import joblib
import pickle
import torch
from torch import nn
import os
# plt.style.use('seaborn-white')


class gaNdalF(object):
    """"""
    def __init__(self, gandalf_logger, flow_cfg, classifier_cfg):
        """"""
        self.gandalf_logger = gandalf_logger
        self.gandalf_logger.log_info_stream(f"Init gaNdalF")
        self.flow_cfg = flow_cfg
        self.classifier_cfg = classifier_cfg

        if self.flow_cfg["RUN_FLOW"] is True:
            self.flow_galaxies = self.init_flow_dataset()
            self.flow_model = self.init_flow_model()
            self.flow_data = self.flow_galaxies.run_dataset

        if self.classifier_cfg["RUN_CLASSIFIER"] is True:
            self.classifier_galaxies = self.init_classifier_dataset()
            self.classifier_model = self.init_classifier_model()
            self.classifier_data = self.classifier_galaxies.run_dataset

    def init_flow_dataset(self):
        galaxies = DESGalaxies(
            dataset_logger=self.gandalf_logger,
            cfg=self.flow_cfg,
        )
        galaxies.apply_log10()
        galaxies.scale_data()
        return galaxies

    def init_classifier_dataset(self):
        galaxies = DESGalaxies(
            dataset_logger=self.gandalf_logger,
            cfg=self.classifier_cfg,
        )
        galaxies.scale_data()
        return galaxies

    def init_flow_model(self):
        modules = []
        num_outputs = len(self.flow_cfg["OUTPUT_COLS"])
        for _ in range(self.flow_cfg["NUMBER_BLOCKS"]):
            modules += [
                fnn.MADE(
                    num_inputs=num_outputs,
                    num_hidden=self.flow_cfg["NUMBER_HIDDEN"],
                    num_cond_inputs=len(self.flow_cfg["INPUT_COLS"]),
                    act=self.flow_cfg["ACTIVATION_FUNCTION"],
                    num_layers=self.flow_cfg["NUMBER_LAYERS"]
                ),
                fnn.BatchNormFlow(num_outputs),
                fnn.Reverse(num_outputs)
            ]
        model = fnn.FlowSequential(*modules)
        model = model.to(dtype=torch.float32)
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)
        model.to(self.flow_cfg["DEVICE"])
        model.load_state_dict(torch.load(
            f"{self.flow_cfg['PATH_TRAINED_NN']}/{self.flow_cfg['FILENAME_NN_FLOW']}",
            map_location="cpu",
            weights_only=True
        ))
        model.num_inputs = num_outputs
        model.eval()
        return model

    def init_classifier_model(self):
        input_dim = len(self.classifier_cfg["INPUT_COLS"])
        output_dim = len(self.classifier_cfg["OUTPUT_COLS"])
        hs = list(self.classifier_cfg["HIDDEN_SIZES"])
        nl = len(hs)
        dp = float(self.classifier_cfg["DROPOUT_PROB"])
        ubn = bool(self.classifier_cfg["BATCH_NORM"])
        device = self.classifier_cfg["DEVICE"]

        activation = self.classifier_cfg.get("ACTIVATION_FUNCTION", "ReLU")
        if isinstance(activation, str):
            ActClass = getattr(nn, activation)
            make_act = lambda: ActClass()
        elif isinstance(activation, type) and issubclass(activation, nn.Module):
            make_act = lambda: activation()
        elif callable(activation):
            make_act = activation
        else:
            raise ValueError(f"Unsupported activation spec: {activation}")

        layers = []
        in_features = input_dim
        assert nl == len(hs), f"len(hidden_sizes)={len(hs)} != number_layer={nl}"

        for out_features in hs:
            out_features = int(out_features)
            layers.append(nn.Linear(in_features, out_features))
            if ubn:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(make_act())
            if dp > 0.0:
                layers.append(nn.Dropout(dp))
            in_features = out_features

        layers.append(nn.Linear(in_features, output_dim))

        model = nn.Sequential(*layers)

        model = model.to(dtype=torch.float32)

        for module in model.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)

        model.to(device)

        state_dict_path = (
            f"{self.classifier_cfg['PATH_TRAINED_NN']}/"
            f"{self.classifier_cfg['FILENAME_NN_CLASSIFIER']}"
        )
        state = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)

        model.eval()
        return model

    def run_classifier(self, data_frame=None):
        """"""
        if data_frame is not None:
            self.classifier_data = data_frame

        input_data = torch.tensor(self.classifier_data[self.classifier_cfg["INPUT_COLS"]].values, dtype=torch.float32)
        input_data = input_data.to(torch.device('cpu'))

        with torch.no_grad():
            arr_classf_gandalf_output = self.classifier_model(input_data).squeeze().numpy()

        arr_gandalf_detected = arr_classf_gandalf_output > np.random.rand(self.classifier_cfg['NUMBER_SAMPLES'])
        arr_gandalf_detected = arr_gandalf_detected.astype(int)

        validation_accuracy = accuracy_score(self.classifier_data[self.classifier_cfg["OUTPUT_COLS"]].values, arr_gandalf_detected)

        df_gandalf = self.classifier_data.copy()
        df_gandalf.loc[:, "detected"] = arr_gandalf_detected.astype(int)
        df_gandalf.loc[:, "probability detected"] = arr_classf_gandalf_output

        self.gandalf_logger.log_info_stream(f"Accuracy sample: {validation_accuracy * 100.0:.2f}%")
        return df_gandalf, self.classifier_data

    def run_flow(self, data_frame=None):
        """"""
        if data_frame is not None:
            self.flow_data = data_frame

        input_data = torch.tensor(self.flow_data[self.flow_cfg["INPUT_COLS"]].values, dtype=torch.float32)
        output_data = torch.tensor(self.flow_data[self.flow_cfg["OUTPUT_COLS"]].values, dtype=torch.float32)

        input_data = input_data.to(torch.device('cpu'))

        with torch.no_grad():
            arr_gandalf_output = self.flow_model.sample(len(input_data), cond_inputs=input_data).detach()

        output_data_np = arr_gandalf_output.cpu().numpy()

        input_data_np_true = input_data.cpu().numpy()
        output_data_np_true = output_data.cpu().numpy()
        arr_all_balrog = np.concatenate([input_data_np_true, output_data_np_true], axis=1)
        arr_all_gandalf = np.concatenate([input_data_np_true, output_data_np], axis=1)

        df_output_balrog = pd.DataFrame(arr_all_balrog, columns=list(self.flow_cfg["INPUT_COLS"]) + list(self.flow_cfg["OUTPUT_COLS"]))
        df_output_balrog = df_output_balrog[self.flow_cfg["COLUMNS_OF_INTEREST"]]

        df_output_gandalf = pd.DataFrame(arr_all_gandalf, columns=list(self.flow_cfg["INPUT_COLS"]) + list(self.flow_cfg["OUTPUT_COLS"]))
        df_output_gandalf = df_output_gandalf[self.flow_cfg["COLUMNS_OF_INTEREST"]]

        df_output_balrog = self.flow_galaxies.inverse_scale_data(df_output_balrog)
        df_output_gandalf = self.flow_galaxies.inverse_scale_data(df_output_gandalf)

        return df_output_gandalf, df_output_balrog

    def save_data(self, data_frame, file_name, protocol=2, tmp_samples=False):
        """"""
