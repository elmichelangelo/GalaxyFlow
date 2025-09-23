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
import os
# plt.style.use('seaborn-white')


class gaNdalF(object):
    """"""
    def __init__(self, gandalf_logger, cfg):
        """"""
        self.gandalf_logger = gandalf_logger
        self.gandalf_logger.log_info_stream(f"Init gaNdalF")
        self.cfg = cfg

        self.galaxies = self.init_dataset(gandalf_logger=gandalf_logger)
        self.flow_model = self.init_flow_model()
        self.flow_data = self.galaxies.run_dataset

    def init_dataset(self, gandalf_logger):

        galaxies = DESGalaxies(
            dataset_logger=gandalf_logger,
            cfg=self.cfg,
        )

        galaxies.apply_log10()
        galaxies.scale_data()

        return galaxies

    def init_flow_model(self):
        modules = []
        num_outputs = len(self.cfg["OUTPUT_COLS"])
        for _ in range(self.cfg["NUMBER_BLOCKS"]):
            modules += [
                fnn.MADE(
                    num_inputs=num_outputs,
                    num_hidden=self.cfg["NUMBER_HIDDEN"],
                    num_cond_inputs=len(self.cfg["INPUT_COLS"]),
                    act=self.cfg["ACTIVATION_FUNCTION"],
                    num_layers=self.cfg["NUMBER_LAYERS"]
                ),
                fnn.BatchNormFlow(num_outputs),
                fnn.Reverse(num_outputs)
            ]
        model = fnn.FlowSequential(*modules)
        model = model.to(dtype=torch.float64)
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)

        model.to(self.cfg["DEVICE"])

        model.load_state_dict(torch.load(
            f"{self.cfg['PATH_TRAINED_NN']}/{self.cfg['FILENAME_NN_FLOW']}",
            map_location="cpu",
            weights_only=True
        ))

        model.num_inputs = num_outputs

        model.eval()

        return model

    def run_classifier(self, data_frame):
        """"""
        with torch.no_grad():
            input_data = torch.tensor(data_frame[self.cfg["INPUT_COLS"]].values, dtype=torch.float32)

            for j, col in enumerate(self.cfg["INPUT_COLS"]):
                if col in self.cfg.get("COLUMNS_LOG1P", []):
                    input_data[:, j] = torch.log1p(input_data[:, j])
                scale = self.scalers_classifier[col].scale_[0]
                min_ = self.scalers_classifier[col].min_[0]
                input_data[:, j] = input_data[:, j] * scale + min_

            arr_classf_gandalf_output = self.gandalf_classifier(input_data).squeeze().numpy()

        arr_gandalf_detected = arr_classf_gandalf_output > np.random.rand(self.cfg['NUMBER_SAMPLES'])
        arr_gandalf_detected = arr_gandalf_detected.astype(int)

        validation_accuracy = accuracy_score(data_frame[self.cfg["OUTPUT_COLS_CLASSIFIER"]].values, arr_gandalf_detected)

        df_gandalf = data_frame.copy()
        df_gandalf.rename(columns={"detected": "true_detected"}, inplace=True)
        df_gandalf.loc[:, "detected"] = arr_gandalf_detected.astype(int)
        df_gandalf.loc[:, "probability detected"] = arr_classf_gandalf_output

        self.gandalf_logger.log_info_stream(f"Accuracy sample: {validation_accuracy * 100.0:.2f}%")
        return df_gandalf

    def run_flow(self, data_frame=None):
        """"""
        if data_frame is not None:
            self.flow_data = data_frame

        input_data = torch.tensor(self.flow_data[self.cfg["INPUT_COLS"]].values, dtype=torch.float64)
        output_data = torch.tensor(self.flow_data[self.cfg["OUTPUT_COLS"]].values, dtype=torch.float64)

        input_data = input_data.to(torch.device('cpu'))

        with torch.no_grad():
            arr_gandalf_output = self.flow_model.sample(len(input_data), cond_inputs=input_data).detach()

        output_data_np = arr_gandalf_output.cpu().numpy()

        input_data_np_true = input_data.cpu().numpy()
        output_data_np_true = output_data.cpu().numpy()
        arr_all_balrog = np.concatenate([input_data_np_true, output_data_np_true], axis=1)
        arr_all_gandalf = np.concatenate([input_data_np_true, output_data_np], axis=1)

        df_output_balrog = pd.DataFrame(arr_all_balrog, columns=list(self.cfg["INPUT_COLS"]) + list(self.cfg["OUTPUT_COLS"]))
        df_output_balrog = df_output_balrog[self.cfg["NF_COLUMNS_OF_INTEREST"]]

        df_output_gandalf = pd.DataFrame(arr_all_gandalf, columns=list(self.cfg["INPUT_COLS"]) + list(self.cfg["OUTPUT_COLS"]))
        df_output_gandalf = df_output_gandalf[self.cfg["NF_COLUMNS_OF_INTEREST"]]

        df_output_balrog = self.galaxies.inverse_scale_data(df_output_balrog)
        df_output_gandalf = self.galaxies.inverse_scale_data(df_output_gandalf)

        return df_output_gandalf, df_output_balrog

        n_features = len(self.cfg["OUTPUT_COLS"])
        ncols = min(n_features, 3)
        nrows = int(np.ceil(n_features / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten()
        for i, k in enumerate(self.cfg["OUTPUT_COLS"]):
            sns.histplot(x=df_output_gandalf[k], bins=100, ax=axes[i], label="gandalf")
            sns.histplot(x=df_output_balrog[k], bins=100, ax=axes[i], label="balrog")
            axes[i].set_yscale("log")
            if k in ["unsheared/mag_err_r", "unsheared/mag_err_i", "unsheared/mag_err_z"]:
                axes[i].set_title(f"log10({k})")
                axes[i].set_xlabel(f"log10({k})")
            else:
                axes[i].set_title(k)
                axes[i].set_xlabel(k)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle(f"Before applying mag and shear cuts", fontsize=16)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.legend()
        # plt.savefig(f"/Users/P.Gebhardt/Development/PhD/output/gaNdalF/before_cuts.pdf", bbox_inches='tight', dpi=300)
        plt.show()
        plt.clf()
        plt.close(fig)


    def save_data(self, data_frame, file_name, protocol=2, tmp_samples=False):
        """"""
