from gandalf_galaxie_dataset import DESGalaxies
from torch.utils.data import DataLoader, TensorDataset
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
from scipy.stats import ks_2samp
# plt.style.use('seaborn-white')


class MagAwarePlatt:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    @staticmethod
    def _safe_logit(p, eps=1e-6):
        p = np.clip(np.asarray(p, float), eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

    def transform(self, p, mag):
        x1 = self._safe_logit(p)
        x2 = np.asarray(mag, float)
        x3 = x2 ** 2
        z = np.c_[x1, x2, x3] @ self.coef_ + self.intercept_
        return self._sigmoid(z)

    def state_dict(self): return {"coef_": self.coef_, "intercept_": self.intercept_}
    def load_state_dict(self, d):
        self.coef_ = np.asarray(d["coef_"], float)
        self.intercept_ = float(d["intercept_"])


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
            self.T_cal = 1.0
            self.mag_platt = None
            self._load_calibration()

    def init_flow_dataset(self):
        galaxies = DESGalaxies(
            dataset_logger=self.gandalf_logger,
            cfg=self.flow_cfg,
        )
        for k in self.flow_cfg["INPUT_COLS"]:
            print(k, galaxies.run_dataset[k].min(), galaxies.run_dataset[k].max())
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

    def _load_calibration(self):
        """Load temperature + mag-aware Platt from joblib artifact if present."""
        try:
            cal_path = os.path.join(
                self.classifier_cfg["PATH_TRAINED_NN"], self.classifier_cfg["FILENAME_CALIBRATION_ARTIFACTS"]
            )
            if os.path.exists(cal_path):
                cal = joblib.load(cal_path)
                self.T_cal = float(cal.get("temperature", 1.0))
                mp_state = cal.get("mag_platt", None)
                if mp_state is not None:
                    self.mag_platt = MagAwarePlatt()
                    self.mag_platt.load_state_dict(mp_state)
                self.gandalf_logger.log_info_stream(
                    f"Loaded calibration (T={self.T_cal:.3f}, mag_platt={'yes' if self.mag_platt else 'no'})"
                )
            else:
                self.gandalf_logger.log_info_stream("No calibration_artifacts.pkl found – using raw probabilities.")
        except Exception as e:
            self.gandalf_logger.log_error(f"Loading calibration failed: {e}")
            self.T_cal = 1.0
            self.mag_platt = None

    def _hidden_sizes_from_filename(self, fname: str):
        m = re.search(r"_hs_(\[.*?\])_", fname)
        if m:
            try:
                import ast
                return list(ast.literal_eval(m.group(1)))
            except Exception:
                pass
        return None

    def init_classifier_model(self):
        input_dim = len(self.classifier_cfg["INPUT_COLS"])
        output_dim = len(self.classifier_cfg["OUTPUT_COLS"])
        device_str = str(self.classifier_cfg.get("DEVICE", "cpu")).lower()
        device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

        state_dict_path = f"{self.classifier_cfg['PATH_TRAINED_NN']}/{self.classifier_cfg['FILENAME_NN_CLASSIFIER']}"

        sd = torch.load(state_dict_path, map_location="cpu", weights_only=True)

        hs = self._hidden_sizes_from_filename(self.classifier_cfg.get("FILENAME_NN_CLASSIFIER", "")) or list(self.classifier_cfg["HIDDEN_SIZES"])

        dp = float(self.classifier_cfg.get("DROPOUT_PROB", 0.0))
        ubn = bool(self.classifier_cfg.get("BATCH_NORM", True))
        act_name = self.classifier_cfg.get("ACTIVATION_FUNCTION", "ReLU")
        ActClass = getattr(nn, act_name) if isinstance(act_name, str) else act_name

        layers, in_features = [], input_dim

        for h in hs:
            h = int(h)
            layers.append(nn.Linear(in_features, h, bias=not ubn))
            if ubn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(ActClass() if isinstance(ActClass, type) else ActClass())
            if dp > 0.0:
                layers.append(nn.Dropout(dp))
            in_features = h

        layers.append(nn.Linear(in_features, output_dim, bias=True))
        model = nn.Sequential(*layers)
        model.load_state_dict(sd, strict=True)

        model.to(dtype=torch.float32, device=device)
        model.eval()
        return model

    def run_classifier(self, dataset:str="test", threshold:float=None):
        """
        If calibrated=True, returns AND samples from calibrated probabilities.
        Otherwise uses raw probs.
        """
        if dataset == "test":
            self.classifier_data = self.classifier_galaxies.run_dataset
        elif dataset == "valid":
            self.classifier_data = self.classifier_galaxies.valid_dataset
        elif dataset == "train":
            self.classifier_data = self.classifier_galaxies.train_dataset
        else:
            raise ValueError("Unkonwn dataset")

        device = torch.device(str(self.classifier_cfg.get("DEVICE", "cpu")).lower())
        mag_col = self.classifier_cfg.get("MAG_COL", "BDF_MAG_DERED_CALIB_I")

        X_np = self.classifier_data[self.classifier_cfg["INPUT_COLS"]].to_numpy(dtype=np.float32, copy=False)
        mag_np = self.classifier_data[mag_col].to_numpy(dtype=np.float32, copy=False)

        X_t = torch.from_numpy(X_np)
        mag_t = torch.from_numpy(mag_np)

        bs = int(self.classifier_cfg.get("BATCH_SIZE_INFER", 131072))
        pin = (device.type == "cuda")
        loader = DataLoader(
            TensorDataset(X_t, mag_t),
            batch_size=bs, shuffle=False,
            num_workers=max(1, os.cpu_count() // 2),
            pin_memory=pin, persistent_workers=False
        )

        self.classifier_model.eval()
        probs_raw_chunks = []
        probs_cal_chunks = []

        with torch.inference_mode():
            for xb, mb in loader:
                xb = xb.to(device, non_blocking=pin)
                logits = self.classifier_model(xb).squeeze(-1)

                # raw
                p_raw = torch.sigmoid(logits).float().cpu().numpy()

                # temperature scaling
                if self.T_cal != 1.0:
                    p_T = torch.sigmoid((logits / self.T_cal)).float().cpu().numpy()
                else:
                    p_T = p_raw

                # mag-aware platt
                if self.mag_platt is not None:
                    mb_np = mb.numpy()
                    p_cal = self.mag_platt.transform(p_T, mb_np)
                else:
                    p_cal = p_T

                probs_raw_chunks.append(p_raw)
                probs_cal_chunks.append(p_cal)

        p_raw = np.concatenate(probs_raw_chunks).astype(np.float32)
        p_cal = np.concatenate(probs_cal_chunks).astype(np.float32)

        # which probs should drive the Bernoulli?
        p_for_sampling = p_cal if self.classifier_cfg.get("CALIB_CLASSIFIER", True) else p_raw

        thr = threshold
        if thr is None:
            thr = getattr(self, "thr_best", None)  # aus _load_calibration gesetzt, s.u.
        if thr is None:
            thr = 0.5

        y_true = self.classifier_data[self.classifier_cfg["OUTPUT_COLS"]].to_numpy(int).ravel()
        y_pred = ((p_for_sampling if self.classifier_cfg.get("CALIB_CLASSIFIER", True) else p_raw) >= thr).astype(int)

        rng = np.random.default_rng(int(self.classifier_cfg.get('BERNOULLI_SEED', 123)))
        y_sampled = ((p_for_sampling if self.classifier_cfg.get("CALIB_CLASSIFIER", True) else p_raw) > rng.random(p_raw.shape[0])).astype(int)

        df_gandalf = self.classifier_data.copy()
        df_gandalf["true detected"] = y_true
        df_gandalf["threshold detected"] = y_pred
        df_gandalf["sampled mcal_galaxy"] = y_sampled
        df_gandalf["probability detected raw"] = p_raw
        df_gandalf["probability detected"] = p_for_sampling

        acc = accuracy_score(y_true, y_pred)
        self.gandalf_logger.log_info_stream(f"Accuracy (deterministic, thr={thr:.3f}): {acc * 100:.2f}%")
        return df_gandalf, self.classifier_data

    def run_flow(self, data_frame=None):
        """"""
        if data_frame is not None:
            self.flow_data = data_frame

        df_gandalf_input = self.flow_data.copy()
        df_gandalf_input.loc[:, self.flow_cfg["OUTPUT_COLS"]] = np.nan
        df_gandalf_input["gandalf_id"] = np.arange(len(df_gandalf_input))
        try:
            flow_detected = df_gandalf_input[df_gandalf_input[self.flow_cfg["DETECTION_TYPE"]]==1].copy()
        except KeyError:
            self.gandalf_logger.log_info_stream(f"using detected instead if {self.flow_cfg["DETECTION_TYPE"]} due to key error")
            self.gandalf_logger.log_error(f"using detected instead if {self.flow_cfg["DETECTION_TYPE"]} due to key error")
            flow_detected = df_gandalf_input[df_gandalf_input["detected"] == 1].copy()

        input_data = torch.tensor(flow_detected[self.flow_cfg["INPUT_COLS"]].values, dtype=torch.float32)
        input_data = input_data.to(torch.device('cpu'))

        self.flow_model.eval()
        with torch.no_grad():
           arr_gandalf_output = self.flow_model.sample(len(input_data), cond_inputs=input_data).detach()

        arr_gandalf = np.concatenate([flow_detected["gandalf_id"].values.reshape(-1, 1), arr_gandalf_output.cpu().numpy()], axis=1)
        df_generated = pd.DataFrame(arr_gandalf, columns=["gandalf_id"] + list(self.flow_cfg["OUTPUT_COLS"]))
        df_gandalf_input.drop(self.flow_cfg["OUTPUT_COLS"], axis=1, inplace=True)
        df_gandalf = pd.merge(df_gandalf_input, df_generated, on='gandalf_id', how="left")
        df_gandalf = self.flow_galaxies.inverse_scale_data(df_gandalf)
        self.flow_data = self.flow_galaxies.inverse_scale_data(self.flow_data)

        return df_gandalf, self.flow_data

    def save_data(self, data_frame, file_name, protocol=2, tmp_samples=False):
        """"""
