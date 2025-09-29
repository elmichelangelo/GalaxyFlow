import copy
import torch
import torch.optim as optim
import torch.utils.data

import torch.nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Handler import count_parameters, fnn, calc_color, plot_compare_corner, LoggerHandler, plot_features, luptize_inverse_fluxes, calc_mag, plot_binning_statistics_combined, plot_balrog_histogram_with_error
from gandalf_galaxie_dataset import DESGalaxies
import pandas as pd
import numpy as np
import seaborn as sns
import os
import joblib
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import math
from datetime import datetime
from ray import tune
import random
import logging
import signal

class gaNdalFFlow(object):

    def __init__(self,
                 cfg,
                 learning_rate,
                 number_hidden,
                 number_blocks,
                 batch_size,
                 number_layers
                 ):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg["DEVICE"])
        self.act = cfg["ACTIVATION_FUNCTION"]
        self.lst_epochs = []
        self.lst_mean_mag_r = []
        self.lst_mean_mag_i = []
        self.lst_mean_mag_z = []
        self.lst_mean_snr = []
        self.lst_mean_size_ratio = []
        self.lst_mean_t = []
        self.lst_std_mag_r = []
        self.lst_std_mag_i = []
        self.lst_std_mag_z = []
        self.lst_std_snr = []
        self.lst_std_size_ratio = []
        self.lst_std_t = []
        self.lst_mean_mag_r_cut = []
        self.lst_mean_mag_i_cut = []
        self.lst_mean_mag_z_cut = []
        self.lst_mean_snr_cut = []
        self.lst_mean_size_ratio_cut = []
        self.lst_mean_t_cut = []
        self.lst_std_mag_r_cut = []
        self.lst_std_mag_i_cut = []
        self.lst_std_mag_z_cut = []
        self.lst_std_snr_cut = []
        self.lst_std_size_ratio_cut = []
        self.lst_std_t_cut = []
        self.lst_train_loss_per_batch = []
        self.lst_train_loss_per_epoch = []
        self.lst_valid_loss_per_batch = []
        self.lst_valid_loss_per_epoch = []

        self.dict_delta_color_color = {}
        self.dict_delta_color_color_mcal = {}
        self.dict_delta_unsheared = {}
        self.dict_delta_unsheared_mcal = {}
        self.dict_delta_color_diff = {}
        self.dict_delta_color_diff_mcal = {}

        self.study_root_out = None
        self.study_root_cat = None
        self.study_root_plt = None

        self.start_epoch = 0
        self.current_epoch = 0

        # self.scalers = joblib.load(f"{self.cfg['PATH_TRANSFORMERS']}/{self.cfg['FILENAME_STANDARD_SCALER']}")

        self.bs = batch_size
        self.lr = learning_rate
        self.nh = number_hidden
        self.nb = number_blocks
        self.nl = number_layers

        self.make_dirs()

        log_lvl = logging.INFO
        if self.cfg["LOGGING_LEVEL"] == "DEBUG":
            log_lvl = logging.DEBUG
        elif self.cfg["LOGGING_LEVEL"] == "ERROR":
            log_lvl = logging.ERROR

        self.gandalf_logger = LoggerHandler(
            logger_dict={"logger_name": "train flow logger",
                         "info_logger": self.cfg['INFO_LOGGER'],
                         "error_logger": self.cfg['ERROR_LOGGER'],
                         "debug_logger": self.cfg['DEBUG_LOGGER'],
                         "stream_logger": self.cfg['STREAM_LOGGER'],
                         "stream_logging_level": log_lvl},
            log_folder_path=f"{self.cfg['PATH_LOGS']}/"
        )

        self.galaxies = self.init_dataset()

        flow_dtype_str = str(self.cfg.get("FLOW_DTYPE", "float64")).lower()
        self.dtype = torch.float32 if flow_dtype_str == "float32" else torch.float64

        self.use_amp = (
                bool(self.cfg.get("AMP_FLOW", True))
                and (self.device.type == "cuda")
                and (self.dtype == torch.float32)
        )

        if self.device.type == "cuda" and torch.cuda.is_available():
            bf16_ok = bool(self.cfg.get("AMP_BF16", True)) and torch.cuda.is_bf16_supported()
            self.autocast_dtype = torch.bfloat16 if bf16_ok else torch.float32
        else:
            self.autocast_dtype = torch.float32
        self.cuda_scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        if self.device.type == "cuda":
            self.gandalf_logger.log_info_stream(
                f"Using CUDA: {torch.cuda.get_device_name(0)} | AMP_FLOW={self.use_amp} | model_dtype={self.dtype} | "
                f"autocast={'bf16' if self.autocast_dtype is torch.bfloat16 else ('fp16' if self.use_amp else '-')}"
            )
            torch.backends.cudnn.benchmark = True

        self.model, self.optimizer = self.init_network(
            num_outputs=len(cfg[f"OUTPUT_COLS"]),
            num_input=len(cfg[f"INPUT_COLS"])
        )

        self.best_model_state_dict = copy.deepcopy(self.model.state_dict())

        self.global_step = 0
        self.best_validation_loss = np.float64('inf')
        self.best_validation_epoch = 1
        self.best_train_loss = np.float64('inf')
        self.best_train_epoch = 1
        self.best_model = self.model

        self._terminate_requested = False
        self._already_plotted = False
        self._finalized = False

        self.load_checkpoint_if_any()
        self._install_sigterm_handler()

    def make_dirs(self):
        """"""
        self.study_root_out = os.path.join(self.cfg['PATH_OUTPUT_BASE'], f"study_{self.cfg['RUN_ID']}")
        self.study_root_cat = os.path.join(self.cfg['PATH_OUTPUT_CATALOGS_BASE'], f"study_{self.cfg['RUN_ID']}")
        self.study_root_plt = os.path.join(self.cfg['PATH_PLOTS'], f"study_{self.cfg['RUN_ID']}")
        self.study_root_snn = os.path.join(self.cfg['PATH_TRAINED_NN'], f"study_{self.cfg['RUN_ID']}")
        self.study_root_log = os.path.join(self.cfg['PATH_LOGS'], f"study_{self.cfg['RUN_ID']}")

        self.study_root_plt = os.path.join(self.study_root_plt, f"trial_{self.cfg['TRIAL_ID']}")
        self.study_root_cat = os.path.join(self.study_root_cat, f"trial_{self.cfg['TRIAL_ID']}")
        self.study_root_snn = os.path.join(self.study_root_snn, f"trial_{self.cfg['TRIAL_ID']}")
        self.study_root_log = os.path.join(self.study_root_log, f"trial_{self.cfg['TRIAL_ID']}")

        self.cfg['PATH_OUTPUT_CATALOGS'] = os.path.join(self.study_root_cat, self.cfg['FOLDER_CATALOGS'])
        self.cfg['PATH_PLOTS'] = os.path.join(self.study_root_plt, self.cfg['FOLDER_PLOTS'])
        self.cfg['PATH_SAVE_NN'] = os.path.join(self.study_root_snn, self.cfg['FOLDER_SAVE_NN'])
        self.cfg['PATH_LOGS'] = os.path.join(self.study_root_log, self.cfg['FOLDER_LOGS'])

        os.makedirs(self.cfg['PATH_OUTPUT'], exist_ok=True)
        os.makedirs(self.cfg['PATH_OUTPUT_CATALOGS'], exist_ok=True)
        os.makedirs(self.cfg['PATH_PLOTS'], exist_ok=True)
        os.makedirs(self.cfg['PATH_SAVE_NN'], exist_ok=True)
        os.makedirs(self.cfg['PATH_LOGS'], exist_ok=True)

        self.cfg['PATH_PLOTS_FOLDER'] = {}

        for plot in self.cfg['PLOT_FOLDERS_FLOW']:
            self.cfg[f'PATH_PLOTS_FOLDER'][plot.upper()] = f"{self.cfg['PATH_PLOTS']}/{plot}"

        if self.cfg['PLOT_TRAINING'] is True:
            os.makedirs(self.cfg['PATH_PLOTS'], exist_ok=True)
            for path_plot in self.cfg['PATH_PLOTS_FOLDER'].values():
                os.makedirs(path_plot, exist_ok=True)
        if self.cfg['SAVE_NN_FLOW'] is True:
            os.makedirs(self.cfg['PATH_SAVE_NN'], exist_ok=True)

    def init_dataset(self):
        """"""
        galaxies = DESGalaxies(
            dataset_logger=self.gandalf_logger,
            cfg=self.cfg
        )
        galaxies.apply_log10()
        galaxies.scale_data()
        return galaxies

    def init_network(self, num_outputs, num_input):
        modules = []
        for _ in range(self.nb):
            modules += [
                fnn.MADE(num_inputs=num_outputs, num_hidden=self.nh, num_cond_inputs=num_input, act=self.act, num_layers=self.nl),
                fnn.BatchNormFlow(num_outputs),
                fnn.Reverse(num_outputs)
            ]
        model = fnn.FlowSequential(*modules)
        model = model.to(dtype=self.dtype)
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)

        model.to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, )
        return model, optimizer

    def _checkpoint_path(self):
        return os.path.join(self.cfg['PATH_OUTPUT'], "last.ckpt.pt")

    def _state_dict(self, epoch):
        return {
            "epoch": int(epoch),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_validation_loss": float(self.best_validation_loss),
            "best_validation_epoch": int(self.best_validation_epoch),
            "best_train_loss": float(self.best_train_loss),
            "best_train_epoch": int(self.best_train_epoch),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "hist": {
                "lst_epochs": self.lst_epochs,
                "lst_train_loss_per_epoch": self.lst_train_loss_per_epoch,
                "lst_valid_loss_per_epoch": self.lst_valid_loss_per_epoch,
            },
            "cfg_snapshot": {
                "lr": self.lr, "bs": self.bs, "nh": self.nh, "nb": self.nb, "nl": self.nl
            }
        }

    def save_checkpoint(self, epoch):
        path = self._checkpoint_path()
        tmp = path + ".tmp"
        torch.save(self._state_dict(epoch), tmp)
        os.replace(tmp, path)  # atomic

    def load_checkpoint_if_any(self):
        path = self._checkpoint_path()
        if not os.path.exists(path):
            self.start_epoch = 0
            return False
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.best_validation_loss = ckpt.get("best_validation_loss", self.best_validation_loss)
        self.best_validation_epoch = ckpt.get("best_validation_epoch", self.best_validation_epoch)
        self.best_train_loss = ckpt.get("best_train_loss", self.best_train_loss)
        self.best_train_epoch = ckpt.get("best_train_epoch", self.best_train_epoch)
        random.setstate(ckpt["rng"]["python"])
        np.random.set_state(ckpt["rng"]["numpy"])
        torch.random.set_rng_state(ckpt["rng"]["torch"])
        if torch.cuda.is_available() and ckpt["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
        self.start_epoch = int(ckpt["epoch"]) + 1
        self.gandalf_logger.log_info_stream(f"Resumed from checkpoint at epoch {self.start_epoch}")
        return True

    def _install_sigterm_handler(self):
        import threading, os, signal
        if threading.current_thread() is not threading.main_thread():
            self.gandalf_logger.log_info_stream("Skip SIGTERM handler: not in main thread.")
            return
        if os.environ.get("TUNE_ORIG_WORKING_DIR") or os.environ.get("RAY_ADDRESS"):
            self.gandalf_logger.log_info_stream("Skip SIGTERM handler: running under Ray/Tune.")
            return

        self._terminate_requested = False

        def _handler(signum, frame):
            ep = getattr(self, "current_epoch", 0)
            try:
                self.gandalf_logger.log_info_stream("SIGTERM/SIGINT – checkpoint & graceful stop requested.")
                self.save_checkpoint(ep)
            except Exception as e:
                self.gandalf_logger.log_info_stream(f"SIGTERM – checkpoint failed: {e}")
            self._terminate_requested = True

        try:
            signal.signal(signal.SIGTERM, _handler)
            signal.signal(signal.SIGINT, _handler)
            self.gandalf_logger.log_info_stream("Installed SIGTERM/SIGINT handler (graceful).")
        except ValueError as e:
            self.gandalf_logger.log_info_stream(f"Skip SIGTERM handler ({e}).")

    def run_training(self, on_epoch_end=None):
        today = datetime.now().strftime("%Y%m%d_%H%M%S")
        min_delta = 1e-4
        patience = int(self.cfg.get("EARLY_STOP_PATIENCE_FLOW", self.cfg.get("EARLY_STOP_PATIENCE", 0)))
        epochs_no_improve = 0

        self._finalized = False

        if self.start_epoch >= int(self.cfg["EPOCHS_FLOW"]):
            self.gandalf_logger.log_info_stream(
                f"No epochs to run (start_epoch={self.start_epoch} >= EPOCHS_FLOW={self.cfg['EPOCHS_FLOW']})."
            )
            if not self._already_plotted:
                try:
                    self.model.load_state_dict(self.best_model_state_dict)
                    self.run_tests(epoch=max(self.start_epoch, 1), today=today)
                    self._already_plotted = True
                except Exception as e:
                    self.gandalf_logger.log_info_stream(f"run_tests failed: {e}")
            self._finalize_and_log(ep_for_plots=max(self.start_epoch, 1), reason="no_epochs")
            return self.best_validation_loss

        last_epoch = None
        for epoch in range(self.start_epoch, self.cfg["EPOCHS_FLOW"]):
            self.current_epoch = epoch
            last_epoch = epoch

            self.gandalf_logger.log_info_stream(f"Epoch: {epoch + 1}/{self.cfg['EPOCHS_FLOW']}")
            self.gandalf_logger.log_info_stream("Train")
            train_loss_epoch = self.train(epoch=epoch)

            self.gandalf_logger.log_info_stream("Validation")
            validation_loss = self.validate(epoch=epoch)

            self.lst_epochs.append(epoch)
            self.lst_train_loss_per_epoch.append(train_loss_epoch)
            self.lst_valid_loss_per_epoch.append(validation_loss)

            if validation_loss < self.best_validation_loss - min_delta:
                self.best_validation_epoch = epoch
                self.best_validation_loss = float(validation_loss)
                self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if train_loss_epoch < self.best_train_loss - min_delta:
                self.best_train_epoch = epoch
                self.best_train_loss = float(train_loss_epoch)

            if on_epoch_end is not None:
                try:
                    on_epoch_end(
                        epoch=epoch,
                        train_loss=float(train_loss_epoch),
                        val_loss=float(validation_loss),
                    )
                except Exception as e:
                    self.gandalf_logger.log_info_stream(f"on_epoch_end failed: {e}")

            self.save_checkpoint(epoch)

            term_flag = bool(getattr(self, "_terminate_requested", False))
            early_flag = bool(patience and epochs_no_improve >= patience)
            last_flag = bool((epoch + 1) >= int(self.cfg["EPOCHS_FLOW"]))
            should_finish_now = term_flag or early_flag or last_flag

            if should_finish_now:
                reason = "sigterm" if term_flag else ("early_stop" if early_flag else "last_epoch")
                last_state_dict = copy.deepcopy(self.model.state_dict())

                if not self._already_plotted:
                    try:
                        self.gandalf_logger.log_info_stream(
                            f"Trigger run_tests at epoch {epoch + 1} (reason={reason})."
                        )
                        self.model.load_state_dict(self.best_model_state_dict)
                        self.run_tests(epoch=epoch + 1, today=today)
                        self._already_plotted = True
                    except Exception as e:
                        self.gandalf_logger.log_info_stream(f"run_tests failed (in-loop): {e}")

                self._finalize_and_log(ep_for_plots=epoch + 1, reason=reason, last_state_dict=last_state_dict)
                break

        if not self._finalized:
            try:
                if not self._already_plotted:
                    self.model.load_state_dict(self.best_model_state_dict)
                    ep_for_plots = (last_epoch + 1) if last_epoch is not None else max(self.start_epoch, 1)
                    self.gandalf_logger.log_info_stream(f"Fallback run_tests after loop at epoch {ep_for_plots}.")
                    self.run_tests(epoch=ep_for_plots, today=today)
                    self._already_plotted = True
            except Exception as e:
                self.gandalf_logger.log_info_stream(f"run_tests (post-loop) failed: {e}")

            self._finalize_and_log(
                ep_for_plots=(last_epoch + 1) if last_epoch is not None else max(self.start_epoch, 1),
                reason="post_loop"
            )

        return self.best_validation_loss

    def train(self, epoch):
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 0.1

        train_loss = 0.0
        total_samples = 0

        df_train = self.galaxies.train_dataset

        if self.cfg["DROPPED"] is True:
            df_train[self.cfg["INPUT_COLS"]] = df_train[self.cfg["INPUT_COLS"]].astype(np.float64)
            df_train[self.cfg["OUTPUT_COLS"]] = df_train[self.cfg["OUTPUT_COLS"]].astype(np.float64)

        input_data = torch.tensor(df_train[self.cfg["INPUT_COLS"]].values, dtype=self.dtype)
        output_data = torch.tensor(df_train[self.cfg["OUTPUT_COLS"]].values, dtype=self.dtype)

        train_dataset = TensorDataset(input_data, output_data)
        train_loader = DataLoader(train_dataset, batch_size=self.bs, shuffle=True)

        dev_type = "cuda" if self.device.type == "cuda" else ("mps" if self.device.type == "mps" else "cpu")

        for input_data, output_data in train_loader:
            input_data = input_data.to(self.device)
            output_data = output_data.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=dev_type,
                                    dtype=(self.autocast_dtype if dev_type == "cuda" else None),
                                    enabled=(self.use_amp if dev_type == "cuda" else False)):
                loss = -self.model.log_probs(output_data, input_data).mean()

            if self.use_amp:
                self.cuda_scaler.scale(loss).backward()
                self.cuda_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, error_if_nonfinite=False)
                self.cuda_scaler.step(self.optimizer)
                self.cuda_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, error_if_nonfinite=False)
                self.optimizer.step()

            bs_curr = output_data.size(0)
            total_samples += bs_curr
            train_loss += loss.item() * bs_curr
            self.lst_train_loss_per_batch.append(train_loss / max(1, total_samples))
            self.global_step += 1

            if getattr(self, "_terminate_requested", False):
                break

        train_loss = train_loss / len(self.galaxies.train_dataset)
        self.gandalf_logger.log_info_stream(f"Training,\t"
                                               f"Epoch: {epoch + 1},\t"
                                               f"learning rate: {self.lr},\t"
                                               f"number hidden: {self.nh},\t"
                                               f"number blocks: {self.nb},\t"
                                               f"number layers: {self.nl},\t"
                                               f"batch size: {self.bs},\t"
                                               f"training loss: {train_loss}")

        self.model.train()
        for module in self.model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 0
        with torch.no_grad():
            y = torch.tensor(
                self.galaxies.train_dataset[self.cfg["OUTPUT_COLS"]].values,
                dtype=self.dtype, device=self.device
            )
            x = torch.tensor(
                df_train[self.cfg["INPUT_COLS"]].values,
                dtype=self.dtype, device=self.device
            )
            self.model(y, cond_inputs=x)
        for module in self.model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 0.1
        self.model.eval()

        return train_loss

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        total_samples = 0

        df_valid = self.galaxies.valid_dataset

        if self.cfg["DROPPED"] is True:
            df_valid[self.cfg["INPUT_COLS"]] = df_valid[self.cfg["INPUT_COLS"]].astype(np.float64)
            df_valid[self.cfg["OUTPUT_COLS"]] = df_valid[self.cfg["OUTPUT_COLS"]].astype(np.float64)

        input_data = torch.tensor(df_valid[self.cfg["INPUT_COLS"]].values, dtype=self.dtype)
        output_data = torch.tensor(df_valid[self.cfg["OUTPUT_COLS"]].values, dtype=self.dtype)

        valid_dataset = TensorDataset(input_data, output_data)
        valid_loader = DataLoader(valid_dataset, batch_size=self.bs, shuffle=False)

        dev_type = "cuda" if self.device.type == "cuda" else ("mps" if self.device.type == "mps" else "cpu")

        for input_data, output_data in valid_loader:
            input_data = input_data.to(self.device)
            output_data = output_data.to(self.device)

            with torch.no_grad():
                with torch.amp.autocast(device_type=dev_type,
                                        dtype=(self.autocast_dtype if dev_type == "cuda" else None),
                                        enabled=(self.use_amp if dev_type == "cuda" else False)):
                    loss = -self.model.log_probs(output_data, input_data).mean()

                bs_curr = output_data.size(0)
                val_loss += loss.item() * bs_curr

            total_samples += bs_curr
            self.lst_valid_loss_per_batch.append(val_loss / max(1, total_samples))

            if getattr(self, "_terminate_requested", False):
                break

        val_loss = val_loss / len(self.galaxies.valid_dataset)
        self.gandalf_logger.log_info_stream(f"Validation,\t"
                                               f"Epoch: {epoch + 1},\t"
                                               f"learning rate: {self.lr},\t"
                                               f"number hidden: {self.nh},\t"
                                               f"number blocks: {self.nb},\t"
                                               f"number layers: {self.nl},\t"
                                               f"batch size: {self.bs},\t"
                                               f"validation loss: {val_loss}")
        return val_loss

    def run_tests(self, epoch, today):
        """"""
        plt.rcParams["text.usetex"] = False

        self.gandalf_logger.log_info_stream("Plot test")
        self.model.eval()
        df_balrog = self.galaxies.test_dataset

        if self.cfg["DROPPED"] is True:
            df_balrog[self.cfg["INPUT_COLS"]] = df_balrog[self.cfg["INPUT_COLS"]].astype(np.float64)
            df_balrog[self.cfg["OUTPUT_COLS"]] = df_balrog[self.cfg["OUTPUT_COLS"]].astype(np.float64)

        input_data = torch.tensor(df_balrog[self.cfg["INPUT_COLS"]].values, dtype=self.dtype)
        output_data = torch.tensor(df_balrog[self.cfg["OUTPUT_COLS"]].values, dtype=self.dtype)

        df_gandalf = df_balrog.copy()

        input_data = input_data.to(self.device)
        dev_type = "cuda" if self.device.type == "cuda" else ("mps" if self.device.type == "mps" else "cpu")
        with torch.no_grad():
            with torch.amp.autocast(device_type=dev_type,
                                    dtype=(self.autocast_dtype if dev_type == "cuda" else None),
                                    enabled=(self.use_amp if dev_type == "cuda" else False)):
                arr_gandalf_output = self.model.sample(len(input_data), cond_inputs=input_data).detach()

        output_data_np = arr_gandalf_output.cpu().numpy()

        input_data_np_true = input_data.cpu().numpy()
        output_data_np_true = output_data.cpu().numpy()
        arr_all_true = np.concatenate([input_data_np_true, output_data_np_true], axis=1)
        arr_all = np.concatenate([input_data_np_true, output_data_np], axis=1)


        df_output_true = pd.DataFrame(arr_all_true, columns=list(self.cfg["INPUT_COLS"]) + list(self.cfg["OUTPUT_COLS"]))
        df_output_true = df_output_true[self.cfg["NF_COLUMNS_OF_INTEREST"]]

        df_output_gandalf = pd.DataFrame(arr_all, columns=list(self.cfg["INPUT_COLS"]) + list(self.cfg["OUTPUT_COLS"]))
        df_output_gandalf = df_output_gandalf[self.cfg["NF_COLUMNS_OF_INTEREST"]]

        df_output_true = self.galaxies.inverse_scale_data(df_output_true)
        df_output_gandalf = self.galaxies.inverse_scale_data(df_output_gandalf)

        if self.cfg['PLOT_TRAINING_FEATURES'] is True:
            plot_features(
                cfg=self.cfg,
                plot_log=self.gandalf_logger,
                df_gandalf=df_output_gandalf,
                df_balrog=df_output_true,
                columns=self.cfg["OUTPUT_PLOT_COLS"],
                title_prefix=f"epoch {epoch}; bs {self.bs}; lr {self.lr:.6f}; nh {self.nh}; nb {self.nb} ; nl {self.nl}",
                savename=f"{self.cfg['PATH_PLOTS_FOLDER']['FEATURE_HIST_PLOT']}/{today}_{epoch}_compare_output.pdf"
            )

        self.gandalf_logger.log_info_stream(f"gaNdalF NaNs: {df_gandalf.isna().sum()}")

        if self.cfg['PLOT_CHAIN_FLOW'] is True:
            try:
                img_grid, self.dict_delta_unsheared = plot_compare_corner(
                    data_frame_generated=df_output_gandalf,
                    data_frame_true=df_output_true,
                    dict_delta=self.dict_delta_unsheared,
                    epoch=epoch,
                    title=f"chain plot",
                    show_plot=self.cfg['SHOW_PLOT'],
                    save_plot=self.cfg['SAVE_PLOT'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['CHAIN_PLOT']}/{today}_{epoch}_chainplot.pdf",
                    columns=self.cfg["OUTPUT_PLOT_COLS"],
                    labels=[
                        f"mag_r",
                        f"mag_i",
                        f"mag_z",
                        f"log10(mag_err_r)",
                        f"log10(mag_err_i)",
                        f"log10(mag_err_z)",
                        "e_1",
                        "e_1",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    ranges=None, # [(15, 30), (15, 30), (15, 30), (-15, 75), (-1.5, 4), (-1.5, 2), (-8, 8), (-8, 8)]
                )
                # self.writer.add_image("chain plot", img_grid, epoch + 1)
            except Exception as e:
                self.gandalf_logger.log_info_stream(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['CHAIN_PLOT']}/chainplot_{epoch}.pdf")

        for b in self.cfg['BANDS_FLOW']:
            df_output_gandalf[f"meas {b} - true {b}"] = df_output_gandalf[f"unsheared/mag_{b}"] - df_output_gandalf[f"BDF_MAG_DERED_CALIB_{b.upper()}"]
            df_output_true[f"meas {b} - true {b}"] = df_output_true[f"unsheared/mag_{b}"] - df_output_true[f"BDF_MAG_DERED_CALIB_{b.upper()}"]

        df_output_gandalf = calc_color(
            data_frame=df_output_gandalf,
            colors=self.cfg['COLORS_FLOW'],
            column_name=f"unsheared/mag"
        )
        df_output_true = calc_color(
            data_frame=df_output_true,
            colors=self.cfg['COLORS_FLOW'],
            column_name=f"unsheared/mag"
        )

        if self.cfg['PLOT_COLOR_COLOR_FLOW'] is True:
            try:
                img_grid, self.dict_delta_color_color = plot_compare_corner(
                    data_frame_generated=df_output_gandalf,
                    data_frame_true=df_output_true,
                    dict_delta=self.dict_delta_color_color,
                    epoch=epoch,
                    title=f"color-color plot",
                    columns=["r-i", "i-z"],
                    labels=["r-i", "i-z"],
                    show_plot=self.cfg['SHOW_PLOT'],
                    save_plot=self.cfg['SAVE_PLOT'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['COLOR_COLOR_PLOT']}/color_color_{epoch}.pdf",
                    ranges=[(-4, 4), (-4, 4)]
                )
                # self.writer.add_image("color color plot", img_grid, epoch + 1)
            except Exception as e:
                self.gandalf_logger.log_info_stream(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['COLOR_COLOR_PLOT']}/color_color_{epoch}.pdf")

        if self.cfg["PLOT_BINNING_STATS"] is True:
            plot_binning_statistics_combined(
                df_gandalf=df_output_gandalf,
                df_balrog=df_output_true,
                sample_size=10000,
                plot_scatter=False,
                show_plot=self.cfg["SHOW_PLOT"],
                save_plot=self.cfg["SAVE_PLOT"],
                title="gaNdalF vs. Balrog: Measured Photometric Property Distribution Comparison",
                save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['BINNING_STATS']}/binning_statistics_combined_{epoch}",
            )

        if self.cfg["PLOT_HIST_W_ERROR"] is True:
            plot_balrog_histogram_with_error(
                df_gandalf=df_output_gandalf,
                df_balrog=df_output_true,
                columns=self.cfg["HIST_PLOT_COLS"],
                labels=[
                    "mag r-i",
                    "mag i-z",
                    "mag r",
                    "mag i",
                    "mag z",
                    "log10(mag err r)",
                    "log10(mag err i)",
                    "log10(mag err z)",
                    "e_1",
                    "e_2",
                    "snr",
                    "size ratio",
                    "T"
                ],
                ranges=[
                    [-0.5, 1.5],  # mag r-i
                    [-0.5, 1.5],  # mag i-z
                    [18, 24.5],  # mag r
                    [18, 24.5],  # mag i
                    [18, 24.5],  # mag z
                    None,  # mag err r
                    None,  # mag err i
                    None,  # mag err z
                    None,  # e_1
                    None,  # e_2
                    [2, 100],  # snr
                    [-0.5, 5],  # size ratio
                    [0, 3.5]  # T
                ],
                binwidths=[
                    0.08,  # mag r-i
                    0.08,  # mag i-z
                    None,  # mag r
                    None,  # mag i
                    None,  # mag z
                    None,  # mag err r
                    None,  # mag err i
                    None,  # mag err z
                    None,  # e_1
                    None,  # e_2
                    2,  # snr
                    0.2,  # size ratio
                    0.2,  # T
                ],
                title=r"gaNdalF vs. Balrog: Property Distribution Comparison",
                show_plot=self.cfg["SHOW_PLOT"],
                save_plot=self.cfg["SAVE_PLOT"],
                save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['HIST_W_ERROR']}/hist_plot_{epoch}.pdf"
            )

    def _finalize_and_log(self, ep_for_plots: int, reason: str, last_state_dict=None):
        if self._finalized:
            return
        try:
            self.gandalf_logger.log_info_stream(f"Finalize training at epoch {ep_for_plots} (reason={reason}).")
            self.gandalf_logger.log_info_stream("End Training")
            self.gandalf_logger.log_info_stream(
                f"Best validation epoch: {self.best_validation_epoch + 1}\t"
                f"best validation loss: {self.best_validation_loss}\t"
                f"learning rate: {self.lr}\t"
                f"num_hidden: {self.nh}\t"
                f"num_blocks: {self.nb}\t"
                f"num_layers: {self.nl}\t"
                f"batch_size: {self.bs}"
            )
            if self.cfg.get('SAVE_NN_FLOW', False):
                try:
                    torch.save(
                        self.best_model_state_dict,
                        f"{self.cfg['PATH_SAVE_NN']}/best_model_state_e_{self.best_validation_epoch + 1}"
                        f"_lr_{self.lr}_bs_{self.bs}_run_{self.cfg['RUN_DATE']}.pt"
                    )
                    torch.save(
                        (last_state_dict if last_state_dict is not None else self.model.state_dict()),
                        f"{self.cfg['PATH_SAVE_NN']}/last_model_state_e_{self.cfg['EPOCHS_FLOW']}"
                        f"_lr_{self.lr}_bs_{self.bs}_run_{self.cfg['RUN_DATE']}.pt"
                    )
                except Exception as e:
                    self.gandalf_logger.log_info_stream(f"Saving models failed: {e}")
        finally:
            self._finalized = True