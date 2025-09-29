import torch
import os
import seaborn as sns
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from Handler import plot_classification_results, plot_confusion_matrix, plot_roc_curve, plot_recall_curve, plot_probability_hist, plot_multivariate_clf, LoggerHandler
from sklearn.metrics import accuracy_score, f1_score
from gandalf_galaxie_dataset import DESGalaxies
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import random
import logging
from datetime import datetime
import copy
import csv


class gaNdalFClassifier(nn.Module):
    def __init__(self,
                 cfg,
                 batch_size,
                 learning_rate,
                 hidden_sizes,
                 dropout_prob,
                 batch_norm
                 ):
        super().__init__()
        self.cfg = cfg
        self.best_loss = float('inf')
        self.best_acc = 0.0
        self.best_epoch = 0
        self.lst_epochs = []
        self.lst_valid_acc_per_epoch = []

        self.bs = int(batch_size)
        self.lr = float(learning_rate)
        self.hs = list(hidden_sizes)
        self.nl = len(self.hs)
        self.ubn = bool(batch_norm)
        self.dp = float(dropout_prob)

        self.activation = nn.ReLU
        self.weight_decay = float(self.cfg.get("WEIGHT_DECAY", 1e-4))
        self.number_hidden = []
        self.lst_loss = []
        self.lst_train_loss_per_batch = []
        self.lst_train_loss_per_epoch = []
        self.lst_valid_loss_per_batch = []
        self.lst_valid_loss_per_epoch = []
        self.lst_valid_f1_per_epoch = []
        self._finalized = False
        self.make_dirs()

        log_lvl = logging.INFO
        if self.cfg["LOGGING_LEVEL"] == "DEBUG":
            log_lvl = logging.DEBUG
        elif self.cfg["LOGGING_LEVEL"] == "ERROR":
            log_lvl = logging.ERROR

        self.classifier_logger = LoggerHandler(
            logger_dict={"logger_name": "train classifier logger",
                         "info_logger": self.cfg['INFO_LOGGER'],
                         "error_logger": self.cfg['ERROR_LOGGER'],
                         "debug_logger": self.cfg['DEBUG_LOGGER'],
                         "stream_logger": self.cfg['STREAM_LOGGER'],
                         "stream_logging_level": log_lvl},
            log_folder_path=f"{self.cfg['PATH_LOGS']}/"
        )

        dev_str = str(self.cfg.get("DEVICE_CLASSF", "auto")).lower()
        if dev_str == "auto":
            if torch.cuda.is_available():
                dev_str = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev_str = "mps"
            else:
                dev_str = "cpu"

        self.device = torch.device(dev_str)
        if self.device.type == "cuda":
            self.classifier_logger.log_info_stream(
                f"Using CUDA device: {torch.cuda.get_device_name(0)} "
                f"(visible: {os.environ.get('CUDA_VISIBLE_DEVICES')})"
            )
            torch.backends.cudnn.benchmark = True
        elif self.device.type == "cpu" and dev_str.startswith("cuda"):
            self.classifier_logger.log_info_stream("CUDA not available – using CPU.")

        self.galaxies = self.init_dataset()

        self.model = self.init_network(
            input_dim=len(self.cfg['INPUT_COLS']),
            output_dim=1
        ).float().to(self.device)

        self.best_model_state_dict = copy.deepcopy(self.model.state_dict())

        # self.loss_function = nn.BCELoss()
        n_positive = int((self.galaxies.train_dataset["detected"] == 1).sum())
        n_negative = int((self.galaxies.train_dataset["detected"] == 0).sum())
        pos_weight = torch.tensor([n_negative / max(1, n_positive)], device=self.device, dtype=torch.float32)
        self.loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.classifier_logger.log_info_stream(f"#########################################################################")
        self.classifier_logger.log_info_stream(f"Hidden Sizes: {self.number_hidden}")
        self.classifier_logger.log_info_stream(f"Activation Functions: {self.activation}")
        self.classifier_logger.log_info_stream(f"Learning Rate: {self.lr}")
        self.classifier_logger.log_info_stream(f"Batch Size: {self.bs}")

        self.loss = 0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.global_step = 0
        self.best_validation_loss = np.float64('inf')
        self.best_validation_epoch = 1
        self.best_train_loss = np.float64('inf')
        self.best_train_epoch = 1
        self.best_model = self.model
        self._terminate_requested = False
        self._already_plotted = False

        self.use_amp = bool(self.cfg.get("AMP", True)) and (self.device.type == "cuda")
        self.cuda_scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.load_checkpoint_if_any()

        if bool(self.cfg.get("TENSORBOARD_GRAPH", True)):
            try:
                from torch.utils.tensorboard import SummaryWriter
                dummy = torch.zeros(1, len(self.cfg['INPUT_COLS']), dtype=torch.float32).to(self.device)
                writer = SummaryWriter(log_dir=self.cfg['PATH_OUTPUT'])
                self.model.eval()
                writer.add_graph(self.model, dummy)
                writer.flush()
                writer.close()
                self.classifier_logger.log_info_stream("TensorBoard: graph written (add_graph).")
            except Exception as e:
                self.classifier_logger.log_info_stream(f"TensorBoard-Graph error writing graph: {e}")


        self._install_sigterm_handler()

    def init_dataset(self):
        """"""
        galaxies = DESGalaxies(
            dataset_logger=self.classifier_logger,
            cfg=self.cfg
        )

        galaxies.scale_data()
        return galaxies

    def init_network(self, input_dim, output_dim):
        if isinstance(self.activation, str):
            ActClass = getattr(nn, self.activation)
            make_act = lambda: ActClass()
        elif isinstance(self.activation, type) and issubclass(self.activation, nn.Module):
            make_act = lambda: self.activation()
        elif callable(self.activation):
            make_act = self.activation
        else:
            raise ValueError(f"Unsupported activation spec: {self.activation}")

        layers = []
        in_features = input_dim

        assert self.nl == len(self.hs), \
            f"len(hidden_sizes)={len(self.hs)} != number_layer={self.nl}"

        for out_features in self.hs:
            out_features = int(out_features)
            self.number_hidden.append(out_features)

            layers.append(nn.Linear(in_features, out_features))

            if self.ubn:
                layers.append(nn.BatchNorm1d(out_features))

            layers.append(make_act())
            if self.dp > 0.0:
                layers.append(nn.Dropout(self.dp))

            in_features = out_features
        layers.append(nn.Linear(in_features, output_dim))
        # layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

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
                "lr": self.lr, "bs": self.bs, "hs": self.hs, "ubn": self.ubn, "dp": self.dp, "nl": self.nl,
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
        self.classifier_logger.log_info_stream(f"Resumed from checkpoint at epoch {self.start_epoch}")
        return True

    def _install_sigterm_handler(self):
        import threading, signal
        if threading.current_thread() is not threading.main_thread():
            self.classifier_logger.log_info_stream("Skip SIGTERM handler: not in main thread.")
            return

        self._terminate_requested = False

        def _handler(signum, frame):
            ep = getattr(self, "current_epoch", 0)
            try:
                self.classifier_logger.log_info_stream("SIGTERM – checkpoint & graceful stop requested.")
                self.save_checkpoint(ep)
            except Exception as e:
                self.classifier_logger.log_info_stream(f"SIGTERM – checkpoint failed: {e}")
            # Kein os._exit(0)! Wir beenden die Epoche und plotten danach:
            self._terminate_requested = True

        try:
            signal.signal(signal.SIGTERM, _handler)
            signal.signal(signal.SIGINT, _handler)  # falls manuell abgebrochen wird
            self.classifier_logger.log_info_stream("Installed SIGTERM handler (graceful).")
        except ValueError as e:
            self.classifier_logger.log_info_stream(f"Skip SIGTERM handler ({e}).")

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

        for plot in self.cfg['PLOTS_CF']:
            self.cfg[f'PATH_PLOTS_FOLDER'][plot.upper()] = f"{self.cfg['PATH_PLOTS']}/{plot}"

        if self.cfg['PLOT_TRAINING'] is True:
            os.makedirs(self.cfg['PATH_PLOTS'], exist_ok=True)
            for path_plot in self.cfg['PATH_PLOTS_FOLDER'].values():
                os.makedirs(path_plot, exist_ok=True)
        if self.cfg['SAVE_NN'] is True:
            os.makedirs(self.cfg['PATH_SAVE_NN'], exist_ok=True)

    def run_training(self, on_epoch_end=None):
        """"""
        today = datetime.now().strftime("%Y%m%d_%H%M%S")
        min_delta = 1e-4
        patience = int(self.cfg.get("EARLY_STOP_PATIENCE", 0))
        epochs_no_improve = 0

        self._finalized = False

        if self.start_epoch >= int(self.cfg["EPOCHS"]):
            self.classifier_logger.log_info_stream(
                f"No epochs to run (start_epoch={self.start_epoch} >= EPOCHS={self.cfg['EPOCHS']})."
            )
            if not self._already_plotted:
                try:
                    self.model.load_state_dict(self.best_model_state_dict)
                    self.run_tests(epoch=max(self.start_epoch, 1), today=today)
                    self._already_plotted = True
                except Exception as e:
                    self.classifier_logger.log_info_stream(f"run_tests failed: {e}")
            self._finalize_and_log(ep_for_plots=max(self.start_epoch, 1), reason="no_epochs")
            return self.best_validation_loss

        last_epoch = None
        for epoch in range(self.start_epoch, self.cfg["EPOCHS"]):
            self.current_epoch = epoch
            last_epoch = epoch

            self.classifier_logger.log_info_stream(f"Epoch: {epoch + 1}/{self.cfg['EPOCHS']}")
            self.classifier_logger.log_info_stream("Train")
            train_loss_epoch = self.train_cf(epoch=epoch)

            self.classifier_logger.log_info_stream("Validation")
            val_loss, val_acc, val_f1 = self.validate_cf(epoch=epoch)

            self.lst_epochs.append(epoch)
            self.lst_train_loss_per_epoch.append(train_loss_epoch)
            self.lst_valid_loss_per_epoch.append(val_loss)
            self.lst_valid_acc_per_epoch.append(val_acc)
            self.lst_valid_f1_per_epoch.append(val_f1)

            if val_loss < self.best_validation_loss - min_delta:
                self.best_validation_loss = float(val_loss)
                self.best_validation_epoch = epoch
                self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if train_loss_epoch < self.best_train_loss - min_delta:
                self.best_train_loss = float(train_loss_epoch)
                self.best_train_epoch = epoch

            self.save_checkpoint(epoch)

            term_flag = bool(getattr(self, "_terminate_requested", False))
            early_flag = bool(patience and epochs_no_improve >= patience)
            last_flag = bool((epoch + 1) >= int(self.cfg["EPOCHS"]))
            should_finish_now = term_flag or early_flag or last_flag

            if should_finish_now:
                reason = "sigterm" if term_flag else ("early_stop" if early_flag else "last_epoch")

                last_state_dict = copy.deepcopy(self.model.state_dict())

                if not self._already_plotted:
                    try:
                        self.classifier_logger.log_info_stream(
                            f"Trigger run_tests at epoch {epoch + 1} (reason={reason})."
                        )
                        self.model.load_state_dict(self.best_model_state_dict)
                        self.run_tests(epoch=epoch + 1, today=today)
                        self._already_plotted = True
                    except Exception as e:
                        self.classifier_logger.log_info_stream(f"run_tests failed (in-loop): {e}")

                self._finalize_and_log(ep_for_plots=epoch + 1, reason=reason, last_state_dict=last_state_dict)

            if on_epoch_end is not None:
                try:
                    on_epoch_end(
                        epoch=epoch,
                        train_loss=float(train_loss_epoch),
                        val_loss=float(val_loss),
                        accuracy=float(val_acc),
                        f1=float(val_f1),
                    )
                except BaseException as e:
                    if not self._finalized:
                        self._finalize_and_log(ep_for_plots=epoch + 1, reason=f"on_epoch_end_raised:{type(e).__name__}")
                    raise

            if should_finish_now:
                break

        if not self._finalized:
            try:
                if not self._already_plotted:
                    self.model.load_state_dict(self.best_model_state_dict)
                    ep_for_plots = (last_epoch + 1) if last_epoch is not None else max(self.start_epoch, 1)
                    self.classifier_logger.log_info_stream(f"Fallback run_tests after loop at epoch {ep_for_plots}.")
                    self.run_tests(epoch=ep_for_plots, today=today)
                    self._already_plotted = True
            except Exception as e:
                self.classifier_logger.log_info_stream(f"run_tests (post-loop) failed: {e}")
            self._finalize_and_log(ep_for_plots=(last_epoch + 1) if last_epoch is not None else max(self.start_epoch, 1),
                              reason="post_loop")

        return self.best_validation_loss

    def train_cf(self, epoch):
        """"""
        self.model.train()
        train_loss = 0.0
        seen = 0

        df_train = self.galaxies.train_dataset

        input_data = torch.tensor(df_train[self.cfg["INPUT_COLS"]].values, dtype=torch.float32)
        output_data = torch.tensor(df_train[self.cfg["OUTPUT_COLS"]].values, dtype=torch.float32)

        train_dataset = TensorDataset(input_data, output_data)

        pin = (self.device.type == "cuda")
        nw = max(1, os.cpu_count() // 2)
        train_loader = DataLoader(train_dataset, batch_size=self.bs, shuffle=True,
                                  num_workers=nw, pin_memory=pin, persistent_workers=True)

        for input_data, output_data in train_loader:
            input_data = input_data.to(self.device, non_blocking=pin)
            output_data = output_data.to(self.device, non_blocking=pin)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(input_data)
                loss = self.loss_function(outputs.squeeze(), output_data.squeeze()).mean()

            if self.use_amp:
                self.cuda_scaler.scale(loss).backward()
                self.cuda_scaler.step(self.optimizer)
                self.cuda_scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            bs_curr = output_data.size(0)
            train_loss += loss.item() * bs_curr
            seen += bs_curr

            if getattr(self, "_terminate_requested", False):
                break

        avg_train_loss = train_loss / max(1, seen)
        self.lst_loss.append(avg_train_loss)
        self.classifier_logger.log_info_stream(f'Epoch {epoch + 1} \t Training Loss: {avg_train_loss:.6f}')
        return avg_train_loss

    def validate_cf(self, epoch):
        self.model.eval()
        val_loss, correct, seen = 0.0, 0, 0

        df_valid = self.galaxies.valid_dataset
        input_data = torch.tensor(df_valid[self.cfg["INPUT_COLS"]].values, dtype=torch.float32)
        output_data = torch.tensor(df_valid[self.cfg["OUTPUT_COLS"]].values, dtype=torch.float32)
        valid_dataset = TensorDataset(input_data, output_data)

        pin = (self.device.type == "cuda")
        nw = max(1, os.cpu_count() // 2)
        valid_loader = DataLoader(valid_dataset, batch_size=self.bs, shuffle=False,
                                  num_workers=nw, pin_memory=pin, persistent_workers=True)

        use_random_threshold = bool(self.cfg.get("VALID_RANDOM_THRESHOLD", False))
        all_true, all_pred = [], []

        with torch.no_grad():
            for input_data, output_data in valid_loader:
                input_data = input_data.to(self.device, non_blocking=pin)
                output_data = output_data.to(self.device, non_blocking=pin)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    logits = self.model(input_data).squeeze()
                    loss = self.loss_function(logits.squeeze(), output_data.squeeze()).mean()

                bs_curr = output_data.size(0)
                val_loss += loss.item() * bs_curr

                probability = torch.sigmoid(logits).float().cpu().numpy()
                detected = (probability > np.random.rand(len(probability))).astype(int) if use_random_threshold \
                    else (probability >= 0.5).astype(int)

                true_detected = output_data.float().cpu().numpy().astype(int).ravel()
                correct += (detected == true_detected).sum()
                seen += true_detected.size

                all_true.append(true_detected)
                all_pred.append(detected)

                if getattr(self, "_terminate_requested", False):
                    break

        val_loss = val_loss / max(1, seen)
        avg_accuracy = correct / max(1, seen)
        all_true = np.concatenate(all_true) if all_true else np.array([], dtype=int)
        all_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=int)
        f1 = f1_score(all_true, all_pred) if all_true.size > 0 else 0.0

        self.classifier_logger.log_info_stream(f"F1 for lr={self.lr}, bs={self.bs}: {f1:.3f}")
        self.classifier_logger.log_info_stream(f"Accuracy for lr={self.lr}, bs={self.bs}: {avg_accuracy * 100.0:.2f}%")
        if val_loss <= self.best_loss:
            self.classifier_logger.log_info_stream(
                f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f})')
            self.best_loss = val_loss
        self.classifier_logger.log_info_stream(f'Epoch {epoch + 1} \t Validation Loss: {val_loss:.4f}')
        return val_loss, avg_accuracy, f1

    def run_tests(self, epoch, today):
        """"""
        self.classifier_logger.log_info_stream("run_tests: start")
        df_test = self.galaxies.test_dataset
        input_data = torch.tensor(df_test[self.cfg["INPUT_COLS"]].values, dtype=torch.float32)
        detected_true = df_test["detected"]

        with torch.no_grad():
            logits = self.model(input_data.to(self.device)).squeeze()
            probability = torch.sigmoid(logits).cpu().numpy()
        detected = (probability > np.random.rand(len(probability))).astype(int)
        detected = detected.astype(int)

        df_test['gandalf_probability'] = probability
        df_test['gandalf_detected'] = detected

        df_test['true_detected'] = detected_true
        df_test['detected_calibrated'] = detected
        df_test['detected_true'] = detected

        accuracy = accuracy_score(detected_true, detected)
        f1_Score = f1_score(detected_true, detected)
        self.classifier_logger.log_info_stream(
            f"Accuracy for lr={self.lr}, bs={self.bs}: {accuracy * 100.0:.2f}%")
        self.classifier_logger.log_info_stream(
            f'f1_Score for lr={self.lr}, bs={self.bs}: {f1_Score:.3f}')

        self.classifier_logger.log_info_stream(f"detected shape: {detected.shape}")
        self.classifier_logger.log_info_stream(f"detected_true shape: {detected_true.shape}")

        false_positives = np.sum((detected == 1) & (detected_true == 0))
        false_negatives = np.sum((detected == 0) & (detected_true == 1))

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.classifier_logger.log_info_stream(f"trainable params: {trainable_params}")
        self.classifier_logger.log_info_stream(f"false positives: {false_positives}")
        self.classifier_logger.log_info_stream(f"false negatives: {false_negatives}")

        lst_cols = [
            ['BDF_MAG_DERED_CALIB_R', 'BDF_MAG_DERED_CALIB_I'],
            ['BDF_MAG_DERED_CALIB_I', 'BDF_MAG_DERED_CALIB_Z'],
            ['Color BDF MAG R-I', 'Color BDF MAG I-Z'],
            ['BDF_T', 'BDF_G'],
            ['FWHM_WMEAN_R', 'FWHM_WMEAN_I'],
            ['FWHM_WMEAN_I', 'FWHM_WMEAN_Z'],
            ['AIRMASS_WMEAN_R', 'AIRMASS_WMEAN_I'],
            ['AIRMASS_WMEAN_I', 'AIRMASS_WMEAN_Z'],
            ['MAGLIM_R', 'MAGLIM_I'],
            ['MAGLIM_I', 'MAGLIM_Z']
        ]

        lst_save_names = [
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_BDF_RI_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_BDF_IZ_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_Color_RI_IZ_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_BDF_TG_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_FWHM_RI_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_FWHM_IZ_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_AIRMASS_RI_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_AIRMASS_IZ_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_MAGLIM_RI_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_MAGLIM_IZ_epoch_{epoch}.png"
        ]

        if self.cfg['PLOT_MISS_CLASSF'] is True:
            for idx_cols, cols in enumerate(lst_cols):
                plot_classification_results(
                    data_frame=df_test,
                    cols=cols,
                    show_plot=self.cfg['SHOW_PLOT'],
                    save_plot=self.cfg['SAVE_PLOT'],
                    save_name=lst_save_names[idx_cols],
                    title=f"Classification Results, lr={self.lr}, bs={self.bs}, epoch={epoch}"
                )

        if self.cfg['PLOT_MATRIX'] is True:
            plot_confusion_matrix(
                data_frame=df_test,
                show_plot=self.cfg['SHOW_PLOT'],
                save_plot=self.cfg['SAVE_PLOT'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'CONFUSION_MATRIX']}/confusion_matrix_epoch_{epoch}.png",
                title=f"Confusion Matrix, lr={self.lr}, bs={self.bs}, epoch={epoch}"
            )

        if self.cfg['PLOT_MULTIVARIATE_CLF'] is True:
            plot_multivariate_clf(
                df_balrog_detected=df_test[df_test['detected_true'] == 1],
                df_gandalf_detected=df_test[df_test['detected_calibrated'] == 1],
                df_balrog_not_detected=df_test[df_test['detected_true'] == 0],
                df_gandalf_not_detected=df_test[df_test['detected_calibrated'] == 0],
                train_plot=False,
                columns={
                    "BDF_MAG_DERED_CALIB_R": {
                        "label": "BDF Mag R",
                        "range": [17.5, 26.5],
                        "position": [0, 0]
                    },
                    "BDF_MAG_DERED_CALIB_Z": {
                        "label": "BDF Mag Z",
                        "range": [17.5, 26.5],
                        "position": [0, 1]
                    },
                    "BDF_T": {
                        "label": "BDF T",
                        "range": [-2, 3],
                        "position": [0, 2]
                    },
                    "BDF_G": {
                        "label": "BDF G",
                        "range": [-0.1, 0.9],
                        "position": [1, 0]
                    },
                    "FWHM_WMEAN_R": {
                        "label": "FWHM R",
                        "range": [0.7, 1.3],
                        "position": [1, 1]
                    },
                    "FWHM_WMEAN_I": {
                        "label": "FWHM I",
                        "range": [0.7, 1.1],
                        "position": [1, 2]
                    },
                    "FWHM_WMEAN_Z": {
                        "label": "FWHM Z",
                        "range": [0.6, 1.16],
                        "position": [2, 0]
                    },
                    "AIRMASS_WMEAN_R": {
                        "label": "AIRMASS R",
                        "range": [0.95, 1.45],
                        "position": [2, 1]
                    },
                    "AIRMASS_WMEAN_I": {
                        "label": "AIRMASS I",
                        "range": [1, 1.45],
                        "position": [2, 2]
                    },
                    "AIRMASS_WMEAN_Z": {
                        "label": "AIRMASS Z",
                        "range": [1, 1.4],
                        "position": [2, 3]
                    },
                    "MAGLIM_R": {
                        "label": "MAGLIM R",
                        "range": [23, 24.8],
                        "position": [3, 0]
                    },
                    "MAGLIM_I": {
                        "label": "MAGLIM I",
                        "range": [22.4, 24.0],
                        "position": [3, 1]
                    },
                    "MAGLIM_Z": {
                        "label": "MAGLIM Z",
                        "range": [21.8, 23.2],
                        "position": [3, 2]
                    },
                    "EBV_SFD98": {
                        "label": "EBV SFD98",
                        "range": [-0.01, 0.10],
                        "position": [3, 3]
                    }
                },
                show_plot=self.cfg["SHOW_PLOT"],
                save_plot=self.cfg["SAVE_PLOT"],
                save_name=f"{self.cfg['PATH_OUTPUT']}/balrog_classifier_multiv.pdf",
                sample_size=None,
                x_range=(17.5, 26.5),
                title=f"nl: {self.nl}; nh: {self.number_hidden}; af: {self.activation}; lr: {self.lr}; bs: {self.bs}; YJ: {self.cfg['APPLY_YJ_TRANSFORM_CLASSF']}; scaler: {self.cfg['APPLY_SCALER_CLASSF']}"
            )

        if self.cfg['PLOT_ROC_CURVE'] is True:
            plot_roc_curve(
                data_frame=df_test,
                show_plot=self.cfg['SHOW_PLOT'],
                save_plot=self.cfg['SAVE_PLOT'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'ROC_CURVE']}/roc_curve_epoch_{epoch}.png",
                title=f"Receiver Operating Characteristic (ROC) Curve, lr={self.lr}, bs={self.bs}, epoch={epoch}"
            )

        if self.cfg['PLOT_PRECISION_RECALL_CURVE'] is True:
            plot_recall_curve(
                data_frame=df_test,
                show_plot=self.cfg['SHOW_PLOT'],
                save_plot=self.cfg['SAVE_PLOT'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'PRECISION_RECALL_CURVE']}/precision_recall_curve_epoch_{epoch}.png",
                title=f"recision-Recall Curve, lr={self.lr}, bs={self.bs}, epoch={epoch}"
            )

        if self.cfg['PLOT_PROBABILITY_HIST'] is True:
            plot_probability_hist(
                data_frame=df_test,
                show_plot=self.cfg['SHOW_PLOT'],
                save_plot=self.cfg['SAVE_PLOT'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'PROB_HIST']}/probability_histogram{epoch}.png",
                title=f"probability histogram, lr={self.lr}, bs={self.bs}, epoch={epoch}"
            )

        self.classifier_logger.log_info_stream("run_tests: done")

    def _finalize_and_log(self, ep_for_plots: int, reason: str, last_state_dict=None):
        if self._finalized:
            return
        try:
            self.classifier_logger.log_info_stream(f"Finalize training at epoch {ep_for_plots} (reason={reason}).")
            self.classifier_logger.log_info_stream("End Training")
            self.classifier_logger.log_info_stream(
                f"Best validation epoch: {self.best_validation_epoch + 1}\t"
                f"best validation loss: {self.best_validation_loss}\t"
                f"learning rate: {self.lr}\t"
                f"hidden_size: {self.hs}\t"
                f"batch_norm: {self.ubn}\t"
                f"dropout_prob: {self.dp}\t"
                f"num_layers: {self.nl}\t"
                f"batch_size: {self.bs}"
            )
            if self.cfg.get('SAVE_NN', False):
                try:
                    torch.save(
                        self.best_model_state_dict,
                        f"{self.cfg['PATH_SAVE_NN']}/best_model_state_e_{self.best_validation_epoch + 1}"
                        f"_lr_{self.lr}_bs_{self.bs}_hs_{self.hs}_ubn_{self.ubn}_dp_{self.dp}"
                        f"_nl_{self.nl}_run_{self.cfg['RUN_DATE']}.pt"
                    )
                    torch.save(
                        (last_state_dict if last_state_dict is not None else self.model.state_dict()),
                        f"{self.cfg['PATH_SAVE_NN']}/last_model_state_e_{self.cfg['EPOCHS']}"
                        f"_lr_{self.lr}_bs_{self.bs}_hs_{self.hs}_ubn_{self.ubn}_dp_{self.dp}"
                        f"_nl_{self.nl}_run_{self.cfg['RUN_DATE']}.pt"
                    )
                except Exception as e:
                    self.classifier_logger.log_info_stream(f"Saving models failed: {e}")
        finally:
            self._finalized = True


if __name__ == '__main__':
    pass
