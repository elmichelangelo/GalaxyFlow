#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import copy
import joblib
import torch
import random
import logging
import numpy as np
from datetime import datetime

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast

from sklearn.metrics import accuracy_score, f1_score, brier_score_loss

from Handler import (
    plot_classification_results,
    plot_confusion_matrix,
    plot_roc_curve_gandalf,
    plot_recall_curve_gandalf,
    LoggerHandler,
    plot_reliability_curve,
    plot_calibration_by_mag_singlepanel,
    plot_rate_ratio_curve,
    plot_rate_ratio_by_mag,
    expected_calibration_error,
    mag_rate_mae,
    neg_log_loss,
    quantile_edges
)
from gandalf_calibration_model.gaNdalF_calibration_model import MagAwarePlatt, ModelWithTemperature
from gandalf_galaxie_dataset import DESGalaxies

class gaNdalFClassifier(nn.Module):
    def __init__(self, cfg, batch_size, learning_rate, hidden_sizes, dropout_prob, batch_norm, weight_decay):
        super().__init__()
        self.cfg = cfg
        self.mag_col = str(self.cfg.get("MAG_COL", "BDF_MAG_DERED_CALIB_I"))

        # Hyperparams
        self.bs = int(batch_size)
        self.lr = float(learning_rate)
        self.hs = list(hidden_sizes)
        self.nl = len(self.hs)
        self.ubn = bool(batch_norm)
        self.dp = float(dropout_prob)
        self.weight_decay = float(weight_decay)
        self.activation = nn.ReLU

        # trackers
        self.lst_epochs = []; self.lst_train_loss_per_epoch = []
        self.lst_valid_loss_per_epoch = []; self.lst_valid_acc_per_epoch = []
        self.lst_valid_f1_per_epoch = []; self.best_validation_loss = float("inf")
        self.best_validation_epoch = 0; self.best_threshold_for_best_model = 0.5
        self.best_model_state_dict = None; self._finalized = False; self._already_plotted = False

        self.lst_valid_f1_cal_per_epoch = []
        self.lst_valid_acc_cal_per_epoch = []
        self.lst_valid_brier_cal_per_epoch = []
        self.lst_valid_nll_cal_per_epoch = []
        self.lst_valid_ece_cal_per_epoch = []
        self.lst_valid_mre_cal_per_epoch = []
        self.lst_valid_gre_cal_per_epoch = []

        self.make_dirs()

        # logger + device
        log_lvl = {"DEBUG": logging.DEBUG, "ERROR": logging.ERROR}.get(self.cfg.get("LOGGING_LEVEL", "INFO"), logging.INFO)
        self.classifier_logger = LoggerHandler(
            logger_dict={
                "logger_name": "train classifier logger",
                "info_logger": self.cfg.get("INFO_LOGGER", True),
                "error_logger": self.cfg.get("ERROR_LOGGER", True),
                "debug_logger": self.cfg.get("DEBUG_LOGGER", True),
                "stream_logger": self.cfg.get("STREAM_LOGGER", True),
                "stream_logging_level": log_lvl,
            },
            log_folder_path=f"{self.cfg['PATH_LOGS']}/",
        )
        dev = str(self.cfg.get("DEVICE", "auto")).lower()

        if int(self.cfg.get("RESOURCE_GPU", 0)) == 0 and dev == "cuda":
            dev = "cpu"

        if dev == "auto":
            if torch.cuda.is_available(): dev = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): dev = "mps"
            else: dev = "cpu"
        self.device = torch.device(dev)
        if self.device.type == "cuda":
            self.classifier_logger.log_info_stream(f"Using CUDA device: {torch.cuda.get_device_name(0)}"); torch.backends.cudnn.benchmark = True

        self.galaxies = self.init_dataset()

        # model
        self.model = self.init_network(input_dim=len(self.cfg["INPUT_COLS"]), output_dim=1).float().to(self.device)
        self.best_model_state_dict = copy.deepcopy(self.model.state_dict())

        # loss with class pos_weight
        n_pos = int((self.galaxies.train_dataset["mcal_galaxy"] == 1).sum())
        n_neg = int((self.galaxies.train_dataset["mcal_galaxy"] == 0).sum())
        pos_weight_tensor = torch.tensor([n_neg / max(1, n_pos)], device=self.device, dtype=torch.float32)
        self.loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor, reduction="none")

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.use_amp = bool(self.cfg.get("AMP", True)) and (self.device.type == "cuda")
        self.cuda_scaler = GradScaler(enabled=self.use_amp)

        # calibration state
        self.temperature_value = 1.0
        self.scaled_model = None
        self.mag_platt = None

        self.load_checkpoint_if_any()
        self._install_sigterm_handler()

    # ---------------- utils ----------------
    def init_dataset(self):
        galaxies = DESGalaxies(dataset_logger=self.classifier_logger, cfg=self.cfg)
        galaxies.scale_data()
        return galaxies

    def init_network(self, input_dim, output_dim):
        layers, in_features = [], input_dim
        self.number_hidden = []
        for out_features in self.hs:
            out_features = int(out_features)
            self.number_hidden.append(out_features)
            layers.append(nn.Linear(in_features, out_features, bias=not self.ubn))
            if self.ubn: layers.append(nn.BatchNorm1d(out_features))
            layers.append(self.activation())
            if self.dp > 0.0: layers.append(nn.Dropout(self.dp))
            in_features = out_features
        layers.append(nn.Linear(in_features, output_dim))
        return nn.Sequential(*layers)

    def _checkpoint_path(self): return os.path.join(self.cfg["PATH_OUTPUT"], "last.ckpt.pt")

    def _state_dict(self, epoch):
        return {
            "epoch": int(epoch),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_validation_loss": float(self.best_validation_loss),
            "best_validation_epoch": int(self.best_validation_epoch),
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
        }

    def compute_metrics(self, probs, y_true, mag, ece_bins=15, val_rate_n_bins=5):
        probs = np.asarray(probs, float).ravel()
        y_true = np.asarray(y_true, int).ravel()
        mag = np.asarray(mag, float).ravel()

        # Best F1 + Acc auf gleichem Grid wie im Modell
        thr_grid = np.linspace(0.0, 1.0, 1001)[1:-1]
        best_f1, best_thr = -1.0, 0.5
        for t in thr_grid:
            pred = (probs >= t).astype(int)
            f1t = f1_score(y_true, pred)
            if f1t > best_f1:
                best_f1, best_thr = f1t, float(t)
        acc_star = ((probs >= best_thr).astype(int) == y_true).mean() if y_true.size else 0.0

        # Proper scoring + Kalibrierung
        brier = float(np.mean((probs - y_true) ** 2)) if y_true.size else 1.0
        nll = neg_log_loss(probs, y_true) if y_true.size else 10.0
        ece = expected_calibration_error(probs, y_true, n_bins=ece_bins) if y_true.size else 1.0

        # mag-abhängige Raten-Fehler
        edges_for_rate = quantile_edges(mag, max(5, int(val_rate_n_bins)))
        mre = mag_rate_mae(probs, y_true, mag, edges_for_rate) if y_true.size else 1.0
        gre = float(abs(probs.mean() - y_true.mean())) if y_true.size else 1.0

        return {
            "thr": best_thr, "f1": float(best_f1), "acc": float(acc_star),
            "brier": brier, "nll": nll, "ece": ece, "mre": mre, "gre": gre
        }

    def save_checkpoint(self, epoch):
        path = self._checkpoint_path(); tmp = path + ".tmp"
        torch.save(self._state_dict(epoch), tmp)
        os.replace(tmp, path)

    def load_checkpoint_if_any(self):
        path = self._checkpoint_path()
        if not os.path.exists(path):
            self.start_epoch = 0; return False
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"]); self.optimizer.load_state_dict(ckpt["optimizer"])
        self.best_validation_loss = ckpt.get("best_validation_loss", self.best_validation_loss)
        self.best_validation_epoch = ckpt.get("best_validation_epoch", self.best_validation_epoch)
        random.setstate(ckpt["rng"]["python"]); np.random.set_state(ckpt["rng"]["numpy"])
        torch.random.set_rng_state(ckpt["rng"]["torch"])
        if torch.cuda.is_available() and ckpt["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
        self.start_epoch = int(ckpt["epoch"]) + 1
        self.classifier_logger.log_info_stream(f"Resumed from checkpoint at epoch {self.start_epoch}")
        return True

    def _install_sigterm_handler(self):
        import threading, signal
        if threading.current_thread() is not threading.main_thread():
            self.classifier_logger.log_info_stream("Skip SIGTERM handler: not in main thread."); return
        self._terminate_requested = False
        def _handler(signum, frame):
            ep = getattr(self, "current_epoch", 0)
            try: self.classifier_logger.log_info_stream("SIGTERM – checkpoint & graceful stop requested."); self.save_checkpoint(ep)
            except Exception as e: self.classifier_logger.log_error(f"SIGTERM – checkpoint failed: {e}")
            self._terminate_requested = True
        try:
            signal.signal(signal.SIGTERM, _handler); signal.signal(signal.SIGINT, _handler)
            self.classifier_logger.log_info_stream("Installed SIGTERM handler (graceful).")
        except ValueError as e:
            self.classifier_logger.log_info_stream(f"Skip SIGTERM handler ({e}).")

    def make_dirs(self):
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
            self.cfg['PATH_PLOTS_FOLDER'][plot.upper()] = f"{self.cfg['PATH_PLOTS']}/{plot}"
            os.makedirs(self.cfg['PATH_PLOTS_FOLDER'][plot.upper()], exist_ok=True)

    # ---------------- train/val/test ----------------
    def run_training(self, on_epoch_end=None):
        today = datetime.now().strftime("%Y%m%d_%H%M%S")
        min_delta = 1e-4
        patience = int(self.cfg.get("EARLY_STOP_PATIENCE", 0))
        epochs_no_improve = 0
        self._finalized = False

        if getattr(self, "start_epoch", 0) >= int(self.cfg["EPOCHS"]):
            self.classifier_logger.log_info_stream(f"No epochs to run (start_epoch={self.start_epoch} >= EPOCHS={self.cfg['EPOCHS']}).")
            if not self._already_plotted:
                try: self.model.load_state_dict(self.best_model_state_dict); self.run_tests(epoch=max(self.start_epoch, 1), today=today); self._already_plotted = True
                except Exception as e: self.classifier_logger.log_error(f"run_tests failed: {e}")
            self._finalize_and_log(ep_for_plots=max(self.start_epoch, 1), reason="no_epochs"); return self.best_validation_loss

        last_epoch = None
        for epoch in range(getattr(self, "start_epoch", 0), self.cfg["EPOCHS"]):
            self.current_epoch = epoch; last_epoch = epoch
            self.classifier_logger.log_info_stream(f"Epoch: {epoch + 1}/{self.cfg['EPOCHS']}")
            train_loss_epoch = self.train_cf(epoch=epoch)

            val_loss, val_acc, val_f1, thr_star, val_extra = self.validate_cf(epoch=epoch)
            val_obj = float(val_extra["val_objective"])

            self.lst_epochs.append(epoch)
            self.lst_train_loss_per_epoch.append(train_loss_epoch)
            self.lst_valid_loss_per_epoch.append(val_loss)
            self.lst_valid_acc_per_epoch.append(val_acc)
            self.lst_valid_f1_per_epoch.append(val_f1)

            # Selektion NACH Obj (kleiner ist besser)
            if (epoch == getattr(self, "start_epoch", 0)) or (
                    val_obj < getattr(self, "best_val_objective", float("inf")) - min_delta):
                self.best_val_objective = val_obj
                self.best_validation_epoch = epoch
                self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
                self.best_threshold_for_best_model = float(thr_star)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            self.save_checkpoint(epoch)

            term_flag = bool(getattr(self, "_terminate_requested", False))
            early_flag = bool(patience and epochs_no_improve >= patience)
            last_flag = bool((epoch + 1) >= int(self.cfg["EPOCHS"]))
            should_finish_now = term_flag or early_flag or last_flag

            if should_finish_now:
                reason = "sigterm" if term_flag else ("early_stop" if early_flag else "last_epoch")
                self.model.load_state_dict(self.best_model_state_dict)
                self.calibrate_cf()
                try:
                    _m_raw, _m_cal = self.validate_cf_calibrated(epoch_plus_one=epoch + 1)
                except Exception as e:
                    self.classifier_logger.log_error(f"validate_cf_calibrated failed: {e}")
                last_state_dict = copy.deepcopy(self.model.state_dict())
                if not self._already_plotted:
                    try: self.run_tests(epoch=epoch + 1, today=today); self._already_plotted = True
                    except Exception as e: self.classifier_logger.log_error(f"run_tests failed (in-loop): {e}")
                self._finalize_and_log(ep_for_plots=epoch + 1, reason=reason, last_state_dict=last_state_dict)

            if on_epoch_end is not None:
                try:
                    on_epoch_end(
                        epoch=epoch,
                        train_loss=float(train_loss_epoch),
                        val_loss=float(val_loss),
                        accuracy=float(val_acc),
                        f1=float(val_f1),
                        brier=float(val_extra["brier"]),
                        nll=float(val_extra["nll"]),
                        ece=float(val_extra["ece"]),
                        mre=float(val_extra["mre"]),
                        gre=float(val_extra["gre"]),
                        val_objective=float(val_extra["val_objective"]),
                    )
                except BaseException as e:
                    if not self._finalized:
                        self._finalize_and_log(ep_for_plots=epoch + 1, reason=f"on_epoch_end_raised:{type(e).__name__}")
                    raise

            if should_finish_now: break

        if not self._finalized:
            try:
                if not self._already_plotted:
                    self.model.load_state_dict(self.best_model_state_dict)
                    ep_for_plots = (last_epoch + 1) if last_epoch is not None else max(getattr(self, "start_epoch", 0), 1)
                    self.run_tests(epoch=ep_for_plots, today=today); self._already_plotted = True
            except Exception as e: self.classifier_logger.log_error(f"run_tests (post-loop) failed: {e}")
            self._finalize_and_log(ep_for_plots=(last_epoch + 1) if last_epoch is not None else max(getattr(self, "start_epoch", 0), 1),
                                   reason="post_loop")
        return self.best_validation_loss

    def train_cf(self, epoch):
        self.model.train()
        df = self.galaxies.train_dataset
        x = torch.tensor(df[self.cfg["INPUT_COLS"]].values, dtype=torch.float32)
        y = torch.tensor(df[self.cfg["OUTPUT_COLS"]].values, dtype=torch.float32).squeeze()
        dataset = TensorDataset(x, y)

        nw_cfg = self.cfg.get("NUM_WORKERS", "auto")
        nw = (os.cpu_count() // 2) if str(nw_cfg).lower() == "auto" else int(nw_cfg)
        pin = bool(self.cfg.get("PIN_MEMORY", True)) and (self.device.type == "cuda")
        persist = bool(self.cfg.get("PERSISTENT_WORKERS", True))
        loader = DataLoader(
            dataset, batch_size=self.bs, shuffle=True,
            num_workers=max(1, nw), pin_memory=pin, persistent_workers=persist
        )

        train_loss, seen = 0.0, 0
        for batch in loader:
            self.optimizer.zero_grad(set_to_none=True)
            xb, yb = batch; wb = None

            xb = xb.to(self.device, non_blocking=pin); yb = yb.to(self.device, non_blocking=pin)
            with autocast(enabled=self.use_amp):
                logits = self.model(xb).squeeze()
                loss_vec = self.loss_function(logits, yb)
                loss = loss_vec.mean()

            if self.use_amp:
                self.cuda_scaler.scale(loss).backward()
                self.cuda_scaler.step(self.optimizer)
                self.cuda_scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            bs_curr = yb.size(0); train_loss += float(loss.item()) * bs_curr; seen += bs_curr
            if getattr(self, "_terminate_requested", False): break

        avg = train_loss / max(1, seen)
        self.lst_train_loss_per_epoch.append(avg)
        self.classifier_logger.log_info_stream(f"Epoch {epoch + 1}\tTraining Loss: {avg:.6f}")
        return avg

    def _best_threshold_f1(self, probs: np.ndarray, y_true: np.ndarray):
        thr_grid = np.linspace(0.0, 1.0, 1001)[1:-1]
        best_f1, best_thr = -1.0, 0.5
        for t in thr_grid:
            f1 = f1_score(y_true, (probs >= t).astype(int))
            if f1 > best_f1: best_f1, best_thr = f1, float(t)
        return best_thr, best_f1

    def validate_cf(self, epoch):
        self.model.eval()
        df = self.galaxies.valid_dataset
        x = torch.tensor(df[self.cfg["INPUT_COLS"]].values, dtype=torch.float32)
        y = torch.tensor(df[self.cfg["OUTPUT_COLS"]].values, dtype=torch.float32).squeeze()
        dataset = TensorDataset(x, y)
        pin = (self.device.type == "cuda")
        nw = max(1, os.cpu_count() // 2)
        loader = DataLoader(dataset, batch_size=self.bs, shuffle=False, num_workers=nw, pin_memory=pin,
                            persistent_workers=True)

        all_probs, all_true = [], [];
        val_loss, seen = 0.0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=pin); yb = yb.to(self.device, non_blocking=pin)
                with autocast(enabled=self.use_amp):
                    logits = self.model(xb).squeeze()
                loss = self.loss_function(logits, yb).mean()
                val_loss += loss.item() * yb.size(0); seen += yb.size(0)
                all_probs.append(torch.sigmoid(logits).float().cpu().numpy().ravel())
                all_true.append(yb.float().cpu().numpy().astype(int).ravel())

        val_loss = val_loss / max(1, seen)
        probs = np.concatenate(all_probs) if all_probs else np.array([], float)
        y_true = np.concatenate(all_true) if all_true else np.array([], int)

        # Klassifikations-Kennzahlen (für Logging)
        thr_star, f1_star = self._best_threshold_f1(probs, y_true) if y_true.size else (0.5, 0.0)
        acc_star = ((probs >= thr_star).astype(int) == y_true).mean() if y_true.size else 0.0

        # Proper scoring + Kalibrationsmaße (auf unkalibrierten probs)
        brier = float(np.mean((probs - y_true) ** 2)) if y_true.size else 1.0
        nll = neg_log_loss(probs, y_true) if y_true.size else 10.0
        ece20 = expected_calibration_error(probs, y_true, n_bins=int(self.cfg.get("ECE_N_BINS", 5))) if y_true.size else 1.0

        mag_val = df[self.mag_col].values.astype(float)
        edges_for_rate = quantile_edges(mag_val, max(5, int(self.cfg.get("VAL_RATE_N_BINS", 5))))
        mre = mag_rate_mae(probs, y_true, mag_val, edges_for_rate) if y_true.size else 1.0
        gre = float(abs(probs.mean() - y_true.mean())) if y_true.size else 1.0  # global rate error

        # Kombiniertes Selektions-Objektiv (gewichtet)
        w_ece = float(self.cfg.get("VALOBJ_W_ECE", 0.5))
        w_mre = float(self.cfg.get("VALOBJ_W_MRE", 1.0))
        w_gre = float(self.cfg.get("VALOBJ_W_GRE", 0.5))
        val_objective = float(brier + w_ece * ece20 + w_mre * mre + w_gre * gre)

        self.classifier_logger.log_info_stream(
            f"Val epoch {epoch + 1}: Acc*={acc_star * 100:.2f} F1*={f1_star:.3f} | "
            f"Brier={brier:.5f} NLL={nll:.5f} ECE={ece20:.5f} MRE={mre:.5f} GRE={gre:.5f} "
            f"| Obj={val_objective:.6f} (w_ece={w_ece}, w_mre={w_mre}, w_gre={w_gre})"
        )
        self.classifier_logger.log_info_stream(f"Epoch {epoch + 1}\tValidation BCE: {val_loss:.4f}")

        self.last_val_extra = {
            "brier": brier,
            "nll": nll,
            "ece": ece20,
            "mre": mre,
            "gre": gre,
            "val_objective": val_objective,
        }
        return val_loss, float(acc_star), float(f1_star), float(thr_star), dict(self.last_val_extra)

    def validate_cf_calibrated(self, epoch_plus_one: int):
        """
        Rechnet die Metriken auf der VALID-Partition mit kalibrierten Wahrscheinlichkeiten neu
        und logged/plotet sie als 'epoch+1'.
        """
        self.model.eval()
        df = self.galaxies.valid_dataset

        # Daten
        x = torch.tensor(df[self.cfg["INPUT_COLS"]].values, dtype=torch.float32).to(self.device)
        y_true = df[self.cfg["OUTPUT_COLS"]].to_numpy().astype(int).ravel()
        mag_val = df[self.mag_col].to_numpy(float)

        # Probs: raw, temp, cal
        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                logits_raw = self.model(x).squeeze()
            p_raw = torch.sigmoid(logits_raw).cpu().numpy().ravel()
            logits_T = logits_raw / max(self.temperature_value, 1e-6)
            p_T = torch.sigmoid(logits_T).cpu().numpy().ravel()

        if self.mag_platt is not None:
            p_cal = self.mag_platt.transform(p_T, mag_val)
        else:
            p_cal = p_T

        # Metriken roh vs kalibriert
        ece_bins = int(self.cfg.get("ECE_N_BINS", 15))
        rate_bins = int(self.cfg.get("VAL_RATE_N_BINS", 5))
        m_raw = self.compute_metrics(p_raw, y_true, mag_val, ece_bins=ece_bins, val_rate_n_bins=rate_bins)
        m_cal = self.compute_metrics(p_cal, y_true, mag_val, ece_bins=ece_bins, val_rate_n_bins=rate_bins)

        # Logging
        self.classifier_logger.log_info_stream(
            f"VAL raw (epoch {epoch_plus_one} pre-cal): "
            f"Acc*={m_raw['acc'] * 100:.2f} F1*={m_raw['f1']:.3f} | "
            f"Brier={m_raw['brier']:.5f} NLL={m_raw['nll']:.5f} ECE={m_raw['ece']:.5f} "
            f"MRE={m_raw['mre']:.5f} GRE={m_raw['gre']:.5f}"
        )
        self.classifier_logger.log_info_stream(
            f"VAL cal (epoch {epoch_plus_one} post-cal): "
            f"Acc*={m_cal['acc'] * 100:.2f} F1*={m_cal['f1']:.3f} | "
            f"Brier={m_cal['brier']:.5f} NLL={m_cal['nll']:.5f} ECE={m_cal['ece']:.5f} "
            f"MRE={m_cal['mre']:.5f} GRE={m_cal['gre']:.5f} | T={getattr(self, 'temperature_value', 1.0):.3f}"
        )

        return m_raw, m_cal

    # ---------------- calibration ----------------
    def calibrate_cf(self):
        df_valid = self.galaxies.valid_dataset
        x_val = torch.tensor(df_valid[self.cfg["INPUT_COLS"]].values, dtype=torch.float32)
        y_val = torch.tensor(df_valid[self.cfg["OUTPUT_COLS"]].values, dtype=torch.float32)
        valid_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=self.bs, shuffle=False)

        # (1) Temperature scaling
        self.scaled_model = ModelWithTemperature(model=self.model, logger=self.classifier_logger).to(self.device)
        self.scaled_model.set_temperature(valid_loader, self.device)
        self.temperature_value = float(self.scaled_model.temperature.detach().cpu().item())

        # (2) Mag-aware Platt auf temp-skalierte Probs
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x_val.to(self.device)).squeeze()
            p_T = torch.sigmoid(logits / max(self.temperature_value, 1e-6)).cpu().numpy().ravel()
            y_true = y_val.cpu().numpy().astype(int).ravel()
        mag_val = df_valid[self.mag_col].values.astype(float)

        self.mag_platt = MagAwarePlatt().fit(p_T, mag_val, y_true, max_iter=400, lr=0.05)
        self.classifier_logger.log_info_stream("Fitted mag-aware Platt calibration.")

        if self.cfg.get('SAVE_NN', False):
            try:
                joblib.dump(
                    {"temperature": self.temperature_value,
                     "mag_platt": self.mag_platt.state_dict()},
                    f"{self.cfg['PATH_SAVE_NN']}/calib_arts_e_{self.best_validation_epoch + 1}"
                    f"_lr_{self.lr}_bs_{self.bs}_hs_{self.hs}_ubn_{self.ubn}_dp_{self.dp}"
                    f"_nl_{self.nl}_run_{self.cfg['RUN_DATE']}.pkl",
                )
            except Exception as e:
                self.classifier_logger.log_error(f"Save calibration artifacts failed: {e}")

    # ---------------- test/eval ----------------
    def run_tests(self, epoch, today):
        self.classifier_logger.log_info_stream("run_tests: start")
        df_test = self.galaxies.test_dataset

        n_max = int(self.cfg.get("NUMBER_TEST_SAMPLES", 0) or 0)
        n_full = len(df_test)
        if n_max > 0 and n_full > n_max:
            df_test = df_test.sample(n=n_max, random_state=2025).reset_index(drop=True)
            self.classifier_logger.log_info_stream(
                f"Subsampled test set from {n_full} to {len(df_test)} objects for run_tests."
            )

        df_out = df_test

        x = torch.tensor(df_test[self.cfg["INPUT_COLS"]].values, dtype=torch.float32).to(self.device)
        y_true = df_test["mcal_galaxy"].to_numpy().astype(int).ravel()
        mag_te = df_test[self.mag_col].to_numpy(float)

        self.model.eval()
        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                logits_raw = self.model(x).squeeze()
            p_raw = torch.sigmoid(logits_raw).cpu().numpy().ravel()
            logits_T = logits_raw / max(self.temperature_value, 1e-6)
            p_T = torch.sigmoid(logits_T).cpu().numpy().ravel()

        if self.mag_platt is not None:
            p_cal = self.mag_platt.transform(p_T, mag_te)
            self.classifier_logger.log_info_stream("Applied mag-aware Platt calibration.")
        else:
            p_cal = p_T
            self.classifier_logger.log_info_stream("No mag-aware calibrator – using temp-scaled probs.")

        # Statt threshold-basierter Klassifikation:
        rng = np.random.default_rng(2025)
        y_pred_raw = (p_raw >= rng.random(len(p_raw))).astype(int)
        y_pred_cal = (p_cal >= rng.random(len(p_cal))).astype(int)

        # Kalibrierungs-Metriken
        brier_raw = brier_score_loss(y_true, p_raw)
        brier_cal = brier_score_loss(y_true, p_cal)
        nll_raw = neg_log_loss(p_raw, y_true)
        nll_cal = neg_log_loss(p_cal, y_true)
        ece_raw = expected_calibration_error(p_raw, y_true, int(self.cfg.get("ECE_N_BINS", 5)))
        ece_cal = expected_calibration_error(p_cal, y_true, int(self.cfg.get("ECE_N_BINS", 5)))

        edges_for_rate = quantile_edges(mag_te, max(5, int(self.cfg.get("VAL_RATE_N_BINS", 5))))
        mre_raw = mag_rate_mae(p_raw, y_true, mag_te, edges_for_rate)
        mre_cal = mag_rate_mae(p_cal, y_true, mag_te, edges_for_rate)
        gre_raw = abs(p_raw.mean() - y_true.mean())
        gre_cal = abs(p_cal.mean() - y_true.mean())

        self.classifier_logger.log_info_stream(
            f"PROBS raw:  Brier={brier_raw:.5f} NLL={nll_raw:.5f} ECE={ece_raw:.5f} MRE={mre_raw:.5f} GRE={gre_raw:.5f}"
        )
        self.classifier_logger.log_info_stream(
            f"PROBS cal:  Brier={brier_cal:.5f} NLL={nll_cal:.5f} ECE={ece_cal:.5f} MRE={mre_cal:.5f} GRE={gre_cal:.5f}"
        )

        acc_raw = accuracy_score(y_true, y_pred_raw); f1_raw = f1_score(y_true, y_pred_raw)
        acc_cal = accuracy_score(y_true, y_pred_cal); f1_cal = f1_score(y_true, y_pred_cal)

        self.classifier_logger.log_info_stream(f"Temp scaling T={getattr(self, 'temperature_value', 1.0):.3f}")
        self.classifier_logger.log_info_stream(f"Accuracy raw: {acc_raw*100:.2f}% | F1 raw: {f1_raw:.3f}")
        self.classifier_logger.log_info_stream(f"Accuracy cal: {acc_cal*100:.2f}% | F1 cal: {f1_cal:.3f}")
        self.classifier_logger.log_info_stream(f"Brier raw: {brier_score_loss(y_true, p_raw):.5f} | Brier cal: {brier_score_loss(y_true, p_cal):.5f}")

        # Outputs
        df_out['mcal_galaxy probability raw'] = p_raw
        df_out['mcal_galaxy probability'] = p_cal
        df_out['mcal_galaxy raw'] = y_pred_raw
        df_out['mcal_galaxy'] = y_pred_cal
        df_out['true mcal_galaxy'] = y_true

        # Optional plots
        if self.cfg['PLOT_MISS_CLASSF'] is True:
            try:
                pairs = [
                    ['BDF_MAG_DERED_CALIB_R', 'BDF_MAG_DERED_CALIB_I'],
                    ['BDF_MAG_DERED_CALIB_I', 'BDF_MAG_DERED_CALIB_Z'],
                    ['Color BDF MAG R-I', 'Color BDF MAG I-Z'],
                    ['BDF_T', 'BDF_G'],
                    ['FWHM_WMEAN_R', 'FWHM_WMEAN_I'],
                    ['FWHM_WMEAN_I', 'FWHM_WMEAN_Z'],
                    ['AIRMASS_WMEAN_R', 'AIRMASS_WMEAN_I'],
                    ['AIRMASS_WMEAN_I', 'AIRMASS_WMEAN_Z'],
                    ['MAGLIM_R', 'MAGLIM_I'],
                    ['MAGLIM_I', 'MAGLIM_Z'],
                ]
                for i, cols in enumerate(pairs):
                    plot_classification_results(
                        data_frame=df_out, cols=cols,
                        show_plot=self.cfg['SHOW_PLOT'], save_plot=self.cfg['SAVE_PLOT'],
                        save_name=f"{self.cfg['PATH_PLOTS_FOLDER']['MISS-CLASSIFICATION']}/classf_{i}.pdf",
                        title=f"Classification Results, lr={self.lr}, bs={self.bs}, epoch={epoch}",
                    )
            except Exception as e:
                self.classifier_logger.log_info_stream(f"Error {e} plotting plot_classification_results")

        if self.cfg['PLOT_MATRIX'] is True:
            try:
                plot_confusion_matrix(
                    df_gandalf=df_out,
                    # df_balrog=df_test,
                    show_plot=self.cfg['SHOW_PLOT'],
                    save_plot=self.cfg['SAVE_PLOT'],
                    y_true_col = "true mcal_galaxy",
                    y_pred_col = "mcal_galaxy",
                    save_name=f"{self.cfg['PATH_PLOTS_FOLDER']['CONFUSION_MATRIX']}/confusion_matrix.pdf",
                    title=f"Confusion Matrix, lr={self.lr}, bs={self.bs}, epoch={epoch}",
                )
            except Exception as e:
                self.classifier_logger.log_info_stream(f"Error {e} plotting plot_confusion_matrix")

        if self.cfg['PLOT_ROC_CURVE'] is True:
            try:
                plot_roc_curve_gandalf(
                    df_gandalf=df_out,
                    # df_balrog=df_test,
                    y_true_col="true mcal_galaxy",
                    y_pred_col="mcal_galaxy",
                    show_plot=self.cfg['SHOW_PLOT'], save_plot=self.cfg['SAVE_PLOT'],
                    save_name=f"{self.cfg['PATH_PLOTS_FOLDER']['ROC_CURVE']}/roc_curve.pdf",
                    title=f"Receiver Operating Characteristic (ROC) Curve, lr={self.lr}, bs={self.bs}, epoch={epoch}",
                )
            except Exception as e:
                self.classifier_logger.log_info_stream(f"Error {e} plotting plot_roc_curve_gandalf")

        if self.cfg['PLOT_PRECISION_RECALL_CURVE'] is True:
            try:
                plot_recall_curve_gandalf(
                    df_gandalf=df_out,
                    # df_balrog=df_test,
                    show_plot=self.cfg['SHOW_PLOT'], save_plot=self.cfg['SAVE_PLOT'],
                    y_true_col="true mcal_galaxy",
                    y_pred_col="mcal_galaxy",
                    save_name=f"{self.cfg['PATH_PLOTS_FOLDER']['PRECISION_RECALL_CURVE']}/precision_recall_curve.pdf",
                    title=f"Precision-Recall Curve, lr={self.lr}, bs={self.bs}, epoch={epoch}",
                )
            except Exception as e:
                self.classifier_logger.log_info_stream(f"Error {e} plotting plot_recall_curve_gandalf")

            try:
                plot_calibration_by_mag_singlepanel(
                    mag=mag_te,
                    p_cal=p_cal,
                    p_raw=p_raw,
                    y_true=y_true,
                    n_bins=max(7, int(self.cfg.get("REWEIGHT_N_BINS", 5))),  # oder feste Zahl
                    title=f"Calibration by magnitude, lr={self.lr}, bs={self.bs}, epoch={epoch}",
                    xlabel=self.mag_col + " (quantile bins)",
                    show_plot=self.cfg['SHOW_PLOT'],
                    save_plot=self.cfg['SAVE_PLOT'],
                    save_name=f"{self.cfg['PATH_PLOTS_FOLDER']['PRECISION_RECALL_CURVE']}/calib_by_mag_single.pdf".replace(
                        "probability_histogram", "calib_by_mag")
                )
            except Exception as e:
                self.classifier_logger.log_info_stream(f"Error {e} plotting plot_calibration_by_mag_singlepanel")

        if self.cfg['PLOT_PROBABILITY_HIST'] is True:
            try:
                plot_reliability_curve(
                    df_gandalf=df_out,
                    df_balrog=df_test,  # Ground truth!
                    show_plot=self.cfg['SHOW_PLOT'],
                    save_plot=self.cfg['SAVE_PLOT'],
                    save_name=f"{self.cfg['PATH_PLOTS_FOLDER']['PROB_HIST']}/reliability.pdf",
                    prob_cols=("mcal_galaxy probability", "mcal_galaxy probability raw"),
                    labels=("calibrated", "raw"),
                    n_bins=15,
                    title=f"Reliability, lr={self.lr}, bs={self.bs}, epoch={epoch}"
                )
            except Exception as e:
                self.classifier_logger.log_info_stream(f"Error {e} plotting plot_reliability_curve")
            try:
                plot_rate_ratio_curve(
                    mag=mag_te,
                    probs_cal=p_cal,
                    probs_raw=p_raw,
                    y_true=y_true,
                    n_bins=max(7, int(self.cfg.get("REWEIGHT_N_BINS", 5))),
                    title=f"Rate ratio by magnitude, lr={self.lr}, bs={self.bs}, epoch={epoch}",
                    xlabel=self.mag_col + " (quantile bins)",
                    show_plot=self.cfg['SHOW_PLOT'],
                    save_plot=self.cfg['SAVE_PLOT'],
                    save_name=f"{self.cfg['PATH_PLOTS_FOLDER']['PROB_HIST']}/rate_ratio_by_mag_curve.pdf"
                )
            except Exception as e:
                self.classifier_logger.log_info_stream(f"Error {e} plotting plot_rate_ratio_curve")
            try:
                plot_rate_ratio_by_mag(
                    mag=mag_te,
                    y_true=y_true,
                    probs_raw=p_raw,
                    probs_cal=p_cal,
                    calibrated=True,  # default
                    bin_width=0.25,
                    mag_label="BDF_MAG_DERED_CALIB_I",
                    show_density_ratio=True,
                    save_name=f"{self.cfg['PATH_PLOTS_FOLDER']['PROB_HIST']}/rate_ratio_by_mag.pdf"
                )
            except Exception as e:
                self.classifier_logger.log_info_stream(f"Error {e} plotting plot_rate_ratio_by_mag")

        self.classifier_logger.log_info_stream("run_tests: done")

    # ---------------- finalize ----------------
    def _finalize_and_log(self, ep_for_plots: int, reason: str, last_state_dict=None):
        if self._finalized: return
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
                        f"_nl_{self.nl}_run_{self.cfg['RUN_DATE']}.pt",
                    )
                    torch.save(
                        last_state_dict if last_state_dict is not None else self.model.state_dict(),
                        f"{self.cfg['PATH_SAVE_NN']}/last_model_state_e_{self.cfg['EPOCHS']}"
                        f"_lr_{self.lr}_bs_{self.bs}_hs_{self.hs}_ubn_{self.ubn}_dp_{self.dp}"
                        f"_nl_{self.nl}_run_{self.cfg['RUN_DATE']}.pt",
                    )
                except Exception as e:
                    self.classifier_logger.log_error(f"Saving models failed: {e}")
        finally:
            self._finalized = True


if __name__ == '__main__':
    pass