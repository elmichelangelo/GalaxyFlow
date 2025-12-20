#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gaNdalF_get_calibration_model.py

Fits an isotonic calibration model for gaNdalF classifier outputs and saves it.
Also creates a reliability plot comparing uncalibrated vs isotonic on a held-out test split.
"""

import os
import sys
import gc
import pickle
import logging
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# --- your project imports ---
from Handler import get_os, LoggerHandler
from gandalf import gaNdalF

# Reuse your existing helpers/calibrators
from gandalf_calibration_model.calibration_benchmark import (
    collect_classifier_outputs,
    IsotonicCalibrator,
)

# -----------------------------
# Small utilities
# -----------------------------
def safe_clip(p, eps=1e-12):
    p = np.asarray(p, dtype=np.float64)
    return np.clip(p, eps, 1.0 - eps)

def reliability_curve_quantile(y, p, n_bins=20, min_count=500):
    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)

    # quantile bins in p
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(0, 1, 3)

    ids = np.digitize(p, edges) - 1
    ids = np.clip(ids, 0, len(edges) - 2)

    mean_p, frac_pos, counts = [], [], []
    for b in range(len(edges) - 1):
        m = (ids == b)
        c = int(np.sum(m))
        if c < min_count:
            continue
        counts.append(c)
        mean_p.append(float(np.mean(p[m])))
        frac_pos.append(float(np.mean(y[m])))

    return np.array(mean_p), np.array(frac_pos), np.array(counts)

def _auto_xlim(p_all, q=(0.001, 0.999), pad=0.02):
    lo, hi = np.quantile(p_all, q)
    lo = max(0.0, lo - pad)
    hi = min(1.0, hi + pad)
    # if spread is large, show full [0,1]
    if hi - lo > 0.5:
        return (0.0, 1.0)
    return (lo, hi)

def plot_reliability_uncal_vs_iso(y_test, p_uncal, p_iso, *, title, outpath, n_bins=20, min_count=500):
    y = np.asarray(y_test, int)
    p_uncal = safe_clip(p_uncal)
    p_iso = safe_clip(p_iso)

    plt.figure(figsize=(16, 9))
    plt.plot([0, 1], [0, 1], linewidth=1.5)

    for name, p in [("uncalibrated", p_uncal), ("isotonic", p_iso)]:
        mean_p, frac_pos, _ = reliability_curve_quantile(y, p, n_bins=n_bins, min_count=min_count)
        plt.plot(mean_p, frac_pos, marker="o", linestyle="-", label=name)

    # nicer scaling (avoid “everything in the lower left corner”)
    p_stack = np.concatenate([p_uncal, p_iso])
    xlo, xhi = _auto_xlim(p_stack)
    plt.xlim(xlo, xhi)
    plt.ylim(0.0, 1.0)

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=250)
    plt.close()


# -----------------------------
# Config loader (keep it simple)
# -----------------------------
import argparse
import yaml

def load_config(system_path):
    if get_os() == "Mac":
        default_cfg = "MAC_get_calibration_model.cfg"
    elif get_os() == "Linux":
        default_cfg = "LMU_get_calibration_model.cfg"
    else:
        raise RuntimeError("Undefined operating system")

    parser = argparse.ArgumentParser(description="Fit & save isotonic calibration model for gaNdalF")
    parser.add_argument("--config_filename", "-cf", type=str, default=default_cfg)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--test_size", type=float, default=0.5)
    parser.add_argument("--max_fit", type=int, default=500_000)
    parser.add_argument("--batch_size", type=int, default=131072)
    parser.add_argument("--n_bins_plot", type=int, default=20)
    parser.add_argument("--min_count_plot", type=int, default=500)
    args = parser.parse_args()

    cfg_path = os.path.join(system_path, "..", "conf", args.config_filename)
    with open(cfg_path, "r") as fp:
        cfg = yaml.safe_load(fp)

    cfg["RUN_DATE"] = datetime.now().strftime("%Y-%m-%d_%H-%M")
    cfg["_ARGS"] = vars(args)
    return cfg


def init_output_paths(cfg):
    # keep your naming, but ensure dirs exist
    cfg["PATH_OUTPUT"] = f'{cfg["PATH_OUTPUT"]}/{cfg["RUN_DATE"]}_RUN_GANDALF'
    os.makedirs(cfg["PATH_OUTPUT"], exist_ok=True)

    cfg["PATH_PLOTS"] = os.path.join(cfg["PATH_OUTPUT"], "Plots")
    os.makedirs(cfg["PATH_PLOTS"], exist_ok=True)

    cfg["PATH_LOGS"] = os.path.join(cfg["PATH_OUTPUT"], "Logs")
    os.makedirs(cfg["PATH_LOGS"], exist_ok=True)

    cfg["PATH_CALIB"] = os.path.join(cfg["PATH_OUTPUT"], "CalibrationModels")
    os.makedirs(cfg["PATH_CALIB"], exist_ok=True)


# -----------------------------
# Main
# -----------------------------
def main(cfg, logger):
    args = cfg["_ARGS"]
    seed = int(args["seed"])
    test_size = float(args["test_size"])
    max_fit = int(args["max_fit"]) if args["max_fit"] is not None else None
    batch_size = int(args["batch_size"])

    logger.log_info_stream("Init gaNdalF classifier...")
    model = gaNdalF(logger, cfg=cfg)
    model.init_classifier()

    # IMPORTANT: scale input data (as in your pipeline)
    logger.log_info_stream("Scale classifier data...")
    model.classifier_galaxies.scale_data(
        cfg_key_cols_interest="COLUMNS_OF_INTEREST_CF",
        cfg_key_filename_scaler="FILENAME_STANDARD_SCALER_CF",
    )

    mag_col = model.cfg.get("MAG_COL", "BDF_MAG_DERED_CALIB_I")

    logger.log_info_stream("Collect classifier outputs (logits, p_raw, y)...")
    outs = collect_classifier_outputs(
        model,
        cov_cols=None,
        batch_size=batch_size,
        mag_col=mag_col,
    )

    y = np.asarray(outs["y"], int)
    p_raw = safe_clip(outs["p_raw"])

    # Split indices
    idx = np.arange(len(y))
    idx_cal, idx_test = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)

    # optional subsample calibration set
    if max_fit is not None and len(idx_cal) > max_fit:
        rng = np.random.default_rng(seed)
        idx_cal = rng.choice(idx_cal, size=max_fit, replace=False)

    y_cal, y_test = y[idx_cal], y[idx_test]
    p_cal, p_test = p_raw[idx_cal], p_raw[idx_test]

    logger.log_info_stream(f"Split: N_total={len(y)}  N_cal={len(idx_cal)}  N_test={len(idx_test)}  pos_rate={y.mean():.6f}")

    # Fit isotonic
    logger.log_info_stream("Fit isotonic calibrator...")
    iso = IsotonicCalibrator()
    iso.fit(p_cal, y_cal)

    # Predict on test
    p_iso_test = safe_clip(iso.predict_proba(p_test))

    # Save model (+ metadata)
    model_name = f"isotonic_seed{seed}_cal{len(idx_cal)}_test{len(idx_test)}.pkl"
    save_path = os.path.join(cfg["PATH_CALIB"], model_name)

    payload = {
        "name": "isotonic",
        "calibrator": iso,
        "mag_col": mag_col,
        "seed": seed,
        "test_size": test_size,
        "max_fit": max_fit,
        "n_total": int(len(y)),
        "n_cal": int(len(idx_cal)),
        "n_test": int(len(idx_test)),
        "pos_rate": float(np.mean(y)),
        "created": cfg["RUN_DATE"],
        # optional: remember sklearn version to diagnose future warnings
        "sklearn_version": __import__("sklearn").__version__,
    }

    with open(save_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.log_info_stream(f"Saved isotonic calibration model to: {save_path}")

    # Reliability plot
    plot_path = os.path.join(cfg["PATH_PLOTS"], "reliability_uncalibrated_vs_isotonic.png")
    logger.log_info_stream("Create reliability plot (uncalibrated vs isotonic)...")
    plot_reliability_uncal_vs_iso(
        y_test=y_test,
        p_uncal=p_test,
        p_iso=p_iso_test,
        title=f"Reliability (test) — uncalibrated vs isotonic | N_test={len(y_test)}",
        outpath=plot_path,
        n_bins=int(args["n_bins_plot"]),
        min_count=int(args["min_count_plot"]),
    )
    logger.log_info_stream(f"Saved reliability plot to: {plot_path}")

    # clean up
    del outs
    gc.collect()

    logger.log_info_stream("Done.")


if __name__ == "__main__":
    system_path = os.path.abspath(os.path.dirname(__file__))
    cfg = load_config(system_path)
    init_output_paths(cfg)

    log_lvl = logging.INFO
    if cfg.get("LOGGING_LEVEL", "INFO") == "DEBUG":
        log_lvl = logging.DEBUG

    run_logger = LoggerHandler(
        logger_dict={
            "logger_name": "train flow logger",
            "info_logger": cfg["INFO_LOGGER"],
            "error_logger": cfg["ERROR_LOGGER"],
            "debug_logger": cfg["DEBUG_LOGGER"],
            "stream_logger": cfg["STREAM_LOGGER"],
            "stream_logging_level": log_lvl,
        },
        log_folder_path=f"{cfg['PATH_LOGS']}/",
    )

    main(cfg=cfg, logger=run_logger)