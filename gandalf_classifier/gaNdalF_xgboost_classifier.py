#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, yaml, math, json, joblib, datetime
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score, brier_score_loss
from xgboost import XGBClassifier

# Optional: dein Logger, ansonsten fallback auf print
try:
    from Handler.logger import LoggerHandler
    HAVE_HANDLER_LOGGER = True
except Exception:
    HAVE_HANDLER_LOGGER = False

# Dein Dataset-Wrapper
from gandalf_galaxie_dataset import DESGalaxies

# ---------- helpers ----------

def setup_logger(cfg, out_dir):
    if HAVE_HANDLER_LOGGER:
        log = LoggerHandler(
            logger_dict={
                "logger_name": "xgb_scratch",
                "info_logger": cfg.get("INFO_LOGGER", True),
                "error_logger": cfg.get("ERROR_LOGGER", True),
                "debug_logger": cfg.get("DEBUG_LOGGER", True),
                "stream_logger": cfg.get("STREAM_LOGGER", True),
                "stream_logging_level": {"ERROR":40,"INFO":20,"DEBUG":10}.get(cfg.get("LOGGING_LEVEL","INFO"),20)
            },
            log_folder_path=os.path.join(out_dir, "Logs")
        )
        return log
    class _P:  # simple print-logger
        def log_info_stream(self, msg): print(msg, flush=True)
    os.makedirs(os.path.join(out_dir, "Logs"), exist_ok=True)
    return _P()

def _compute_slice_neg_weights(x: np.ndarray, quantiles: list, neg_weights: list):
    edges = np.quantile(x, np.array(quantiles, dtype=float))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = edges[i-1] + 1e-8
    w = np.ones_like(x, dtype=np.float32)
    for i in range(len(edges)-1):
        m = (x >= edges[i]) & (x <= edges[i+1])
        w[m] = float(neg_weights[i])
    return w, edges

def best_threshold_f1(probs: np.ndarray, y_true: np.ndarray):
    thr_grid = np.linspace(0.0, 1.0, 1001)[1:-1]
    best_f1, best_thr = -1.0, 0.5
    for t in thr_grid:
        f1 = f1_score(y_true, (probs >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    return best_thr, best_f1

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser("XGBoost scratch run for DESGalaxies")
    ap.add_argument("-cf","--config_filename", type=str, required=True,
                    help="Pfad zur CFG (yaml)")
    ap.add_argument("--n_estimators", type=int, default=1000)
    ap.add_argument("--max_depth", type=int, default=6)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample_bytree", type=float, default=0.8)
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    ap.add_argument("--reg_alpha", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--early_stopping_rounds", type=int, default=50)
    ap.add_argument("--tree_method", type=str, default="auto", choices=["auto","hist","gpu_hist"])
    ap.add_argument("--out", type=str, default=None, help="Ausgabeordner überschreiben (optional)")
    args = ap.parse_args()

    # --- CFG laden ---
    with open(args.config_filename, "r") as fp:
        cfg = yaml.safe_load(fp)

    run_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root_out = args.out or os.path.join(cfg["PATH_OUTPUT"], f"{run_date}_xgb_scratch")
    os.makedirs(root_out, exist_ok=True)

    lgr = setup_logger(cfg, root_out)
    lgr.log_info_stream(f"Output: {root_out}")

    # --- Daten laden & skalieren über deinen Wrapper ---
    galaxies = DESGalaxies(dataset_logger=lgr, cfg=cfg)
    galaxies.scale_data()

    # Pandas → Numpy
    X_cols = cfg["INPUT_COLS"]
    y_col  = cfg["OUTPUT_COLS"][0]  # "detected"

    df_train = galaxies.train_dataset
    df_valid = galaxies.valid_dataset
    df_test  = galaxies.test_dataset

    X_tr = df_train[X_cols].to_numpy(dtype=np.float32)
    y_tr = df_train[y_col].to_numpy(dtype=np.int32).ravel()
    X_va = df_valid[X_cols].to_numpy(dtype=np.float32)
    y_va = df_valid[y_col].to_numpy(dtype=np.int32).ravel()
    X_te = df_test[X_cols].to_numpy(dtype=np.float32)
    y_te = df_test[y_col].to_numpy(dtype=np.int32).ravel()

    # --- Imbalance-Handling ---
    n_pos = max(1, int((y_tr == 1).sum()))
    n_neg = max(1, int((y_tr == 0).sum()))
    scale_pos_weight = n_neg / float(n_pos)
    lgr.log_info_stream(f"Class balance train: pos={n_pos}, neg={n_neg}, scale_pos_weight={scale_pos_weight:.3f}")

    # --- optionale Slice-Weights für NEGATIVE Beispiele ---
    sample_weight = None
    if bool(cfg.get("SLICE_WEIGHTING_ENABLE", False)):
        feat_name = cfg.get("SLICE_WEIGHTING_FEATURE", "BDF_MAG_DERED_CALIB_I")
        qtls      = cfg.get("SLICE_WEIGHTING_QUANTILES", [0.0, 0.6, 0.85, 1.0])
        neg_ws    = cfg.get("SLICE_NEG_WEIGHTS", [1.0, 1.5, 2.0])
        x_feat = df_train[feat_name].to_numpy(dtype=float)
        wneg, edges = _compute_slice_neg_weights(x_feat, qtls, neg_ws)
        sw = np.ones_like(y_tr, dtype=np.float32)
        sw[y_tr == 0] = wneg[y_tr == 0]  # nur Negative hochgewichten
        sample_weight = sw
        lgr.log_info_stream(f"Slice weighting ON ({feat_name}); edges={edges}")

    # --- tree_method wählen ---
    tree_method = args.tree_method
    if tree_method == "auto":
        try:
            import xgboost as xgb
            gpu_ok = any("GPU" in d.upper() for d in xgb.rabit.get_world_size.__doc__ or [])
        except Exception:
            gpu_ok = False
        tree_method = "gpu_hist" if gpu_ok else "hist"
    lgr.log_info_stream(f"tree_method={tree_method}")

    # --- XGB Modell ---
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        gamma=args.gamma,
        objective="binary:logistic",
        eval_metric=["logloss","aucpr"],
        scale_pos_weight=scale_pos_weight,
        tree_method=tree_method,
        n_jobs=os.cpu_count(),
        early_stopping_rounds=50
    )

    lgr.log_info_stream("Fit XGBoost …")
    model.fit(
        X_tr, y_tr,
        sample_weight=sample_weight,
        eval_set=[(X_va, y_va)],
        verbose=True
    )

    # --- Best threshold anhand Validation (F1) ---
    p_va = model.predict_proba(X_va)[:,1]
    thr_star, f1_star = best_threshold_f1(p_va, y_va)
    lgr.log_info_stream(f"Validation: best F1-threshold={thr_star:.3f}, F1={f1_star:.3f}")

    # --- Testevaluation ---
    p_te = model.predict_proba(X_te)[:,1]
    y_pred = (p_te >= thr_star).astype(int)

    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred)
    brier = brier_score_loss(y_te, p_te)

    tp = int(((y_pred == 1) & (y_te == 1)).sum())
    tn = int(((y_pred == 0) & (y_te == 0)).sum())
    fp = int(((y_pred == 1) & (y_te == 0)).sum())
    fn = int(((y_pred == 0) & (y_te == 1)).sum())

    lgr.log_info_stream(f"Test: Acc={acc*100:.2f}% | F1={f1:.3f} | Brier={brier:.4f}")
    lgr.log_info_stream(f"Confusion (thr={thr_star:.3f}): TP={tp} TN={tn} FP={fp} FN={fn}")

    # --- Artefakte speichern ---
    meta = {
        "run_date": run_date,
        "params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_lambda": args.reg_lambda,
            "reg_alpha": args.reg_alpha,
            "gamma": args.gamma,
            "tree_method": tree_method,
            "scale_pos_weight": scale_pos_weight,
            "early_stopping_rounds": args.early_stopping_rounds
        },
        "validation": {"thr_star": thr_star, "f1": f1_star},
        "test": {"acc": float(acc), "f1": float(f1), "brier": float(brier),
                 "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    }

    os.makedirs(root_out, exist_ok=True)
    model_path = os.path.join(root_out, "xgb_model.json")
    preds_path = os.path.join(root_out, "test_predictions.parquet")
    meta_path  = os.path.join(root_out, "metadata.json")

    # Modell + Meta + Testpredictions
    model.save_model(model_path)
    pd.DataFrame({"prob": p_te, "y_true": y_te, "y_pred": y_pred}).to_parquet(preds_path, index=False)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    lgr.log_info_stream(f"Saved model → {model_path}")
    lgr.log_info_stream(f"Saved test predictions → {preds_path}")
    lgr.log_info_stream(f"Saved meta → {meta_path}")

if __name__ == "__main__":
    main()