# gaNdalF Conditional Diagnostics (script version)

import os, sys, math, json, logging
import numpy as np
import pandas as pd
import torch
from contextlib import nullcontext
import matplotlib.pyplot as plt

# Optional lokal:
USE_SEABORN = False
if USE_SEABORN:
    import seaborn as sns
    sns.set()

from Handler import fnn, get_os, unsheared_shear_cuts, unsheared_mag_cut, LoggerHandler, calc_color
from gandalf import gaNdalF
import yaml
from datetime import datetime

plt.rcParams['figure.figsize'] = (12, 6)



def load_config_pair(system_path):
    now = datetime.now()
    if get_os() == "Mac":
        cfg_cls_name = "MAC_run_classifier.cfg"
        cfg_flow_name = "MAC_run_flow.cfg"
    elif get_os() == "Linux":
        cfg_cls_name = "LMU_run_classifier.cfg"
        cfg_flow_name = "LMU_run_flow.cfg"
    else:
        raise RuntimeError("Undefined operating system")

    with open(f"{system_path}/conf/{cfg_cls_name}", 'r') as fp:
        cfg_cls = yaml.safe_load(fp)
    with open(f"{system_path}/conf/{cfg_flow_name}", 'r') as fp:
        cfg_flow = yaml.safe_load(fp)

    now_str = now.strftime('%Y-%m-%d_%H-%M')
    cfg_cls['RUN_DATE']  = now_str
    cfg_flow['RUN_DATE'] = now_str
    return cfg_cls, cfg_flow

PROJECT_ROOT = os.path.abspath(sys.path[-1])
classifier_cfg, flow_cfg = load_config_pair(PROJECT_ROOT)

log_lvl = logging.INFO
if flow_cfg.get("LOGGING_LEVEL") == "DEBUG":
    log_lvl = logging.DEBUG
elif flow_cfg.get("LOGGING_LEVEL") == "ERROR":
    log_lvl = logging.ERROR

run_flow_logger = LoggerHandler(
    logger_dict={"logger_name": "run_flow logger",
                 "info_logger": flow_cfg['INFO_LOGGER'],
                 "error_logger": flow_cfg['ERROR_LOGGER'],
                 "debug_logger": flow_cfg['DEBUG_LOGGER'],
                 "stream_logger": flow_cfg['STREAM_LOGGER'],
                 "stream_logging_level": log_lvl},
    log_folder_path=f"{flow_cfg['PATH_OUTPUT']}/"
)



flow_model = gaNdalF(run_flow_logger, classifier_cfg=classifier_cfg, flow_cfg=flow_cfg)
df_gandalf, df_balrog = flow_model.run_flow()

df_gandalf = calc_color(df_gandalf, colors=flow_cfg['COLORS_FLOW'], column_name="unsheared/mag")
df_balrog  = calc_color(df_balrog,  colors=flow_cfg['COLORS_FLOW'], column_name="unsheared/mag")

df_gandalf_cut = unsheared_mag_cut(unsheared_shear_cuts(df_gandalf.copy()))
df_balrog_cut  = unsheared_mag_cut(unsheared_shear_cuts(df_balrog.copy()))



def extract_flow_core(maybe_wrapper):
    for name in ["flow_model","flow","nf","model","flow_nf","flow_core"]:
        if hasattr(maybe_wrapper, name):
            obj = getattr(maybe_wrapper, name)
            if hasattr(obj, "log_probs") and hasattr(obj, "sample"):
                return obj
            # recursive
            inner = extract_flow_core(obj)
            if inner is not None:
                return inner
    if hasattr(maybe_wrapper, "log_probs") and hasattr(maybe_wrapper, "sample"):
        return maybe_wrapper
    return None

flow_core = extract_flow_core(flow_model)
assert flow_core is not None, "Flow-Modell nicht gefunden – bitte extract_flow_core anpassen."

in_cols  = list(flow_cfg["INPUT_COLS"])
out_cols = list(flow_cfg["OUTPUT_COLS"])

X_valid = torch.tensor(df_balrog[in_cols].values, dtype=next(flow_core.parameters()).dtype)
Y_valid = torch.tensor(df_balrog[out_cols].values, dtype=next(flow_core.parameters()).dtype)

DEVICE = flow_cfg.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
flow_core = flow_core.to(DEVICE).eval()



@torch.no_grad()
def mean_nll(model, Y, X, bs=4096, use_amp=False, device="cpu"):
    model.eval()
    tot, n = 0.0, 0
    X = X.to(device); Y = Y.to(device)
    autocast_ctx = (torch.amp.autocast(device_type="cuda") if (use_amp and device=='cuda') else nullcontext())
    for i in range(0, X.size(0), bs):
        xb = X[i:i+bs]; yb = Y[i:i+bs]
        with autocast_ctx:
            nll = -model.log_probs(yb, xb)
        tot += nll.sum().item(); n += yb.size(0)
    return tot / max(n,1)

def perm_importance(model, Y, X, col_names, repeats=3, bs=4096, use_amp=False, device="cpu", rng=None):
    base = mean_nll(model, Y, X, bs, use_amp, device)
    deltas = {}
    Xcpu = X.detach().cpu(); N = Xcpu.shape[0]
    import numpy as _np, torch as _torch
    rng = _np.random.default_rng(None if rng is None else rng)
    for j, name in enumerate(col_names):
        incs = []
        for _ in range(repeats):
            Xperm = Xcpu.clone()
            idx = _torch.from_numpy(rng.permutation(N))
            Xperm[:, j] = Xperm[idx, j]
            inc = mean_nll(model, Y, Xperm.to(device), bs, use_amp, device) - base
            incs.append(float(inc))
        deltas[name] = (float(_np.mean(incs)), float(_np.std(incs)))
    return base, deltas

def all_shuffle_nll(model, Y, X, bs=4096, use_amp=False, device="cpu", rng=None):
    import numpy as _np, torch as _torch
    rng = _np.random.default_rng(None if rng is None else rng)
    Xcpu = X.detach().cpu().clone(); N = Xcpu.shape[0]
    for j in range(Xcpu.shape[1]):
        idx = _torch.from_numpy(rng.permutation(N))
        Xcpu[:, j] = Xcpu[idx, j]
    return mean_nll(model, Y, Xcpu.to(device), bs, use_amp, device)

def context_grad_sensitivity(model, Y, X, bs=2048, device="cpu"):
    model.eval()
    grads_sum = torch.zeros(X.size(1), device=device); count = 0
    for i in range(0, X.size(0), bs):
        xb = X[i:i+bs].to(device).detach().requires_grad_(True)
        yb = Y[i:i+bs].to(device)
        lp = model.log_probs(yb, xb).mean()
        g, = torch.autograd.grad(lp, xb, retain_graph=False, create_graph=False)
        grads_sum += g.abs().mean(dim=0); count += 1
    return (grads_sum / max(count,1)).detach().cpu()

@torch.no_grad()
def counterfactual_shift(model, X, j, delta, sample_bs=8192, device="cpu"):
    model.eval(); N = X.size(0)
    Ys = []
    for i in range(0, N, sample_bs):
        xb = X[i:i+sample_bs].to(device)
        Ys.append(model.sample(num_samples=xb.size(0), cond_inputs=xb))
    Y0 = torch.cat(Ys, dim=0).cpu()

    Xp = X.clone(); Xp[:, j] += delta
    Ys = []
    for i in range(0, N, sample_bs):
        xb = Xp[i:i+sample_bs].to(device)
        Ys.append(model.sample(num_samples=xb.size(0), cond_inputs=xb))
    Y1 = torch.cat(Ys, dim=0).cpu()

    mean_shift = (Y1.mean(dim=0) - Y0.mean(dim=0))
    var_shift  = (Y1.var(dim=0, unbiased=False) - Y0.var(dim=0, unbiased=False))
    return mean_shift, var_shift



BATCH_SIZE_EVAL = 4096
PERM_REPEATS = 2
COUNTERFACTUAL_DELTA = 0.1
TOPK = min(5, len(in_cols))
USE_AMP = False

device = "cuda" if torch.cuda.is_available() else "cpu"

base_nll, deltas = perm_importance(flow_core, Y_valid, X_valid, in_cols, repeats=PERM_REPEATS, bs=BATCH_SIZE_EVAL, use_amp=USE_AMP, device=device)
print(f"Base NLL: {base_nll:.6f}")
d_rows = [(name, deltas[name][0], deltas[name][1]) for name in in_cols]
d_rows.sort(key=lambda t: t[1], reverse=True)
df_perm = pd.DataFrame(d_rows, columns=["feature", "delta_nll_mean", "delta_nll_std"])
display(df_perm.head(10))

# Plot ΔNLL
names = [r[0] for r in d_rows]; vals = [r[1] for r in d_rows]; errs = [r[2] for r in d_rows]
idx = np.arange(len(names))
plt.figure(figsize=(max(12, len(names)*0.5), 6))
plt.bar(idx, vals, yerr=errs)
plt.xticks(idx, names, rotation=45, ha="right")
plt.ylabel("ΔNLL (↑ schlechter)"); plt.title("Permutation Importance (ΔNLL)")
plt.tight_layout(); plt.show()

# Gradienten
grads = context_grad_sensitivity(flow_core, Y_valid, X_valid, bs=2048, device=device).numpy()
g_rows = list(zip(in_cols, grads))
g_rows.sort(key=lambda t: t[1], reverse=True)
df_grad = pd.DataFrame(g_rows, columns=["feature", "grad_sensitivity"])
display(df_grad.head(10))

plt.figure(figsize=(max(12, len(in_cols)*0.5), 6))
plt.bar(np.arange(len(in_cols)), grads)
plt.xticks(np.arange(len(in_cols)), in_cols, rotation=45, ha="right")
plt.ylabel("⟨|∂ log p/∂x_j|⟩"); plt.title("Gradient Sensitivity")
plt.tight_layout(); plt.show()

# Counterfactual Top-K
top_features = [r[0] for r in d_rows[:TOPK]]
subset = X_valid[: min(10000, X_valid.size(0))].clone()
for name in top_features:
    j = in_cols.index(name)
    mean_shift, var_shift = counterfactual_shift(flow_core, subset, j, delta=COUNTERFACTUAL_DELTA, device=device)
    print(f"Feature: {name} (Δx={COUNTERFACTUAL_DELTA})")
    xs = np.arange(len(out_cols))
    plt.figure(figsize=(12,5))
    plt.plot(xs, mean_shift.numpy(), marker="o", linestyle="-", label="Δ mean(Y)")
    plt.plot(xs, var_shift.numpy(),  marker="s", linestyle="--", label="Δ var(Y)")
    plt.xticks(xs, out_cols, rotation=45, ha="right")
    plt.legend(); plt.xlabel("Output-Dimension")
    plt.title(f"Counterfactual Shift: {name} (Δx={COUNTERFACTUAL_DELTA})")
    plt.tight_layout(); plt.show()

# All-Shuffle
nll_all_shuffle = all_shuffle_nll(flow_core, Y_valid, X_valid, bs=BATCH_SIZE_EVAL, use_amp=USE_AMP, device=device)
print(f"NLL (all-shuffle): {nll_all_shuffle:.6f} | Base: {base_nll:.6f} | Δ = {nll_all_shuffle - base_nll:.6f}")



def ensure_outdir(base_dir, sub="diagnostics"):
    outdir = os.path.join(base_dir, sub); os.makedirs(outdir, exist_ok=True); return outdir

out_dir = ensure_outdir(flow_cfg["PATH_PLOTS"])
df_perm.to_csv(os.path.join(out_dir, "permutation_importance.csv"), index=False)
df_grad.to_csv(os.path.join(out_dir, "gradient_sensitivity.csv"), index=False)
print("CSV gespeichert unter:", out_dir)


