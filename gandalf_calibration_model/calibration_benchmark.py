import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, accuracy_score, f1_score

# ----------------------------
# Metrics
# ----------------------------
def brier_score(y, p):
    y = np.asarray(y).astype(np.float64)
    p = np.asarray(p).astype(np.float64)
    return float(np.mean((p - y) ** 2))

def expected_calibration_error(y, p, n_bins=20):
    """ECE with equal-width bins in probability space."""
    y = np.asarray(y).astype(np.float64)
    p = np.asarray(p).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        m = (bin_ids == b)
        if not np.any(m):
            continue
        conf = np.mean(p[m])
        acc = np.mean(y[m])
        ece += (np.sum(m) / len(y)) * abs(acc - conf)
    return float(ece)

def safe_clip(p, eps=1e-12):
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)

def safe_logit(p, eps=1e-12):
    p = safe_clip(p, eps)
    return np.log(p / (1.0 - p))

def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------
# Calibrators
# ----------------------------
class IdentityCalibrator:
    name = "uncalibrated"
    def fit(self, p, y, **kwargs):
        return self
    def predict_proba(self, p, **kwargs):
        return np.asarray(p, dtype=np.float64)

class PlattCalibrator:
    """Logistic regression on logit(p) (or logits if you pass logits as p)."""
    name = "platt"
    def __init__(self, C=1e6):
        self.lr = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
    def fit(self, p, y, **kwargs):
        x = safe_logit(p).reshape(-1, 1)
        self.lr.fit(x, y)
        return self
    def predict_proba(self, p, **kwargs):
        x = safe_logit(p).reshape(-1, 1)
        return self.lr.predict_proba(x)[:, 1]

class BetaCalibrator:
    """Beta calibration via logistic regression on [log(p), log(1-p)]."""
    name = "beta"
    def __init__(self, C=1e6):
        self.lr = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
    def fit(self, p, y, **kwargs):
        p = safe_clip(p)
        X = np.column_stack([np.log(p), np.log(1.0 - p)])
        self.lr.fit(X, y)
        return self
    def predict_proba(self, p, **kwargs):
        p = safe_clip(p)
        X = np.column_stack([np.log(p), np.log(1.0 - p)])
        return self.lr.predict_proba(X)[:, 1]

class IsotonicCalibrator:
    name = "isotonic"
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")
    def fit(self, p, y, **kwargs):
        self.iso.fit(np.asarray(p, float), np.asarray(y, int))
        return self
    def predict_proba(self, p, **kwargs):
        return np.asarray(self.iso.predict(np.asarray(p, float)), dtype=np.float64)

class HistogramCalibrator:
    name = "histogram"
    def __init__(self, n_bins=30, strategy="quantile"):
        self.n_bins = int(n_bins)
        self.strategy = strategy
        self.bin_edges = None
        self.bin_value = None
    def fit(self, p, y, **kwargs):
        p = np.asarray(p, float)
        y = np.asarray(y, int)
        if self.strategy == "quantile":
            self.bin_edges = np.quantile(p, np.linspace(0, 1, self.n_bins + 1))
        else:
            self.bin_edges = np.linspace(0, 1, self.n_bins + 1)
        # make edges strictly increasing
        self.bin_edges = np.unique(self.bin_edges)
        if len(self.bin_edges) < 3:
            self.bin_edges = np.linspace(0, 1, 3)

        ids = np.digitize(p, self.bin_edges) - 1
        ids = np.clip(ids, 0, len(self.bin_edges) - 2)

        self.bin_value = np.zeros(len(self.bin_edges) - 1, dtype=float)
        for b in range(len(self.bin_value)):
            m = (ids == b)
            self.bin_value[b] = float(np.mean(y[m])) if np.any(m) else float(np.mean(y))
        return self
    def predict_proba(self, p, **kwargs):
        p = np.asarray(p, float)
        ids = np.digitize(p, self.bin_edges) - 1
        ids = np.clip(ids, 0, len(self.bin_edges) - 2)
        return self.bin_value[ids].astype(np.float64)

class TemperatureScaler:
    """Temperature scaling on logits: p = sigmoid(logits / T)."""
    name = "temperature"
    def __init__(self):
        self.T = 1.0
    def fit(self, logits, y, **kwargs):
        # fit T by minimizing NLL on calibration set
        logits_t = torch.tensor(np.asarray(logits, np.float32))
        y_t = torch.tensor(np.asarray(y, np.float32))

        log_T = torch.zeros((), requires_grad=True)  # optimize log(T)
        opt = torch.optim.LBFGS([log_T], lr=0.5, max_iter=100)

        bce = torch.nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            T = torch.exp(log_T)
            loss = bce(logits_t / T, y_t)
            loss.backward()
            return loss

        opt.step(closure)
        self.T = float(torch.exp(log_T).detach().cpu().numpy())
        return self
    def predict_proba(self, logits, **kwargs):
        logits = np.asarray(logits, dtype=np.float64)
        return sigmoid(logits / self.T)

class MagAwarePlattCalibrator:
    """
    Mag-aware Platt: logistic regression on [logit(p), mag, mag^2]
    (matches your MagAwarePlatt form; here we fit it with sklearn).
    """
    name = "magaware_platt"
    def __init__(self, C=1e6):
        self.lr = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)
    def fit(self, p, y, mag, **kwargs):
        x1 = safe_logit(p)
        x2 = np.asarray(mag, float)
        X = np.column_stack([x1, x2, x2**2])
        self.lr.fit(X, y)
        return self
    def predict_proba(self, p, mag, **kwargs):
        x1 = safe_logit(p)
        x2 = np.asarray(mag, float)
        X = np.column_stack([x1, x2, x2**2])
        return self.lr.predict_proba(X)[:, 1].astype(np.float64)

class CovariatePlattCalibrator:
    """
    logistic regression on [logit(p), x1, x1^2, x2, x2^2, ...]
    """
    def __init__(self, cov_idx=None, C=1e6, name="cov_platt"):
        self.cov_idx = cov_idx
        self.lr = LogisticRegression(C=C, solver="lbfgs", max_iter=3000)
        self.name = name

    def fit(self, p, y, cov, **kwargs):
        x1 = safe_logit(p)
        X = [x1]

        cov = np.asarray(cov, float)
        if cov.ndim == 1:
            cov = cov[:, None]

        if self.cov_idx is None:
            cols = range(cov.shape[1])
        else:
            cols = [self.cov_idx]

        for j in cols:
            cj = cov[:, j]
            X.append(cj)
            X.append(cj**2)

        X = np.column_stack(X)
        self.lr.fit(X, y)
        return self

    def predict_proba(self, p, cov, **kwargs):
        x1 = safe_logit(p)
        X = [x1]

        cov = np.asarray(cov, float)
        if cov.ndim == 1:
            cov = cov[:, None]

        if self.cov_idx is None:
            cols = range(cov.shape[1])
        else:
            cols = [self.cov_idx]

        for j in cols:
            cj = cov[:, j]
            X.append(cj)
            X.append(cj**2)

        X = np.column_stack(X)
        return self.lr.predict_proba(X)[:, 1].astype(np.float64)

class MagBinnedWrapper:
    """
    Fit a base calibrator per magnitude bin.
    base_factory must return a *fresh* calibrator instance.
    """
    def __init__(self, base_factory, n_mag_bins=10, mag_strategy="quantile", name=None):
        self.base_factory = base_factory
        self.n_mag_bins = int(n_mag_bins)
        self.mag_strategy = mag_strategy
        self.mag_edges = None
        self.models = None
        self.name = name or f"magbinned_{getattr(base_factory(), 'name', 'cal')}"
    def fit(self, p, y, mag, **kwargs):
        p = np.asarray(p, float); y = np.asarray(y, int); mag = np.asarray(mag, float)
        if self.mag_strategy == "quantile":
            self.mag_edges = np.quantile(mag, np.linspace(0, 1, self.n_mag_bins + 1))
        else:
            self.mag_edges = np.linspace(mag.min(), mag.max(), self.n_mag_bins + 1)
        self.mag_edges = np.unique(self.mag_edges)
        if len(self.mag_edges) < 3:
            self.mag_edges = np.linspace(mag.min(), mag.max(), 3)

        ids = np.digitize(mag, self.mag_edges) - 1
        ids = np.clip(ids, 0, len(self.mag_edges) - 2)

        self.models = []
        global_fallback = self.base_factory().fit(p, y)
        for b in range(len(self.mag_edges) - 1):
            m = (ids == b)
            if np.sum(m) < 50:  # too few points -> fallback
                self.models.append(global_fallback)
            else:
                self.models.append(self.base_factory().fit(p[m], y[m]))
        return self
    def predict_proba(self, p, mag, **kwargs):
        p = np.asarray(p, float); mag = np.asarray(mag, float)
        ids = np.digitize(mag, self.mag_edges) - 1
        ids = np.clip(ids, 0, len(self.mag_edges) - 2)

        out = np.empty_like(p, dtype=np.float64)
        for b, model in enumerate(self.models):
            m = (ids == b)
            if np.any(m):
                out[m] = model.predict_proba(p[m])
        return out


# ----------------------------
# Collect outputs from your gaNdalF classifier
# ----------------------------
def collect_classifier_outputs(gandalf, *, cov_cols=None, mag_col=None, batch_size=131072):
    device = torch.device(str(gandalf.cfg.get("DEVICE", "cpu")).lower())
    pin = (device.type == "cuda")

    mag_col = mag_col or gandalf.cfg.get("MAG_COL", "BDF_MAG_DERED_CALIB_I")
    df = gandalf.classifier_data

    X_np = df[gandalf.cfg["INPUT_COLS"]].to_numpy(dtype=np.float32, copy=False)
    mag_np = df[mag_col].to_numpy(dtype=np.float32, copy=False)
    y_true = df[gandalf.cfg["OUTPUT_COLS_CF"]].to_numpy(int).ravel()

    cov = None
    if cov_cols is not None and len(cov_cols) > 0:
        cov = df[cov_cols].to_numpy(dtype=np.float32, copy=False)

    X_t = torch.from_numpy(X_np)
    mag_t = torch.from_numpy(mag_np)
    loader = DataLoader(
        TensorDataset(X_t, mag_t),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=max(1, (os.cpu_count() or 4) // 2),
        pin_memory=pin,
        persistent_workers=False,
    )

    gandalf.classifier_model.eval()
    logits_chunks, mags_chunks = [], []
    with torch.inference_mode():
        for xb, mb in loader:
            xb = xb.to(device, non_blocking=pin)
            logits = gandalf.classifier_model(xb).squeeze(-1).detach().float().cpu().numpy()
            logits_chunks.append(logits)
            mags_chunks.append(mb.numpy())

    logits = np.concatenate(logits_chunks).astype(np.float64)
    mag = np.concatenate(mags_chunks).astype(np.float64)
    p_raw = sigmoid(logits)

    out = dict(y=y_true.astype(int), mag=mag, logits=logits, p_raw=p_raw)
    if cov is not None:
        out["cov"] = cov.astype(np.float64)
        out["cov_cols"] = list(cov_cols)
    return out


# ----------------------------
# Benchmark runner
# ----------------------------
def evaluate_method(name, y_test, p_test, thr=0.5):
    yhat = (p_test >= thr).astype(int)
    return {
        "method": name,
        "brier": brier_score(y_test, p_test),
        "logloss": float(log_loss(y_test, safe_clip(p_test), labels=[0, 1])),
        "ece(20)": expected_calibration_error(y_test, p_test, n_bins=20),
        "acc@0.5": float(accuracy_score(y_test, yhat)),
        "f1@0.5": float(f1_score(y_test, yhat)),
    }


def run_covariate_sweep(y, p_raw, logits, cov, cov_cols, *, seed=41, test_size=0.5, max_fit=500_000):
    y = np.asarray(y, int)
    p_raw = np.asarray(p_raw, float)
    logits = np.asarray(logits, float)
    cov = np.asarray(cov, float)

    idx = np.arange(len(y))
    idx_cal, idx_test = train_test_split(idx, test_size=float(test_size), random_state=int(seed), stratify=y)

    if max_fit is not None and len(idx_cal) > int(max_fit):
        rng = np.random.default_rng(int(seed))
        idx_cal = rng.choice(idx_cal, size=int(max_fit), replace=False)

    y_cal, y_test = y[idx_cal], y[idx_test]
    p_cal, p_test = p_raw[idx_cal], p_raw[idx_test]
    cov_cal, cov_test = cov[idx_cal], cov[idx_test]

    rows = []

    # baseline uncalibrated
    rows.append(evaluate_method("uncalibrated", y_test, safe_clip(p_test)))

    # each covariate alone
    for j, col in enumerate(cov_cols):
        cal = CovariatePlattCalibrator(cov_idx=j, name=f"cov_platt[{col}]")
        cal.fit(p_cal, y_cal, cov=cov_cal)
        p_hat = safe_clip(cal.predict_proba(p_test, cov=cov_test))
        rows.append(evaluate_method(cal.name, y_test, p_hat))

    # all covariates together (can overfit a bit, but often very strong with L2)
    cal_all = CovariatePlattCalibrator(cov_idx=None, name="cov_platt[ALL]")
    cal_all.fit(p_cal, y_cal, cov=cov_cal)
    p_hat_all = safe_clip(cal_all.predict_proba(p_test, cov=cov_test))
    rows.append(evaluate_method(cal_all.name, y_test, p_hat_all))

    df = pd.DataFrame(rows).sort_values("brier").reset_index(drop=True)
    print("\n=== Covariate sweep (sorted by Brier) ===")
    print(df.head(20).to_string(index=False, float_format=lambda x: f"{x:0.6f}"))
    return df


def run_calibration_suite(y, p_raw, logits, mag, *, seed=41, test_size=0.5, max_fit=None):
    """
    Splits data -> fit calibrators on calibration split -> evaluate on test split.
    max_fit: optionally subsample calibration split for faster fitting (None = use all).
    """
    y = np.asarray(y, int)
    p_raw = np.asarray(p_raw, float)
    logits = np.asarray(logits, float)
    mag = np.asarray(mag, float)

    idx = np.arange(len(y))
    idx_cal, idx_test = train_test_split(
        idx, test_size=float(test_size), random_state=int(seed), stratify=y
    )

    # optionally subsample calibration set for speed
    if max_fit is not None and len(idx_cal) > int(max_fit):
        rng = np.random.default_rng(int(seed))
        idx_cal = rng.choice(idx_cal, size=int(max_fit), replace=False)

    y_cal, y_test = y[idx_cal], y[idx_test]
    p_cal, p_test = p_raw[idx_cal], p_raw[idx_test]
    log_cal, log_test = logits[idx_cal], logits[idx_test]
    mag_cal, mag_test = mag[idx_cal], mag[idx_test]

    calibrators = []

    calibrators.append(IdentityCalibrator())
    calibrators.append(PlattCalibrator())
    calibrators.append(BetaCalibrator())
    calibrators.append(IsotonicCalibrator())
    calibrators.append(HistogramCalibrator(n_bins=40, strategy="quantile"))

    # temperature scaling (needs logits)
    calibrators.append(TemperatureScaler())

    # mag-aware
    calibrators.append(MagAwarePlattCalibrator())

    # mag-binned variants (nice for selection functions)
    calibrators.append(MagBinnedWrapper(lambda: PlattCalibrator(), n_mag_bins=10, name="magbinned_platt"))
    calibrators.append(MagBinnedWrapper(lambda: BetaCalibrator(), n_mag_bins=10, name="magbinned_beta"))
    calibrators.append(MagBinnedWrapper(lambda: IsotonicCalibrator(), n_mag_bins=10, name="magbinned_isotonic"))

    rows = []

    print("\n=== Calibration benchmark ===")
    print(f"  N total     : {len(y)}")
    print(f"  N cal-fit   : {len(idx_cal)}")
    print(f"  N test      : {len(idx_test)}")
    print(f"  pos rate    : {np.mean(y):.6f}\n")

    for cal in calibrators:
        name = getattr(cal, "name", cal.__class__.__name__)

        # fit
        if isinstance(cal, TemperatureScaler):
            cal.fit(log_cal, y_cal)
            p_hat = cal.predict_proba(log_test)
        elif isinstance(cal, (MagAwarePlattCalibrator, MagBinnedWrapper)):
            cal.fit(p_cal, y_cal, mag=mag_cal)
            p_hat = cal.predict_proba(p_test, mag=mag_test)
        else:
            cal.fit(p_cal, y_cal)
            p_hat = cal.predict_proba(p_test)

        p_hat = safe_clip(p_hat)
        row = evaluate_method(name, y_test, p_hat, thr=0.5)

        # extra info
        if isinstance(cal, TemperatureScaler):
            row["T"] = getattr(cal, "T", np.nan)
        else:
            row["T"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("brier").reset_index(drop=True)

    # pretty print
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 140):
        print(df.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))

    best = df.iloc[0]
    print("\nBest by Brier:", best["method"], f"(brier={best['brier']:.6f}, logloss={best['logloss']:.6f}, ece={best['ece(20)']:.6f})")
    return df


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    import os

    # You already have these in your environment:
    # from Handler import *
    # from gandalf_galaxie_dataset import DESGalaxies
    # from gandalf_calibration_model.gaNdalF_calibration_model import MagAwarePlatt
    #
    # Here we assume you already create gandalf + cfg + logger as in your pipeline.
    #
    # gandalf = gaNdalF(gandalf_logger, cfg)
    # gandalf.init_classifier()

    raise SystemExit(
        "Edit the __main__ part: create `gandalf`, call init_classifier(), then:\n"
        "outs = collect_classifier_outputs(gandalf)\n"
        "run_calibration_suite(**outs, seed=41, test_size=0.5, max_fit=None)\n"
    )