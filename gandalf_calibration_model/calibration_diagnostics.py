import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, f1_score

# Reuse helpers from your benchmark file if available; otherwise keep these:
def safe_clip(p, eps=1e-12):
    return np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)

def brier_score(y, p):
    y = np.asarray(y).astype(np.float64)
    p = np.asarray(p).astype(np.float64)
    return float(np.mean((p - y) ** 2))

def expected_calibration_error(y, p, n_bins=20):
    y = np.asarray(y).astype(np.float64)
    p = np.asarray(p).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(p, bins) - 1
    ids = np.clip(ids, 0, n_bins - 1)

    ece = 0.0
    n = len(y)
    for b in range(n_bins):
        m = (ids == b)
        if not np.any(m):
            continue
        conf = np.mean(p[m])
        acc = np.mean(y[m])
        ece += (np.sum(m) / n) * abs(acc - conf)
    return float(ece)

def reliability_curve_quantile(y, p, n_bins=20, min_count=500):
    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)

    # Quantile bins in p
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


# --------- Main: fit calibrators and collect predictions on TEST split ---------
def fit_and_predict_on_test(
    y, p_raw, logits, mag, cov=None, cov_cols=None,
    *,
    seed=41, test_size=0.5, max_fit=500_000,
    calibrator_factories=None
):
    """
    Fits a set of calibrators on a calibration split and returns a dataframe with
    p_hat(method) on the test split + test metadata (mag, cov).
    """
    y = np.asarray(y, int)
    p_raw = np.asarray(p_raw, float)
    logits = np.asarray(logits, float)
    mag = np.asarray(mag, float)

    idx = np.arange(len(y))
    idx_cal, idx_test = train_test_split(
        idx, test_size=float(test_size), random_state=int(seed), stratify=y
    )

    if max_fit is not None and len(idx_cal) > int(max_fit):
        rng = np.random.default_rng(int(seed))
        idx_cal = rng.choice(idx_cal, size=int(max_fit), replace=False)

    y_cal, y_test = y[idx_cal], y[idx_test]
    p_cal, p_test = p_raw[idx_cal], p_raw[idx_test]
    log_cal, log_test = logits[idx_cal], logits[idx_test]
    mag_cal, mag_test = mag[idx_cal], mag[idx_test]

    cov_cal = cov_test = None
    if cov is not None:
        cov = np.asarray(cov, float)
        cov_cal = cov[idx_cal]
        cov_test = cov[idx_test]

    if calibrator_factories is None:
        raise ValueError("Please pass `calibrator_factories` (see example below).")

    # dataframe holding test set
    # dataframe holding test set
    df = pd.DataFrame({
        "y": y_test.astype(int),
        "mag": mag_test.astype(float),
        "p_raw": safe_clip(p_test),
        "logits": log_test.astype(float),
    }, index=idx_test)  # <--- WICHTIG
    df.index.name = "orig_idx"  # optional, aber praktisch
    if cov_test is not None and cov_cols is not None:
        for j, c in enumerate(cov_cols):
            df[c] = cov_test[:, j]

    fitted = {}
    for name, factory in calibrator_factories.items():
        cal = factory()

        # Fit/predict routing based on what calibrator expects
        # Conventions:
        #  - temperature: fit(logits, y), predict_proba(logits)
        #  - mag-aware/binned: fit(p, y, mag=...), predict_proba(p, mag=...)
        #  - cov-aware: fit(p, y, cov=...), predict_proba(p, cov=...)
        #  - otherwise: fit(p, y), predict_proba(p)
        if name.startswith("temperature"):
            cal.fit(log_cal, y_cal)
            p_hat = cal.predict_proba(log_test)
        elif "mag" in name and ("binned" in name or "aware" in name):
            cal.fit(p_cal, y_cal, mag=mag_cal)
            p_hat = cal.predict_proba(p_test, mag=mag_test)
        elif "cov" in name:
            if cov_cal is None:
                raise ValueError("cov is required for cov calibrators.")
            cal.fit(p_cal, y_cal, cov=cov_cal)
            p_hat = cal.predict_proba(p_test, cov=cov_test)
        else:
            cal.fit(p_cal, y_cal)
            p_hat = cal.predict_proba(p_test)

        df[f"p_{name}"] = safe_clip(p_hat)
        fitted[name] = cal

    return df, fitted


# --------- Per-bin metric tables ---------
def per_bin_metrics(df_test, *, var, methods, n_bins=10, binning="quantile", ece_bins=20):
    """
    Returns a long-form table: one row per (bin, method) with Brier/ECE/LogLoss/Acc/F1.
    """
    x = np.asarray(df_test[var], float)
    y = np.asarray(df_test["y"], int)

    if binning == "quantile":
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    else:
        edges = np.linspace(np.nanmin(x), np.nanmax(x), n_bins + 1)

    edges = np.unique(edges)
    if len(edges) < 3:
        raise ValueError(f"Not enough unique edges for binning variable '{var}'.")

    bin_ids = np.digitize(x, edges) - 1
    bin_ids = np.clip(bin_ids, 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        m = (bin_ids == b)
        if np.sum(m) < 100:  # skip tiny bins
            continue

        yb = y[m]
        x_lo, x_hi = edges[b], edges[b + 1]
        x_mid = 0.5 * (x_lo + x_hi)

        for method in methods:
            p = np.asarray(df_test[f"p_{method}"][m], float)
            p = safe_clip(p)

            yhat = (p >= 0.5).astype(int)
            rows.append({
                "var": var,
                "bin": b,
                "bin_lo": float(x_lo),
                "bin_hi": float(x_hi),
                "bin_mid": float(x_mid),
                "n": int(np.sum(m)),
                "pos_rate": float(np.mean(yb)),
                "method": method,
                "brier": brier_score(yb, p),
                "logloss": float(log_loss(yb, p, labels=[0, 1])),
                "ece": expected_calibration_error(yb, p, n_bins=ece_bins),
                "acc@0.5": float(accuracy_score(yb, yhat)),
                "f1@0.5": float(f1_score(yb, yhat)),
            })

    return pd.DataFrame(rows)


# --------- Reliability plots ---------
def plot_reliability(df_test, methods, *, title="Reliability diagram", n_bins=20, outpath=None):
    y = df_test["y"].to_numpy(int)

    plt.figure()
    # perfect line
    plt.plot([0, 1], [0, 1])

    for method in methods:
        p = df_test[f"p_{method}"].to_numpy(float)
        mean_p, frac_pos, counts = reliability_curve_quantile(y, p, n_bins=n_bins, min_count=500)
        plt.plot(mean_p, frac_pos, marker="o", linestyle="-", label=method)

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=300)
    plt.show()


def add_delta_to_uncal(df_bins, baseline="uncalibrated"):
    base = df_bins[df_bins["method"] == baseline][["var","bin","brier","ece","logloss"]].rename(
        columns={"brier":"brier_base", "ece":"ece_base", "logloss":"logloss_base"}
    )
    out = df_bins.merge(base, on=["var","bin"], how="left")
    out["delta_brier"] = out["brier"] - out["brier_base"]
    out["delta_ece"] = out["ece"] - out["ece_base"]
    out["delta_logloss"] = out["logloss"] - out["logloss_base"]
    return out

def summarize_delta(df_bins_delta):
    # weighted means over bins
    def wmean(g, col):
        w = g["n"].to_numpy(float)
        x = g[col].to_numpy(float)
        return float(np.sum(w * x) / np.sum(w))

    summ = (
        df_bins_delta
        .groupby(["var", "method"], as_index=False)
        .apply(lambda g: pd.Series({
            "N_total": int(g["n"].sum()),
            "mean_delta_brier": wmean(g, "delta_brier"),
            "mean_delta_ece": wmean(g, "delta_ece"),
            "mean_delta_logloss": wmean(g, "delta_logloss"),
        }))
        .reset_index(drop=True)
        .sort_values(["var", "mean_delta_brier"])
    )
    return summ


def _auto_xlim(p_all, q=(0.001, 0.999), pad=0.02):
    lo, hi = np.quantile(p_all, q)
    lo = max(0.0, lo - pad)
    hi = min(1.0, hi + pad)
    # wenn Range groß genug ist, zeig trotzdem [0,1]
    if hi - lo > 0.5:
        return (0.0, 1.0)
    return (lo, hi)


def plot_reliability_slices(df_test, methods, *, slice_var="mag", n_slices=4, n_bins=20, outpath=None):
    """
    Reliability diagram in slices of slice_var (quantile slices).
    """
    x = df_test[slice_var].to_numpy(float)
    y = df_test["y"].to_numpy(int)

    edges = np.quantile(x, np.linspace(0, 1, n_slices + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        raise ValueError(f"Not enough unique edges for slicing '{slice_var}'.")

    plt.figure()
    plt.plot([0, 1], [0, 1])

    # To avoid too many lines, we plot each method averaged across slices as separate figures
    # -> better: one figure per slice
    plt.close()

    for s in range(len(edges) - 1):
        lo, hi = edges[s], edges[s + 1]
        m = (x >= lo) & (x <= hi) if s == len(edges) - 2 else (x >= lo) & (x < hi)
        if np.sum(m) < 500:
            continue

        plt.figure()
        plt.plot([0, 1], [0, 1])

        for method in methods:
            p = df_test.loc[m, f"p_{method}"].to_numpy(float)
            yy = y[m]
            mean_p, frac_pos, counts = reliability_curve_quantile(yy, p, n_bins=n_bins, min_count=500)
            plt.plot(mean_p, frac_pos, marker="o", label=method)

        # sammle alle p im Slice über alle Methoden für sinnvolle Achsenlimits
        p_stack = np.concatenate([df_test.loc[m, f"p_{method}"].to_numpy(float) for method in methods])
        xlo, xhi = _auto_xlim(p_stack)

        plt.xlim(xlo, xhi)
        plt.ylim(0.0, 1.0)  # y meist ok so; optional auch auto-ylim

        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title(f"Reliability slices: {slice_var} in [{lo:.3f}, {hi:.3f}]  (N={np.sum(m)})")
        plt.legend()
        plt.tight_layout()
        if outpath:
            base, ext = outpath.rsplit(".", 1)
            plt.savefig(f"{base}_slice{s}.{ext}", dpi=200)
        plt.show()