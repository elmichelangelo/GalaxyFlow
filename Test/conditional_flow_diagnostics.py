# Conditional Flow Diagnostics (script version)
# Note: enable seaborn locally by uncommenting its import and calls.


# --- Imports ---
import math, json, os
import numpy as np
import torch
from contextlib import nullcontext
import matplotlib.pyplot as plt

# Optional (lokal aktivierbar):
# import seaborn as sns
# sns.set()


# --- Konfiguration (anpassbar) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE_EVAL = 4096
PERM_REPEATS = 2           # wie oft eine Spalte permutiert wird
COUNTERFACTUAL_DELTA = 0.1 # Schrittweite für Δx_j
TOPK = 5                   # wie viele Top-Features wir detailliert anschauen
USE_AMP = False            # für Diagnostik besser aus
PRINT_WIDTH = 120


# --- Utility-Funktionen ---
@torch.no_grad()
def mean_nll(model, Y, X, bs=4096, use_amp=False, device="cpu"):
    model.eval()
    tot, n = 0.0, 0
    X = X.to(device)
    Y = Y.to(device)
    autocast_ctx = (torch.amp.autocast(device_type="cuda") if (use_amp and device=="cuda") else nullcontext())
    for i in range(0, X.size(0), bs):
        xb = X[i:i+bs]
        yb = Y[i:i+bs]
        with autocast_ctx:
            nll = -model.log_probs(yb, xb)  # shape [B,1]
        tot += nll.sum().item()
        n += yb.size(0)
    return tot / max(n,1)

def perm_importance(model, Y, X, col_names, repeats=3, bs=4096, use_amp=False, device="cpu", rng=None):
    base = mean_nll(model, Y, X, bs, use_amp, device)
    deltas = {}
    Xcpu = X.detach().cpu()
    N = Xcpu.shape[0]
    rng = np.random.default_rng(None if rng is None else rng)
    for j, name in enumerate(col_names):
        incs = []
        for _ in range(repeats):
            Xperm = Xcpu.clone()
            idx = torch.from_numpy(rng.permutation(N))
            Xperm[:, j] = Xperm[idx, j]   # nur Spalte j mischen
            inc = mean_nll(model, Y, Xperm.to(device), bs, use_amp, device) - base
            incs.append(float(inc))
        deltas[name] = (float(np.mean(incs)), float(np.std(incs)))
    return base, deltas

def all_shuffle_nll(model, Y, X, bs=4096, use_amp=False, device="cpu", rng=None):
    rng = np.random.default_rng(None if rng is None else rng)
    Xcpu = X.detach().cpu().clone()
    N = Xcpu.shape[0]
    # jede Spalte unabhängig permutieren
    for j in range(Xcpu.shape[1]):
        idx = torch.from_numpy(rng.permutation(N))
        Xcpu[:, j] = Xcpu[idx, j]
    return mean_nll(model, Y, Xcpu.to(device), bs, use_amp, device)

def context_grad_sensitivity(model, Y, X, bs=2048, device="cpu"):
    model.eval()
    grads_sum = torch.zeros(X.size(1), device=device)
    count = 0
    for i in range(0, X.size(0), bs):
        xb = X[i:i+bs].to(device).detach().requires_grad_(True)
        yb = Y[i:i+bs].to(device)
        lp = model.log_probs(yb, xb).mean()   # scalar
        g, = torch.autograd.grad(lp, xb, retain_graph=False, create_graph=False)
        grads_sum += g.abs().mean(dim=0)
        count += 1
    return (grads_sum / max(count,1)).detach().cpu()  # Größe [C]

@torch.no_grad()
def counterfactual_shift(model, X, j, delta, sample_bs=8192, device="cpu"):
    model.eval()
    N = X.size(0)
    # a) Original-Samples
    Ys = []
    for i in range(0, N, sample_bs):
        xb = X[i:i+sample_bs].to(device)
        Ys.append(model.sample(num_samples=xb.size(0), cond_inputs=xb))  # [B,Dy]
    Y0 = torch.cat(Ys, dim=0).cpu()

    # b) Perturbierte Kontexte
    Xp = X.clone()
    Xp[:, j] += delta
    Ys = []
    for i in range(0, N, sample_bs):
        xb = Xp[i:i+sample_bs].to(device)
        Ys.append(model.sample(num_samples=xb.size(0), cond_inputs=xb))
    Y1 = torch.cat(Ys, dim=0).cpu()

    mean_shift = (Y1.mean(dim=0) - Y0.mean(dim=0))       # [Dy]
    var_shift  = (Y1.var(dim=0, unbiased=False) - Y0.var(dim=0, unbiased=False))  # [Dy]
    return mean_shift, var_shift

def topk_by_delta(deltas, k=5):
    arr = [(name, v[0], v[1]) for name, v in deltas.items()]
    arr.sort(key=lambda t: t[1], reverse=True)  # nach ΔNLL-mean absteigend
    return arr[:k], arr

def print_table(rows, headers):
    colw = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i,h in enumerate(headers)]
    fmt = " | ".join("{:%d}"%w for w in colw)
    line = "-+-".join("-"*w for w in colw)
    print(fmt.format(*headers))
    print(line)
    for r in rows:
        print(fmt.format(*r))

# --- Plot-Funktionen (matplotlib-only) ---
def barplot_perm_importance(deltas, title="Permutation Importance (ΔNLL)", figsize=(10, 5)):
    names = list(deltas.keys())
    vals  = [deltas[n][0] for n in names]
    errs  = [deltas[n][1] for n in names]
    idx = np.arange(len(names))
    plt.figure(figsize=figsize)
    plt.bar(idx, vals, yerr=errs)
    plt.xticks(idx, names, rotation=45, ha="right")
    plt.ylabel("ΔNLL (↑ schlimmer)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def barplot_grad_sensitivity(names, grads, title="Gradient Sensitivity ⟨|∂ log p/∂x_j|⟩", figsize=(10,5)):
    idx = np.arange(len(names))
    plt.figure(figsize=figsize)
    plt.bar(idx, grads)
    plt.xticks(idx, names, rotation=45, ha="right")
    plt.ylabel("⟨|∂ log p/∂x_j|⟩")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def mean_var_shift_plot(mean_shift, var_shift, out_names=None, title="Counterfactual Shift (Δx_j)", figsize=(10,6)):
    Dy = mean_shift.numel()
    xs = np.arange(Dy)
    plt.figure(figsize=figsize)
    plt.plot(xs, mean_shift.numpy(), marker="o", linestyle="-", label="Δ mean(Y)")
    plt.plot(xs, var_shift.numpy(),  marker="s", linestyle="--", label="Δ var(Y)")
    if out_names is not None and len(out_names)==Dy:
        plt.xticks(xs, out_names, rotation=45, ha="right")
    plt.legend()
    plt.xlabel("Output-Dimension")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# --- Einbindung deiner Daten/Models ---
# Ersetze diesen Block durch dein tatsächliches Laden:
# Beispiel für dein Projekt (Pseudo-Code; bitte anpassen):
#
# from gandalf_flow import gaNdalFFlow
# train_flow = gaNdalFFlow(cfg=cfg, learning_rate=..., number_hidden=..., number_blocks=..., number_layers=..., batch_size=...)
# model = train_flow.model.to(DEVICE).eval()
# dfv = train_flow.galaxies.valid_dataset
# col_in  = train_flow.cfg["INPUT_COLS"]
# col_out = train_flow.cfg["OUTPUT_COLS"]
# X_valid = torch.tensor(dfv[col_in].values, dtype=next(model.parameters()).dtype)
# Y_valid = torch.tensor(dfv[col_out].values, dtype=next(model.parameters()).dtype)
# names_in = list(col_in)
# names_out = list(col_out)
#
# Für eine Demo mit synthetischen Daten (funktioniert ohne dein Projekt):
torch.manual_seed(0)
N, C, Dy = 4000, 20, 8
# künstliches lineares Mapping mit Nichtlinearität
W = torch.randn(C, Dy)*0.4
b = torch.randn(Dy)*0.1
def gen_y(x):
    return torch.tanh(x @ W + b) + 0.05*torch.randn(x.size(0), Dy)

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_inputs = Dy
    def forward(self, y, cond_inputs=None, mode='direct'):
        return y, torch.zeros(y.size(0), 1, device=y.device, dtype=y.dtype)
    def log_probs(self, y, cond_inputs=None):
        # Gauß um die deterministische gen_y(cond_inputs)
        mu = gen_y(cond_inputs)
        ll = -0.5*((y-mu)**2).sum(dim=1, keepdim=True)  # var=1
        return ll
    @torch.no_grad()
    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        mu = gen_y(cond_inputs)
        return mu + torch.randn_like(mu)

model = ToyModel().to(DEVICE).eval()
X_valid = torch.randn(N, C)
Y_valid = gen_y(X_valid)  # aus "wahrer" cond. Verteilung
names_in = [f"x{j}" for j in range(C)]
names_out = [f"y{k}" for k in range(Dy)]
print(f"Demo-Daten: X_valid={tuple(X_valid.shape)}, Y_valid={tuple(Y_valid.shape)}")


# --- 1) Permutation-Importance ---
base_nll, deltas = perm_importance(model, Y_valid, X_valid, names_in, repeats=PERM_REPEATS, bs=BATCH_SIZE_EVAL, use_amp=USE_AMP, device=DEVICE)
topk, all_rows = topk_by_delta(deltas, k=min(TOPK, len(names_in)))

print(f"Base NLL: {base_nll:.6f}")
rows = [(n, f"{m:.6f}", f"{s:.6f}") for (n,m,s) in [(n,)+deltas[n] for n in deltas]]
rows.sort(key=lambda r: float(r[1]), reverse=True)
print_table(rows, headers=["Feature", "ΔNLL (mean)", "ΔNLL (std)"])

# Plot (alle Features)
barplot_perm_importance(deltas, title="Permutation Importance (ΔNLL)")


# --- 2) Gradienten-Sensitivität ---
grads = context_grad_sensitivity(model, Y_valid, X_valid, bs=2048, device=DEVICE).numpy()
rows = list(zip(names_in, [f"{g:.6e}" for g in grads]))
rows.sort(key=lambda t: float(t[1]), reverse=True)
print_table(rows, headers=["Feature", "⟨|∂ log p/∂x_j|⟩"])
barplot_grad_sensitivity(names_in, grads, title="Gradient Sensitivity ⟨|∂ log p/∂x_j|⟩")


# --- 3) Counterfactual‑Sampling auf Top-K Features ---
# Wähle Top-K nach ΔNLL
top_features = [name for name,_,_ in topk]
print("Top-Features (nach ΔNLL):", top_features)

subset = X_valid[: min(10000, X_valid.size(0))].clone()
for name in top_features:
    j = names_in.index(name)
    mean_shift, var_shift = counterfactual_shift(model, subset, j, delta=COUNTERFACTUAL_DELTA, device=DEVICE)
    print(f"\nFeature: {name} (Δx={COUNTERFACTUAL_DELTA})")
    print("Δ mean(Y):", mean_shift.numpy())
    print("Δ var(Y): ", var_shift.numpy())
    mean_var_shift_plot(mean_shift, var_shift, out_names=names_out, title=f"Counterfactual Shift: {name} (Δx={COUNTERFACTUAL_DELTA})")


# --- 4) All‑Shuffle Sanity Check ---
nll_all_shuffle = all_shuffle_nll(model, Y_valid, X_valid, bs=BATCH_SIZE_EVAL, use_amp=USE_AMP, device=DEVICE)
print(f"NLL (all-shuffle contexts): {nll_all_shuffle:.6f}  |  Base NLL: {base_nll:.6f}  |  Δ = {nll_all_shuffle - base_nll:.6f}")

