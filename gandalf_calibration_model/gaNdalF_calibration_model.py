from sklearn.linear_model import LogisticRegression
from torch import nn, optim
from torch.utils.data import DataLoader
from gandalf_galaxie_dataset import DESGalaxies
from Handler import *
import torch
import os
import seaborn as sns
import numpy as np
import joblib
import pandas as pd
import sys
import matplotlib.pyplot as plt


# =========================== Mag-aware Platt =================================

class MagAwarePlatt:
    """2D Platt: Logistic Regression auf [logit(p), mag, mag^2]."""
    def __init__(self):
        self.coef_ = None  # (3,)
        self.intercept_ = None  # scalar

    @staticmethod
    def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, p: np.ndarray, mag: np.ndarray, y: np.ndarray, max_iter: int = 200, lr: float = 0.1):
        p = p.astype(float).ravel()
        mag = mag.astype(float).ravel()
        y = y.astype(int).ravel()
        x1 = self._safe_logit(p); x2 = mag; x3 = mag ** 2
        X = np.c_[x1, x2, x3]
        w = np.zeros(3, dtype=float); b = 0.0
        for _ in range(max_iter):
            z = X @ w + b
            yhat = self._sigmoid(z)
            err = yhat - y
            grad_w = X.T @ err / X.shape[0]
            grad_b = err.mean()
            w -= lr * grad_w; b -= lr * grad_b
        self.coef_ = w; self.intercept_ = b
        return self

    def transform(self, p: np.ndarray, mag: np.ndarray) -> np.ndarray:
        x1 = self._safe_logit(np.asarray(p, float))
        x2 = np.asarray(mag, float)
        x3 = x2 ** 2
        z = np.c_[x1, x2, x3] @ self.coef_ + self.intercept_
        return self._sigmoid(z)

    def state_dict(self):
        return {"coef_": self.coef_, "intercept_": self.intercept_}

    def load_state_dict(self, d):
        self.coef_ = np.asarray(d["coef_"], float)
        self.intercept_ = float(d["intercept_"])

# =========================== Temperature wrapper =============================

class ModelWithTemperature(nn.Module):
    def __init__(self, model, logger):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.logger = logger

    def forward(self, x): return self.model(x) / self.temperature

    def set_temperature(self, valid_loader, device):
        self.to(device)
        nll = nn.BCEWithLogitsLoss().to(device)
        opt = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        logits_list, labels_list = [], []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits_list.append(self.model(xb)); labels_list.append(yb)
        logits = torch.cat(logits_list); labels = torch.cat(labels_list)

        def eval():
            opt.zero_grad()
            loss = nll(logits / self.temperature, labels)
            loss.backward(); return loss

        opt.step(eval)
        self.logger.log_info_stream(f"Optimal temperature: {self.temperature.item():.3f}")
        return self