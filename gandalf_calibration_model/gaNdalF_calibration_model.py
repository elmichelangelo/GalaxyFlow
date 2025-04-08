from sklearn.linear_model import LogisticRegression
from torch import nn
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


class gaNdalFCalibModel(nn.Module):
    def __init__(self, cfg, performance_logger):
        super().__init__()
        self.cfg = cfg
        self.performance_logger = performance_logger
        self.device = self.cfg["DEVICE"]
        self.batch_size = self.cfg["BATCH_SIZE"]
        self.calibration_model = None
        self.df_balrog = None
        self.df_gandalf = None

        self.galaxies, self.valid_loader, self.test_loader = self.init_dataset()
        self.gandalf_classifier = self.init_classifier_model()
        self.gandalf_classifier = self.gandalf_classifier.to(self.device)

    def init_dataset(self):
        galaxies = DESGalaxies(
            cfg=self.cfg,
            kind="classifier_training"
        )

        test_dataset = galaxies.test_dataset
        desired_test_size = 20208363
        num_test_samples = len(test_dataset)

        sampled_indices = torch.randint(0, num_test_samples, size=(desired_test_size,))
        sampler = torch.utils.data.SubsetRandomSampler(sampled_indices)

        valid_loader = DataLoader(galaxies.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, sampler=sampler, num_workers=0)

        return galaxies, valid_loader, test_loader

    def init_classifier_model(self):
        """"""
        gandalf_classifier = torch.load(f"{self.cfg['PATH_TRAINED_NN']}/{self.cfg['FILENAME_NN_CLASSF']}")
        gandalf_classifier = gandalf_classifier.float()
        return gandalf_classifier

    def calibrate(self):
        self.gandalf_classifier.eval()
        predictions = []

        print(next(self.gandalf_classifier.parameters()).dtype)

        tsr_input, tsr_output = dataset_to_tensor(self.valid_loader.dataset)
        with torch.no_grad():
            outputs = self.gandalf_classifier(tsr_input.float().to(self.device))
            predictions = outputs.squeeze().cpu().numpy()

        # Drop samples with NaN predictions
        # mask = ~np.isnan(predictions)
        predictions = np.array(predictions).reshape(-1, 1)  # [mask].reshape(-1, 1)
        labels = tsr_output.numpy().ravel()#[mask]

        self.calibration_model = LogisticRegression(class_weight='balanced')  # LogisticRegression(penalty='l2', C=1.0, max_iter=1000, class_weight='balanced')
        self.calibration_model.fit(predictions, labels)

        y_calibrated = self.calibration_model.predict_proba(predictions)[:, 1]

        from sklearn.metrics import brier_score_loss
        score_uncal = brier_score_loss(labels, predictions.squeeze())
        score_cal = brier_score_loss(labels, y_calibrated)
        ece_uncal = compute_ece(labels, predictions.squeeze(), n_bins=15)
        ece_cal = compute_ece(labels, y_calibrated, n_bins=50)
        print(f"Brier Score non calibrated: {score_uncal:.6f}")
        print(f"Brier Score calibrated: {score_cal:.6f}")
        print(f"ECE non calibrated: {ece_uncal:.4f}")
        print(f"ECE calibrated:   {ece_cal:.4f}")

        if self.cfg["PLOT"] is True:
            if self.cfg["PLOT_RELIABILITY"] is True:
                plot_combined_reliability_diagram(
                    y_true=labels,
                    y_prob_uncal=predictions,
                    y_prob_cal=y_calibrated,
                    n_bins=50,
                    title="Reliability Diagram: Calibrated vs. Uncalibrated",
                    show_hist=True,
                    score_uncal=score_uncal,
                    score_cal=score_cal,
                    ece_uncal=ece_uncal,
                    ece_cal=ece_cal,
                    show_plot=self.cfg["SHOW_PLOT"],
                    save_plot=self.cfg["SAVE_PLOT"],
                    save_name=f"{self.cfg['PATH_SAVE_PLOTS']}/reliability_diagram_calib_vs_uncalib.pdf"
                )

        return self.calibration_model

    def predict_calibrated(self, y_pred):
        calibrated_outputs = self.calibration_model.predict_proba(y_pred.reshape(-1, 1))[:, 1]
        return calibrated_outputs

    def run_training_calib_model(self):
        self.calibration_model = self.calibrate()

        self.test_calibration_model()

        if  self.cfg["PLOT"] is True:
            self.plot_calib_model(
                df_balrog=self.df_balrog,
                df_gandalf=self.df_gandalf,
                path_save_plots=self.cfg["PATH_SAVE_PLOTS"]
            )

        if self.cfg["SAVE_CALIB_MODEL"]is True:
            self.save_calibration_model()

    def test_calibration_model(self):
        self.gandalf_classifier.eval()
        tsr_input, tsr_output = dataset_to_tensor(self.test_loader.dataset)
        input_data = tsr_input.numpy()
        arr_flow_output = np.ones((len(input_data), len(self.cfg["OUTPUT_COLS_MAG_FLOW"])))
        self.df_balrog = pd.DataFrame(
            np.concatenate((input_data, arr_flow_output), axis=1),
            columns=self.cfg['INPUT_COLS_MAG_CLASSF'] + self.cfg['OUTPUT_COLS_MAG_FLOW']
        )
        if self.cfg['APPLY_SCALER_CLASSF'] is True:
            self.df_balrog = pd.DataFrame(
                self.galaxies.scaler.inverse_transform(self.df_balrog),
                columns=self.df_balrog.keys()
            )

        if self.cfg['APPLY_YJ_TRANSFORM_CLASSF'] is True:
            if self.cfg['TRANSFORM_COLS_CLASSF'] is None:
                trans_col = self.df_balrog.keys()
            else:
                trans_col = self.cfg['TRANSFORM_COLS_CLASSF']
            self.df_balrog = self.galaxies.yj_inverse_transform_data(
                data_frame=self.df_balrog,
                columns=trans_col
            )
        self.df_gandalf = self.df_balrog.copy()
        detected_true = tsr_output.numpy().reshape(-1).astype(int)
        self.df_balrog['detected'] = detected_true

        with torch.no_grad():
            probability = self.gandalf_classifier(tsr_input.float().to(self.device)).squeeze().cpu().numpy()
        probability_calibrated = self.predict_calibrated(probability)

        detected = probability > np.random.rand(len(detected_true))
        detected_calibrated = probability_calibrated > np.random.rand(len(tsr_input))

        self.df_gandalf['detected non calibrated'] = detected
        self.df_gandalf['detected'] = detected_calibrated
        self.df_gandalf['probability'] = probability
        self.df_gandalf['probability_calibrated'] = probability_calibrated

    def save_calibration_model(self):
        """"""
        # save config model
        joblib.dump(
            self.calibration_model,
            f"{self.cfg['PATH_SAVE_NN']}/gaNdalF_classifier_{self.batch_size}_scr_{self.cfg['APPLY_SCALER_CLASSF']}_yjt_{self.cfg['APPLY_YJ_TRANSFORM_CLASSF']}_run_{self.cfg['RUN_DATE']}.pkl"
        )

    def plot_calib_model(self, df_balrog, df_gandalf, path_save_plots):
        if self.cfg["PLOT_MISS"] is True:
            plot_multivariate_clf(
                df_balrog_detected=df_balrog[df_balrog['detected'] == 1],
                df_gandalf_detected=df_gandalf[df_gandalf['detected'] == 1],
                df_balrog_not_detected=df_balrog[df_balrog['detected'] == 0],
                df_gandalf_not_detected=df_gandalf[df_gandalf['detected'] == 0],
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
                save_name=f"{path_save_plots}/{self.cfg['RUN_DATE']}_classifier_multiv.pdf",
                sample_size=100000,
                x_range=(17.5, 26.5),
                title=f"gaNdalF vs. Balrog: Photometric Property Distribution Comparison calibrated"
            )

            plot_multivariate_clf(
                df_balrog_detected=df_balrog[df_balrog['detected'] == 1],
                df_gandalf_detected=df_gandalf[df_gandalf['detected non calibrated'] == 1],
                df_balrog_not_detected=df_balrog[df_balrog['detected'] == 0],
                df_gandalf_not_detected=df_gandalf[df_gandalf['detected non calibrated'] == 0],
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
                save_name=f"{path_save_plots}/{self.cfg['RUN_DATE']}_classifier_multiv.pdf",
                sample_size=100000,  # None,
                x_range=(17.5, 26.5),
                title=f"gaNdalF vs. Balrog: Photometric Property Distribution Comparison not calibrated"
            )