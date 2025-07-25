from gandalf_galaxie_dataset import DESGalaxies
from torch.utils.data import DataLoader, RandomSampler
from scipy.stats import binned_statistic, median_abs_deviation
from Handler import *
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import h5py
import joblib
import pickle
import torch
import os
# plt.style.use('seaborn-white')


class gaNdalF(object):
    """"""
    def __init__(self, gandalf_logger, cfg):
        """"""
        self.gandalf_logger = gandalf_logger
        self.gandalf_logger.log_info_stream(f"Init gaNdalF")
        self.scalers_classifier = joblib.load(f"{cfg['PATH_TRANSFORMERS']}/{cfg['FILENAME_SCALER_CLASSIFIER']}")
        self.scalers_nf = joblib.load(f"{cfg['PATH_TRANSFORMERS']}/{cfg['FILENAME_SCALER_NF']}")
        self.cfg = cfg

        self.galaxies = self.init_dataset(gandalf_logger=gandalf_logger)
        self.gandalf_flow, self.gandalf_classifier = self.init_trained_models()  # , self.calibration_model

    def init_dataset(self, gandalf_logger):

        galaxies = DESGalaxies(
            dataset_logger=gandalf_logger,
            cfg=self.cfg,
        )

        if self.cfg['NUMBER_SAMPLES'] == -1:
            self.cfg['NUMBER_SAMPLES'] = len(galaxies.run_dataset)
        elif self.cfg['NUMBER_SAMPLES'] == -666:
            self.cfg['NUMBER_SAMPLES'] = 20208363

        return galaxies

    def init_trained_models(self):
        """"""
        gandalf_flow = None
        gandalf_classifier = None

        self.gandalf_logger.log_info(f"Init trained models")
        self.gandalf_logger.log_stream(f"Init trained models")

        # TODO for security reasons (warning) save model as state dict and load with weights_only=True
        if self.cfg['EMULATE_GALAXIES'] is True:
            self.gandalf_logger.log_info_stream(f"Load flow model {self.cfg['FILENAME_NN_FLOW']}")
            gandalf_flow = torch.load(f"{self.cfg['PATH_TRAINED_NN']}/{self.cfg['FILENAME_NN_FLOW']}", weights_only=False, map_location=torch.device('cpu'))

        if self.cfg['CLASSF_GALAXIES'] is True:
            self.gandalf_logger.log_info_stream(f"Load classifier model {self.cfg['FILENAME_NN_CLASSF']}")
            gandalf_classifier = torch.load(f"{self.cfg['PATH_TRAINED_NN']}/{self.cfg['FILENAME_NN_CLASSF']}", weights_only=False, map_location=torch.device('cpu'))

        return gandalf_flow, gandalf_classifier  # , calibration_model

    def run_classifier(self, data_frame):
        """"""
        with torch.no_grad():
            input_data = torch.tensor(data_frame[self.cfg["INPUT_COLS"]].values, dtype=torch.float32)

            for j, col in enumerate(self.cfg["INPUT_COLS"]):
                if col in self.cfg.get("COLUMNS_LOG1P", []):
                    input_data[:, j] = torch.log1p(input_data[:, j])
                scale = self.scalers_classifier[col].scale_[0]
                min_ = self.scalers_classifier[col].min_[0]
                input_data[:, j] = input_data[:, j] * scale + min_

            arr_classf_gandalf_output = self.gandalf_classifier(input_data).squeeze().numpy()

        arr_gandalf_detected = arr_classf_gandalf_output > np.random.rand(self.cfg['NUMBER_SAMPLES'])
        arr_gandalf_detected = arr_gandalf_detected.astype(int)

        validation_accuracy = accuracy_score(data_frame[self.cfg["OUTPUT_COLS_CLASSIFIER"]].values, arr_gandalf_detected)

        df_gandalf = data_frame.copy()
        df_gandalf.rename(columns={"detected": "true_detected"}, inplace=True)
        df_gandalf.loc[:, "detected"] = arr_gandalf_detected.astype(int)
        df_gandalf.loc[:, "probability detected"] = arr_classf_gandalf_output

        self.gandalf_logger.log_info_stream(f"Accuracy sample: {validation_accuracy * 100.0:.2f}%")
        return df_gandalf

    def run_emulator(self, data_frame):
        """"""
        with torch.no_grad():
            input_data = torch.tensor(data_frame[self.cfg["INPUT_COLS"]].values, dtype=torch.float32)
            output_data = torch.tensor(data_frame[self.cfg["INPUT_COLS"]].values, dtype=torch.float32)
            for j, col in enumerate(self.cfg["INPUT_COLS"]):
                if col in self.cfg.get("COLUMNS_LOG1P", []):
                    input_data[:, j] = torch.log1p(input_data[:, j])
                scale = self.scalers_nf[col].scale_[0]
                min_ = self.scalers_nf[col].min_[0]
                input_data[:, j] = input_data[:, j] * scale + min_

            for j, col in enumerate(self.cfg["OUTPUT_COLS_NF"]):
                if col in self.cfg.get("COLUMNS_LOG1P", []):
                    output_data[:, j] = torch.log1p(output_data[:, j])
                scale = self.scalers_nf[col].scale_[0]
                min_ = self.scalers_nf[col].min_[0]
                output_data[:, j] = output_data[:, j] * scale + min_

        arr_flow_gandalf_output = self.gandalf_flow.sample(len(data_frame), cond_inputs=input_data).detach().numpy()

        if self.cfg["PLOT_FLOW_TRANSFORM_OUTPUT"] is True:
            arr_flow = arr_flow_gandalf_output
            arr_true = output_data.detach().numpy()

            for j, col in enumerate(self.cfg["OUTPUT_COLS_NF"]):
                plt.figure(figsize=(6, 3))
                sns.histplot(arr_flow[:, j], bins=100, color="red", label="gandalf", stat="density", kde=False,
                             element="step")
                sns.histplot(arr_true[:, j], bins=100, color="green", label="balrog", stat="density", kde=False,
                             element="step")
                plt.legend()
                plt.yscale("log")
                plt.title(f"Feature: {col}")
                plt.tight_layout()
                plt.show()
            exit()

        # Invers transformieren
        for j, col in enumerate(self.cfg["OUTPUT_COLS_NF"]):
            scale = self.scalers_nf[col].scale_[0]
            min_ = self.scalers_nf[col].min_[0]

            if col in self.cfg.get("COLUMNS_LOG1P", []):
                # log1p und Skalierung rückgängig machen
                arr_flow_gandalf_output[:, j] = np.expm1((arr_flow_gandalf_output[:, j] - min_) / scale)
            else:
                arr_flow_gandalf_output[:, j] = (arr_flow_gandalf_output[:, j] - min_) / scale

        print(f"Number of NaNs before inverse transformation in df_gandalf: {data_frame.isna().sum().sum()}")
        data_frame.loc[:, self.cfg[f'OUTPUT_COLS_NF']] = arr_flow_gandalf_output
        print(f"Length gandalf catalog: {len(data_frame)}")
        print(f"Number of NaNs after inverse transformation in df_gandalf: {data_frame.isna().sum().sum()}")

        return data_frame

    def save_data(self, data_frame, file_name, protocol=2, tmp_samples=False):
        """"""
        print(f"Save file to: {self.cfg['PATH_CATALOGS']}/{file_name}")
        if tmp_samples is True:
            duplicates = data_frame.columns[data_frame.columns.duplicated()]
            if len(duplicates) > 0:
                print("Doppelte Spaltennamen:", duplicates)
            pd.to_pickle(
                data_frame,
                f"{self.cfg['PATH_CATALOGS']}/{file_name}"
            )
        else:
            data_frame.rename(columns={"ID": "true_id"}, inplace=True)
            if "h5" in file_name:
                data_frame.to_hdf(f"{self.cfg['PATH_CATALOGS']}/{file_name}", key='df', mode='w')
                # with h5py.File(f"{self.cfg['PATH_CATALOGS']}/{file_name}", 'w') as hf:
                #     for column in data_frame.columns:
                #         # Convert the column data to a numpy array
                #         data = np.array(data_frame[column])
                #         # Create a dataset in the file and save the numpy array
                #         hf.create_dataset(column, data=data)
            elif "pkl" in file_name:
                if protocol == 2:
                    with open(f"{self.cfg['PATH_CATALOGS']}/{file_name}", "wb") as f:
                        pickle.dump(data_frame, f, protocol=protocol)
                else:
                    data_frame.to_pickle(f"{self.cfg['PATH_CATALOGS']}/{file_name}")

    def apply_cuts(self, data_frame):
        """"""
        data_frame = unsheared_object_cuts(data_frame=data_frame)
        data_frame = flag_cuts(data_frame=data_frame)
        data_frame = unsheared_shear_cuts(data_frame=data_frame)
        data_frame = binary_cut(data_frame=data_frame)
        if self.cfg['MASK_CUT_FUNCTION'] == "HEALPY":
            data_frame = mask_cut_healpy(
                data_frame=data_frame,
                master=f"{self.cfg['PATH_DATA']}/{self.cfg['FILENAME_MASTER_CAT']}"
            )
        elif self.cfg['MASK_CUT_FUNCTION'] == "ASTROPY":
            # Todo there is a bug here, I cutout to many galaxies
            data_frame = mask_cut(
                data_frame=data_frame,
                master=f"{self.cfg['PATH_DATA']}/{self.cfg['FILENAME_MASTER_CAT']}"
            )
        else:
            print("No mask cut function defined!!!")
            exit()
        data_frame = unsheared_mag_cut(data_frame=data_frame)
        return data_frame

    def apply_deep_cuts(self, data_frame):
        """"""
        data_frame = flag_cuts(data_frame=data_frame)
        if self.cfg['MASK_CUT_FUNCTION'] == "HEALPY":
            data_frame = mask_cut_healpy(
                data_frame=data_frame,
                master=f"{self.cfg['PATH_DATA']}/{self.cfg['FILENAME_MASTER_CAT']}"
            )
        elif self.cfg['MASK_CUT_FUNCTION'] == "ASTROPY":
            # Todo there is a bug here, I cutout to many galaxies
            data_frame = mask_cut(
                data_frame=data_frame,
                master=f"{self.cfg['PATH_DATA']}/{self.cfg['FILENAME_MASTER_CAT']}"
            )
        else:
            print("No mask cut function defined!!!")
            exit()
        return data_frame

    def plot_classf_data(self, df_balrog, df_gandalf):
        """"""

        print("Start plotting classf data")
        if self.cfg['PLOT_MATRIX_RUN'] is True:
            plot_confusion_matrix_gandalf(
                df_balrog=df_balrog,
                df_gandalf=df_gandalf,
                show_plot=self.cfg['SHOW_PLOT_RUN'],
                save_plot=self.cfg['SAVE_PLOT_RUN'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'CONFUSION_MATRIX']}/confusion_matrix.png",
                title=f"Confusion Matrix"
            )

        if self.cfg['PLOT_CALIBRATION_CURVE'] is True:
            plot_calibration_curve_gandalf(
                true_detected=df_balrog["detected"],
                probability=df_gandalf["probability detected"],
                n_bins=10,
                show_plot=self.cfg['SHOW_PLOT_RUN'],
                save_plot=self.cfg['SAVE_PLOT_RUN'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'CALIBRATION_CURVE']}/calibration_curve.png",
                title=f"Calibration Curve"
            )

        # ROC und AUC
        if self.cfg['PLOT_ROC_CURVE_RUN'] is True:
            plot_roc_curve_gandalf(
                df_balrog=df_balrog,
                df_gandalf=df_gandalf,
                show_plot=self.cfg['SHOW_PLOT_RUN'],
                save_plot=self.cfg['SAVE_PLOT_RUN'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'ROC_CURVE']}/roc_curve.png",
                title=f"Receiver Operating Characteristic (ROC) Curve"
            )

        if self.cfg['PLOT_CLASSF_BOX'] is True:
            plot_box(
                df_balrog=df_balrog,
                df_gandalf=df_gandalf,
                columns=[
                    "BDF_LUPT_DERED_CALIB_R",
                    "BDF_LUPT_DERED_CALIB_I",
                    "BDF_LUPT_DERED_CALIB_Z",
                    "BDF_LUPT_ERR_DERED_CALIB_R",
                    "BDF_LUPT_ERR_DERED_CALIB_I",
                    "BDF_LUPT_ERR_DERED_CALIB_Z",
                    "BDF_T",
                    "BDF_G",
                    "FWHM_WMEAN_R",
                    "FWHM_WMEAN_I",
                    "FWHM_WMEAN_Z",
                    "AIRMASS_WMEAN_R",
                    "AIRMASS_WMEAN_I",
                    "AIRMASS_WMEAN_Z",
                    "MAGLIM_R",
                    "MAGLIM_I",
                    "MAGLIM_Z",
                    "EBV_SFD98"
                ],
                labels=[
                    "BDF Lupt R",
                    "BDF Lupt I",
                    "BDF Lupt Z",
                    "BDF Lupt Err R",
                    "BDF Lupt Err I",
                    "BDF Lupt Err Z",
                    "BDF T",
                    "BDF G",
                    "FWHM R",
                    "FWHM I",
                    "FWHM Z",
                    "AIRMASS R",
                    "AIRMASS I",
                    "AIRMASS Z",
                    "MAGLIM R",
                    "MAGLIM I",
                    "MAGLIM Z",
                    "EBV SFD98"
                ],
                show_plot=self.cfg['SHOW_PLOT_RUN'],
                save_plot=self.cfg['SAVE_PLOT_RUN'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'CLASSF_BOX']}/boxplot.png",
                title=f"Box Plot Gandalf vs. Balrog"
            )
        if self.cfg['PLOT_NUMBER_DENSITY_RUN'] is True:
            plot_number_density_fluctuation(
                df_balrog=df_balrog,
                df_gandalf=df_gandalf,
                columns=[
                    "BDF_LUPT_DERED_CALIB_R",
                    "BDF_LUPT_DERED_CALIB_I",
                    "BDF_LUPT_DERED_CALIB_Z",
                    "BDF_T",
                    "BDF_G",
                    "FWHM_WMEAN_R",
                    "FWHM_WMEAN_I",
                    "FWHM_WMEAN_Z",
                    "AIRMASS_WMEAN_R",
                    "AIRMASS_WMEAN_I",
                    "AIRMASS_WMEAN_Z",
                    "MAGLIM_R",
                    "MAGLIM_I",
                    "MAGLIM_Z",
                    "EBV_SFD98"
                ],
                labels=[
                    "BDF Lupt R",
                    "BDF Lupt I",
                    "BDF Lupt Z",
                    "BDF T",
                    "BDF G",
                    "FWHM R",
                    "FWHM I",
                    "FWHM Z",
                    "AIRMASS R",
                    "AIRMASS I",
                    "AIRMASS Z",
                    "MAGLIM R",
                    "MAGLIM I",
                    "MAGLIM Z",
                    "EBV SFD98"
                ],
                ranges=[
                    [18, 26],
                    [18, 26],
                    [18, 26],
                    [-1, 1.5],
                    [-0.1, 0.8],
                    [0.8, 1.2],
                    [0.7, 1.1],
                    [0.7, 1.0],
                    [1, 1.4],
                    [1, 1.4],
                    [1, 1.4],
                    [23.5, 24.5],
                    [23, 23.75],
                    [22, 23],
                    [0, 0.05]
                ],
                show_plot=self.cfg['SHOW_PLOT_RUN'],
                save_plot=self.cfg['SAVE_PLOT_RUN'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'CLASSF_NUMBER_DENSITY']}/number_density_fluctuation.png",
                title=f"Comparison Between Gandalf and Balrog"
            )
        # Multivariate classifier
        if self.cfg['PLOT_MULTI_CLASSF_RUN'] is True:
            plot_multivariate_classifier(
                df_balrog=df_balrog,
                df_gandalf=df_gandalf,
                columns=[
                    "BDF_MAG_DERED_CALIB_R",
                    "BDF_MAG_DERED_CALIB_I",
                    "BDF_MAG_DERED_CALIB_Z",
                    "BDF_T",
                    "BDF_G",
                    "FWHM_WMEAN_R",
                    "FWHM_WMEAN_I",
                    "FWHM_WMEAN_Z",
                    "AIRMASS_WMEAN_R",
                    "AIRMASS_WMEAN_I",
                    "AIRMASS_WMEAN_Z",
                    "MAGLIM_R",
                    "MAGLIM_I",
                    "MAGLIM_Z",
                    "EBV_SFD98"
                ],
                # labels=[
                #     "BDF Mag R",
                #     "BDF Mag I",
                #     "BDF Mag Z",
                #     "BDF T",
                #     "BDF G",
                #     "FWHM R",
                #     "FWHM I",
                #     "FWHM Z",
                #     "AIRMASS R",
                #     "AIRMASS I",
                #     "AIRMASS Z",
                #     "MAGLIM R",
                #     "MAGLIM I",
                #     "MAGLIM Z",
                #     "EBV SFD98"
                # ],
                # ranges=[
                #     [18, 26],
                #     [18, 26],
                #     [18, 26],
                #     [-1, 1.5],
                #     [-0.1, 0.8],
                #     [0.8, 1.2],
                #     [0.7, 1.1],
                #     [0.7, 1.0],
                #     [1, 1.4],
                #     [1, 1.4],
                #     [1, 1.4],
                #     [23.5, 24.5],
                #     [23, 23.75],
                #     [22, 23],
                #     [0, 0.05]
                # ],
                show_plot=self.cfg['SHOW_PLOT_RUN'],
                save_plot=self.cfg['SAVE_PLOT_RUN'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'CLASSF_MULTIVARIATE_GAUSSIAN']}/classifier_multiv.png",
                title=f"Comparison Between Gandalf and Balrog"
            )

        if self.cfg['PLOT_MULTI_CLASSF_CUT_RUN'] is True:
            df_balrog_deep_cut = self.apply_deep_cuts(df_balrog)
            df_gandalf_deep_cut = self.apply_deep_cuts(df_gandalf)

            plot_multivariate_classifier(
                df_balrog=df_balrog_deep_cut,
                df_gandalf=df_gandalf_deep_cut,
                columns=[
                    "BDF_MAG_DERED_CALIB_R",
                    "BDF_MAG_DERED_CALIB_Z",
                    "BDF_T",
                    "BDF_G",
                    "FWHM_WMEAN_R",
                    "FWHM_WMEAN_I",
                    "FWHM_WMEAN_Z",
                    "AIRMASS_WMEAN_R",
                    "AIRMASS_WMEAN_I",
                    "AIRMASS_WMEAN_Z",
                    "MAGLIM_R",
                    "MAGLIM_I",
                    "MAGLIM_Z",
                    "EBV_SFD98"
                ],
                # labels=[
                #     "BDF Mag R",
                #     "BDF Mag Z",
                #     "BDF T",
                #     "BDF G",
                #     "FWHM R",
                #     "FWHM I",
                #     "FWHM Z",
                #     "AIRMASS R",
                #     "AIRMASS I",
                #     "AIRMASS Z",
                #     "MAGLIM R",
                #     "MAGLIM I",
                #     "MAGLIM Z",
                #     "EBV SFD98"
                # ],
                # ranges=[
                #     [18, 26],
                #     [18, 26],
                #     [-1, 1.5],
                #     [-0.1, 0.8],
                #     [0.8, 1.2],
                #     [0.7, 1.1],
                #     [0.7, 1.0],
                #     [1, 1.4],
                #     [1, 1.4],
                #     [1, 1.4],
                #     [23.5, 24.5],
                #     [23, 23.75],
                #     [22, 23],
                #     [0, 0.05]
                # ],
                show_plot=self.cfg['SHOW_PLOT_RUN'],
                save_plot=self.cfg['SAVE_PLOT_RUN'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'CLASSF_MULTIVARIATE_GAUSSIAN']}/classifier_multiv_cut.png",
                title=f"Comparison Between Gandalf and Balrog after applying cuts"
            )

            del df_balrog_deep_cut
            del df_gandalf_deep_cut

        if self.cfg['PLOT_CLASSF_HISTOGRAM'] is True:
            plot_classifier_histogram(
                df_balrog=df_balrog,
                df_gandalf=df_gandalf,
                columns=self.cfg["INPUT_COLS_MAG_RUN"],
                show_plot=self.cfg['SHOW_PLOT_RUN'],
                save_plot=self.cfg['SAVE_PLOT_RUN'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'CLASSF_HIST']}/All.png",
                xlim=[
                    (15, 30),
                    (15, 30),
                    (15, 30),
                    (24, 30),
                    (24, 30),
                    (24, 30),
                    (-5, 5),
                    (-5, 5),
                    (-5, 5),
                    (-5, 5),
                    (-5, 5),
                    (-5, 5),
                    (-5, 5),
                    (-5, 5),
                    (-0.1, 1.25),
                    (0.5, 1.4),
                    (0.5, 1.25),
                    (0.5, 1.25),
                    (1, 1.5),
                    (1, 1.5),
                    (1, 1.5),
                    (22.5, 24.5),
                    (22.5, 24.5),
                    (20.5, 23.5),
                    (0, 0.1),
                ],
                title=f"Classifier detection histogram"
            )

            plot_classifier_histogram(
                df_balrog=df_balrog,
                df_gandalf=df_gandalf,
                columns=["BDF_MAG_DERED_CALIB_R", "BDF_MAG_DERED_CALIB_I", "BDF_MAG_DERED_CALIB_Z"],
                show_plot=self.cfg['SHOW_PLOT_RUN'],
                save_plot=self.cfg['SAVE_PLOT_RUN'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'CLASSF_HIST']}/magnitude.png",
                title=f"magnitude"
            )
            plot_classifier_histogram(
                df_balrog=df_balrog,
                df_gandalf=df_gandalf,
                columns=["BDF_MAG_ERR_DERED_CALIB_R", "BDF_MAG_ERR_DERED_CALIB_I", "BDF_MAG_ERR_DERED_CALIB_Z"],
                show_plot=self.cfg['SHOW_PLOT_RUN'],
                save_plot=self.cfg['SAVE_PLOT_RUN'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'CLASSF_HIST']}/magnitude_error.png",
                title=f"magnitude_error"
            )
            plot_classifier_histogram(
                df_balrog=df_balrog,
                df_gandalf=df_gandalf,
                columns=["Color BDF MAG U-G", "Color BDF MAG G-R", "Color BDF MAG R-I", "Color BDF MAG I-Z", "Color BDF MAG Z-J", "Color BDF MAG J-H", "Color BDF MAG H-K"],
                show_plot=self.cfg['SHOW_PLOT_RUN'],
                save_plot=self.cfg['SAVE_PLOT_RUN'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'CLASSF_HIST']}/Color.png",
                title=f"color"
            )
            plot_classifier_histogram(
                df_balrog=df_balrog,
                df_gandalf=df_gandalf,
                columns=["BDF_T", "BDF_G", "EBV_SFD98"],
                show_plot=self.cfg['SHOW_PLOT_RUN'],
                save_plot=self.cfg['SAVE_PLOT_RUN'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'CLASSF_HIST']}/shape_and_dust.png",
                title=f"shape_and_dust"
            )
            plot_classifier_histogram(
                df_balrog=df_balrog,
                df_gandalf=df_gandalf,
                columns=["FWHM_WMEAN_R", "FWHM_WMEAN_I", "FWHM_WMEAN_Z"],
                show_plot=self.cfg['SHOW_PLOT_RUN'],
                save_plot=self.cfg['SAVE_PLOT_RUN'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'CLASSF_HIST']}/FWHM.png",
                title=f"FWHM"
            )
            plot_classifier_histogram(
                df_balrog=df_balrog,
                df_gandalf=df_gandalf,
                columns=["AIRMASS_WMEAN_R", "AIRMASS_WMEAN_I", "AIRMASS_WMEAN_Z"],
                show_plot=self.cfg['SHOW_PLOT_RUN'],
                save_plot=self.cfg['SAVE_PLOT_RUN'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'CLASSF_HIST']}/AIRMASS.png",
                title=f"AIRMASS"
            )
            plot_classifier_histogram(
                df_balrog=df_balrog,
                df_gandalf=df_gandalf,
                columns=["MAGLIM_R", "MAGLIM_I", "MAGLIM_Z"],
                show_plot=self.cfg['SHOW_PLOT_RUN'],
                save_plot=self.cfg['SAVE_PLOT_RUN'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'CLASSF_HIST']}/MAGLIM.png",
                title=f"MAGLIM"
            )
        print("End plotting classf data")

    def plot_data_flow(self, df_gandalf, df_balrog, mcal=''):
        """"""
        if self.cfg['PLOT_BALROG_HIST_RUN'] is True:
            plot_balrog_histogram_with_error(
                df_gandalf=df_gandalf,
                df_balrog=df_balrog,
                columns=[
                    "unsheared/mag_r",
                    "unsheared/mag_i",
                    "unsheared/mag_z",
                    "Color unsheared MAG r-i",
                    "Color unsheared MAG i-z",
                    "unsheared/snr",
                    "unsheared/size_ratio",
                    "unsheared/weight",
                    "unsheared/T",
                ],
                labels=[
                    "mag r",
                    "mag i",
                    "mag z",
                    "mag r-i",
                    "mag i-z",
                    "snr",
                    "size ratio",
                    "weight",
                    "T",
                ],
                ranges=[
                    [18, 24.5],  # mag r
                    [18, 24.5],  # mag i
                    [18, 24.5],  # mag z
                    [-0.5, 1.5],  # mag r-i
                    [-0.5, 1.5],  # mag i-z
                    [2, 100],  # snr
                    [-0.5, 5],  # size ratio
                    [10, 80],  # weight
                    [0, 3.5]  # T
                ],
                binwidths=[
                    None,  # mag r
                    None,  # mag i
                    None,  # mag z
                    0.08,  # mag r-i
                    0.08,  # mag i-z
                    2,  # snr
                    0.2,  # size ratio
                    2,  # weight
                    0.2  # T
                ],
                title="Hist compare",
                show_plot=self.cfg['SHOW_PLOT_RUN'],
                save_plot=self.cfg['SAVE_PLOT_RUN'],
                save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER'][f'{mcal.upper()}BALROG_HIST_PLOT']}/{self.cfg[f'RUN_NUMBER']}_{mcal}balrog_hist_plot.png"
            )

        if self.cfg['PLOT_COLOR_COLOR_RUN'] is True:
            try:
                plot_compare_corner(
                    data_frame_generated=df_gandalf,
                    data_frame_true=df_balrog,
                    dict_delta=None,
                    epoch=None,
                    title=f"{mcal} color-color plot",
                    columns=["Color unsheared MAG r-i", "Color unsheared MAG i-z"],
                    labels=["r-i", "i-z"],
                    show_plot=self.cfg['SHOW_PLOT_RUN'],
                    save_plot=self.cfg['SAVE_PLOT_RUN'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER'][f'{mcal.upper()}COLOR_COLOR_PLOT']}/{self.cfg[f'RUN_NUMBER']}_{mcal}color_color_{self.cfg['RUN_NUMBER']}.png",
                    ranges=[(-8, 8), (-8, 8)]
                )
            except Exception as e:
                print(e)

        if self.cfg['PLOT_RESIDUAL_RUN'] is True:
            try:
                hist_figure, ((stat_ax1), (stat_ax2), (stat_ax3)) = plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
                hist_figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
                hist_figure.suptitle(r"residual", fontsize=16)

                lst_axis_res = [
                    stat_ax1,
                    stat_ax2,
                    stat_ax3
                ]

                lst_xlim_res = [
                    (-2.5, 2.5),
                    (-2.5, 2.5),
                    (-2.5, 2.5)
                ]

                df_hist_Balrog = pd.DataFrame({
                    "dataset": ["Balrog" for _ in range(len(df_balrog[f"unsheared/mag_r"]))]
                })
                df_hist_generated = pd.DataFrame({
                    "dataset": ["gaNdalF" for _ in range(len(df_gandalf[f"unsheared/mag_r"]))]
                })
                for band in self.cfg['BANDS_RUN']:
                    df_hist_Balrog[f"BDF_MAG_DERED_CALIB - unsheared/mag {band}"] = \
                        df_balrog[f"BDF_MAG_DERED_CALIB_{band.upper()}"] - df_balrog[f"unsheared/mag_{band}"]
                    df_hist_generated[f"BDF_MAG_DERED_CALIB - unsheared/mag {band}"] = \
                        df_balrog[f"BDF_MAG_DERED_CALIB_{band.upper()}"] - df_gandalf[f"unsheared/mag_{band}"]

                for idx, band in enumerate(self.cfg['BANDS_RUN']):
                    sns.histplot(
                        data=df_hist_Balrog,
                        x=f"BDF_MAG_DERED_CALIB - unsheared/mag {band}",
                        ax=lst_axis_res[idx],
                        element="step",
                        stat="density",
                        color="dodgerblue",
                        bins=50,
                        label="Balrog"
                    )
                    sns.histplot(
                        data=df_hist_generated,
                        x=f"BDF_MAG_DERED_CALIB - unsheared/mag {band}",
                        ax=lst_axis_res[idx],
                        element="step",
                        stat="density",
                        color="darkorange",
                        fill=False,
                        bins=50,
                        label="gaNdalF"
                    )
                    lst_axis_res[idx].axvline(
                        x=df_hist_Balrog[f"BDF_MAG_DERED_CALIB - unsheared/mag {band}"].median(),
                        color='dodgerblue',
                        ls='--',
                        lw=1.5,
                        label="Mean Balrog"
                    )
                    lst_axis_res[idx].axvline(
                        x=df_hist_generated[f"BDF_MAG_DERED_CALIB - unsheared/mag {band}"].median(),
                        color='darkorange',
                        ls='--',
                        lw=1.5,
                        label="Mean gaNdalF"
                    )
                    lst_axis_res[idx].set_xlim(lst_xlim_res[idx][0], lst_xlim_res[idx][1])
                    if idx == 0:
                        lst_axis_res[idx].legend()
                    else:
                        lst_axis_res[idx].legend([], [], frameon=False)
                hist_figure.tight_layout()
                if self.cfg['SAVE_PLOT_RUN'] is True:
                    plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER'][f'{mcal.upper()}RESIDUAL_PLOT']}/{mcal}residual_plot_{self.cfg['RUN_NUMBER']}.png")
                if self.cfg['SHOW_PLOT_RUN'] is True:
                    plt.show()
                plt.clf()
                plt.close()
            except Exception as e:
                print(e)
        if self.cfg['PLOT_CHAIN_RUN'] is True:
            try:

                # for b in ["r", "i", "z"]:
                #     df_gandalf[f"unsheared/mag_{b}"] = flux2mag(df_gandalf[f"unsheared/flux_{b}"])

                plot_compare_corner(
                    data_frame_generated=df_gandalf,
                    data_frame_true=df_balrog,
                    dict_delta=None,
                    epoch=None,
                    title=f"Observed Properties gaNdalF compared to Balrog",
                    show_plot=False,
                    save_plot=True,
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER'][f'{mcal.upper()}CHAIN_PLOT']}/{mcal}chainplot_slide_{self.cfg['RUN_NUMBER']}.png",
                    columns=[
                        f"unsheared/{self.lum_type.lower()}_r",
                        f"unsheared/{self.lum_type.lower()}_i",
                        f"unsheared/{self.lum_type.lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio"
                    ],
                    labels=[
                        f"{self.lum_type.lower()}_r",
                        f"{self.lum_type.lower()}_i",
                        f"{self.lum_type.lower()}_z",
                        "snr",
                        "size_ratio"
                    ],
                    ranges=[(17, 25), (17, 25), (17, 25), (-2, 300), (0, 6)]
                )
            except Exception as e:
                print(e)
            try:
                plot_compare_corner(
                    data_frame_generated=df_gandalf,
                    data_frame_true=df_balrog,
                    dict_delta=None,
                    epoch=None,
                    title=f"{mcal} chain plot",
                    show_plot=self.cfg['SHOW_PLOT_RUN'],
                    save_plot=self.cfg['SAVE_PLOT_RUN'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER'][f'{mcal.upper()}CHAIN_PLOT']}/{mcal}chainplot_{self.cfg['RUN_NUMBER']}.png",
                    columns=[
                        f"unsheared/{self.lum_type.lower()}_r",
                        f"unsheared/{self.lum_type.lower()}_i",
                        f"unsheared/{self.lum_type.lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T",
                        # "unsheared/weight",
                    ],
                    labels=[
                        f"{self.lum_type.lower()}_r",
                        f"{self.lum_type.lower()}_i",
                        f"{self.lum_type.lower()}_z",
                        "snr",
                        "size_ratio",
                        "T",
                        # "weight"
                    ],
                    ranges=[(17, 25), (17, 25), (17, 25), (-2, 300), (0, 6), (0, 3)]  # , (10, 80)
                )
            except Exception as e:
                print(e)

        if self.cfg['PLOT_CONDITIONS'] is True:
            for condition in self.cfg['CONDITIONS']:
                try:
                    cond_figure, (
                        (stat_ax1), (stat_ax2), (stat_ax3)) = \
                        plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
                    cond_figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
                    cond_figure.suptitle(f"BDF_MAG_DERED_CALIB - unsheared/mag", fontsize=16)

                    outputs = ['unsheared/mag_' + b for b in self.cfg['BANDS_RUN']]
                    true_outputs = ['BDF_MAG_DERED_CALIB_' + b.upper() for b in self.cfg['BANDS_RUN']]
                    output_errs = ['unsheared/mag_err_' + b for b in self.cfg['BANDS_RUN']]

                    lst_axis_con = [
                        stat_ax1,
                        stat_ax2,
                        stat_ax3
                    ]

                    if df_balrog[condition].size == 0:
                        print("df_balrog[condition] empty")
                    else:
                        cond_lims = np.percentile(df_balrog[condition], [2, 98])
                        standard_levels = 10 #  [0.393, 0.865, 0.989]

                    for idx, out in enumerate(zip(outputs, output_errs, true_outputs)):
                        output_ = out[0]
                        output_err_ = out[1]
                        true_output_ = out[2]

                        diff_true = (df_balrog[true_output_] - df_balrog[output_]) / df_balrog[output_err_]
                        df_conditional_true = pd.DataFrame({
                            condition: df_balrog[condition],
                            f"residual band {self.cfg['BANDS_RUN'][idx]}": diff_true,
                            "dataset": ["Balrog" for _ in range(len(df_balrog[condition]))]
                        })
                        bin_means_true, bin_edges_mean_true, binnumber_true = binned_statistic(
                            df_balrog[condition], diff_true, statistic='median', bins=10, range=cond_lims)
                        bin_stds_true, bin_edges_true, binnumber_true = binned_statistic(
                            df_balrog[condition], diff_true, statistic=median_abs_deviation, bins=10, range=cond_lims)
                        xerr_true = (bin_edges_mean_true[1:] - bin_edges_mean_true[:-1]) / 2
                        xmean_true = (bin_edges_mean_true[1:] + bin_edges_mean_true[:-1]) / 2
                        lst_axis_con[idx].errorbar(
                            xmean_true, bin_means_true, xerr=xerr_true, yerr=bin_stds_true, color='dodgerblue', lw=2,
                            label='Balrog')

                        diff_generated = (df_gandalf[true_output_] - df_gandalf[output_]) / df_gandalf[
                            output_err_]
                        df_conditional_generated = pd.DataFrame({
                            condition: df_gandalf[condition],
                            f"residual band {self.cfg['BANDS_RUN'][idx]}": diff_generated,
                            "dataset": ["gaNdalF" for _ in range(len(df_gandalf[condition]))]
                        })
                        bin_means_generated, bin_edges_mean_generated, binnumber_mean_generated = binned_statistic(
                            df_gandalf[condition], diff_generated, statistic='median', bins=10, range=cond_lims)
                        bin_stds_generated, bin_edges_generated, binnumber_generated = binned_statistic(
                            df_gandalf[condition], diff_generated, statistic=median_abs_deviation, bins=10,
                            range=cond_lims)
                        xerr_generated = (bin_edges_mean_generated[1:] - bin_edges_mean_generated[:-1]) / 2
                        xmean_generated = (bin_edges_mean_generated[1:] + bin_edges_mean_generated[:-1]) / 2
                        lst_axis_con[idx].errorbar(
                            xmean_generated, bin_means_generated, xerr=xerr_generated, yerr=bin_stds_generated,
                            color='darkorange', lw=2, label='gaNdalF')
                        m, s = np.median(diff_generated), median_abs_deviation(diff_generated)
                        range_ = [m - 4 * s, m + 4 * s]

                        sns.kdeplot(
                            data=df_conditional_true,
                            x=condition,
                            y=f"residual band {self.cfg['BANDS_RUN'][idx]}",
                            fill=True,
                            thresh=0,
                            alpha=.4,
                            levels=standard_levels,  # 10
                            color="limegreen",
                            legend="Balrog",
                            ax=lst_axis_con[idx]
                        )
                        sns.kdeplot(
                            data=df_conditional_generated,
                            x=condition,
                            y=f"residual band {self.cfg['BANDS_RUN'][idx]}",
                            fill=False,
                            thresh=0,
                            levels=standard_levels,  # 10
                            alpha=.8,
                            color="orangered",
                            legend="gaNdalF",
                            ax=lst_axis_con[idx]
                        )
                        lst_axis_con[idx].set_xlim(cond_lims)
                        lst_axis_con[idx].set_ylim(range_)
                        lst_axis_con[idx].axhline(np.median(diff_true), c='dodgerblue', ls='--', label='median Balrog')
                        lst_axis_con[idx].axhline(0, c='grey', ls='--', label='zero')
                        lst_axis_con[idx].axhline(np.median(diff_generated), c='darkorange', ls='--',
                                                  label='median gaNdalF')
                        lst_axis_con[idx].axvline(np.median(df_balrog[condition]), c='grey', ls='--',
                                                  label='median conditional')
                    lst_axis_con[0].legend()
                    cond_figure.tight_layout()
                    if self.cfg['SAVE_PLOT_RUN'] is True:
                        plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER'][f'{mcal.upper()}CONDITIONS_PLOT']}/{mcal}{condition}_plot_{self.cfg['RUN_NUMBER']}.png")
                    if self.cfg['SHOW_PLOT_RUN'] is True:
                        plt.show()
                    plt.clf()
                    plt.close()

                except ValueError:
                    print(f"Value Error for {condition}")

        if self.cfg['PLOT_HIST'] is True:
            try:
                hist_figure_2, ((hist_ax1), (hist_ax2), (hist_ax3)) = \
                    plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
                hist_figure_2.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
                hist_figure_2.suptitle(r"magnitude histogram", fontsize=16)

                lst_axis_his = [
                    hist_ax1,
                    hist_ax2,
                    hist_ax3
                ]

                for ax_idx, his_ax in enumerate(lst_axis_his):
                    sns.histplot(
                        data=df_balrog,
                        x=f"unsheared/mag_{self.cfg['BANDS_RUN'][ax_idx]}",
                        ax=his_ax,
                        element="step",
                        stat="count",
                        color="dodgerblue",
                        fill=True,
                        binwidth=0.2,
                        log_scale=(False, True),
                        label="balrog"
                    )
                    sns.histplot(
                        data=df_gandalf,
                        x=f"unsheared/mag_{self.cfg['BANDS_RUN'][ax_idx]}",
                        ax=his_ax,
                        element="step",
                        stat="count",
                        color="darkorange",
                        fill=False,
                        log_scale=(False, True),
                        binwidth=0.2,
                        label="gaNdalF"
                    )
                    his_ax.axvline(
                        x=df_balrog[f"unsheared/mag_{self.cfg['BANDS_RUN'][ax_idx]}"].median(),
                        color='dodgerblue',
                        ls='--',
                        lw=1.5,
                        label="Mean Balrog"
                    )
                    his_ax.axvline(
                        x=df_gandalf[f"unsheared/mag_{self.cfg['BANDS_RUN'][ax_idx]}"].median(),
                        color='darkorange',
                        ls='--',
                        lw=1.5,
                        label="Mean gaNdalF"
                    )
                plt.legend()
                if self.cfg['SAVE_PLOT_RUN'] == True:
                    plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER'][f'{mcal.upper()}HIST_PLOT']}/{mcal}magnitude_histogram_{self.cfg['RUN_NUMBER']}.png")
                if self.cfg['SHOW_PLOT_RUN'] == True:
                    plt.show()
                plt.clf()
                plt.close()
            except Exception as e:
                print(e)

            print("End plotting data flow")

        return df_balrog, df_gandalf
