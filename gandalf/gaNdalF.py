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
plt.style.use('seaborn-white')


class gaNdalF(object):
    """"""
    def __init__(self, cfg):
        """"""
        self.cfg = cfg
        self.lum_type = self.cfg['LUM_TYPE_RUN']
        if "yjt_True" in self.cfg['FILENAME_NN_FLOW']:
            self.cfg['APPLY_YJ_TRANSFORM_FLOW_RUN'] = True
        if "scr_True" in self.cfg['FILENAME_NN_FLOW']:
            self.cfg['APPLY_SCALER_FLOW_RUN'] = True
        if "yjt_True" in self.cfg['FILENAME_NN_CLASSF']:
            self.cfg['APPLY_YJ_TRANSFORM_CLASSF_RUN'] = True
        if "scr_True" in self.cfg['FILENAME_NN_CLASSF']:
            self.cfg['APPLY_SCALER_CLASSF_RUN'] = True

        self.galaxies = self.init_dataset()
        self.gandalf_flow, self.gandalf_classifier, self.calibration_model = self.init_trained_models()

    def init_dataset(self):
        galaxies = DESGalaxies(
            cfg=self.cfg,
            kind="run_gandalf"
        )

        if self.cfg['NUMBER_SAMPLES'] == -1:
            self.cfg['NUMBER_SAMPLES'] = len(galaxies.run_dataset)

        return galaxies

    def init_trained_models(self):
        """"""
        gandalf_flow = None
        gandalf_classifier = None
        calibration_model = None

        if self.cfg['EMULATE_GALAXIES'] is True:
            gandalf_flow = torch.load(f"{self.cfg['PATH_TRAINED_NN']}/{self.cfg['FILENAME_NN_FLOW']}")

        if self.cfg['CLASSF_GALAXIES'] is True:
            gandalf_classifier = torch.load(f"{self.cfg['PATH_TRAINED_NN']}/{self.cfg['FILENAME_NN_CLASSF']}")
            calibration_model = joblib.load(f"{self.cfg['PATH_TRAINED_NN']}/{self.cfg['FILENAME_CLASSF_CALIBRATION']}")

        return gandalf_flow, gandalf_classifier, calibration_model

    def predict_calibrated(self, y_pred):
        calibrated_outputs = self.calibration_model.predict_proba(y_pred.reshape(-1, 1))[:, 1]
        return calibrated_outputs

    def run_classifier(self, data_frame):
        """"""
        with torch.no_grad():
            arr_classf_gandalf_output = self.gandalf_classifier(torch.tensor(data_frame[self.cfg["INPUT_COLS_MAG_RUN"]].values)).squeeze().numpy()
        arr_gandalf_prob_calib = self.predict_calibrated(arr_classf_gandalf_output)
        arr_gandalf_detected_calib = arr_gandalf_prob_calib > np.random.rand(self.cfg['NUMBER_SAMPLES'])
        validation_accuracy = accuracy_score(data_frame[self.cfg["OUTPUT_COLS_CLASSF_RUN"]].values, arr_gandalf_detected_calib)

        gandalf_detected = np.count_nonzero(arr_gandalf_detected_calib)
        gandalf_not_detected = arr_gandalf_detected_calib.size - gandalf_detected

        balrog_detected = np.count_nonzero(data_frame[self.cfg["OUTPUT_COLS_CLASSF_RUN"]].values)
        balrog_not_detected = data_frame[self.cfg["OUTPUT_COLS_CLASSF_RUN"]].values.size - balrog_detected

        df_balrog = data_frame[self.cfg[f'INPUT_COLS_{self.lum_type}_RUN'] + self.cfg[f'OUTPUT_COLS_{self.lum_type}_RUN']]

        if self.cfg['APPLY_SCALER_CLASSF_RUN'] is True:
            df_balrog = self.galaxies.inverse_scale_data(df_balrog)

        if self.cfg['APPLY_YJ_TRANSFORM_CLASSF_RUN'] is True:
            if self.cfg['TRANSFORM_COLS_RUN'] is None:
                trans_col =df_balrog.columns
            else:
                trans_col = self.cfg['TRANSFORM_COLS_RUN']

            df_balrog = self.galaxies.yj_inverse_transform_data(
                data_frame=df_balrog,
                columns=trans_col
            )

        df_balrog.loc[:, self.cfg["OUTPUT_COLS_CLASSF_RUN"]] = data_frame[self.cfg["OUTPUT_COLS_CLASSF_RUN"]].values
        df_balrog.loc[:, self.cfg["CUT_COLS_RUN"]] = data_frame[self.cfg["CUT_COLS_RUN"]].values

        df_gandalf = df_balrog.copy()

        df_gandalf.loc[:, "detected"] = arr_gandalf_detected_calib.astype(int)
        df_gandalf.loc[:, "probability detected"] = arr_gandalf_prob_calib

        print(f"Accuracy sample: {validation_accuracy * 100.0:.2f}%")
        print(f"Number of NOT true_detected galaxies gandalf: {gandalf_not_detected} of {self.cfg['NUMBER_SAMPLES']}")
        print(f"Number of true_detected galaxies gandalf: {gandalf_detected} of {self.cfg['NUMBER_SAMPLES']}")
        print(f"Number of NOT true_detected galaxies balrog: {balrog_not_detected} of {self.cfg['NUMBER_SAMPLES']}")
        print(f"Number of true_detected galaxies balrog: {balrog_detected} of {self.cfg['NUMBER_SAMPLES']}")

        return df_balrog, df_gandalf

    def run_emulator(self, df_balrog, df_gandalf):
        """"""
        if self.cfg['CLASSF_GALAXIES'] is True:
            if self.cfg['APPLY_YJ_TRANSFORM_FLOW_RUN'] is True:
                self.galaxies.applied_yj_transform = "_YJ"
            else:
                self.galaxies.applied_yj_transform = ""
            filename = self.cfg[f'FILENAME_SCALER_ODET_{self.lum_type}{self.galaxies.applied_yj_transform}']
            self.galaxies.scaler = joblib.load(
                f"{self.cfg['PATH_TRANSFORMERS']}/{filename}"
            )
            self.galaxies.name_scaler = filename
            self.galaxies.dict_pt = joblib.load(
                f"{self.cfg['PATH_TRANSFORMERS']}/{self.cfg['FILENAME_YJ_TRANSFORMER_ODET']}"
            )
            self.galaxies.name_yj_transformer = self.cfg['FILENAME_YJ_TRANSFORMER_ODET']

            df_gandalf_flow = df_gandalf[self.cfg[f'INPUT_COLS_{self.lum_type}_RUN'] + self.cfg[f'OUTPUT_COLS_{self.lum_type}_RUN']]
            if self.cfg['APPLY_YJ_TRANSFORM_FLOW_RUN'] is True:
                df_gandalf_flow = self.galaxies.yj_transform_data_on_fly(
                    data_frame=df_gandalf_flow,
                    columns=df_gandalf_flow.columns,
                    dict_pt=self.galaxies.dict_pt
                )

            if self.cfg['APPLY_SCALER_FLOW_RUN'] is True:
                df_gandalf_flow = self.galaxies.scale_data_on_fly(
                    data_frame=df_gandalf_flow,
                    scaler=self.galaxies.scaler
                )

        else:
            df_gandalf_flow = df_gandalf[self.cfg[f'INPUT_COLS_{self.lum_type}_RUN'] + self.cfg[f'OUTPUT_COLS_{self.lum_type}_RUN']].copy()
            df_balrog_flow = df_balrog[self.cfg[f'INPUT_COLS_{self.lum_type}_RUN'] + self.cfg[f'OUTPUT_COLS_{self.lum_type}_RUN']].copy()
            print(f"Number of NaNs in df_balrog: {df_balrog_flow.isna().sum().sum()}")
            if self.cfg['APPLY_SCALER_FLOW_RUN'] is True:
                print("apply inverse scaler on balrog")
                df_balrog_flow = self.galaxies.inverse_scale_data(df_balrog_flow)
            print(f"Number of NaNs in df_balrog after scaler: {df_balrog_flow.isna().sum().sum()}")
            if self.cfg['APPLY_YJ_TRANSFORM_FLOW_RUN'] is True:
                if self.cfg['TRANSFORM_COLS_RUN'] is None:
                    trans_col = df_balrog_flow.keys()
                else:
                    trans_col = self.cfg['TRANSFORM_COLS_RUN']
                print("apply inverse yj transform on balrog")
                df_balrog_flow = self.galaxies.yj_inverse_transform_data(
                    data_frame=df_balrog_flow,
                    columns=trans_col
                )
            print(f"Number of NaNs in df_balrog after yj inverse transformation: {df_balrog_flow.isna().sum().sum()}")
            df_balrog.loc[:, self.cfg[f'OUTPUT_COLS_{self.lum_type}_RUN']] = df_balrog_flow[self.cfg[f'OUTPUT_COLS_{self.lum_type}_RUN']].values
            df_balrog.loc[:, self.cfg[f'INPUT_COLS_{self.lum_type}_RUN']] = df_balrog_flow[self.cfg[f'INPUT_COLS_{self.lum_type}_RUN']].values
        arr_flow_gandalf_output = self.gandalf_flow.sample(
            len(df_gandalf_flow),
            cond_inputs=torch.from_numpy(df_gandalf_flow[self.cfg[f'INPUT_COLS_{self.lum_type}_RUN']].values).double()
        ).detach().numpy()

        df_gandalf_flow.loc[:, self.cfg[f'OUTPUT_COLS_{self.lum_type}_RUN']] = arr_flow_gandalf_output
        print(f"Length gandalf catalog: {len(df_gandalf_flow)}")
        print(f"Number of NaNs in df_gandalf: {df_gandalf_flow.isna().sum().sum()}")
        if self.cfg['APPLY_SCALER_FLOW_RUN'] is True:
            print("apply scaler on df_gandalf")
            df_gandalf_flow = self.galaxies.inverse_scale_data(df_gandalf_flow)
        print(f"Number of NaNs in df_gandalf after scaler: {df_gandalf_flow.isna().sum().sum()}")
        if self.cfg['APPLY_YJ_TRANSFORM_FLOW_RUN'] is True:
            if self.cfg['TRANSFORM_COLS_RUN'] is None:
                trans_col = df_gandalf_flow.keys()
            else:
                trans_col = self.cfg['TRANSFORM_COLS_RUN']
            print("apply yj transform on df_gandalf")
            df_gandalf_flow = self.galaxies.yj_inverse_transform_data(
                data_frame=df_gandalf_flow,
                columns=trans_col
            )
        print(f"Number of NaNs in df_gandalf after yj inverse transformation: {df_gandalf_flow.isna().sum().sum()}")

        df_gandalf.loc[:, self.cfg[f'OUTPUT_COLS_{self.lum_type}_RUN']] = df_gandalf_flow[self.cfg[f'OUTPUT_COLS_{self.lum_type}_RUN']].values
        df_gandalf.loc[:, self.cfg[f'INPUT_COLS_{self.lum_type}_RUN']] = df_gandalf_flow[self.cfg[f'INPUT_COLS_{self.lum_type}_RUN']].values

        print(f"Length gandalf catalog: {len(df_gandalf)}")
        print(f"Length balrog catalog: {len(df_balrog)}")
        if df_gandalf.isna().sum().sum() > 0:
            print("Warning: NaNs in df_gandalf_rescaled")
            print(f"Number of NaNs in df_gandalf: {df_gandalf.isna().sum().sum()}")
            df_gandalf.dropna(inplace=True)

        if df_balrog.isna().sum().sum() > 0:
            print("Warning: NaNs in df_gandalf_rescaled")
            print(f"Number of NaNs in df_gandalf: {df_balrog.isna().sum().sum()}")
            df_balrog.dropna(inplace=True)

        print(f"Length gandalf catalog: {len(df_gandalf)}")
        print(f"Length balrog catalog: {len(df_balrog)}")

        for col in df_gandalf.keys():
            if "unsheared" in col:
                print(f"{col}: {df_gandalf[col].min()}/{df_balrog[col].min()}\t{df_gandalf[col].max()}/{df_balrog[col].max()}")

        # if "unsheared/flux_r" not in df_gandalf.keys():
        #     df_gandalf.loc[:, "unsheared/flux_r"] = mag2flux(df_gandalf["unsheared/mag_r"])
        #
        # if "unsheared/flux_r" not in df_balrog.keys():
        #     df_balrog.loc[:, "unsheared/flux_r"] = mag2flux(df_balrog["unsheared/mag_r"])
        print(f"Length gandalf catalog: {len(df_gandalf)}")
        print(f"Length balrog catalog: {len(df_balrog)}")

        return df_balrog, df_gandalf

    def save_data(self, data_frame, file_name, tmp_samples=False):
        """"""
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
                with open(f"{self.cfg['PATH_CATALOGS']}/{file_name}", "wb") as f:
                    pickle.dump(data_frame, f, protocol=2)

    @staticmethod
    def apply_cuts(cfg, data_frame):
        """"""
        data_frame = unsheared_object_cuts(data_frame=data_frame)
        data_frame = flag_cuts(data_frame=data_frame)
        data_frame = unsheared_shear_cuts(data_frame=data_frame)
        data_frame = binary_cut(data_frame=data_frame)
        if cfg['MASK_CUT_FUNCTION'] == "HEALPY":
            data_frame = mask_cut_healpy(data_frame=data_frame, master=f"{cfg['PATH_DATA']}/{cfg['FILENAME_MASTER_CAT']}")
        elif cfg['MASK_CUT_FUNCTION'] == "ASTROPY":
            # Todo there is a bug here, I cutout to many galaxies
            data_frame = mask_cut(data_frame=data_frame, master=f"{cfg['PATH_DATA']}/{cfg['FILENAME_MASTER_CAT']}")
        else:
            print("No mask cut function defined!!!")
            exit()
        data_frame = unsheared_mag_cut(data_frame=data_frame)
        return data_frame

    def plot_classf_data(self, df_balrog, df_gandalf, df_balrog_detected, df_gandalf_detected):
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
