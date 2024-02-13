from gandalf_galaxie_dataset import DESGalaxies
from torch.utils.data import DataLoader, RandomSampler
from scipy.stats import binned_statistic, median_abs_deviation
from Handler import calc_color, plot_compare_corner, plot_confusion_matrix_gandalf, plot_roc_curve_gandalf, plot_classifier_histogram, plot_calibration_curve_gandalf
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
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
        # self.make_dirs()
        self.run_loader, self.galaxies = self.init_dataset()
        self.gandalf_flow, self.gandalf_classifier, self.calibration_model = self.init_trained_models()

    # def make_dirs(self):
    #     """"""
    #     self.cfg['PATH_PLOTS_FOLDER'] = {}
    #     self.cfg['PATH_OUTPUT'] = f"{self.cfg['PATH_OUTPUT']}/gandalf_run_{self.cfg['RUN_DATE']}"
    #     self.cfg['PATH_PLOTS'] = f"{self.cfg['PATH_OUTPUT']}/{self.cfg['FOLDER_PLOTS']}"
    #     self.cfg['PATH_CATALOGS'] = f"{self.cfg['PATH_OUTPUT']}/{self.cfg['FOLDER_CATALOGS']}"
    #     if not os.path.exists(self.cfg['PATH_OUTPUT']):
    #         os.mkdir(self.cfg['PATH_OUTPUT'])
    #     if not os.path.exists(self.cfg['PATH_PLOTS']):
    #         os.mkdir(self.cfg['PATH_PLOTS'])
    #     if not os.path.exists(self.cfg['PATH_CATALOGS']):
    #         os.mkdir(self.cfg['PATH_CATALOGS'])
    #     for plot in self.cfg['PLOTS_RUN']:
    #         self.cfg[f'PATH_PLOTS_FOLDER'][plot.upper()] = f"{self.cfg['PATH_PLOTS']}/{plot}"
    #         if not os.path.exists(self.cfg[f'PATH_PLOTS_FOLDER'][plot.upper()]):
    #             os.mkdir(self.cfg[f'PATH_PLOTS_FOLDER'][plot.upper()])

    def init_dataset(self):
        galaxies = DESGalaxies(
            cfg=self.cfg,
            kind="run_gandalf"
        )
        run_loader = DataLoader(
            galaxies.run_dataset,
            batch_size=self.cfg['NUMBER_SAMPLES'],
            shuffle=False,
            num_workers=0
        )

        return run_loader, galaxies

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

    def sample_random_data_from_dataset(self, dataset):
        sampler = RandomSampler(dataset, replacement=True, num_samples=self.cfg['NUMBER_SAMPLES'])
        dataloader = DataLoader(dataset, batch_size=self.cfg['NUMBER_SAMPLES'], sampler=sampler)
        return dataloader

    def predict_calibrated(self, y_pred):
        calibrated_outputs = self.calibration_model.predict_proba(y_pred.reshape(-1, 1))[:, 1]
        return calibrated_outputs

    def run(self):
        """"""
        print(f"Length sample dataset: {len(self.run_loader.dataset)}")

        for batch_idx, data in enumerate(self.run_loader):
            tsr_input = data[0].double()
            arr_flow_true_output = data[1].numpy()
            arr_true_detected = data[2].numpy()
            arr_cut_cols = data[3].numpy()

        df_balrog = pd.DataFrame(
            np.concatenate((tsr_input.numpy(), arr_flow_true_output, arr_true_detected, arr_cut_cols), axis=1),
            columns=self.cfg[f'INPUT_COLS_{self.lum_type}_RUN'] +
                    self.cfg[f'OUTPUT_COLS_{self.lum_type}_RUN'] +
                    self.cfg[f'OUTPUT_COLS_CLASSF_RUN'] +
                    self.cfg[f'CUT_COLS_RUN']
        )

        if self.cfg['CLASSF_GALAXIES'] is True:
            with torch.no_grad():
                arr_classf_gandalf_output = self.gandalf_classifier(tsr_input).squeeze().numpy()
            arr_gandalf_prob_calib = self.predict_calibrated(arr_classf_gandalf_output)
            arr_gandalf_detected_calib = arr_gandalf_prob_calib > np.random.rand(self.cfg['NUMBER_SAMPLES'])
            validation_accuracy = accuracy_score(arr_true_detected, arr_gandalf_detected_calib)
            gandalf_detected = np.count_nonzero(arr_gandalf_detected_calib)
            gandalf_not_detected = arr_gandalf_detected_calib.size - gandalf_detected
            balrog_detected = np.count_nonzero(arr_true_detected)
            balrog_not_detected = arr_true_detected.size - balrog_detected

            arr_masked_input = tsr_input.numpy()[arr_gandalf_detected_calib]
            arr_masked_output = arr_flow_true_output[arr_gandalf_detected_calib]
            arr_masked_cut_cols = arr_cut_cols[arr_gandalf_detected_calib]
            df_balrog_all = df_balrog[self.cfg[f'INPUT_COLS_{self.lum_type}_RUN'] +
                                      self.cfg[f'OUTPUT_COLS_{self.lum_type}_RUN']]
            df_gandalf_all = pd.DataFrame(np.concatenate((tsr_input.numpy(), arr_flow_true_output), axis=1),
                                          columns=self.cfg[f'INPUT_COLS_{self.lum_type}_RUN'] +
                                                  self.cfg[f'OUTPUT_COLS_{self.lum_type}_RUN'])
            # df_balrog_true = df_balrog[arr_true_detected.astype(bool)]
            # df_balrog_true = df_balrog_true[self.cfg[f'INPUT_COLS_{self.lum_type}_RUN'] + self.cfg[f'OUTPUT_COLS_{self.lum_type}_RUN']]
            df_balrog_all.reset_index(drop=True, inplace=True)
            df_gandalf_all.reset_index(drop=True, inplace=True)

            df_balrog = df_balrog[arr_gandalf_detected_calib]
            df_balrog = df_balrog[self.cfg[f'INPUT_COLS_{self.lum_type}_RUN'] + self.cfg[f'OUTPUT_COLS_{self.lum_type}_RUN']]
            df_balrog.reset_index(drop=True, inplace=True)

            df_flow_input = pd.DataFrame(
                np.concatenate((arr_masked_input, arr_masked_output), axis=1),
                columns=self.cfg[f'INPUT_COLS_{self.lum_type}_RUN'] + self.cfg[f'OUTPUT_COLS_{self.lum_type}_RUN']
            )

            if self.cfg['APPLY_SCALER_CLASSF_RUN'] is True:
                df_flow_input = self.galaxies.inverse_scale_data(df_flow_input)
                df_balrog = self.galaxies.inverse_scale_data(df_balrog)
                # df_balrog_true = self.galaxies.inverse_scale_data(df_balrog_true)
                df_balrog_all = self.galaxies.inverse_scale_data(df_balrog_all)
                df_gandalf_all = self.galaxies.inverse_scale_data(df_gandalf_all)

            if self.cfg['APPLY_YJ_TRANSFORM_CLASSF_RUN'] is True:
                if self.cfg['TRANSFORM_COLS_RUN'] is None:
                    trans_col = df_flow_input.keys()
                else:
                    trans_col = self.cfg['TRANSFORM_COLS_RUN']
                df_flow_input = self.galaxies.yj_inverse_transform_data(
                    data_frame=df_flow_input,
                    columns=trans_col
                )
                df_balrog = self.galaxies.yj_inverse_transform_data(
                    data_frame=df_balrog,
                    columns=trans_col
                )
                # df_balrog_true = self.galaxies.yj_inverse_transform_data(
                #     data_frame=df_balrog_true,
                #     columns=trans_col
                # )
                df_balrog_all = self.galaxies.yj_inverse_transform_data(
                    data_frame=df_balrog_all,
                    columns=trans_col
                )
                df_gandalf_all = self.galaxies.yj_inverse_transform_data(
                    data_frame=df_gandalf_all,
                    columns=trans_col
                )

            print(f"Accuracy sample: {validation_accuracy * 100.0:.2f}%")
            print(f"Number of NOT true_detected galaxies gandalf: {gandalf_not_detected} of {self.cfg['NUMBER_SAMPLES']}")
            print(f"Number of true_detected galaxies gandalf: {gandalf_detected} of {self.cfg['NUMBER_SAMPLES']}")
            print(f"Number of NOT true_detected galaxies balrog: {balrog_not_detected} of {self.cfg['NUMBER_SAMPLES']}")
            print(f"Number of true_detected galaxies balrog: {balrog_detected} of {self.cfg['NUMBER_SAMPLES']}")

            df_classf_plot = pd.DataFrame({
                "gandalf_detected": arr_gandalf_detected_calib.ravel(),
                "balrog_detected": arr_true_detected.ravel(),
            })

            df_balrog_all["true_detected"] = arr_true_detected
            df_gandalf_all["true_detected"] = arr_gandalf_detected_calib
            df_gandalf_all["probability"] = arr_gandalf_prob_calib

            self.plot_classf_data(df_classf_plot=df_classf_plot, df_balrog=df_balrog_all, df_gandalf=df_gandalf_all)

            if self.cfg['EMULATE_GALAXIES'] is False:
                exit()

        if self.cfg['EMULATE_GALAXIES'] is True:
            if self.cfg['CLASSF_GALAXIES'] is True:
                if self.cfg['APPLY_YJ_TRANSFORM_FLOW_RUN'] is True:
                    self.galaxies.applied_yj_transform = "_YJ"
                else:
                    self.galaxies.applied_yj_transform = ""
                if self.cfg['APPLY_SCALER_FLOW_RUN'] is True:
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

                if self.cfg['APPLY_YJ_TRANSFORM_FLOW_RUN'] is True:
                    df_flow_input = self.galaxies.yj_transform_data_on_fly(
                        data_frame=df_flow_input,
                        columns=df_flow_input.keys(),
                        dict_pt=self.galaxies.dict_pt
                    )

                if self.cfg['APPLY_SCALER_FLOW_RUN'] is True:
                    df_flow_input = self.galaxies.scale_data_on_fly(
                        data_frame=df_flow_input,
                        scaler=self.galaxies.scaler
                    )

                tsr_masked_input = torch.from_numpy(
                    df_flow_input[self.cfg[f'INPUT_COLS_{self.lum_type}_RUN']].values).double()

                del df_flow_input

            else:
                tsr_masked_input = torch.from_numpy(
                    df_balrog[self.cfg[f'INPUT_COLS_{self.lum_type}_RUN']].values).double()
                arr_masked_output = df_balrog[self.cfg[f'OUTPUT_COLS_{self.lum_type}_RUN']].values
                arr_masked_cut_cols = df_balrog[self.cfg[f'CUT_COLS_RUN']].values
                arr_gandalf_prob_calib = np.ones(len(df_balrog))
                arr_gandalf_detected_calib = np.array([True for _ in range(len(df_balrog))])
                df_balrog = df_balrog[self.cfg[f'INPUT_COLS_{self.lum_type}_RUN']+self.cfg[f'OUTPUT_COLS_{self.lum_type}_RUN']]

            arr_flow_gandalf_output = self.gandalf_flow.sample(len(tsr_masked_input), cond_inputs=tsr_masked_input).detach().numpy()
            df_gandalf = pd.DataFrame(
                np.concatenate(
                    (tsr_masked_input.numpy(), arr_flow_gandalf_output),
                    axis=1
                ),
                columns=self.cfg[f'INPUT_COLS_{self.lum_type}_RUN'] + self.cfg[f'OUTPUT_COLS_{self.lum_type}_RUN']
            )
            print(f"Length gandalf catalog: {len(df_gandalf)}")
            print(f"Number of NaNs in df_gandalf: {df_gandalf.isna().sum().sum()}")
            print(f"Number of NaNs in df_balrog: {df_balrog.isna().sum().sum()}")
            if self.cfg['APPLY_SCALER_FLOW_RUN'] is True:
                print("apply scaler on df_gandalf")
                df_gandalf = self.galaxies.inverse_scale_data(df_gandalf)
            print(f"Number of NaNs in df_gandalf after scaler: {df_gandalf.isna().sum().sum()}")
            print(f"Number of NaNs in df_balrog after scaler: {df_balrog.isna().sum().sum()}")
            if self.cfg['APPLY_YJ_TRANSFORM_FLOW_RUN'] is True:
                if self.cfg['TRANSFORM_COLS_RUN'] is None:
                    trans_col = df_gandalf.keys()
                else:
                    trans_col = self.cfg['TRANSFORM_COLS_RUN']
                print("apply yj transform on df_gandalf")
                df_gandalf = self.galaxies.yj_inverse_transform_data(
                    data_frame=df_gandalf,
                    columns=trans_col
                )
            print(f"Number of NaNs in df_gandalf after yj inverse transformation: {df_gandalf.isna().sum().sum()}")
            print(f"Number of NaNs in df_balrog after yj inverse transformation: {df_balrog.isna().sum().sum()}")

            df_gandalf['true_detected'] = np.ones(len(df_gandalf))
            df_gandalf["probability"] = arr_gandalf_prob_calib[arr_gandalf_detected_calib]
            df_gandalf[self.cfg['CUT_COLS_RUN']] = arr_masked_cut_cols

            df_balrog['true_detected'] = arr_true_detected[arr_gandalf_detected_calib]
            df_balrog["probability"] = np.ones(len(df_balrog))
            df_balrog[self.cfg['CUT_COLS_RUN']] = arr_masked_cut_cols

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
                print(
                    f"{col}: {df_gandalf[col].min()}/{df_balrog[col].min()}\t{df_gandalf[col].max()}/{df_balrog[col].max()}")

        if self.cfg['APPLY_GANDALF_CUTS'] is True:
            df_gandalf = self.galaxies.remove_outliers(data_frame=df_gandalf)
            df_balrog = self.galaxies.remove_outliers(data_frame=df_balrog)

            for col in df_gandalf.keys():
                if "unsheared" in col:
                    print(
                        f"{col}: {df_gandalf[col].min()}/{df_balrog[col].min()}\t{df_gandalf[col].max()}/{df_balrog[col].max()}")

        # if self.cfg['PLOT_RUN'] is True:
        #     print(f"Start plotting data")
        #     self.plot_data_flow(df_gandalf=df_gandalf, df_balrog=df_balrog)

        df_balrog_cut = df_balrog.copy()
        df_gandalf_cut = df_gandalf.copy()

        if self.cfg['APPLY_OBJECT_CUT_RUN'] is not True:
            df_balrog_cut = self.galaxies.unsheared_object_cuts(data_frame=df_balrog_cut)
            df_gandalf_cut = self.galaxies.unsheared_object_cuts(data_frame=df_gandalf_cut)
        if self.cfg['APPLY_FLAG_CUT_RUN'] is not True:
            df_balrog_cut = self.galaxies.flag_cuts(data_frame=df_balrog_cut)
            df_gandalf_cut = self.galaxies.flag_cuts(data_frame=df_gandalf_cut)
        if self.cfg['APPLY_UNSHEARED_MAG_CUT_RUN'] is not True:
            df_balrog_cut = self.galaxies.unsheared_mag_cut(data_frame=df_balrog_cut)
            df_gandalf_cut = self.galaxies.unsheared_mag_cut(data_frame=df_gandalf_cut)
        if self.cfg['APPLY_UNSHEARED_SHEAR_CUT_RUN'] is not True:
            df_balrog_cut = self.galaxies.unsheared_shear_cuts(data_frame=df_balrog_cut)
            df_gandalf_cut = self.galaxies.unsheared_shear_cuts(data_frame=df_gandalf_cut)

        print(f"Length gandalf catalog: {len(df_gandalf)}")
        if self.cfg['PLOT_RUN'] is True:
            self.plot_data_flow(df_gandalf=df_gandalf_cut, df_balrog=df_balrog_cut, mcal='mcal_')

        self.save_data(df_gandalf=df_gandalf_cut)

    def save_data(self, df_gandalf):
        """"""
        df_gandalf.rename(columns={"ID": "true_id"}, inplace=True)
        with open(f"{self.cfg['PATH_CATALOGS']}/{self.cfg['FILENAME_GANDALF_CATALOG']}_{self.cfg['NUMBER_SAMPLES']}.pkl", "wb") as f:
            pickle.dump(df_gandalf.to_dict(), f, protocol=2)

    @staticmethod
    def dataset_to_tensor(dataset):
        data_list = [dataset[i] for i in range(len(dataset))]
        input_data, output_data_flow, output_data_classf, cut_cols = zip(*data_list)
        tsr_input = torch.stack(input_data)
        tsr_output_flow = torch.stack(output_data_flow)
        tsr_output_classf = torch.stack(output_data_classf)
        tsr_cut_cols = torch.stack(cut_cols)
        return tsr_input, tsr_output_flow, tsr_output_classf, tsr_cut_cols

    @staticmethod
    def dataloader_to_tensor(dataloader):
        input_data, output_data_flow, output_data_classf, cut_cols = [], [], [], []
        for data in dataloader:
            input_data.append(data[0])
            output_data_flow.append(data[1])
            output_data_classf.append(data[2])
            cut_cols.append(data[3])
        tsr_input = torch.stack(input_data)
        tsr_output_flow = torch.stack(output_data_flow)
        tsr_output_classf = torch.stack(output_data_classf)
        tsr_cut_cols = torch.stack(cut_cols)
        return tsr_input, tsr_output_flow, tsr_output_classf, tsr_cut_cols

    def plot_classf_data(self, df_classf_plot, df_balrog, df_gandalf):
        """"""

        print("Start plotting classf data")
        if self.cfg['PLOT_MATRIX_RUN'] is True:
            plot_confusion_matrix_gandalf(
                df_classf_plot=df_classf_plot,
                show_plot=self.cfg['SHOW_PLOT_RUN'],
                save_plot=self.cfg['SAVE_PLOT_RUN'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'CONFUSION_MATRIX']}/confusion_matrix.png",
                title=f"Confusion Matrix"
            )

        if self.cfg['PLOT_CALIBRATION_CURVE'] is True:
            plot_calibration_curve_gandalf(
                true_detected=df_balrog["true_detected"],
                probability=df_gandalf["probability"],
                n_bins=10,
                show_plot=self.cfg['SHOW_PLOT_RUN'],
                save_plot=self.cfg['SAVE_PLOT_RUN'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'CALIBRATION_CURVE']}/calibration_curve.png",
                title=f"Calibration Curve"
            )

        # ROC und AUC
        if self.cfg['PLOT_ROC_CURVE_RUN'] is True:
            plot_roc_curve_gandalf(
                data_frame=df_classf_plot,
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
                title=f"All"
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


    def plot_data_flow(self, df_gandalf, df_balrog, mcal=''):
        """"""

        if self.cfg['PLOT_COLOR_COLOR_RUN'] is True:
            df_gandalf = calc_color(
                data_frame=df_gandalf,
                colors=self.cfg['COLORS_RUN'],
                column_name=f"unsheared/{self.lum_type.lower()}"
            )
            df_balrog = calc_color(
                data_frame=df_balrog,
                colors=self.cfg['COLORS_RUN'],
                column_name=f"unsheared/{self.lum_type.lower()}"
            )

            try:
                plot_compare_corner(
                    data_frame_generated=df_gandalf,
                    data_frame_true=df_balrog,
                    dict_delta=None,
                    epoch=None,
                    title=f"{mcal} color-color plot",
                    columns=["r-i", "i-z"],
                    labels=["r-i", "i-z"],
                    show_plot=self.cfg['SHOW_PLOT_RUN'],
                    save_plot=self.cfg['SAVE_PLOT_RUN'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER'][f'{mcal.upper()}COLOR_COLOR_PLOT']}/{self.cfg[f'RUN_NUMBER']}_{mcal}color_color.png",
                    ranges=[(-8, 8), (-8, 8)]
                )
                # plot_chain_compare(
                #     data_frame_generated=df_gandalf,
                #     data_frame_true=df_balrog,
                #     title=f"{mcal} color-color plot",
                #     columns=["r-i", "i-z"],
                #     labels=["r-i", "i-z"],
                #     sigma2d=True,
                #     extents={
                #         "r-i": (-4, 4),
                #         "i-z": (-4, 4)
                #     },
                #     show_plot=self.cfg['SHOW_PLOT_RUN'],
                #     save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER'][f'{mcal.upper()}COLOR_COLOR_PLOT']}/{self.cfg[f'RUN_NUMBER']}_{mcal}color_color_chain.png"
                # )
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
                    plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER'][f'{mcal.upper()}RESIDUAL_PLOT']}/{mcal}residual_plot.png")
                if self.cfg['SHOW_PLOT_RUN'] is True:
                    plt.show()
                plt.clf()
                plt.close()
            except Exception as e:
                print(e)
        if self.cfg['PLOT_CHAIN_RUN'] is True:
            try:
                plot_compare_corner(
                    data_frame_generated=df_gandalf,
                    data_frame_true=df_balrog,
                    dict_delta=None,
                    epoch=None,
                    title=f"{mcal} chain plot",
                    show_plot=self.cfg['SHOW_PLOT_RUN'],
                    save_plot=self.cfg['SAVE_PLOT_RUN'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER'][f'{mcal.upper()}CHAIN_PLOT']}/{mcal}chainplot.png",
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
                        standard_levels = 10  # [0.393, 0.865, 0.989]

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
                            levels=standard_levels,  # 10
                            color="dodgerblue",
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
                            alpha=.5,
                            color="darkorange",
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
                        plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER'][f'{mcal.upper()}CONDITIONS_PLOT']}/{mcal}{condition}_plot.png")
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
                    plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER'][f'{mcal.upper()}HIST_PLOT']}/{mcal}magnitude_histogram.png")
                if self.cfg['SHOW_PLOT_RUN'] == True:
                    plt.show()
                plt.clf()
                plt.close()
            except Exception as e:
                print(e)

        return df_balrog, df_gandalf
