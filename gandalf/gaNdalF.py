from gandalf_galaxie_dataset import DESGalaxies
from torch.utils.data import DataLoader, RandomSampler
from scipy.stats import binned_statistic, median_abs_deviation
from Handler import calc_color, plot_compare_corner
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
import pickle
import torch
import os


class gaNdalF(object):
    """"""
    def __init__(self, cfg):
        """"""
        self.cfg = cfg

        self.make_dirs()

        self.galaxies = self.init_dataset()

        self.gandalf_flow, self.gandalf_classifier, self.calibration_model = self.init_trained_models()

    def make_dirs(self):
        """"""
        self.cfg['PATH_OUTPUT_RUN'] = f"{self.cfg['PATH_OUTPUT_RUN']}/gandalf_run_{self.cfg['RUN_DATE_RUN']}"
        self.cfg['PATH_PLOTS_RUN'] = f"{self.cfg['PATH_OUTPUT_RUN']}/{self.cfg['PATH_PLOTS_RUN']}"
        self.cfg['PATH_CATALOG_RUN'] = f"{self.cfg['PATH_OUTPUT_RUN']}/{self.cfg['PATH_CATALOG_RUN']}"
        if not os.path.exists(self.cfg['PATH_OUTPUT_RUN']):
            os.mkdir(self.cfg['PATH_OUTPUT_RUN'])
        if not os.path.exists(self.cfg['PATH_PLOTS_RUN']):
            os.mkdir(self.cfg['PATH_PLOTS_RUN'])
        if not os.path.exists(self.cfg['PATH_CATALOG_RUN']):
            os.mkdir(self.cfg['PATH_CATALOG_RUN'])
        for plot in self.cfg['PLOTS_RUN']:
            self.cfg[f'PATH_PLOTS_FOLDER_RUN'][plot.upper()] = f"{self.cfg['PATH_PLOTS_RUN']}/{plot}"
            if not os.path.exists(self.cfg[f'PATH_PLOTS_FOLDER_RUN'][plot.upper()]):
                os.mkdir(self.cfg[f'PATH_PLOTS_FOLDER_RUN'][plot.upper()])

    def init_dataset(self):
        galaxies = DESGalaxies(
            cfg=self.cfg,
            kind="run_gandalf",
            lst_split=[]
        )
        return galaxies

    def init_trained_models(self):
        """"""
        gandalf_flow = None
        gandalf_classifier = None
        calibration_model = None

        if self.cfg['EMULATE_GALAXIES'] is True:
            gandalf_flow = torch.load(f"{self.cfg['PATH_Trained_NN']}/{self.cfg['NN_FILE_NAME_FLOW']}")

        if self.cfg['CLASSF_GALAXIES'] is True:
            gandalf_classifier = torch.load(f"{self.cfg['PATH_Trained_NN']}/{self.cfg['NN_FILE_NAME_CLASSF']}")
            calibration_model = joblib.load(f"{self.cfg['PATH_Trained_NN']}/{self.cfg['NN_FILE_CALIBRATION']}")

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
        sample_dataset = self.sample_random_data_from_dataset(dataset=self.galaxies.tsr_data)
        print(f"Length sample dataset: {len(sample_dataset.dataset)}")

        for batch_idx, data in enumerate(sample_dataset):
            tsr_input = data[0].double()
            tsr_output_flow = data[1].double()
            tsr_output_classf = data[2].double()
            tsr_cut_cols = data[3].double()
            print(f"Länge des gezogenen Datensatzes: {len(tsr_input)}")

        arr_gandalf = np.concatenate(
            (tsr_input.numpy(), tsr_cut_cols.numpy()),
            axis=1
        )
        arr_balrog = np.concatenate(
            (tsr_input.numpy(), tsr_output_flow.numpy()),
            axis=1
        )

        df_gandalf = pd.DataFrame(
            arr_gandalf,
            columns=self.cfg['INPUT_COLS_MAG_RUN'] + self.cfg['CUT_COLS_RUN']
        )

        df_balrog = pd.DataFrame(
            arr_balrog,
            columns=self.cfg['INPUT_COLS_MAG_RUN'] + self.cfg['OUTPUT_COLS_MAG_RUN']
        )
        print(f"Length sample dataset: {df_gandalf.shape}")
        if self.cfg['APPLY_SCALER_RUN'] is True:
            df_balrog = pd.DataFrame(self.galaxies.scaler.inverse_transform(df_balrog), columns=df_balrog.keys())

        if self.cfg['APPLY_YJ_TRANSFORM_RUN'] is True:
            if self.cfg['TRANSFORM_COLS_RUN'] is None:
                trans_col = df_balrog.keys()
            else:
                trans_col = self.cfg['TRANSFORM_COLS_RUN']
            df_balrog = self.galaxies.yj_inverse_transform_data(
                data_frame=df_balrog,
                columns=trans_col
            )
        df_balrog[self.cfg['OUTPUT_COLS_CLASSF_RUN']] = tsr_output_classf.numpy()
        df_balrog[self.cfg['CUT_COLS_RUN']] = tsr_cut_cols.numpy()

        if self.cfg['CLASSF_GALAXIES'] is True:
            input_data = torch.tensor(df_gandalf[self.cfg['INPUT_COLS_MAG_RUN']].values).double()
            with torch.no_grad():
                tsr_output_gandalf_classifier = self.gandalf_classifier(input_data).squeeze().numpy()
            # probability_calibrated_sample = self.predict_calibrated(tsr_output_gandalf_classifier)
            probability_calibrated_sample = tsr_output_gandalf_classifier
            detected_calibrated_sample = probability_calibrated_sample > np.random.rand(self.cfg['NUMBER_SAMPLES'])
            df_gandalf["detected"] = detected_calibrated_sample
            df_gandalf["probability"] = probability_calibrated_sample
            validation_accuracy_calibrated_sample = accuracy_score(df_balrog["detected"], df_gandalf["detected"])
            print(f"Accuracy sample: {validation_accuracy_calibrated_sample * 100.0:.2f}%")
            df_balrog = df_balrog[df_balrog['detected'] == 1]
            df_gandalf = df_gandalf[df_gandalf['detected'] == 1]
            detected_calibrated_sample = df_gandalf["detected"].values
            probability_calibrated_sample = df_gandalf["probability"].values
            df_gandalf.drop("detected", axis=1, inplace=True)
            df_balrog.drop("detected", axis=1, inplace=True)
            df_cut_cols_gandalf = df_gandalf[self.cfg['CUT_COLS_RUN']]
            df_gandalf = df_gandalf[self.cfg['INPUT_COLS_MAG_RUN']]
            print(f"Number of NOT detected galaxies gandalf: {self.cfg['NUMBER_SAMPLES'] - len(df_gandalf)} of {self.cfg['NUMBER_SAMPLES']}")
            print(f"Number of detected galaxies gandalf: {len(df_gandalf)} of {self.cfg['NUMBER_SAMPLES']}")
            print(f"Number of NOT detected galaxies balrog: {self.cfg['NUMBER_SAMPLES'] - len(df_balrog)} of {self.cfg['NUMBER_SAMPLES']}")
            print(f"Number of detected galaxies balrog: {len(df_balrog)} of {self.cfg['NUMBER_SAMPLES']}")
            print(f"Length gandalf catalog: {len(df_gandalf)}")
            if self.cfg['EMULATE_GALAXIES'] is False:
                exit()

        if self.cfg['EMULATE_GALAXIES'] is True:
            if self.cfg['CLASSF_GALAXIES'] is False:
                df_balrog.drop("detected", axis=1, inplace=True)
                df_gandalf = df_gandalf[self.cfg['INPUT_COLS_MAG_RUN']]
            # tsr_samples = tsr_input
            output_data_gandalf_flow = self.gandalf_flow.sample(len(tsr_input), cond_inputs=tsr_input).detach()
            df_gandalf = pd.DataFrame(
                np.concatenate(
                    (tsr_input.numpy(), output_data_gandalf_flow.numpy()),
                    axis=1
                ),
                columns=self.cfg['INPUT_COLS_MAG_RUN'] + self.cfg['OUTPUT_COLS_MAG_RUN']
            )
            print(f"Length gandalf catalog: {len(df_gandalf)}")
            print(f"Number of NaNs in df_gandalf before inverse scaler: {df_gandalf.isna().sum().sum()}")
            print(f"Length gandalf catalog: {len(df_gandalf)}")
            if self.cfg['APPLY_SCALER_RUN'] is True:
                df_gandalf = pd.DataFrame(self.galaxies.scaler.inverse_transform(df_gandalf), columns=df_gandalf.keys())
            print(f"Number of NaNs in df_gandalf before yj inverse transformation: {df_gandalf.isna().sum().sum()}")
            if self.cfg['APPLY_YJ_TRANSFORM_RUN'] is True:
                if self.cfg['TRANSFORM_COLS_RUN'] is None:
                    trans_col = df_gandalf.keys()
                else:
                    trans_col = self.cfg['TRANSFORM_COLS_RUN']
                df_gandalf = self.galaxies.yj_inverse_transform_data(
                    data_frame=df_gandalf,
                    columns=trans_col
                )
            print(f"Length gandalf catalog: {len(df_gandalf)}")
            df_gandalf[self.cfg['OUTPUT_COLS_MAG_RUN']] = output_data_gandalf_flow.numpy()
            if self.cfg['CLASSF_GALAXIES'] is True:
                df_gandalf['detected'] = detected_calibrated_sample
                df_gandalf["probability"] = probability_calibrated_sample
                df_gandalf[self.cfg['CUT_COLS_RUN']] = df_cut_cols_gandalf.values
            else:
                df_gandalf[self.cfg['CUT_COLS_RUN']] = tsr_cut_cols.numpy()
                df_gandalf['detected'] = np.ones(len(df_gandalf))
                df_gandalf["probability"] = np.ones(len(df_gandalf))
            print(f"Length gandalf catalog: {len(df_gandalf)}")
            if df_gandalf.isna().sum().sum() > 0:
                print("Warning: NaNs in df_gandalf_rescaled")
                print(f"Number of NaNs in df_gandalf: {df_gandalf.isna().sum().sum()}")
                df_gandalf.dropna(inplace=True)
        else:
            df_gandalf = df_balrog.copy()
            df_balrog = df_balrog.copy()
        print(f"Length gandalf catalog: {len(df_gandalf)}")
        if self.cfg['PLOT_RUN'] is True:
            print(f"Start plotting data")
            self.plot_data(df_gandalf=df_gandalf, df_balrog=df_balrog)

        df_balrog_cut = df_balrog.copy()
        df_gandalf_cut = df_gandalf.copy()

        if self.cfg['APPLY_OBJECT_CUT'] is not True:
            df_balrog_cut = self.galaxies.unsheared_object_cuts(data_frame=df_balrog_cut)
            df_gandalf_cut = self.galaxies.unsheared_object_cuts(data_frame=df_gandalf_cut)
        if self.cfg['APPLY_FLAG_CUT'] is not True:
            df_balrog_cut = self.galaxies.flag_cuts(data_frame=df_balrog_cut)
            df_gandalf_cut = self.galaxies.flag_cuts(data_frame=df_gandalf_cut)
        if self.cfg['APPLY_UNSHEARED_MAG_CUT'] is not True:
            df_balrog_cut = self.galaxies.unsheared_mag_cut(data_frame=df_balrog_cut)
            df_gandalf_cut = self.galaxies.unsheared_mag_cut(data_frame=df_gandalf_cut)
        if self.cfg['APPLY_UNSHEARED_SHEAR_CUT'] is not True:
            df_balrog_cut = self.galaxies.unsheared_shear_cuts(data_frame=df_balrog_cut)
            df_gandalf_cut = self.galaxies.unsheared_shear_cuts(data_frame=df_gandalf_cut)
        if self.cfg['APPLY_AIRMASS_CUT'] is not True:
            df_balrog_cut = self.galaxies.airmass_cut(data_frame=df_balrog_cut)
            df_gandalf_cut = self.galaxies.airmass_cut(data_frame=df_gandalf_cut)
        print(f"Length gandalf catalog: {len(df_gandalf)}")
        if self.cfg['PLOT_RUN'] is True:
            self.plot_data(df_gandalf=df_gandalf_cut, df_balrog=df_balrog_cut, mcal='mcal_')

        self.save_data(df_gandalf=df_gandalf_cut)

    def save_data(self, df_gandalf):
        """"""
        df_gandalf.rename(columns={"ID": "true_id"}, inplace=True)
        with open(f"{self.cfg['PATH_CATALOG_RUN']}/{self.cfg['NAME_EMULATED_DATA']}_{self.cfg['NUMBER_SAMPLES']}.pkl", "wb") as f:
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

    def plot_data(self, df_gandalf, df_balrog, mcal=''):
        if self.cfg['PLOT_COLOR_COLOR_RUN'] is True:
            df_gandalf = calc_color(
                data_frame=df_gandalf,
                colors=self.cfg['COLORS_RUN'],
                column_name=f"unsheared/{self.cfg['LUM_TYPE'].lower()}"
            )
            df_balrog = calc_color(
                data_frame=df_balrog,
                colors=self.cfg['COLORS_RUN'],
                column_name=f"unsheared/{self.cfg['LUM_TYPE'].lower()}"
            )

            try:
                plot_compare_corner(
                    data_frame_generated=df_gandalf,
                    data_frame_true=df_balrog,
                    dict_delta=None,
                    epoch=None,
                    title=f"color-color plot",
                    columns=["r-i", "i-z"],
                    labels=["r-i", "i-z"],
                    show_plot=self.cfg['SHOW_PLOT_RUN'],
                    save_plot=self.cfg['SAVE_PLOT_RUN'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER_RUN'][f'{mcal.upper()}COLOR_COLOR_PLOT']}/{mcal}color_color.png",
                    ranges=[(-4, 4), (-4, 4)]
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
                    plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER_RUN'][f'{mcal.upper()}RESIDUAL_PLOT']}/{mcal}residual_plot.png")
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
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER_RUN'][f'{mcal.upper()}CHAIN_PLOT']}/{mcal}chainplot.png",
                    columns=[
                        f"unsheared/{self.cfg['LUM_TYPE_RUN'].lower()}_r",
                        f"unsheared/{self.cfg['LUM_TYPE_RUN'].lower()}_i",
                        f"unsheared/{self.cfg['LUM_TYPE_RUN'].lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T"
                    ],
                    labels=[
                        f"{self.cfg['LUM_TYPE_RUN'].lower()}_r",
                        f"{self.cfg['LUM_TYPE_RUN'].lower()}_i",
                        f"{self.cfg['LUM_TYPE_RUN'].lower()}_z",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    ranges=[(15, 30), (15, 30), (15, 30), (-15, 75), (-1.5, 4), (-1.5, 2)]
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

                    cond_lims = np.percentile(df_balrog[condition], [2, 98])

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
                            levels=10,
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
                            levels=10,
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
                        plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER_RUN'][f'{mcal.upper()}CONDITIONS_PLOT']}/{mcal}{condition}_plot.png")
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
                    plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER_RUN'][f'{mcal.upper()}HIST_PLOT']}/{mcal}magnitude_histogram.png")
                if self.cfg['SHOW_PLOT_RUN'] == True:
                    plt.show()
                plt.clf()
                plt.close()
            except Exception as e:
                print(e)

        return df_balrog, df_gandalf
