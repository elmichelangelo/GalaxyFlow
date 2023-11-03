from _Old_scripts.data_loader import load_test_data
from scipy.stats import binned_statistic, median_abs_deviation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml
import torch
import os
import sys
import pandas as pd
import pickle


def load_model(path_model):
    model = torch.load(path_model)
    return model


def main(path_test_data, path_model, path_save_plots, path_save_generated_data, run, number_samples, plot_chain,
         plot_residual, plot_luptize_conditions, conditions, bands, colors, save_generated_data, plot_hist, cfg):
    col_label_flow = [
        "BDF_MAG_DERED_CALIB_U",
        "BDF_MAG_DERED_CALIB_G",
        "BDF_MAG_DERED_CALIB_R",
        "BDF_MAG_DERED_CALIB_I",
        "BDF_MAG_DERED_CALIB_Z",
        "BDF_MAG_DERED_CALIB_J",
        "BDF_MAG_DERED_CALIB_H",
        "BDF_MAG_DERED_CALIB_K",
        "BDF_MAG_ERR_DERED_CALIB_U",
        "BDF_MAG_ERR_DERED_CALIB_G",
        "BDF_MAG_ERR_DERED_CALIB_R",
        "BDF_MAG_ERR_DERED_CALIB_I",
        "BDF_MAG_ERR_DERED_CALIB_Z",
        "BDF_MAG_ERR_DERED_CALIB_J",
        "BDF_MAG_ERR_DERED_CALIB_H",
        "BDF_MAG_ERR_DERED_CALIB_K",
        "Color Mag U-G",
        "Color Mag G-R",
        "Color Mag R-I",
        "Color Mag I-Z",
        "Color Mag Z-J",
        "Color Mag J-H",
        "Color Mag H-K",
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
    ]

    col_output_flow = [
        "unsheared/mag_r",
        "unsheared/mag_i",
        "unsheared/mag_z",
        "unsheared/mag_err_r",
        "unsheared/mag_err_i",
        "unsheared/mag_err_z",
        "unsheared/snr",
        "unsheared/size_ratio",
        "unsheared/flags",
        "unsheared/T"
    ]
    print("Load data...")

    path_save_plots, path_save_generated_data = create_folders(
        path_save_plots=path_save_plots,
        path_save_generated_data=path_save_generated_data,
        run=run
    )

    dict_test_data, df_test_data = load_test_data(
        path_test_data=path_test_data
    )

    scaler = dict_test_data["scaler"]
    # # Write data as torch loader
    test_tensor = torch.from_numpy(np.array(df_test_data[cfg["OUTPUT_COLS_MAG"]]))
    test_labels = torch.from_numpy(np.array(df_test_data[cfg["INPUT_COLS_MAG"]]))
    test_dataset = torch.utils.data.TensorDataset(test_tensor, test_labels)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
        drop_last=False,
        # **kwargs
    )

    print("Load model...")
    model = load_model(path_model)
    for batch_idx, data in enumerate(test_loader):
        if number_samples is None:
            number_samples = len(test_dataset)
            true_data = data[0]
            cond_data = data[1]
        else:
            d_random_real = np.random.choice(len(test_dataset), size=number_samples, replace=True)
            true_data = data[0][d_random_real]
            cond_data = data[1][d_random_real]
        with torch.no_grad():
            print(f"Sample {number_samples} objects")
            test_output = model.sample(num_samples=number_samples, cond_inputs=cond_data).detach()
        print("Plot data")
        df_true, df_generated = plot_data(
            path_save_plots=path_save_plots,
            df_test_data=df_test_data,
            cond_data=cond_data,
            test_output=test_output,
            col_label_flow=col_label_flow,
            col_output_flow=col_output_flow,
            scaler=scaler,
            true_data=true_data,
            plot_chain=plot_chain,
            plot_residual=plot_residual,
            plot_luptize_conditions=plot_luptize_conditions,
            conditions=conditions,
            bands=bands,
            colors=colors,
            plot_hist=plot_hist
        )
    #
    #     if save_generated_data is True:
    #         save_emulated_cat(
    #             path_save_catalog=path_save_generated_data,
    #             df_generated=df_generated,
    #             df_true=df_true
    #         )
    #     break

    # for batch_idx, data in enumerate(all_loader):
    #     if number_samples is None:
    #         number_samples = len(all_loader)
    #         true_data = data[0]
    #         cond_data = data[1]
    #     else:
    #         d_random_real = np.random.choice(len(all_loader), size=number_samples, replace=True)
    #         true_data = data[0][d_random_real]
    #         cond_data = data[1][d_random_real]
    #     with torch.no_grad():
    #         print("Sample data")
    #         test_output = model.sample(num_samples=number_samples, cond_inputs=cond_data).detach()
    #     print("Plot data")
    #     df_true, df_generated = plot_data(
    #         path_save_plots=path_save_plots,
    #         cond_data=cond_data,
    #         test_output=test_output,
    #         col_label_flow=col_label_flow,
    #         col_output_flow=col_output_flow,
    #         scaler=scaler,
    #         true_data=true_data,
    #         do_chain_plot=do_chain_plot,
    #         do_residual_plot=do_residual_plot,
    #         plot_luptize_conditions=plot_luptize_conditions,
    #         conditions=conditions,
    #         bands=bands,
    #         colors=colors,
    #         plot_hist=plot_hist
    #     )

        if save_generated_data is True:
            save_emulated_cat(
                path_save_generated_data=path_save_generated_data,
                df_generated=df_generated,
                df_true=df_true,
                run=run
            )
        break


def plot_data(path_save_plots, df_test_data, cond_data, test_output, col_label_flow, col_output_flow, scaler, true_data,
              plot_chain, plot_residual, plot_luptize_conditions, conditions, bands, colors, plot_hist):

    df_generated_scaled = df_test_data.copy()
    df_true_scaled = df_test_data.copy()

    lst_del_cols = [
        "Color Mag U-G",
        "Color Mag G-R",
        "Color Mag R-I",
        "Color Mag I-Z",
        "Color Mag Z-J",
        "Color Mag J-H",
        "Color Mag H-K"
    ]

    df_generated_label = pd.DataFrame(cond_data.numpy(), columns=col_label_flow)
    df_generated_output = pd.DataFrame(test_output.numpy(), columns=col_output_flow)
    df_generated_output = pd.concat([df_generated_label, df_generated_output], axis=1)
    df_generated_scaled[col_label_flow + col_output_flow] = df_generated_output
    for col in lst_del_cols:
        del df_generated_scaled[col]
    generator_rescaled = scaler.inverse_transform(df_generated_scaled)
    df_generated = pd.DataFrame(generator_rescaled, columns=df_generated_scaled.columns)

    df_true_label = df_generated_label
    df_true_output = pd.DataFrame(true_data, columns=col_output_flow)
    df_true_output = pd.concat([df_true_label, df_true_output], axis=1)
    df_true_scaled[col_label_flow + col_output_flow] = df_true_output
    for col in lst_del_cols:
        del df_true_scaled[col]
    true_rescaled = scaler.inverse_transform(df_true_scaled)
    df_true = pd.DataFrame(true_rescaled, columns=df_true_scaled.columns)

    if plot_chain is True:
        df_generated_measured = pd.DataFrame({})
        df_true_measured = pd.DataFrame({})
        for color in colors:
            df_generated_measured[f"{color[0]}-{color[1]}"] = \
                np.array(df_generated[f"unsheared/mag_{color[0]}"]) - np.array(df_generated[f"unsheared/mag_{color[1]}"])
            df_generated_measured[f"dataset"] = "gaNdalF"
            df_true_measured[f"{color[0]}-{color[1]}"] = \
                np.array(df_true[f"unsheared/mag_{color[0]}"]) - np.array(df_true[f"unsheared/mag_{color[1]}"])
            df_true_measured[f"dataset"] = "Balrog"

        df_color = pd.concat([df_generated_measured, df_true_measured])

        sns.jointplot(
            data=df_color,
            x="r-i",
            y="i-z",
            hue="dataset",
            kind="kde",
            xlim=(-1.5, 2.5),
            ylim=(-2.5, 2.5),
            n_levels=10
        )

        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.savefig(f"{path_save_plots}/color_color_plot.png")
        plt.show()
        plt.clf()
        plt.close()

    if plot_residual is True:
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
            "dataset": ["Balrog" for _ in range(len(df_true[f"unsheared/mag_r"]))]
        })
        df_hist_generated = pd.DataFrame({
            "dataset": ["gaNdalF" for _ in range(len(df_generated[f"unsheared/mag_r"]))]
        })
        for band in bands:
            df_hist_Balrog[f"BDF_MAG_DERED_CALIB - unsheared/mag {band}"] = \
                df_true[f"BDF_MAG_DERED_CALIB_{band.upper()}"] - df_true[f"unsheared/mag_{band}"]
            df_hist_generated[f"BDF_MAG_DERED_CALIB - unsheared/mag {band}"] = \
                df_true[f"BDF_MAG_DERED_CALIB_{band.upper()}"] - df_generated[f"unsheared/mag_{band}"]

        for idx, band in enumerate(bands):
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
        plt.savefig(f"{path_save_plots}/residual_plot.png")
        plt.show()
        plt.clf()
        plt.close()

    if plot_luptize_conditions is True:
        for condition in conditions:
            try:
                cond_figure, (
                (stat_ax1), (stat_ax2), (stat_ax3)) = \
                    plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
                cond_figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
                cond_figure.suptitle(f"BDF_MAG_DERED_CALIB - unsheared/mag", fontsize=16)

                outputs = ['unsheared/mag_' + b for b in bands]
                true_outputs = ['BDF_MAG_DERED_CALIB_' + b.upper() for b in bands]
                output_errs = ['unsheared/mag_err_' + b for b in bands]

                lst_axis_con = [
                    stat_ax1,
                    stat_ax2,
                    stat_ax3
                ]

                cond_lims = np.percentile(df_true[condition], [2, 98])

                for idx, out in enumerate(zip(outputs, output_errs, true_outputs)):
                    output_ = out[0]
                    output_err_ = out[1]
                    true_output_ = out[2]

                    diff_true = (df_true[true_output_] - df_true[output_]) / df_true[output_err_]
                    df_conditional_true = pd.DataFrame({
                        condition: df_true[condition],
                        f"residual band {bands[idx]}": diff_true,
                        "dataset": ["Balrog" for _ in range(len(df_true[condition]))]
                    })
                    bin_means_true, bin_edges_mean_true, binnumber_true = binned_statistic(
                        df_true[condition], diff_true, statistic='median', bins=10, range=cond_lims)
                    bin_stds_true, bin_edges_true, binnumber_true = binned_statistic(
                        df_true[condition], diff_true, statistic=median_abs_deviation, bins=10, range=cond_lims)
                    xerr_true = (bin_edges_mean_true[1:] - bin_edges_mean_true[:-1]) / 2
                    xmean_true = (bin_edges_mean_true[1:] + bin_edges_mean_true[:-1]) / 2
                    lst_axis_con[idx].errorbar(
                        xmean_true, bin_means_true, xerr=xerr_true, yerr=bin_stds_true, color='dodgerblue', lw=2, label='Balrog')

                    diff_generated = (df_generated[true_output_] - df_generated[output_]) / df_generated[output_err_]
                    df_conditional_generated = pd.DataFrame({
                        condition: df_generated[condition],
                        f"residual band {bands[idx]}": diff_generated,
                        "dataset": ["gaNdalF" for _ in range(len(df_true[condition]))]
                    })
                    bin_means_generated, bin_edges_mean_generated, binnumber_mean_generated = binned_statistic(
                        df_generated[condition], diff_generated, statistic='median', bins=10, range=cond_lims)
                    bin_stds_generated, bin_edges_generated, binnumber_generated = binned_statistic(
                        df_generated[condition], diff_generated, statistic=median_abs_deviation, bins=10, range=cond_lims)
                    xerr_generated = (bin_edges_mean_generated[1:] - bin_edges_mean_generated[:-1]) / 2
                    xmean_generated = (bin_edges_mean_generated[1:] + bin_edges_mean_generated[:-1]) / 2
                    lst_axis_con[idx].errorbar(
                        xmean_generated, bin_means_generated, xerr=xerr_generated, yerr=bin_stds_generated, color='darkorange', lw=2, label='gaNdalF')
                    m, s = np.median(diff_generated), median_abs_deviation(diff_generated)
                    range_ = [m - 4 * s, m + 4 * s]

                    sns.kdeplot(
                        data=df_conditional_true,
                        x=condition,
                        y=f"residual band {bands[idx]}",
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
                        y=f"residual band {bands[idx]}",
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
                    lst_axis_con[idx].axhline(np.median(diff_generated), c='darkorange', ls='--', label='median gaNdalF')
                    lst_axis_con[idx].axvline(np.median(df_true[condition]), c='grey', ls='--', label='median conditional')
                lst_axis_con[0].legend()
                cond_figure.tight_layout()
                plt.savefig(f"{path_save_plots}/condition_{condition}_plot.png")
                plt.show()
                plt.clf()
                plt.close()


            except ValueError:
                print(f"Value Error for {condition}")

    if plot_hist is True:
        hist_figure_2, ((hist_ax1), (hist_ax2), (hist_ax3)) = \
            plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
        hist_figure_2.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        hist_figure_2.suptitle(r"magnitude histogram", fontsize=16)

        lst_axis_his = [
            hist_ax1,
            hist_ax2,
            hist_ax3
        ]

        lst_unsheared_bands = [
            "r",
            "i",
            "z"
        ]

        for ax_idx, his_ax in enumerate(lst_axis_his):
            sns.histplot(
                data=df_true,
                x=f"unsheared/mag_{lst_unsheared_bands[ax_idx]}",
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
                data=df_generated,
                x=f"unsheared/mag_{lst_unsheared_bands[ax_idx]}",
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
                x=df_true[f"unsheared/mag_{lst_unsheared_bands[ax_idx]}"].median(),
                color='dodgerblue',
                ls='--',
                lw=1.5,
                label="Mean Balrog"
            )
            his_ax.axvline(
                x=df_generated[f"unsheared/mag_{lst_unsheared_bands[ax_idx]}"].median(),
                color='darkorange',
                ls='--',
                lw=1.5,
                label="Mean gaNdalF"
            )
        plt.legend()
        plt.savefig(f"{path_save_plots}/mag_histogram_plot.png")
        plt.show()
        plt.clf()
        plt.close()

    return df_true, df_generated


def create_folders(path_save_plots, path_save_generated_data, run):
    """"""
    if not os.path.exists(path_save_generated_data):
        os.mkdir(path_save_generated_data)
    if not os.path.exists(path_save_plots):
        os.mkdir(path_save_plots)
    if run is not None:
        path_save_generated_data = f"{path_save_generated_data}/run_{run}"
        path_save_plots = f"{path_save_plots}/run_{run}"
        if not os.path.exists(path_save_generated_data):
            os.mkdir(path_save_generated_data)
        if not os.path.exists(path_save_plots):
            os.mkdir(path_save_plots)
    return path_save_plots, path_save_generated_data


def save_emulated_cat(path_save_generated_data, df_generated, df_true, run):
    # df_generated_sompz = pd.concat([df_generated], axis=1)
    # df_true_sompz = pd.concat([df_true], axis=1)
    with open(f"{path_save_generated_data}/df_gaNdalF_sompz_run_{run}.pkl", "wb") as f:
        pickle.dump(df_generated.to_dict(), f, protocol=2)
    with open(f"{path_save_generated_data}/df_balrog_nf_sompz_run_{run}.pkl", "wb") as f:
        pickle.dump(df_true.to_dict(), f, protocol=2)
    # df_data.to_pickle(path_save_catalog)


if __name__ == '__main__':
    path = os.path.abspath(sys.path[0])
    path_data = "/Users/P.Gebhardt/Development/PhD/data"
    path_output = "/Users/P.Gebhardt/Development/PhD/output/run_gaNdalF"
    path_nn = "/Users/P.Gebhardt/Development/PhD/trained_models"
    lst_conditions = [
        "BDF_MAG_DERED_CALIB_U",
        "BDF_MAG_DERED_CALIB_G",
        "BDF_MAG_DERED_CALIB_R",
        "BDF_MAG_DERED_CALIB_I",
        "BDF_MAG_DERED_CALIB_Z",
        "BDF_MAG_DERED_CALIB_J",
        "BDF_MAG_DERED_CALIB_H",
        "BDF_MAG_DERED_CALIB_K",
        "BDF_MAG_ERR_DERED_CALIB_U",
        "BDF_MAG_ERR_DERED_CALIB_G",
        "BDF_MAG_ERR_DERED_CALIB_R",
        "BDF_MAG_ERR_DERED_CALIB_I",
        "BDF_MAG_ERR_DERED_CALIB_Z",
        "BDF_MAG_ERR_DERED_CALIB_J",
        "BDF_MAG_ERR_DERED_CALIB_H",
        "BDF_MAG_ERR_DERED_CALIB_K",
        "Color Mag U-G",
        "Color Mag G-R",
        "Color Mag R-I",
        "Color Mag I-Z",
        "Color Mag Z-J",
        "Color Mag J-H",
        "Color Mag H-K",
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
    ]
    lst_bands = [
        "r",
        "i",
        "z"
    ]
    lst_colors = [
        ("r", "i"),
        ("i", "z")
    ]
    # for run_date in range(10):
    run = None

    with open(f"{path}/../files/conf/mac.cfg", 'r') as fp:
        cfg = yaml.load(fp, Loader=yaml.Loader)

    main(
        path_test_data=f"{path_data}/df_complete_data_250000.pkl",
        path_model=f"{path_nn}/last_model_des_epoch_10_run_2023-10-08_20-19.pt",
        path_save_plots=f"{path_output}/Plots",
        path_save_generated_data=f"{path_output}/Catalogs",
        run=run,
        number_samples=None,
        plot_chain=True,
        plot_residual=True,
        plot_luptize_conditions=True,
        plot_hist=True,
        conditions=lst_conditions,
        bands=lst_bands,
        colors=lst_colors,
        save_generated_data=True,
        cfg=cfg
    )

