from Handler.data_loader import load_data
from chainconsumer import ChainConsumer
from scipy.stats import binned_statistic, median_abs_deviation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
import sys
import pandas as pd


def load_model(path_model):
    model = torch.load(path_model)
    return model


def main(path_training_data, path_model, path_save_plots, number_samples, plot_chain, plot_residual,
         plot_luptize_conditions, conditions, bands, colors):
    col_label_flow = [
        "BDF_MAG_DERED_CALIB_R",
        "BDF_MAG_DERED_CALIB_I",
        "BDF_MAG_DERED_CALIB_Z",
        "Color Mag U-G",
        "Color Mag G-R",
        "Color Mag R-I",
        "Color Mag I-Z",
        "Color Mag Z-J",
        "Color Mag J-H",
        "Color Mag H-KS",
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
        "unsheared/snr",
        "unsheared/size_ratio",
        "unsheared/flags",
        "unsheared/T",
        "detected"
    ]
    print("Load data...")
    train_data, valid_data, test_data = load_data(
        path_training_data=path_training_data,
        input_flow=col_label_flow,
        output_flow=col_output_flow,
        selected_scaler="MaxAbsScaler"
    )
    scaler = test_data["scaler"]

    # Write data as torch loader
    test_tensor = torch.from_numpy(test_data[f"output flow in order {col_output_flow}"])
    test_labels = torch.from_numpy(test_data[f"label flow in order {col_label_flow}"])
    test_dataset = torch.utils.data.TensorDataset(test_tensor, test_labels)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=number_samples,
        shuffle=False,
        drop_last=False,
        # **kwargs
    )

    print("Load model...")
    model = load_model(path_model)
    for batch_idx, data in enumerate(test_loader):
        cond_data = data[1].float()
        with torch.no_grad():
            print("Sample data")
            test_output = model.sample(num_samples=number_samples, cond_inputs=cond_data).detach()
        print("Plot data")
        plot_data(
            path_save_plots=path_save_plots,
            cond_data=cond_data,
            test_output=test_output,
            col_label_flow=col_label_flow,
            col_output_flow=col_output_flow,
            scaler=scaler,
            data=data[0],
            plot_chain=plot_chain,
            plot_residual=plot_residual,
            plot_luptize_conditions=plot_luptize_conditions,
            conditions=conditions,
            bands=bands,
            colors=colors
        )
        break


def plot_data(path_save_plots, cond_data, test_output, col_label_flow, col_output_flow, scaler, data, plot_chain,
              plot_residual, plot_luptize_conditions, conditions, bands, colors):

    df_generator_label = pd.DataFrame(cond_data.numpy(), columns=col_label_flow)
    df_generator_output = pd.DataFrame(test_output.numpy(), columns=col_output_flow)
    df_generator_scaled = pd.concat([df_generator_label, df_generator_output], axis=1)
    generator_rescaled = scaler.inverse_transform(df_generator_scaled)
    df_generated = pd.DataFrame(generator_rescaled, columns=df_generator_scaled.columns)

    df_true_output = pd.DataFrame(data, columns=col_output_flow)
    df_true_scaled = pd.concat([df_generator_label, df_true_output], axis=1)
    true_rescaled = scaler.inverse_transform(df_true_scaled)
    df_true = pd.DataFrame(true_rescaled, columns=df_true_scaled.columns)

    if plot_chain is True:
        df_generated_measured = pd.DataFrame({})
        df_true_measured = pd.DataFrame({})
        for color in colors:
            df_generated_measured[f"{color[0]}-{color[1]}"] = \
                np.array(df_generated[f"unsheared/mag_{color[0]}"]) - np.array(df_generated[f"unsheared/mag_{color[1]}"])
            df_true_measured[f"{color[0]}-{color[1]}"] = \
                np.array(df_true[f"unsheared/mag_{color[0]}"]) - np.array(df_true[f"unsheared/mag_{color[1]}"])

        arr_true = df_true_measured.to_numpy()
        arr_generated = df_generated_measured.to_numpy()
        parameter = [
            "unsheared/mag r-i",
            "unsheared/mag i-z"
        ]
        chainchat = ChainConsumer()
        chainchat.add_chain(arr_true, parameters=parameter, name="true observed properties: chat")
        chainchat.add_chain(arr_generated, parameters=parameter, name="generated observed properties: chat*")
        chainchat.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
        chainchat.plotter.plot(
            filename=f'{path_save_plots}/chainplot.png',
            figsize="page",
            # extents={
            #     # "unsheared/mag u-g": (-4, 7),
            #     # "unsheared/mag g-r": (-1, 3),
            #     "unsheared/mag r-i": (-3, 3),
            #     "unsheared/mag i-z": (-3, 4),
            #     # "unsheared/mag Z-Y": (-3, 3),
            #     # "unsheared/mag Y-J": (-4, 4),
            #     # "unsheared/mag J-H": (-3, 3),
            #     # "unsheared/mag H-Ks": (-4, 4)
            # }
        )
        plt.savefig(f"{path_save_plots}/chain_color_plot.png")
        plt.show()
        plt.clf()
        plt.close()

    if plot_residual:
        hist_figure, ((stat_ax1), (stat_ax2), (stat_ax3)) = \
            plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
        hist_figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        hist_figure.suptitle(r"residual", fontsize=16)

        lst_axis_res = [
            stat_ax1,
            stat_ax2,
            stat_ax3
        ]

        lst_xlim_res = [
            (-2, 2),
            (-2, 2),
            (-2, 2),
            (-2, 2),
            (-2, 2),
            (-2, 2),
            (-2, 2),
            (-2, 2),
            (-2, 2)
        ]

        df_hist_skills = pd.DataFrame({
            "dataset": ["skillz" for _ in range(len(df_true[f"unsheared/mag_r"]))]
        })
        df_hist_generated = pd.DataFrame({
            "dataset": ["generated" for _ in range(len(df_true[f"unsheared/mag_r"]))]
        })
        for band in bands:
            df_hist_skills[f"BDF_MAG_DERED_CALIB - unsheared/mag {band}"] = df_true[f"BDF_MAG_DERED_CALIB_{band.upper()}"] - df_true[f"unsheared/mag_{band}"]
            df_hist_generated[f"BDF_MAG_DERED_CALIB - unsheared/mag {band}"] = df_true[f"BDF_MAG_DERED_CALIB_{band.upper()}"] - df_generated[f"unsheared/mag_{band}"]

        for idx, band in enumerate(bands):
            sns.histplot(
                data=df_hist_skills,
                x=f"BDF_MAG_DERED_CALIB - unsheared/mag {band}",
                ax=lst_axis_res[idx],
                element="step",
                stat="density",
                color="dodgerblue",
                bins=50,
                label="skills"
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
                label="generated"
            )
            lst_axis_res[idx].axvline(
                x=df_hist_skills[f"BDF_MAG_DERED_CALIB - unsheared/mag {band}"].median(),
                color='dodgerblue',
                ls='--',
                lw=1.5,
                label="Mean skills"
            )
            lst_axis_res[idx].axvline(
                x=df_hist_generated[f"BDF_MAG_DERED_CALIB - unsheared/mag {band}"].median(),
                color='darkorange',
                ls='--',
                lw=1.5,
                label="Mean generated"
            )
            # lst_axis_res[idx].set_xlim(lst_xlim_res[idx][0], lst_xlim_res[idx][1])
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
            cond_figure, (
            (stat_ax1), (stat_ax2), (stat_ax3)) = \
                plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
            cond_figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
            cond_figure.suptitle(f"BDF_MAG_DERED_CALIB - unsheared/mag", fontsize=16)

            outputs = ['unsheared/mag_' + b for b in bands]
            output_errs = ['luptize_err_' + b for b in bands]

            lst_axis_con = [
                stat_ax1,
                stat_ax2,
                stat_ax3
            ]

            cond_lims = np.percentile(df_true[condition], [2, 98])

            for idx, out in enumerate(zip(outputs, output_errs)):
                output_ = out[0]
                # Todo implement real survey error
                output_err_true = 1/df_true[output_] # out[1]
                output_err_generated = 1 / df_generated[output_]  # out[1]

                diff_true = (df_true['BDF_MAG_DERED_CALIB_R'] - df_true[output_]) / output_err_true
                df_conditional_true = pd.DataFrame({
                    condition: df_true[condition],
                    f"residual band {bands[idx]}": diff_true,
                    "dataset": ["skills" for _ in range(len(df_true[condition]))]
                })
                bin_means_true, bin_edges_mean_true, binnumber_true = binned_statistic(
                    df_true[condition], diff_true, statistic='median', bins=10, range=cond_lims)
                bin_stds_true, bin_edges_true, binnumber_true = binned_statistic(
                    df_true[condition], diff_true, statistic=median_abs_deviation, bins=10, range=cond_lims)
                xerr_true = (bin_edges_mean_true[1:] - bin_edges_mean_true[:-1]) / 2
                xmean_true = (bin_edges_mean_true[1:] + bin_edges_mean_true[:-1]) / 2
                lst_axis_con[idx].errorbar(
                    xmean_true, bin_means_true, xerr=xerr_true, yerr=bin_stds_true, color='dodgerblue', lw=2, label='skills')

                diff_generated = (df_generated['BDF_MAG_DERED_CALIB_R'] - df_generated[output_]) / output_err_generated
                df_conditional_generated = pd.DataFrame({
                    condition: df_generated[condition],
                    f"residual band {bands[idx]}": diff_generated,
                    "dataset": ["generated" for _ in range(len(df_true[condition]))]
                })
                bin_means_generated, bin_edges_mean_generated, binnumber_mean_generated = binned_statistic(
                    df_generated[condition], diff_generated, statistic='median', bins=10, range=cond_lims)
                bin_stds_generated, bin_edges_generated, binnumber_generated = binned_statistic(
                    df_generated[condition], diff_generated, statistic=median_abs_deviation, bins=10, range=cond_lims)
                xerr_generated = (bin_edges_mean_generated[1:] - bin_edges_mean_generated[:-1]) / 2
                xmean_generated = (bin_edges_mean_generated[1:] + bin_edges_mean_generated[:-1]) / 2
                lst_axis_con[idx].errorbar(
                    xmean_generated, bin_means_generated, xerr=xerr_generated, yerr=bin_stds_generated, color='darkorange', lw=2, label='generated')
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
                    legend="skills",
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
                    legend="generated",
                    ax=lst_axis_con[idx]
                )
                lst_axis_con[idx].set_xlim(cond_lims)
                lst_axis_con[idx].set_ylim(range_)
                lst_axis_con[idx].axhline(np.median(diff_true), c='dodgerblue', ls='--', label='median skills')
                lst_axis_con[idx].axhline(0, c='grey', ls='--', label='zero')
                lst_axis_con[idx].axhline(np.median(diff_generated), c='darkorange', ls='--', label='median generated')
                lst_axis_con[idx].axvline(np.median(df_true[condition]), c='grey', ls='--', label='median conditional')
            lst_axis_con[0].legend()
            cond_figure.tight_layout()
            plt.savefig(f"{path_save_plots}/condition_{condition}_plot.png")
            plt.show()
            plt.clf()
            plt.close()


if __name__ == '__main__':
    path = os.path.abspath(sys.path[0])
    lst_conditions = [
        "BDF_MAG_DERED_CALIB_R",
        "BDF_MAG_DERED_CALIB_I",
        "BDF_MAG_DERED_CALIB_Z",
        "Color Mag U-G",
        "Color Mag G-R",
        "Color Mag R-I",
        "Color Mag I-Z",
        "Color Mag Z-J",
        "Color Mag J-H",
        "Color Mag H-KS",
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
    main(
        path_training_data=f"{path}/../Data/Balrog_2_data_MAG_250000.pkl",
        path_model=f"{path}/../trained_models/best_model_DES_epoch_96.pt",
        path_save_plots=f"{path}/output_run_flow_DES",
        number_samples=15000,
        plot_chain=True,
        plot_residual=True,
        plot_luptize_conditions=True,
        conditions=lst_conditions,
        bands=lst_bands,
        colors=lst_colors
    )

