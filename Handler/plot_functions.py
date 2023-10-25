import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio
import os
from natsort import natsorted
from chainconsumer import ChainConsumer
from scipy.stats import gaussian_kde
from torchvision.transforms import ToTensor
from io import BytesIO
import corner
from Handler.helper_functions import string_to_tuple
import time
import matplotlib
# matplotlib.use('Agg')


def plot_to_tensor():
    buf = BytesIO()  # Ein Zwischenspeicher, um das Bild zu speichern
    plt.savefig(buf, format='png')  # Speichern Sie das Matplotlib-Bild in den Zwischenspeicher
    buf.seek(0)

    img = plt.imread(buf)  # Lesen Sie das gespeicherte Bild zurück
    img_t = ToTensor()(img)  # Konvertieren Sie das Bild in einen PyTorch Tensor
    return img_t

def plot_corner(data_frame, columns, labels, ranges=None, show_plot=False, save_plot=False, save_name=None):
    """"""
    data = data_frame[columns].values
    ndim = data.shape[1]

    fig, axes = plt.subplots(ndim, ndim, figsize=(18, 10))

    corner.corner(
        data,
        fig=fig,
        bins=20,
        range=ranges,
        color='#51a6fb',
        smooth=True,
        smooth1d=True,
        labels=labels,
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
        scale_hist=False,
        quantiles=[0.16, 0.5, 0.84],
        density=True,
        plot_datapoints=True,
        plot_density=False,
        plot_contours=True,
        fill_contours=True
    )

    # Manually adding a legend using Line2D
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='#51a6fb', lw=4, label='Balrog')]

    # Adjust labels and titles
    for i in range(ndim):
        ax = axes[i, i]

        ax.set_xticklabels(ax.get_xticks(), rotation=45)
        ax.set_yticklabels(ax.get_yticks(), rotation=45)

        # Titel mit Quantilen manuell hinzufügen
        legend_elements.append(
            Line2D([0], [0], color='#51a6fb', lw=0, label=f"mean {labels[i]} = {np.mean(data[:, i]):.2f}"))

    fig.legend(handles=legend_elements, loc='upper right', fontsize='x-large')
    fig.suptitle('Corner Plot', fontsize=16)

    img_tensor = plot_to_tensor()
    if show_plot is True:
        plt.show()
    if save_plot is True:
        plt.savefig(save_name, dpi=200)
    return img_tensor


def plot_compare_corner(data_frame_generated, data_frame_true, columns, labels, title, epoch, dict_delta, ranges=None,
                        show_plot=False, save_plot=False, save_name=None):
    if epoch == 1:
        for label in labels:
            dict_delta[f"delta mean {label}"] = []
            dict_delta[f"delta median {label}"] = []
            dict_delta[f"delta q16 {label}"] = []
            dict_delta[f"delta q84 {label}"] = []

    arr_generated = data_frame_generated[columns].values
    arr_true = data_frame_true[columns].values

    # Quantile für gandalf berechnen
    quantiles_gandalf = np.quantile(arr_generated, q=[0.16, 0.84], axis=0)

    # Quantile für balrog berechnen
    quantiles_balrog = np.quantile(arr_true, q=[0.16, 0.84], axis=0)

    delta_names = ["mean", "median", "q16", "q84"]

    ndim = arr_generated.shape[1]

    # fig, axes = plt.subplots(ndim, ndim, figsize=(16, 9))
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(ndim + 5, ndim, hspace=0.05, wspace=0.1)
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(ndim)] for i in range(ndim)])

    # Plot gandalf
    corner.corner(
        arr_generated,
        fig=fig,
        bins=20,
        range=ranges,
        color='#ff8c00',
        smooth=True,
        smooth1d=True,
        labels=labels,
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
        hist_kwargs={'alpha': 0},
        scale_hist=True,
        quantiles=[0.16, 0.5, 0.84],
        density=True,
        plot_datapoints=True,
        plot_density=False,
        plot_contours=True,
        fill_contours=True
    )

    # Plot balrog
    corner.corner(
        arr_true,
        fig=fig,
        bins=20,
        range=ranges,
        color='#51a6fb',
        smooth=True,
        smooth1d=True,
        labels=labels,
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
        hist_kwargs={'alpha': 0},
        scale_hist=True,
        quantiles=[0.16, 0.5, 0.84],
        density=True,
        plot_datapoints=True,
        plot_density=False,
        plot_contours=True,
        fill_contours=True
    )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#ff8c00', lw=4, label='gaNdalF'),
        Line2D([0], [0], color='#51a6fb', lw=4, label='Balrog')
    ]

    for i in range(ndim):
        ax = axes[i, i]
        sns.histplot(arr_generated[:, i], ax=ax, color='#ff8c00', stat='density', bins=20, alpha=0.5)
        sns.histplot(arr_true[:, i], ax=ax, color='#51a6fb', stat='density', bins=20, alpha=0.5)
        sns.kdeplot(arr_generated[:, i], ax=ax, color='#ff8c00', fill=True, levels=[0.16, 0.5, 0.84])
        sns.kdeplot(arr_true[:, i], ax=ax, color='#51a6fb', fill=True, levels=[0.16, 0.5, 0.84])
        # ax.set_xlim(ranges[i])  # Setzen Sie die Achsenlimits entsprechend Ihren ranges
        ax.set_yticklabels(ax.get_yticks(), rotation=45)

        # Titel mit Quantilen manuell hinzufügen
        delta_mean = np.mean(arr_generated[:, i]) - np.mean(arr_true[:, i])
        delta_median = np.median(arr_generated[:, i]) - np.median(arr_true[:, i])
        delta_q16 = quantiles_gandalf[0, i] - quantiles_balrog[0, i]
        delta_q84 = quantiles_gandalf[1, i] - quantiles_balrog[1, i]

        dict_delta[f"delta mean {labels[i]}"].append(delta_mean)
        dict_delta[f"delta median {labels[i]}"].append(delta_median)
        dict_delta[f"delta q16 {labels[i]}"].append(delta_q16)
        dict_delta[f"delta q84 {labels[i]}"].append(delta_q84)

    if ranges is not None:
        for i in range(ndim):
            for j in range(ndim):
                ax = axes[i, j]
                ax.set_xlim(ranges[j])
                ax.set_ylim(ranges[i])

    fig.suptitle(f'{title}, epoch {epoch}', fontsize=16)

    delta_legend_elements = []
    epochs = list(range(1, epoch + 1))
    for idx, delta_name in enumerate(delta_names):
        delta_ax = fig.add_subplot(gs[ndim + 1 + idx, :])
        for i, label in enumerate(labels):
            line, = delta_ax.plot(epochs, dict_delta[f"delta {delta_name} {label}"], '-o',
                                  label=f"delta {label}")
            if idx == 0:
                delta_legend_elements.append(line)

        delta_ax.axhline(y=0, color='gray', linestyle='--')
        delta_ax.set_ylim(-0.05, 0.05)

        if idx == len(delta_names) - 1:
            delta_ax.set_xlabel('Epoch')
        else:
            delta_ax.set_xticklabels([])

        delta_ax.set_ylabel(f'Delta {delta_name}')

    fig.legend(handles=legend_elements + delta_legend_elements, loc='upper right', fontsize=12)
    img_tensor = plot_to_tensor()
    if show_plot is True:
        plt.show()
    if save_plot is True:
        plt.savefig(save_name, dpi=200)
    fig.clf()
    plt.close(fig)
    return img_tensor, dict_delta


# def plot_chain(data_frame, plot_name, max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12, columns=None,
#                parameter=None, extends=None, show_plot=False):
#     """
#
#     :param extends: extents={
#                 "mag r": (17.5, 26),
#                 "mag i": (17.5, 26),
#                 "mag z": (17.5, 26),
#                 "snr": (-11, 55),
#                 "size ratio": (-1.5, 4),
#                 "T": (-1, 2.5)
#             }
#     :param label_font_size:
#     :param tick_font_size:
#     :param shade_alpha:
#     :param max_ticks:
#     :param plot_name: "generated observed properties: chat*"
#     :param data_frame:
#     :param columns: Mutable list, default values are columns = [
#             "unsheared/mag_r",
#             "unsheared/mag_i",
#             "unsheared/mag_z",
#             "unsheared/snr",
#             "unsheared/size_ratio",
#             "unsheared/T"
#         ]
#     :param parameter: Mutable list, default values are labels = [
#                 "mag r",
#                 "mag i",
#                 "mag z",
#                 "snr",              # signal-noise      Range: min=0.3795, max=38924.4662
#                 "size ratio",       # T/psfrec_T        Range: min=-0.8636, max=4346136.5645
#                 "T"                 # T=<x^2>+<y^2>     Range: min=-0.6693, max=1430981.5103
#             ]
#     :return:
#     """
#     df_plot = pd.DataFrame({})
#
#     if columns is None:
#         columns = [
#             "unsheared/mag_r",
#             "unsheared/mag_i",
#             "unsheared/mag_z",
#             "unsheared/snr",
#             "unsheared/size_ratio",
#             "unsheared/T"
#         ]
#
#     if parameter is None:
#         parameter = [
#                 "mag r",
#                 "mag i",
#                 "mag z",
#                 "snr",              # signal-noise      Range: min=0.3795, max=38924.4662
#                 "size ratio",       # T/psfrec_T        Range: min=-0.8636, max=4346136.5645
#                 "T"                 # T=<x^2>+<y^2>     Range: min=-0.6693, max=1430981.5103
#             ]
#
#     for col in columns:
#         df_plot[col] = np.array(data_frame[col])
#
#     chain = ChainConsumer()
#     chain.add_chain(df_plot.to_numpy(), parameters=parameter, name=plot_name)
#     chain.configure(
#         max_ticks=max_ticks,
#         shade_alpha=shade_alpha,
#         tick_font_size=tick_font_size,
#         label_font_size=label_font_size
#     )
#     # if extends is not None:
#     chain.plotter.plot(
#         figsize="page",
#         extents=extends
#     )
#     img_tensor = plot_to_tensor()
#     if show_plot is True:
#         plt.show()
#     plt.clf()
#     plt.close()
#     return img_tensor


def loss_plot(
        epoch,
        lst_train_loss_per_batch,
        lst_train_loss_per_epoch,
        lst_valid_loss_per_batch,
        lst_valid_loss_per_epoch,
        show_plot,
        save_plot,
        save_name
):
    statistical_figure, (stat_ax1, stat_ax2, stat_ax3) = plt.subplots(nrows=3, ncols=1)
    statistical_figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)
    statistical_figure.suptitle(f"Epoch: {epoch+1}", fontsize=16)

    # Create dataframe of progress list
    df_training_loss_per_batch = pd.DataFrame({
        "training loss": lst_train_loss_per_batch
    })
    df_training_loss_per_epoch = pd.DataFrame({
        "training loss": lst_train_loss_per_epoch
    })
    df_valid_loss_per_batch = pd.DataFrame({
        "validation loss": lst_valid_loss_per_batch
    })
    df_valid_loss_per_epoch = pd.DataFrame({
        "validation loss": lst_valid_loss_per_epoch
    })

    # Create plot
    df_training_loss_per_batch.plot(
        figsize=(16, 9),
        alpha=0.5,
        marker=".",
        grid=True,
        # yticks=(0, 0.25, 0.5, 0.69, 1.0, 5.0),
        ax=stat_ax1)

    stat_ax1.set_xlabel("batch", fontsize=10, loc='right')
    stat_ax1.set_ylabel("loss", fontsize=12, loc='top')
    stat_ax1.set_title(f"Loss per batch")

    df_valid_loss_per_batch.plot(
        figsize=(16, 9),
        alpha=0.5,
        marker=".",
        grid=True,
        # yticks=(0, 0.25, 0.5, 0.69, 1.0, 5.0),
        ax=stat_ax2)

    stat_ax2.set_xlabel("batch", fontsize=10, loc='right')
    stat_ax2.set_ylabel("loss", fontsize=12, loc='top')
    stat_ax2.set_title(f"Loss per batch")

    df_training_loss_per_epoch.plot(
        figsize=(16, 9),
        alpha=0.5,
        marker=".",
        grid=True,
        # yticks=(0, 0.25, 0.5, 0.69, 1.0, 5.0),
        ax=stat_ax3)

    stat_ax3.set_xlabel("epoch", fontsize=10, loc='right')
    stat_ax3.set_ylabel("loss", fontsize=12, loc='top')
    stat_ax3.set_title(f"Loss per epoch")

    df_valid_loss_per_epoch.plot(
        figsize=(16, 9),
        alpha=0.5,
        marker=".",
        grid=True,
        # yticks=(0, 0.25, 0.5, 0.69, 1.0, 5.0),
        ax=stat_ax3)

    stat_ax3.set_xlabel("epoch", fontsize=10, loc='right')
    stat_ax3.set_ylabel("loss", fontsize=12, loc='top')
    stat_ax3.set_title(f"Loss per epoch")

    if show_plot is True:
        statistical_figure.show()
    if save_plot is True:
        statistical_figure.savefig(f"{save_name}", dpi=200)

    # Clear and close open figure to avoid memory overload
    img_tensor = plot_to_tensor()
    statistical_figure.clf()
    plt.clf()
    plt.close(statistical_figure)
    return img_tensor


# def color_color_plot(data_frame_generated, luminosity_type, data_frame_true, colors, show_plot, save_name, extents=None):
#     """"""
#     df_generated_measured = pd.DataFrame({})
#     df_true_measured = pd.DataFrame({})
#     for color in colors:
#         df_generated_measured[f"{color[0]}-{color[1]}"] = \
#             np.array(data_frame_generated[f"unsheared/{luminosity_type.lower()}_{color[0]}"]) - np.array(
#                 data_frame_generated[f"unsheared/{luminosity_type.lower()}_{color[1]}"])
#         df_true_measured[f"{color[0]}-{color[1]}"] = \
#             np.array(data_frame_true[f"unsheared/{luminosity_type.lower()}_{color[0]}"]) - np.array(
#                 data_frame_true[f"unsheared/{luminosity_type.lower()}_{color[1]}"])
#
#     arr_true = df_true_measured.to_numpy()
#     arr_generated = df_generated_measured.to_numpy()
#     labels = [
#         f"unsheared/{luminosity_type.lower()} r-i",
#         f"unsheared/{luminosity_type.lower()} i-z"
#     ]
#     chainchat = ChainConsumer()
#     chainchat.add_chain(arr_true, parameters=labels, name="true observed properties: chat")
#     chainchat.add_chain(arr_generated, parameters=labels,
#                         name="generated observed properties: chat*")
#     chainchat.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
#     chainchat.plotter.plot(
#         filename=save_name,
#         figsize="page",
#         extents=extents
#     )
#     if show_plot is True:
#         plt.show()
#     img_tensor = plot_to_tensor()
#     plt.clf()
#     plt.close()
#     return img_tensor


# def color_color_plot(data_frame_generated, luminosity_type, data_frame_true, colors, show_plot, save_name, save_plot, extents=None):
#     """"""
#     df_generated_measured = pd.DataFrame({})
#     df_true_measured = pd.DataFrame({})
#     for color in string_to_tuple(str(colors)):
#         df_generated_measured[f"{color[0]}-{color[1]}"] = \
#             np.array(data_frame_generated[f"unsheared/{luminosity_type.lower()}_{color[0]}"]) - np.array(
#                 data_frame_generated[f"unsheared/{luminosity_type.lower()}_{color[1]}"])
#         df_true_measured[f"{color[0]}-{color[1]}"] = \
#             np.array(data_frame_true[f"unsheared/{luminosity_type.lower()}_{color[0]}"]) - np.array(
#                 data_frame_true[f"unsheared/{luminosity_type.lower()}_{color[1]}"])
#
#     arr_true = df_true_measured.to_numpy()
#     arr_generated = df_generated_measured.to_numpy()
#     labels = [
#         f"unsheared/{luminosity_type.lower()} r-i",
#         f"unsheared/{luminosity_type.lower()} i-z"
#     ]
#     ranges = None
#     ndim = arr_generated.shape[1]
#
#     fig, axes = plt.subplots(ndim, ndim, figsize=(16, 9))
#
#     # Plot data_1
#     corner.corner(
#         arr_generated,
#         fig=fig,
#         bins=20,
#         range=ranges,
#         color='#ff8c00',
#         smooth=True,
#         smooth1d=True,
#         labels=labels,
#         show_titles=True,
#         title_fmt=".2f",
#         title_kwargs={"fontsize": 12},
#         scale_hist=False,
#         quantiles=[0.16, 0.5, 0.84],
#         density=True,
#         plot_datapoints=True,
#         plot_density=False,
#         plot_contours=True,
#         fill_contours=True,
#         label="gaNdalF"
#     )
#
#     # Plot data_2
#     corner.corner(
#         arr_true,
#         fig=fig,
#         bins=20,
#         range=ranges,
#         color='#51a6fb',
#         smooth=True,
#         smooth1d=True,
#         labels=labels,
#         show_titles=True,
#         title_fmt=".2f",
#         title_kwargs={"fontsize": 12},
#         scale_hist=False,
#         quantiles=[0.16, 0.5, 0.84],
#         density=True,
#         plot_datapoints=True,
#         plot_density=False,
#         plot_contours=True,
#         fill_contours=True,
#         label="Balrog"
#     )
#     if show_plot is True:
#         plt.show()
#     if save_plot is True:
#         plt.savefig(save_name, dpi=200)
#     img_tensor = plot_to_tensor()
#     plt.clf()
#     plt.close()
#     return img_tensor


def residual_plot(data_frame_generated, data_frame_true, luminosity_type, bands, plot_title, show_plot, save_plot, save_name):
    """"""
    hist_figure, ((stat_ax1), (stat_ax2), (stat_ax3)) = \
        plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
    hist_figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    hist_figure.suptitle(plot_title, fontsize=16)

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

    df_hist_balrog = pd.DataFrame({
        "dataset": ["balrog" for _ in range(len(data_frame_true[f"unsheared/{luminosity_type.lower()}_r"]))]
    })
    df_hist_generated = pd.DataFrame({
        "dataset": ["generated" for _ in range(len(data_frame_generated[f"unsheared/{luminosity_type.lower()}_r"]))]
    })
    for band in bands:
        df_hist_balrog[f"BDF_{luminosity_type.upper()}_DERED_CALIB - unsheared/{luminosity_type.lower()} {band}"] = data_frame_true[
                                                                              f"BDF_{luminosity_type.upper()}_DERED_CALIB_{band.upper()}"] - \
                                                                          data_frame_true[f"unsheared/{luminosity_type.lower()}_{band}"]
        df_hist_generated[f"BDF_{luminosity_type.upper()}_DERED_CALIB - unsheared/{luminosity_type.lower()} {band}"] = data_frame_generated[
                                                                                 f"BDF_{luminosity_type.upper()}_DERED_CALIB_{band.upper()}"] - \
                                                                             data_frame_generated[
                                                                                 f"unsheared/{luminosity_type.lower()}_{band}"]

    for idx, band in enumerate(bands):
        sns.histplot(
            data=df_hist_balrog,
            x=f"BDF_{luminosity_type.upper()}_DERED_CALIB - unsheared/{luminosity_type.lower()} {band}",
            ax=lst_axis_res[idx],
            element="step",
            stat="density",
            color="dodgerblue",
            bins=50,
            label="balrog"
        )
        sns.histplot(
            data=df_hist_generated,
            x=f"BDF_{luminosity_type.upper()}_DERED_CALIB - unsheared/{luminosity_type.lower()} {band}",
            ax=lst_axis_res[idx],
            element="step",
            stat="density",
            color="darkorange",
            fill=False,
            bins=50,
            label="generated"
        )
        lst_axis_res[idx].axvline(
            x=df_hist_balrog[f"BDF_{luminosity_type.upper()}_DERED_CALIB - unsheared/{luminosity_type.lower()} {band}"].median(),
            color='dodgerblue',
            ls='--',
            lw=1.5,
            label="Mean balrog"
        )
        lst_axis_res[idx].axvline(
            x=df_hist_generated[f"BDF_{luminosity_type.upper()}_DERED_CALIB - unsheared/{luminosity_type.lower()} {band}"].median(),
            color='darkorange',
            ls='--',
            lw=1.5,
            label="Mean generated"
        )
        lst_axis_res[idx].set_xlim(lst_xlim_res[idx][0], lst_xlim_res[idx][1])
        if idx == 0:
            lst_axis_res[idx].legend()
        else:
            lst_axis_res[idx].legend([], [], frameon=False)
    hist_figure.tight_layout()
    if show_plot is True:
        plt.show()

    if save_plot is True:
        plt.savefig(save_name)
    img_tensor = plot_to_tensor()
    plt.clf()
    plt.close()
    return img_tensor


# def plot_chain_compare(data_frame_generated, data_frame_true, epoch, show_plot, save_name, columns=None, labels=None,
#                        extends=None, max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12):
#     """"""
#     if columns is None:
#         columns = [
#             "unsheared/mag_r",
#             "unsheared/mag_i",
#             "unsheared/mag_z",
#             "unsheared/snr",
#             "unsheared/size_ratio",
#             "unsheared/T"
#         ]
#
#     if labels is None:
#         labels = [
#             "mag r",
#             "mag i",
#             "mag z",
#             "snr",  # signal-noise      Range: min=0.3795, max=38924.4662
#             "size ratio",  # T/psfrec_T        Range: min=-0.8636, max=4346136.5645
#             "T"  # T=<x^2>+<y^2>     Range: min=-0.6693, max=1430981.5103
#         ]
#
#     df_plot_generated = data_frame_generated[columns]
#     df_plot_true = data_frame_true[columns]
#
#     chainchat = ChainConsumer()
#     chainchat.add_chain(df_plot_true.to_numpy(), parameters=labels, name="balrog observed properties: chat")
#     chainchat.add_chain(df_plot_generated.to_numpy(), parameters=labels, name="generated observed properties: chat*")
#     chainchat.configure(
#         max_ticks=max_ticks,
#         shade_alpha=shade_alpha,
#         tick_font_size=tick_font_size,
#         label_font_size=label_font_size
#     )
#     try:
#         chainchat.plotter.plot(
#             filename=save_name,
#             figsize="page",
#             extents=extends
#         )
#     except:
#         print("chain error at epoch", epoch + 1)
#     if show_plot is True:
#         plt.show()
#     img_tensor = plot_to_tensor()
#     plt.clf()
#     plt.close()
#     return img_tensor


def plot_mean_or_std(data_frame_generated, data_frame_true, lists_to_plot, list_epochs, columns, lst_labels, lst_marker,
                     lst_color, plot_title, show_plot, save_plot, save_name, statistic_type="mean"):
    """"""
    y_label = ""
    for idx_col, col in enumerate(columns):
        if statistic_type.upper() == "MEAN":
            lists_to_plot[idx_col].append(data_frame_generated[col].mean() / data_frame_true[col].mean())
            y_label = "mean(chat*) / mean(chat)"
        elif statistic_type.upper() == "STD":
            lists_to_plot[idx_col].append(data_frame_generated[col].std() / data_frame_true[col].std())
            y_label = "std(chat*) / std(chat)"

    for idx_col, col in enumerate(columns):
        plt.plot(
            list_epochs,
            lists_to_plot[idx_col],
            marker=lst_marker[idx_col],
            linestyle='-',
            color=lst_color[idx_col],
            label=lst_labels[idx_col]
        )
    plt.legend()
    plt.title(plot_title)
    plt.xlabel("epoch")
    plt.ylabel(y_label)

    if show_plot is True:
        plt.show()
    if save_plot is True:
        plt.savefig(save_name, dpi=200)
    img_tensor = plot_to_tensor()
    plt.clf()
    plt.close()
    return lists_to_plot, img_tensor


def plot_2d_kde(x, y, manual_levels, limits=None, x_label="", y_label="", title="", color=None):
    """"""

    xmin = -2
    xmax = 2
    ymin = -2
    ymax = 2

    if color is None:
        color = "Blues"

    if limits is not None:
        xmin = limits[0]
        xmax = limits[1]
        ymin = limits[2]
        ymax = limits[3]

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)

    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Contourf plot
    cfset = ax.contourf(xx, yy, f, cmap=color, alpha=0.3, levels=manual_levels)

    # Or kernel density estimate plot instead of the contourf plot
    # ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])

    # Contour plot
    cset = ax.contour(xx, yy, f, colors='k', levels=manual_levels)

    # Label plot

    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(title)
    img_tensor = plot_to_tensor()
    plt.clf()
    plt.close()
    return img_tensor


def plot_2d_kde_compare(x1, y1, x2, y2, manual_levels, limits=None, x_label="", y_label="", title="", color=None):
    """"""

    xmin = -2
    xmax = 2
    ymin = -2
    ymax = 2

    if color is None:
        color = "Blues"

    if limits is not None:
        xmin = limits[0]
        xmax = limits[1]
        ymin = limits[2]
        ymax = limits[3]

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values1 = np.vstack([x1, y1])
    kernel1 = gaussian_kde(values1)
    f1 = np.reshape(kernel1(positions).T, xx.shape)

    values2 = np.vstack([x2, y2])
    kernel2 = gaussian_kde(values2)
    f2 = np.reshape(kernel2(positions).T, xx.shape)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # manual_levels = np.array([0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6])

    # Contourf plot
    ax.contourf(xx, yy, f1, cmap=color[0], alpha=0.3, levels=manual_levels)
    ax.contourf(xx, yy, f2, cmap=color[1], alpha=0.3, levels=manual_levels)

    # Or kernel density estimate plot instead of the contourf plot
    # ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])

    # Contour plot
    cset1 = ax.contour(xx, yy, f1, colors='k', levels=manual_levels)
    cset2 = ax.contour(xx, yy, f2, colors='k', levels=manual_levels)

    # Label plot
    ax.clabel(cset1, inline=1, fontsize=10)
    ax.clabel(cset2, inline=1, fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(title)
    img_tensor = plot_to_tensor()
    plt.clf()
    plt.close()
    return img_tensor


def make_gif(frame_folder, name_save_folder, fps=10):
    filenames = natsorted(os.listdir(frame_folder))
    images_data = []
    for filename in filenames:
        image = imageio.imread(f"{frame_folder}/{filename}")
        images_data.append(image)
    imageio.mimwrite(uri=f"{name_save_folder}", ims=images_data, format='.gif', duration=int(1000*1/fps))
