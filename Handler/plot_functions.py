import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio
import os
from natsort import natsorted
# from chainconsumer import ChainConsumer, Chain, PlotConfig, ChainConfig
from scipy.stats import gaussian_kde
from torchvision.transforms import ToTensor
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, brier_score_loss
from sklearn.calibration import calibration_curve
from io import BytesIO

from Handler.helper_functions import string_to_tuple
import time
import matplotlib
# matplotlib.use('Agg')
plt.style.use('seaborn-white')

def plot_to_tensor():
    buf = BytesIO()  # Ein Zwischenspeicher, um das Bild zu speichern
    plt.savefig(buf, format='png')  # Speichern Sie das Matplotlib-Bild in den Zwischenspeicher
    buf.seek(0)

    img = plt.imread(buf)  # Lesen Sie das gespeicherte Bild zurück
    img_t = ToTensor()(img)  # Konvertieren Sie das Bild in einen PyTorch Tensor
    return img_t

def plot_corner(data_frame, columns, labels, ranges=None, show_plot=False, save_plot=False, save_name=None):
    """"""
    import corner
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
    import corner
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

    if dict_delta is None:
        fig, axes = plt.subplots(ndim, ndim, figsize=(16, 9))
    else:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(ndim + 5, ndim, hspace=0.05, wspace=0.1)
        axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(ndim)] for i in range(ndim)])

    # Plot gandalf
    corner.corner(
        arr_generated,
        fig=fig,
        bins=100,
        range=ranges,
        color='#ff8c00',
        smooth=.8,
        smooth1d=.8,
        labels=labels,
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
        hist_kwargs={'alpha': 1},
        scale_hist=True,
        quantiles=[0.16, 0.5, 0.84],
        levels=[0.393, 0.865, 0.989],  #, 0.989
        density=True,
        plot_datapoints=True,
        plot_density=True,
        plot_contours=True,
        fill_contours=True
    )

    # Plot balrog
    corner.corner(
        arr_true,
        fig=fig,
        bins=100,
        range=ranges,
        color='#51a6fb',
        smooth=.8,
        smooth1d=.8,
        labels=labels,
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
        hist_kwargs={'alpha': 1},
        scale_hist=True,
        quantiles=[0.16, 0.5, 0.84],
        levels=[0.393, 0.865, 0.989],  #, 0.989
        density=True,
        plot_datapoints=True,
        plot_density=True,
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
        ax.set_yticklabels(ax.get_yticks(), rotation=45)

        # Titel mit Quantilen manuell hinzufügen
        delta_mean = np.mean(arr_generated[:, i]) - np.mean(arr_true[:, i])
        delta_median = np.median(arr_generated[:, i]) - np.median(arr_true[:, i])
        delta_q16 = quantiles_gandalf[0, i] - quantiles_balrog[0, i]
        delta_q84 = quantiles_gandalf[1, i] - quantiles_balrog[1, i]

        if dict_delta is not None:
            dict_delta[f"delta mean {labels[i]}"].append(delta_mean)
            dict_delta[f"delta median {labels[i]}"].append(delta_median)
            dict_delta[f"delta q16 {labels[i]}"].append(delta_q16)
            dict_delta[f"delta q84 {labels[i]}"].append(delta_q84)
        else:
            legend_elements.append(Line2D([0], [0], color='#ff8c00', lw=0, label=f'{labels[i]}: $\Delta$ mean = {np.abs(delta_mean):.5f}'), )
            # legend_elements.append(Line2D([0], [0], color='#ff8c00', lw=0, label=f'{labels[i]}: median={delta_median:.5f}'), )
            # legend_elements.append(Line2D([0], [0], color='#ff8c00', lw=0, label=f'{labels[i]}: q16={delta_q16:.5f}'), )
            # legend_elements.append(Line2D([0], [0], color='#ff8c00', lw=0, label=f'{labels[i]}: q84={delta_q84:.5f}'), )

    if epoch is not None:
        fig.suptitle(f'{title}, epoch {epoch}', fontsize=20)
    else:
        fig.suptitle(f'{title}', fontsize=20)

    if dict_delta is not None:
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
    else:
        fig.legend(handles=legend_elements, loc='upper right', fontsize=16)

    img_tensor = plot_to_tensor()
    if show_plot is True:
        plt.show()
    if save_plot is True:
        plt.savefig(save_name, dpi=300)
    plt.clf()
    plt.close(fig)
    return img_tensor, dict_delta


# def plot_chain(df_balrog, plot_name, max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12, columns=None,
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
#     :param df_balrog:
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
#         df_plot[col] = np.array(df_balrog[col])
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
#
#
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


# def plot_chain_compare(data_frame_generated, data_frame_true, columns, labels, title, extents, sigma2d, show_plot, save_name):
#     """"""
#
#     chainchat = ChainConsumer()
#     balrog_chain = Chain(
#         samples=data_frame_true[columns],
#         parameters=labels,
#         name="balrog observed properties: chat",
#         color='#51a6fb',
#         plot_cloud=True,
#         num_cloud=100000,
#         kde=.8,
#         bins=100,
#         smooth=1,
#         shade=True,
#         shade_alpha=.7,
#         show_contour_labels=True
#     )
#     gandalf_chain = Chain(
#         samples=data_frame_generated[columns],
#         parameters=labels,
#         name="generated observed properties: chat*",
#         color='#ff8c00',
#         plot_cloud=True,
#         num_cloud=100000,
#         kde=.8,
#         bins=100,
#         smooth=1,
#         shade=True,
#         shade_alpha=.7,
#         show_contour_labels=True
#     )
#     chainchat.add_chain(balrog_chain)
#     chainchat.add_chain(gandalf_chain)
#     chainchat.set_plot_config(PlotConfig(
#         max_ticks=5,
#         shade_alpha=0.8,
#         tick_font_size=12,
#         label_font_size=12,
#         sigma2d=sigma2d,
#         flip=True,
#         show_legend=True,
#         extents=extents
#     ))
#     chainchat.plotter.plot(
#         filename=save_name,
#         figsize="page"
#     )
#     plt.grid(True)
#     plt.title(title)
#     if show_plot is True:
#         plt.show()
#     plt.clf()
#     plt.close()


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


def plot_classification_results(data_frame, cols, show_plot, save_plot, save_name, title='Classification Results'):
    true_galaxy_values = data_frame[cols].values

    true_positives = (data_frame['true_detected'] == 1) & (data_frame['detected_true'] == 1)
    true_negatives = (data_frame['true_detected'] == 0) & (data_frame['detected_true'] == 0)
    false_positives = (data_frame['true_detected'] == 1) & (data_frame['detected_true'] == 0)
    false_negatives = (data_frame['true_detected'] == 0) & (data_frame['detected_true'] == 1)
    colors = np.empty(true_galaxy_values.shape[0], dtype='object')
    colors[true_positives] = 'True Positives'
    colors[true_negatives] = 'True Negatives'
    colors[false_positives] = 'False Positives'
    colors[false_negatives] = 'False Negatives'

    true_positives_calibrated = (data_frame['detected_calibrated'] == 1) & (data_frame['detected_true'] == 1)
    true_negatives_calibrated = (data_frame['detected_calibrated'] == 0) & (data_frame['detected_true'] == 0)
    false_positives_calibrated = (data_frame['detected_calibrated'] == 1) & (data_frame['detected_true'] == 0)
    false_negatives_calibrated = (data_frame['detected_calibrated'] == 0) & (data_frame['detected_true'] == 1)
    colors_calibrated = np.empty(true_galaxy_values.shape[0], dtype='object')
    colors_calibrated[true_positives_calibrated] = 'True Positives'
    colors_calibrated[true_negatives_calibrated] = 'True Negatives'
    colors_calibrated[false_positives_calibrated] = 'False Positives'
    colors_calibrated[false_negatives_calibrated] = 'False Negatives'

    data = pd.DataFrame({
        cols[0]: true_galaxy_values[:, 0],
        cols[1]: true_galaxy_values[:, 1],
        'Classification': colors
    })

    data_calibrated = pd.DataFrame({
        cols[0]: true_galaxy_values[:, 0],
        cols[1]: true_galaxy_values[:, 1],
        'Classification': colors_calibrated
    })

    data_errors = data[data['Classification'].str.contains('False')]
    data_calibrated_errors = data_calibrated[data_calibrated['Classification'].str.contains('False')]

    color_dict = {'True Positives': 'green', 'True Negatives': 'blue', 'False Positives': 'red',
                  'False Negatives': 'purple'}

    fig_classf, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    sns.scatterplot(
        data=data,
        x=cols[0],
        y=cols[1],
        hue='Classification',
        palette=color_dict,
        alpha=0.5,
        ax=ax1
    )
    sns.scatterplot(
        data=data_calibrated,
        x=cols[0],
        y=cols[1],
        hue='Classification',
        palette=color_dict,
        alpha=0.5,
        ax=ax2
    )
    sns.scatterplot(
        data=data_errors,
        x=cols[0],
        y=cols[1],
        hue='Classification',
        palette=color_dict,
        alpha=0.5,
        ax=ax3
    )
    sns.scatterplot(
        data=data_calibrated_errors,
        x=cols[0],
        y=cols[1],
        hue='Classification',
        palette=color_dict,
        alpha=0.5,
        ax=ax4
    )

    fig_classf.suptitle(title)
    fig_classf.subplots_adjust(right=0.85)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)

    ax2.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
    ax1.get_legend().remove()
    ax3.get_legend().remove()
    ax4.get_legend().remove()

    if show_plot is True:
        plt.show()
    if save_plot is True:
        plt.savefig(save_name, dpi=200)
    plt.clf()
    plt.close(fig_classf)


def plot_confusion_matrix(data_frame, show_plot, save_plot, save_name, title='Confusion matrix'):
    """"""
    matrix = confusion_matrix(
        data_frame['detected_true'].ravel(),
        data_frame['true_detected'].ravel()
    )
    df_cm = pd.DataFrame(matrix, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"])

    matrix_calibrated = confusion_matrix(
        data_frame['detected_true'].ravel(),
        data_frame['detected_calibrated'].ravel()
    )
    df_cm_cali = pd.DataFrame(matrix_calibrated, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"])

    fig_matrix, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt="g", ax=ax1)
    sns.heatmap(df_cm_cali, annot=True, fmt="g", ax=ax2)
    plt.title(title)
    if show_plot is True:
        plt.show()
    if save_plot is True:
        plt.savefig(save_name, dpi=200)
    plt.clf()
    plt.close(fig_matrix)


def plot_calibration_curve_gandalf(true_detected, probability, n_bins=10, show_plot=True, save_plot=False,
                                   save_name="calibration_curve.png", title='Calibration Curve'):
    """
    Plot a calibration curve for the given data.
    """
    y_true = true_detected.to_numpy()
    y_prob = probability.to_numpy()
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # Calculate Brier Score
    brier = brier_score_loss(y_true, y_prob)

    # Calculate ECE
    ece = np.sum(np.abs(prob_true - prob_pred) * (np.histogram(y_prob, bins=n_bins, range=(0, 1))[0] / len(y_prob)))

    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label=f'Calibration plot (Brier={brier:.4f}, ECE={ece:.4f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(title)
    ax.legend(loc='best')
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(save_name, dpi=300)
    plt.clf()
    plt.close(fig)


def plot_confusion_matrix_gandalf(df_balrog, df_gandalf, show_plot, save_plot, save_name, title='Confusion matrix'):
    """"""
    matrix = confusion_matrix(
        df_balrog['detected'].to_numpy(),
        df_gandalf['detected'].to_numpy()
    )
    df_cm = pd.DataFrame(matrix, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"])

    fig_matrix, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt="g", ax=ax1)
    plt.title(title)
    if show_plot is True:
        plt.show()
    if save_plot is True:
        plt.savefig(save_name, dpi=200)
    plt.clf()
    plt.close(fig_matrix)


def plot_roc_curve(data_frame, show_plot, save_plot, save_name, title='Receiver Operating Characteristic (ROC) Curve'):
    """"""
    fpr, tpr, thresholds = roc_curve(
        data_frame['detected_true'].ravel(),
        data_frame['true_detected'].ravel()
    )
    roc_auc = auc(fpr, tpr)
    fpr_calib, tpr_calib, thresholds_calib = roc_curve(
        data_frame['detected_true'].ravel(),
        data_frame['detected_calibrated'].ravel()
    )
    roc_auc_calib = auc(fpr_calib, tpr_calib)

    fig_roc_curve = plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.plot(fpr_calib, tpr_calib, color='darkgreen', lw=2, label=f'ROC curve calibrated (area = {roc_auc_calib:.2f})')
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    if show_plot is True:
        plt.show()
    if save_plot is True:
        plt.savefig(save_name, dpi=200)
    plt.clf()
    plt.close(fig_roc_curve)


def plot_roc_curve_gandalf(df_balrog, df_gandalf, show_plot, save_plot, save_name, title='Receiver Operating Characteristic (ROC) Curve'):
    """"""
    fpr, tpr, thresholds = roc_curve(
        df_balrog['detected'].to_numpy(),
        df_gandalf['detected'].to_numpy()
    )
    roc_auc = auc(fpr, tpr)

    fig_roc_curve = plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    if show_plot is True:
        plt.show()
    if save_plot is True:
        plt.savefig(save_name, dpi=200)
    plt.clf()
    plt.close(fig_roc_curve)


def plot_recall_curve(data_frame, show_plot, save_plot, save_name, title='Precision-Recall Curve'):
    """"""
    precision, recall, thresholds = precision_recall_curve(
        data_frame['detected_true'].ravel(),
        data_frame['true_detected'].ravel()
    )
    precision_calib, recall_calib, thresholds_calib = precision_recall_curve(
        data_frame['detected_true'].ravel(),
        data_frame['detected_calibrated'].ravel()
    )

    fig_recal_curve = plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.plot(recall_calib, precision_calib, color='darkgreen', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    if show_plot is True:
        plt.show()
    if save_plot is True:
        plt.savefig(save_name, dpi=200)
    plt.clf()
    plt.close(fig_recal_curve)


def plot_probability_hist(data_frame, show_plot, save_plot, save_name, title='Histogram of Predicted Probabilities'):
    """"""
    fig_prob_his = plt.figure()
    plt.hist(data_frame['probability'], bins=30, color='red', edgecolor='black')
    plt.hist(data_frame['probability_calibrated'], bins=30, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    if show_plot is True:
        plt.show()
    if save_plot is True:
        plt.savefig(save_name, dpi=200)
    plt.clf()
    plt.close(fig_prob_his)


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


def plot_classifier_histogram(df_balrog, df_gandalf, columns, show_plot, save_plot, save_name, xlim=None, title='Histogram'):
    """
    Plot histograms for each feature in the given columns of df_balrog and df_gandalf.
    """
    df_gandalf_detected = df_gandalf[df_gandalf['detected'] == 1]
    df_balrog_detected = df_balrog[df_balrog['detected'] == 1]
    df_gandalf_not_detected = df_gandalf[df_gandalf['detected'] == 0]
    df_balrog_not_detected = df_balrog[df_balrog['detected'] == 0]

    df_gandalf_detected = df_gandalf_detected[columns]
    df_balrog_detected = df_balrog_detected[columns]
    df_gandalf_not_detected = df_gandalf_not_detected[columns]
    df_balrog_not_detected = df_balrog_not_detected[columns]

    # Number of features/variables in your datasets (columns)
    num_features = len(columns)

    # Grid dimensions (5x5 for 25 features)
    grid_size = int(np.ceil(np.sqrt(num_features)))

    # Create a figure and a grid of subplots
    fig_hist, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

    # Iterate over each feature and plot the histograms
    for i in range(num_features):
        ax = axes[i]
        # Extracting the ith feature/column from each dataset
        feature_gandalf_detected = df_gandalf_detected.iloc[:, i]
        feature_balrog_detected = df_balrog_detected.iloc[:, i]
        feature_gandalf_not_detected = df_gandalf_not_detected.iloc[:, i]
        feature_balrog_not_detected = df_balrog_not_detected.iloc[:, i]

        # Plot histograms for the ith feature from each dataset
        # Gandalf: Not filled, with specific color
        ax.hist(feature_gandalf_detected, bins=100, alpha=1, label='Gandalf detected', color='#ff8c00', histtype='step')
        ax.hist(feature_gandalf_not_detected, bins=100, alpha=1, label='Gandalf not detected', color='darkgrey', histtype='step')
        # Balrog: Filled, with specific color
        ax.hist(feature_balrog_detected, bins=100, alpha=0.5, label='Balrog detected', color='#51a6fb')
        ax.hist(feature_balrog_not_detected, bins=100, alpha=0.5, label='Balrog not detected', color='lightgrey')

        # Set titles, labels, etc.
        ax.set_xlabel(f'{columns[i]}')
        ax.set_ylabel('Counts')
        if xlim is not None:
            ax.set_xlim(xlim[i])
        # ax.legend()

        handles, labels = ax.get_legend_handles_labels()

    by_label = dict(zip(labels, handles))  # Entfernen Sie Duplikate
    fig_hist.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1, 1))

    # Set overall title
    plt.suptitle(title, fontsize=24)

    # Show or save plot based on arguments
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(save_name, dpi=300)

    # Clear the figure to free memory
    plt.clf()
    plt.close(fig_hist)


def make_gif(frame_folder, name_save_folder, fps=10):
    filenames = natsorted(os.listdir(frame_folder))
    images_data = []
    for filename in filenames:
        image = imageio.imread(f"{frame_folder}/{filename}")
        images_data.append(image)
    imageio.mimwrite(uri=f"{name_save_folder}", ims=images_data, format='.gif', duration=int(1000*1/fps))
