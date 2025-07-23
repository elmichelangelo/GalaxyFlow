import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

import pandas as pd
import numpy as np
import imageio
import os
from natsort import natsorted

from scipy.stats import gaussian_kde
from torchvision.transforms import ToTensor
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, brier_score_loss
from sklearn.calibration import calibration_curve
from scipy.stats import binned_statistic, median_abs_deviation
from io import BytesIO

from statsmodels.nonparametric.kernel_density import KDEMultivariate


from Handler.helper_functions import string_to_tuple, calculate_kde
import time
import matplotlib
# matplotlib.use('Agg')
# plt.style.use('seaborn-white')

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
        # bins=100,
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
        # bins=100,
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
            pass
            # legend_elements.append(Line2D([0], [0], color='#ff8c00', lw=0, label=f'{labels[i]}: Delta mean = {np.abs(delta_mean):.5f}'), )
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


def plot_compare_seaborn(data_frame_generated, data_frame_true, columns, labels, title, epoch, dict_delta, ranges=None,
                        show_plot=False, save_plot=False, save_name=None):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.gridspec as gridspec

    # Prepare data arrays
    arr_generated = data_frame_generated[columns].values
    arr_true = data_frame_true[columns].values

    # Compute quantiles
    quantiles_generated = np.quantile(arr_generated, q=[0.16, 0.84], axis=0)
    quantiles_true = np.quantile(arr_true, q=[0.16, 0.84], axis=0)

    delta_names = ["mean", "median", "q16", "q84"]
    ndim = len(columns)

    # Initialize dict_delta if epoch == 1
    if epoch == 1 and dict_delta is not None:
        for label in labels:
            dict_delta[f"delta mean {label}"] = []
            dict_delta[f"delta median {label}"] = []
            dict_delta[f"delta q16 {label}"] = []
            dict_delta[f"delta q84 {label}"] = []

    # Compute deltas
    delta_mean_list = []
    delta_median_list = []
    delta_q16_list = []
    delta_q84_list = []

    for i in range(ndim):
        gen_data = arr_generated[:, i]
        true_data = arr_true[:, i]
        delta_mean = np.mean(gen_data) - np.mean(true_data)
        delta_median = np.median(gen_data) - np.median(true_data)
        delta_q16 = quantiles_generated[0, i] - quantiles_true[0, i]
        delta_q84 = quantiles_generated[1, i] - quantiles_true[1, i]

        delta_mean_list.append(delta_mean)
        delta_median_list.append(delta_median)
        delta_q16_list.append(delta_q16)
        delta_q84_list.append(delta_q84)

        if dict_delta is not None:
            dict_delta[f"delta mean {labels[i]}"].append(delta_mean)
            dict_delta[f"delta median {labels[i]}"].append(delta_median)
            dict_delta[f"delta q16 {labels[i]}"].append(delta_q16)
            dict_delta[f"delta q84 {labels[i]}"].append(delta_q84)

    # Create combined DataFrame
    data_frame_generated_copy = data_frame_generated[columns].copy()
    data_frame_generated_copy['dataset'] = 'gaNdalF'

    data_frame_true_copy = data_frame_true[columns].copy()
    data_frame_true_copy['dataset'] = 'Balrog'

    data_frame_combined = pd.concat([data_frame_generated_copy, data_frame_true_copy], ignore_index=True)

    # Create figure and gridspec
    if dict_delta is None:
        fig, axes = plt.subplots(ndim, ndim, figsize=(16, 9))
    else:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(ndim + 5, ndim, hspace=0.05, wspace=0.1)
        axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(ndim)] for i in range(ndim)])

    # Plot the pairwise plots
    for i, x_var in enumerate(columns):
        for j, y_var in enumerate(columns):
            ax = axes[i, j]
            if i == j:
                # Diagonal: Histograms
                print(f"Plot histogram for col {x_var}. Δmean={delta_mean:.2e}\nΔmedian={delta_median:.2e}")
                sns.histplot(
                    data=data_frame_combined,
                    x=x_var,
                    hue='dataset',
                    ax=ax,
                    binwidth=0.5,
                    element='step',
                    stat='density',
                    common_norm=False,
                    palette={'gaNdalF': '#ff8c00', 'Balrog': '#51a6fb'},
                    legend=False
                )
                # Add delta information in the title
                delta_mean = delta_mean_list[i]
                delta_median = delta_median_list[i]
                ax.set_title(f'Δmean={delta_mean:.2e}\nΔmedian={delta_median:.2e}', fontsize=10)
                ax.set_xlim(ranges[x_var][0], ranges[x_var][1])
            elif i > j:
                try:
                    print(f"Plot gandalf kde for col {x_var} and {y_var}")
                    # Lower triangle: KDE plots
                    sns.kdeplot(
                        data=data_frame_combined,  #[data_frame_combined['dataset'] == 'gaNdalF'],
                        hue='dataset',
                        x=y_var,
                        y=x_var,
                        ax=ax,
                        fill=True,
                        # bw_adjust=1.5,
                        # gridsize=50,
                        levels=5,
                        thresh=0,
                        # color='#ff8c00',
                        alpha=0.5,
                        legend=False
                    )
                    # ax.set_xlim(ranges[y_var][0], ranges[y_var][1])
                    # ax.set_ylim(ranges[x_var][0], ranges[x_var][1])
                except ValueError:
                    pass
                # try:
                #     print(f"Plot balrog kde for col {x_var} and {y_var}")
                #     sns.kdeplot(
                #         data=data_frame_combined[data_frame_combined['dataset'] == 'Balrog'],
                #         x=y_var,
                #         y=x_var,
                #         ax=ax,
                #         fill=True,
                #         # bw_adjust=1.5,
                #         # gridsize=50,
                #         levels=5,
                #         thresh=0,
                #         color='#51a6fb',
                #         alpha=0.5
                #     )
                # except ValueError:
                #     pass
            else:
                # Upper triangle: Hide
                ax.set_visible(False)

            # Set axis labels
            if i == ndim - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xlabel('')
            if j == 0:
                ax.set_ylabel(labels[i])
            else:
                ax.set_ylabel('')

            # Remove ticks where appropriate
            if i != ndim - 1:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])

            # if isinstance(ranges, dict):
            #     if isinstance(ranges[x_var], list):
            #         ax.set_xlim(ranges[x_var][0], ranges[x_var][1])
            #     if isinstance(ranges[y_var], list):
            #         ax.set_ylim(ranges[y_var][0], ranges[y_var][1])

    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#ff8c00', lw=4, label='gaNdalF'),
        Line2D([0], [0], color='#51a6fb', lw=4, label='Balrog')
    ]

    if dict_delta is not None:
        delta_legend_elements = []
        epochs = list(range(1, epoch + 1))
        for idx, delta_name in enumerate(delta_names):
            delta_ax = fig.add_subplot(gs[ndim + 1 + idx, :])
            for i, label in enumerate(labels):
                line, = delta_ax.plot(epochs, dict_delta[f"delta {delta_name} {label}"], '-o',
                                      label=f"Δ {delta_name} {label}")
                if idx == 0:
                    delta_legend_elements.append(line)

            delta_ax.axhline(y=0, color='gray', linestyle='--')
            delta_ax.set_ylim(-0.05, 0.05)

            if idx == len(delta_names) - 1:
                delta_ax.set_xlabel('Epoch')
            else:
                delta_ax.set_xticklabels([])

            delta_ax.set_ylabel(f'Δ {delta_name}')

        # Combine legend elements
        legend_elements.extend(delta_legend_elements)
        fig.legend(handles=legend_elements, loc='upper right', fontsize=12)
    else:
        fig.legend(handles=legend_elements, loc='upper right', fontsize=16)

    # Set the main title
    if epoch is not None:
        fig.suptitle(f'{title}, epoch {epoch}', fontsize=20)
    else:
        fig.suptitle(f'{title}', fontsize=20)

    # Optionally show or save the plot
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(save_name, dpi=300)

    plt.close(fig)

    # Return updated dict_delta
    return dict_delta


def loss_plot(
        epoch,
        title,
        lst_train_loss_per_batch,
        lst_train_loss_per_epoch,
        lst_valid_loss_per_batch,
        lst_valid_loss_per_epoch,
        show_plot,
        save_plot,
        save_name,
        log_scale=False
):
    statistical_figure, (stat_ax1, stat_ax2, stat_ax3) = plt.subplots(nrows=3, ncols=1)
    statistical_figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)
    statistical_figure.suptitle(f"{title} (Epoch: {epoch+1})", fontsize=18)

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
    if log_scale is True:
        stat_ax1.set_yscale("log")

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
    if log_scale is True:
        stat_ax2.set_yscale("log")

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
    if log_scale is True:
        stat_ax3.set_yscale("log")

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
        statistical_figure.savefig(f"{save_name}", bbox_inches='tight', dpi=300)

    # Clear and close open figure to avoid memory overload
    img_tensor = plot_to_tensor()
    statistical_figure.clf()
    plt.clf()
    plt.close(statistical_figure)
    return img_tensor


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


def plot_roc_curve_gandalf_non_calib(df_balrog, df_gandalf, show_plot, save_plot, save_name, title='Receiver Operating Characteristic (ROC) Curve'):
    """"""
    fpr, tpr, thresholds = roc_curve(
        df_balrog['detected'].to_numpy(),
        df_gandalf['detected non calibrated'].to_numpy()
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


def plot_violin_classifier(df_balrog, df_gandalf, columns, ranges,  show_plot, save_plot, save_name, title='Histogram'):
    sns.set_theme(style="whitegrid")

    df_gandalf_detected = df_gandalf[df_gandalf["detected"] == 1]
    df_gandalf_not_detected = df_gandalf[df_gandalf["detected"] == 0]
    df_balrog_detected = df_balrog[df_balrog["detected"] == 1]
    df_balrog_not_detected = df_balrog[df_balrog["detected"] == 0]

    num_rows = len(columns)
    fig, axes = plt.subplots(nrows=num_rows, figsize=(9, 12))
    custom_palette = ["#ff8c00", "#51a6fb", "darkgrey", "lightgrey"]

    for i, col in enumerate(columns):
        ax = axes[i]
        df_violin = pd.DataFrame({
            "gandalf detected": df_gandalf_detected[col],
            "balrog detected": df_balrog_detected[col],
            "gandalf not detected": df_gandalf_not_detected[col],
            "balrog not detected": df_balrog_not_detected[col]
        })

        sns.violinplot(data=df_violin, ax=ax, bw_adjust=.5, cut=1, linewidth=1, palette=custom_palette)
        ax.set_ylim(ranges[i][0], ranges[i][1])
        ax.set_title(col)

        # for j, category in enumerate(df_violin.columns):
        #     mean_val = df_violin[category].mean()
        #     median_val = df_violin[category].median()
        #     std_val = df_violin[category].std()
        #
        #     # Text auf der rechten Seite des Plots platzieren
        #     ax.text(1.05, 0.5 - j * 0.2,
        #             f'{category}:\nMean: {mean_val:.2f}\nMedian: {median_val:.2f}\nStd: {std_val:.2f}',
        #             transform=ax.transAxes, verticalalignment='center', fontsize=10, color=custom_palette[j])

    sns.despine(left=True, bottom=True)

    plt.suptitle(title, fontsize=18)
    if show_plot is True:
        plt.show()
    if save_plot is True:
        plt.savefig(save_name, dpi=300)
    plt.clf()
    plt.close(fig)


def plot_radar_chart(df_balrog, df_gandalf, columns):
    labels = np.array([f'Dim {i + 1}' for i in range(len(columns))])
    stats_gandalf = []
    stats_balrog = []
    for col in columns:
        stats_gandalf.append(df_gandalf[col])
        stats_balrog.append(df_balrog[col])

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    stats_gandalf = np.concatenate((stats_gandalf, [stats_gandalf[0]]))
    stats_balrog = np.concatenate((stats_balrog, [stats_balrog[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, stats_gandalf, 'o-', label='Distribution A')
    ax.plot(angles, stats_balrog, 'o-', label='Distribution B')
    ax.fill(angles, stats_gandalf, alpha=0.25)
    ax.fill(angles, stats_balrog, alpha=0.25)
    ax.set_thetagrids(angles * 180 / np.pi, labels)
    ax.set_title('Radar Chart Comparison')
    ax.legend()
    plt.show()


def plot_box(df_balrog, df_gandalf, columns, labels, show_plot, save_plot, save_name, title):
    fig, axs = plt.subplots(6, 3, figsize=(12, 24))  # Adjust subplots as needed
    axs = axs.ravel()

    for i, col in enumerate(columns):
        axs[i].boxplot([df_balrog[col], df_gandalf[col]], labels=['Balrog', 'gaNdalF'])
        axs[i].set_title(labels[i])

    plt.title(title)
    plt.tight_layout()
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(save_name, dpi=300)
    plt.clf()
    plt.close(fig)


def plot_multivariate_classifier(df_balrog, df_gandalf, columns, grid_size, show_plot, save_plot, save_name,
                                 sample_size=5000, x_range=(18, 26), title='Histogram'):
    import matplotlib.patches as mpatches
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import corner

    # Set the start time
    start_time = time.time()

    # Calculate the number of rows and columns
    num_cols = grid_size[0] # int(np.ceil(np.sqrt(len(columns))))
    num_rows = grid_size[1] # int(np.ceil(len(columns) / num_cols))

    levels = [0.393, 0.865, 0.989]

    # Prepare the dataframes
    df_gandalf_detected = df_gandalf[df_gandalf["detected"] == 1]
    df_gandalf_not_detected = df_gandalf[df_gandalf["detected"] == 0]
    df_balrog_detected = df_balrog[df_balrog["detected"] == 1]
    df_balrog_not_detected = df_balrog[df_balrog["detected"] == 0]

    # Create the grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    color_balrog_detected = '#51a6fb'
    color_gandalf_detected = '#ff8c00'
    color_balrog_not_detected = "grey"  # 'coral'
    color_gandalf_not_detected = "black"  # 'blueviolet'

    hatch_gandalf_not_detected = ['-', '/', '\\']
    linestyle_gandalf_not_detected = '--'

    hatch_balrog_not_detected = ['.', '*', '\\\\']
    linestyle_balrog_not_detected = '-.'

    def get_elapsed_time(start_time):
        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Convert to days, hours, minutes, and seconds
        days = elapsed_time // (24 * 3600)
        elapsed_time = elapsed_time % (24 * 3600)
        hours = elapsed_time // 3600
        elapsed_time %= 3600
        minutes = elapsed_time // 60
        seconds = elapsed_time % 60

        # Print the elapsed time
        print(f"Elapsed time: {int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s")

    if sample_size > 0:
        df_gandalf_detected_sample = df_gandalf_detected.sample(n=sample_size, random_state=42)
        df_balrog_detected_sample = df_balrog_detected.sample(n=sample_size, random_state=42)
        df_gandalf_not_detected_sample = df_gandalf_not_detected.sample(n=sample_size, random_state=42)
        df_balrog_not_detected_sample = df_balrog_not_detected.sample(n=sample_size, random_state=42)
    else:
        df_gandalf_detected_sample = df_gandalf_detected
        df_balrog_detected_sample = df_balrog_detected
        df_gandalf_not_detected_sample = df_gandalf_not_detected
        df_balrog_not_detected_sample = df_balrog_not_detected

    # Add a new column to indicate the type of detector
    df_gandalf_detected_sample["catalog_type"] = "Gandalf Detected"
    df_balrog_detected_sample["catalog_type"] = "Balrog Detected"
    df_gandalf_not_detected_sample["catalog_type"] = "Gandalf Not Detected"
    df_balrog_not_detected_sample["catalog_type"] = "Balrog Not Detected"

    # Concatenate all DataFrames
    df_combined = pd.concat([
        df_gandalf_detected_sample,
        df_balrog_detected_sample,
        df_gandalf_not_detected_sample,
        df_balrog_not_detected_sample
    ], ignore_index=True)
    columns = {"FWHM_WMEAN_R": {
        "label": "FWHM R",
        "range": [0.7, 1.3],
        "position": [1, 1]
    }}
    # Loop over each feature and plot
    for i, col in enumerate(columns.keys()):
        pos = columns[col]["position"]
        ax = axes[pos[0], pos[1]]
        get_elapsed_time(start_time)
        print(f"Plotting column: {col} ({i + 1}/{len(columns)})")

        # Set the plot ranges
        y_range = columns[col]["range"]
        label = columns[col]["label"]

        # Gandalf detected KDE
        # corner.hist2d(
        #     x=df_gandalf_detected_sample["BDF_MAG_DERED_CALIB_I"].values,
        #     y=df_gandalf_detected_sample[col].values,
        #     ax=ax,
        #     bins=50,
        #     range=[x_range, y_range],
        #     levels=levels,
        #     color=color_gandalf_detected,
        #     smooth=1.0,
        #     plot_datapoints=False,
        #     fill_contours=False,
        #     plot_density=True,
        #     plot_contours=True,
        #     contourf_kwargs={'colors': color_gandalf_detected, 'alpha': 0.5}
        # )
        # get_elapsed_time(start_time)

        # Balrog detected KDE
        # corner.hist2d(
        #     x=df_balrog_detected_sample["BDF_MAG_DERED_CALIB_I"].values,
        #     y=df_balrog_detected_sample[col].values,
        #     ax=ax,
        #     bins=50,
        #     range=[x_range, y_range],
        #     levels=levels,
        #     color=color_balrog_detected,
        #     smooth=1.0,
        #     plot_datapoints=False,
        #     fill_contours=False,
        #     plot_density=True,
        #     plot_contours=True,
        #     contourf_kwargs={'colors': color_balrog_detected, 'alpha': 0.5}
        # )
        # get_elapsed_time(start_time)

        # Gandalf not detected KDE
        # corner.hist2d(
        #     x=df_gandalf_not_detected_sample["BDF_MAG_DERED_CALIB_I"].values,
        #     y=df_gandalf_not_detected_sample[col].values,
        #     ax=ax,
        #     bins=50,
        #     range=[x_range, y_range],
        #     levels=levels,
        #     # color=color_gandalf_not_detected,
        #     smooth=1.0,
        #     plot_datapoints=False,
        #     fill_contours=False,
        #     plot_density=True,
        #     plot_contours=True,
        #     contourf_kwargs={
        #         'hatches': hatch_gandalf_not_detected,
        #         'alpha': 0.5,
        #         'color': None
        #     },
        # )
        get_elapsed_time(start_time)

        sns.kdeplot(
            df_combined,
            x="BDF_MAG_DERED_CALIB_I",
            y=col,
            hue="catalog_type",
            ax=ax,
            levels=levels,
            fill = False
        )
        x = df_combined["BDF_MAG_DERED_CALIB_I"].values
        y = df_combined[col].values
        # Calculate the 2D histogram
        H, xedges, yedges = np.histogram2d(x, y, bins=100)  # Adjust bins and range as needed

        # Normalize the histogram
        H = H.T  # Transpose for correct orientation
        H_norm = H / H.max()  # Normalize for contour plotting

        # Create a meshgrid for the contour plot
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

        # Add filled contour with hatching
        ax.contourf(X, Y, H_norm, levels=levels, hatches=['/', '\\', '|'], colors='none', alpha=0.2)
        # Assuming df_balrog_not_detected_sample and related variables are predefined
        # x = df_balrog_not_detected_sample["BDF_MAG_DERED_CALIB_I"].values
        # y = df_balrog_not_detected_sample[col].values
        #
        # # Calculate the 2D histogram
        # H, xedges, yedges = np.histogram2d(x, y, bins=50, range=[x_range, y_range])
        #
        # # Normalize the histogram
        # H = H.T  # Transpose for correct orientation
        # H_norm = H / H.sum()
        #
        # # Create a meshgrid for the contour plot
        # X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
        #
        # # Plot the 2D histogram
        # ax.hist2d(x, y, bins=50, range=[x_range, y_range], cmap='gray', cmin=1)
        #
        # # Plot the contours
        # CS = ax.contour(X, Y, H_norm, levels=levels, colors='black')
        # # ax.clabel(CS, inline=True, fontsize=8) # Optionally add labels to contours
        #
        # # If fill_contours is desired:
        # ax.contourf(X, Y, H_norm, levels=levels, alpha=0.5, hatches=hatch_balrog_not_detected, colors=['none'])

        # ax.set_xlim(x_range)
        # ax.set_ylim(y_range)
        # ax.set_xlabel('BDF_MAG_DERED_CALIB_I')
        # ax.set_ylabel(col)
        # Balrog not detected KDE
        # corner.hist2d(
        #     x=df_balrog_not_detected_sample["BDF_MAG_DERED_CALIB_I"].values,
        #     y=df_balrog_not_detected_sample[col].values,
        #     ax=ax,
        #     bins=50,
        #     range=[x_range, y_range],
        #     levels=levels,
        #     # color=color_balrog_not_detected,
        #     smooth=1.0,
        #     plot_datapoints=False,
        #     fill_contours=False,
        #     plot_density=True,
        #     plot_contours=True,
        #     contourf_kwargs={
        #         'hatches': hatch_balrog_not_detected,
        #         'alpha': 0.5,
        #         'color': None
        #         # 'cmap': 'black'
        #     },
        #     # contour_kwargs={'linestyles': linestyle_balrog_not_detected}
        # )

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_ylabel(label, fontsize=12)

        # Add axis labels only to the bottom row subplots
        if i >= len(axes) - num_cols:
            ax.set_xlabel('BDF Mag I', fontsize=12)

    # Remove any unused subplots
    fig.delaxes(axes[0, 3])
    fig.delaxes(axes[1, 3])

    # Customize layout and legend
    legend_elements = [
        mpatches.Patch(color=color_gandalf_detected, alpha=0.5, label='Gandalf Detected'),
        mpatches.Patch(color=color_balrog_detected, alpha=1, label='Balrog Detected'),
        mpatches.Patch(color=color_gandalf_not_detected, alpha=0.5, label='Gandalf Not Detected'),
        mpatches.Patch(color=color_balrog_not_detected, alpha=1, label='Balrog Not Detected')
    ]

    fig.legend(handles=legend_elements, loc='upper right', fontsize=18, bbox_to_anchor=(0.98, 0.76))

    plt.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(save_name, dpi=300)
    plt.clf()
    plt.close(fig)


def plot_classifier_histogram(df_balrog, df_gandalf, columns, show_plot, save_plot, save_name, xlim=None, title='Histogram'):
    """
    Plot histograms for each feature in the given columns of df_balrog and df_gandalf.
    """
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
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
        ax.yaxis.set_major_formatter(formatter)
        # Extracting the ith feature/column from each dataset
        feature_gandalf_detected = df_gandalf_detected.iloc[:, i]
        feature_balrog_detected = df_balrog_detected.iloc[:, i]
        feature_gandalf_not_detected = df_gandalf_not_detected.iloc[:, i]
        feature_balrog_not_detected = df_balrog_not_detected.iloc[:, i]

        # Plot histograms for the ith feature from each dataset
        # Gandalf: Not filled, with specific color
        ax.hist(feature_gandalf_detected, density=False, bins=100, alpha=1, label='Gandalf detected', color='#ff8c00', histtype='step')
        ax.hist(feature_gandalf_not_detected, density=False, bins=100, alpha=1, label='Gandalf not detected', color='darkgrey', histtype='step')
        # Balrog: Filled, with specific color
        ax.hist(feature_balrog_detected, density=False, bins=100, alpha=0.5, label='Balrog detected', color='#51a6fb')
        ax.hist(feature_balrog_not_detected, density=False, bins=100, alpha=0.5, label='Balrog not detected', color='lightgrey')

        # Set titles, labels, etc.
        ax.set_xlabel(f'{columns[i]}')
        ax.set_ylabel('Counts')
        if xlim is not None:
            ax.set_xlim(xlim[i])
        # ax.legend()

        handles, labels = ax.get_legend_handles_labels()

    by_label = dict(zip(labels, handles))  # Entfernen Sie Duplikate
    fig_hist.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1, 1), fontsize=16)

    # Set overall title
    plt.suptitle(title, fontsize=26)

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # Adjust padding and leave space for the title

    # Show or save plot based on arguments
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(save_name, dpi=300)

    # Clear the figure to free memory
    plt.clf()
    plt.close(fig_hist)


def plot_balrog_histogram(df_gandalf, df_balrog, columns, labels, ranges, binwidths, title, show_plot, save_plot, save_name):
    """"""
    color_gandalf = '#ff8c00'
    color_balrog = '#51a6fb'
    hist_figure_2, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 12))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        ax = axes[i]
        if binwidths[i] is not None:
            binwidth = binwidths[i]
        else:
            binwidth = 0.2
        sns.histplot(
            data=df_balrog,
            x=col,
            ax=ax,
            element="step",
            stat="count",
            color=color_balrog,
            fill=False,
            binwidth=binwidth,
            log_scale=(False, True),
            label="balrog"
        )
        sns.histplot(
            data=df_gandalf,
            x=col,
            ax=ax,
            element="step",
            stat="count",
            color=color_gandalf,
            fill=False,
            binwidth=binwidth,
            log_scale=(False, True),
            label="gandalf"
        )
        if ranges[i] is not None:
            ax.set_xlim(ranges[i][0], ranges[i][1])
        ax.set_xlabel(labels[i])
        ax.grid(True)
        ax.legend()

    plt.suptitle(title)
    plt.tight_layout()

    # Show or save plot based on arguments
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(save_name, dpi=300)

    # Clear the figure to free memory
    plt.clf()
    plt.close(hist_figure_2)


def plot_number_density_fluctuation(
        df_balrog,  # DataFrame containing Balrog data
        df_gandalf,  # DataFrame containing Gandalf data
        columns,  # List of column names to plot
        labels,  # Corresponding labels for x-axes
        ranges,  # List of (min, max) tuples for each column
        title,  # Title for the entire figure
        show_plot,  # Bool: display the figure?
        save_plot,  # Bool: save the figure to disk?
        save_name,  # Path/filename for saving the figure
        calibrated=True
):
    """
    Plot the number density fluctuations for detected and not detected objects
    in the Balrog and Gandalf datasets.

    Approach:
    ----------
    For each column of interest, this function:
    1. Splits Balrog and Gandalf datasets into detected vs. not detected subsets.
    2. Computes binned histograms for each subset (Balrog detected, Gandalf detected, etc.).
    3. Normalizes each binned histogram by its own mean count, giving a "fluctuation" (N/<N>).
    4. Plots both the normalized binned histograms (top sub-panel) and the difference
       between these normalized histograms (bottom sub-panel).

    The difference plotted is:
        (Gandalf_counts / <Gandalf_counts>) - (Balrog_counts / <Balrog_counts>)
    for both detected and not detected objects.

    Notes on "physical correctness":
    --------------------------------
    - This function *does* correctly show relative differences in number counts,
      as a function of some parameter(s). It’s akin to looking at fractional fluctuations.
    - Whether this is “physically correct” depends on your scientific question. Typically,
      to make this more robust, you’d also compute error bars (e.g., Poisson errors) on each
      histogram bin or use a more rigorous approach to comparing two distributions
      (like a ratio plot with uncertainties).
    - As is, this code is a standard approach for visualizing and comparing
      the shapes of distributions, not necessarily capturing detailed statistical
      significance or systematic errors.

    Returns:
        None
    """
    import matplotlib as mpl
    # Use LaTeX fonts in matplotlib
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern Roman']
    mpl.rcParams['axes.labelsize'] = 24
    mpl.rcParams['font.size'] = 28
    mpl.rcParams['legend.fontsize'] = 24
    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['ytick.labelsize'] = 20

    # ----------------------------------------------------------------------
    # Create subsets for detected and not detected objects
    # ----------------------------------------------------------------------
    if calibrated is True:
        gandalf_detection_flag = "detected"
    else:
        gandalf_detection_flag = "detected non calibrated"
    df_balrog_detected = df_balrog[df_balrog["detected"] == 1]
    df_gandalf_detected = df_gandalf[df_gandalf[gandalf_detection_flag] == 1]
    df_balrog_not_detected = df_balrog[df_balrog["detected"] == 0]
    df_gandalf_not_detected = df_gandalf[df_gandalf[gandalf_detection_flag] == 0]

    # ----------------------------------------------------------------------
    # Define colors/markers for different subsets
    # ----------------------------------------------------------------------
    color_marker_gandalf_detected = ['darkgreen', 'o']  # ['#ff8c00', 'o']
    color_marker_balrog_detected = ['purple', 's']  # ['#51a6fb', 's']
    color_marker_gandalf_not_detected = ['green', 'D']  # ['black', 'D']
    color_marker_balrog_not_detected = ['darkviolet', '^']  # ['grey', '^']
    color_marker_difference_detected = ['black', 'v']  # ['purple', 'v']
    color_marker_difference_not_detected = ['grey', 'P']  # ['lightgrey', 'P']

    # ----------------------------------------------------------------------
    # Determine layout for subplots
    # ----------------------------------------------------------------------
    ncols = 5
    nrows = (len(columns) + ncols - 1) // ncols  # Enough rows to accommodate columns

    # Create the main figure
    fig = plt.figure(figsize=(5 * ncols, 8 * nrows))
    main_gs = GridSpec(nrows, ncols, figure=fig)

    # Font sizes
    # font_size_labels = 16
    # font_size_title = 24

    # ----------------------------------------------------------------------
    # Loop over each column to create the subplots
    # ----------------------------------------------------------------------
    for idx, col in enumerate(columns):
        # Identify which row/column in the grid
        row_idx = idx // ncols
        col_idx = idx % ncols

        # Create a "2-row" subplot for each column:
        # top row for normalized counts, bottom row for their difference
        inner_gs = GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=main_gs[row_idx, col_idx],
            height_ratios=[3, 1],
            hspace=0  # No vertical space between subplots in this group
        )

        ax_main = fig.add_subplot(inner_gs[0])
        ax_diff = fig.add_subplot(inner_gs[1], sharex=ax_main)

        # ------------------------------------------------------------------
        # Determine bin ranges:
        # If a range is specified, use it; otherwise use the min/max from the data
        # ------------------------------------------------------------------
        bin_range = ranges[idx]
        if bin_range:
            range_min, range_max = bin_range
        else:
            range_min = min(df_balrog[col].min(), df_gandalf[col].min())
            range_max = max(df_balrog[col].max(), df_gandalf[col].max())

        num_bins = 20
        bins = np.linspace(range_min, range_max, num_bins + 1)

        # ------------------------------------------------------------------
        # Compute histograms for each subset in the same bin edges
        # ------------------------------------------------------------------
        counts_balrog_detected, _ = np.histogram(df_balrog_detected[col], bins=bins)
        counts_gandalf_detected, _ = np.histogram(df_gandalf_detected[col], bins=bins)
        counts_balrog_not_detected, _ = np.histogram(df_balrog_not_detected[col], bins=bins)
        counts_gandalf_not_detected, _ = np.histogram(df_gandalf_not_detected[col], bins=bins)

        # ------------------------------------------------------------------
        # Compute mean counts for each subset (avoid divide-by-zero with epsilon)
        # ------------------------------------------------------------------
        epsilon = 1e-10
        mean_counts_balrog_detected = np.mean(counts_balrog_detected) + epsilon
        mean_counts_gandalf_detected = np.mean(counts_gandalf_detected) + epsilon
        mean_counts_balrog_not_detected = np.mean(counts_balrog_not_detected) + epsilon
        mean_counts_gandalf_not_detected = np.mean(counts_gandalf_not_detected) + epsilon

        # ------------------------------------------------------------------
        # Calculate fractional fluctuations:
        # (counts / mean_counts) for each bin
        # ------------------------------------------------------------------
        fluct_balrog_detected = counts_balrog_detected / mean_counts_balrog_detected
        fluct_gandalf_detected = counts_gandalf_detected / mean_counts_gandalf_detected
        fluct_balrog_not_detected = counts_balrog_not_detected / mean_counts_balrog_not_detected
        fluct_gandalf_not_detected = counts_gandalf_not_detected / mean_counts_gandalf_not_detected

        # ------------------------------------------------------------------
        # Calculate differences in fluctuations for each bin:
        # (Gandalf / mean_Gandalf) - (Balrog / mean_Balrog)
        # ------------------------------------------------------------------
        # detected_diff_values = fluct_gandalf_detected - fluct_balrog_detected
        # not_detected_diff_values = fluct_gandalf_not_detected - fluct_balrog_not_detected

        detected_diff_values = 100*np.abs(fluct_gandalf_detected - fluct_balrog_detected)/(1/2*(fluct_gandalf_detected + fluct_balrog_detected))
        not_detected_diff_values = 100*np.abs(fluct_gandalf_not_detected - fluct_balrog_not_detected)/(1/2*(fluct_gandalf_not_detected + fluct_balrog_not_detected + epsilon))

        # detected_diff_values = 100 * np.abs(fluct_gandalf_detected - fluct_balrog_detected) / fluct_balrog_detected
        # not_detected_diff_values = 100 * np.abs(fluct_gandalf_not_detected - fluct_balrog_not_detected) / fluct_balrog_not_detected

        # Bin centers for plotting
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # ==================================================================
        # Top subplot: fractional fluctuations (N / <N>)
        # ==================================================================
        ax_main.plot(
            bin_centers,
            fluct_balrog_detected,
            color=color_marker_balrog_detected[0],
            marker=color_marker_balrog_detected[1],
            label="Balrog detected",
            linewidth=2,  # Line width
            markersize=8,  # Marker size
            # alpha=0.5
        )
        ax_main.plot(
            bin_centers,
            fluct_gandalf_detected,
            color=color_marker_gandalf_detected[0],
            marker=color_marker_gandalf_detected[1],
            label="gaNdalF detected",
            linewidth=2,  # Line width
            markersize=8,  # Marker size
            # alpha=0.5
        )
        ax_main.plot(
            bin_centers,
            fluct_balrog_not_detected,
            color=color_marker_gandalf_not_detected[0],
            marker=color_marker_gandalf_not_detected[1],
            label="Balrog non-detected",
            linewidth=2,  # Line width
            markersize=8,  # Marker size
            alpha=0.5
        )
        ax_main.plot(
            bin_centers,
            fluct_gandalf_not_detected,
            color=color_marker_balrog_not_detected[0],
            marker=color_marker_balrog_not_detected[1],
            label="gaNdalF non-detected",
            linewidth=2,  # Line width
            markersize=8,  # Marker size
            alpha=0.5
        )

        # Draw a horizontal line at 1 to represent the mean level
        ax_main.axhline(0, color='black', linestyle='--')
        ax_main.set_xlim(range_min, range_max)
        ax_main.set_ylabel("$N / <N>$")  # , fontsize=font_size_labels
        ax_main.grid(True)

        # ==================================================================
        # Bottom subplot: difference in fractional fluctuations
        # ==================================================================
        ax_diff.plot(
            bin_centers,
            detected_diff_values,
            color=color_marker_difference_detected[0],
            marker=color_marker_difference_detected[1],
            # alpha=0.5,
            label="Relative Percentage Difference detected"
        )
        ax_diff.plot(
            bin_centers,
            not_detected_diff_values,
            color=color_marker_difference_not_detected[0],
            marker=color_marker_difference_not_detected[1],
            alpha=0.5,
            label="Relative Percentage Difference non-detected"
        )

        # Draw a horizontal line at 0 to highlight no difference
        ax_diff.axhline(1, color='black', linestyle='--')
        ax_diff.set_xlim(range_min, range_max)
        ax_diff.set_xlabel(labels[idx])  # , fontsize=font_size_labels
        ax_diff.set_ylabel("Difference")  # , fontsize=font_size_labels - 2
        ax_diff.grid(True)

        # Hide the top subplot’s x-axis tick labels to avoid overlap
        plt.setp(ax_main.get_xticklabels(), visible=False)

    # ----------------------------------------------------------------------
    # Create a unified legend outside the subplots
    # ----------------------------------------------------------------------
    handles, labels_legend = ax_main.get_legend_handles_labels()
    handles_diff, labels_diff = ax_diff.get_legend_handles_labels()

    fig.legend(
        handles + handles_diff,
        labels_legend + labels_diff,
        loc='upper right',
        ncol=2,
        # fontsize=font_size_labels,
        bbox_to_anchor=(0.95, 0.95),
        borderaxespad=0.,
    )
    # plt.subplots_adjust(right=0.8)
    # Adjust spacing so the subplots and title fit
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.suptitle(title, y=0.99)  # fontsize=font_size_title,

    # ----------------------------------------------------------------------
    # Save and/or show the final plot
    # ----------------------------------------------------------------------
    if save_plot:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()

    # Close the figure to free memory
    plt.close()


def plot_tomo_bin_redshift_bootstrap(zmean, gandalf_files, balrog_file, gandalf_means, gandalf_stds, balrog_means,
                                     title="Wide n(z)", show_plot=True, save_plot=False, save_name=None, all_bins=True):
    """"""

    if not gandalf_files or balrog_file is None:
        raise ValueError("Required files not found in the folder.")

    num_bins = 1
    lst_x_lims = [
        (0, 0.8),
        (0.05, 0.8),
        (0.5, 1.05),
        (0.6, 1.3)
    ]
    if all_bins is True:
        num_bins = len(balrog_file)  # Number of tomographic bins


    # Define colors
    color_gandalf = 'darkgreen'
    color_balrog = 'purple'

    # Font sizes
    font_size_labels = 16
    font_size_title = 24

    # Create the plot
    fig, axes = plt.subplots(num_bins, 1, figsize=(16, 9))
    if num_bins == 1:
        ax = [axes]
    else:
        ax = axes

    for bin_idx in range(num_bins):  # Iterate over tomographic bins
        # Plot bootstrap lines for gaNdalF in the background
        for idx, gandalf_hist in enumerate(gandalf_files):
            if isinstance(gandalf_hist, pd.DataFrame):  # Handle DataFrame
                ax[bin_idx].plot(
                    zmean,
                    gandalf_hist.iloc[bin_idx, :],  # Row corresponding to the current tomographic bin
                    color=color_gandalf,
                    alpha=0.1,
                    lw=1
                )
            elif isinstance(gandalf_hist, np.ndarray):  # Handle numpy array
                ax[bin_idx].plot(
                    zmean,
                    gandalf_hist[bin_idx, :],  # Row corresponding to the current tomographic bin
                    color=color_gandalf,
                    alpha=0.1,
                    lw=1
                )

        # Plot Balrog distribution in the foreground
        if isinstance(balrog_file, pd.DataFrame):
            ax[bin_idx].plot(
                zmean,
                balrog_file.iloc[bin_idx, :],  # Row corresponding to the current tomographic bin
                color=color_balrog,
                lw=0.5
            )
        elif isinstance(balrog_file, np.ndarray):
            ax[bin_idx].plot(
                zmean,
                balrog_file[bin_idx, :],  # Row corresponding to the current tomographic bin
                color=color_balrog,
                lw=0.5
            )

        # Plot gaNdalF mean line and shaded ±1σ range
        ax[bin_idx].axvline(gandalf_means[bin_idx], color=color_gandalf, linestyle='-', linewidth=1,
                            label=f'<z> gaNdalF {gandalf_means[bin_idx]:.4f} ± {gandalf_stds[bin_idx]:.3f}')
        ax[bin_idx].axvspan(
            gandalf_means[bin_idx] - gandalf_stds[bin_idx],
            gandalf_means[bin_idx] + gandalf_stds[bin_idx],
            color=color_gandalf, alpha=0.2
        )

        # Plot Balrog mean line and shaded ±0.01 range
        ax[bin_idx].axvline(balrog_means[bin_idx], color=color_balrog, linestyle='-', linewidth=1,
                            label=f'<z> Balrog {balrog_means[bin_idx]:.4f} ± 0.01')
        ax[bin_idx].axvspan(
            balrog_means[bin_idx] - 0.01,
            balrog_means[bin_idx] + 0.01,
            color=color_balrog, alpha=0.2
        )

        # Customize each subplot
        ax[bin_idx].set_xlim(lst_x_lims[bin_idx])
        ax[bin_idx].set_ylim(0, 7)
        # Remove y-tick labels but keep tick marks
        ax[bin_idx].set_yticklabels([])  # Remove numerical labels
        ax[bin_idx].tick_params(axis='y', length=5, direction='in', left=True, right=True)

        # Add "Probability Density" label on the y-axis for each subplot
        ax[bin_idx].set_ylabel(f'bin {bin_idx + 1}', fontsize=font_size_labels)
        # ax[bin_idx].set_ylabel(f'$p(z)$ bin {bin_idx + 1}', fontsize=font_size_labels)
        ax[bin_idx].legend()
        ax[bin_idx].grid()

        if bin_idx == num_bins - 1:
            ax[bin_idx].set_xlabel(r'$z$', fontsize=font_size_labels)

    fig.suptitle(title, fontsize=font_size_title)  # Set the title for the whole figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout

    fig.text(
        -0.04, 0.5, 'Probability Density',
        fontsize=font_size_labels,
        va='center', rotation='vertical'
    )

    if show_plot:
        plt.show()
    if save_plot and save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=300)


def plot_tomo_bin_redshift_bootstrap_all_in_one(zmean, gandalf_files, balrog_file, gandalf_means, gandalf_stds,
                                                balrog_means, title="Wide n(z)", show_plot=True, save_plot=False,
                                                save_name=None):
    """"""
    import matplotlib as mpl
    import matplotlib.ticker as ticker
    # Use LaTeX fonts in matplotlib
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern Roman']
    mpl.rcParams['axes.labelsize'] = 24
    mpl.rcParams['font.size'] = 28
    mpl.rcParams['legend.fontsize'] = 20
    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['ytick.labelsize'] = 20

    if not gandalf_files or balrog_file is None:
        raise ValueError("Required files not found in the folder.")

    num_bins = len(balrog_file)  # Number of tomographic bins


    # Define colors
    color_gandalf = [
        'forestgreen',
        'orangered',
        'royalblue',
        'deeppink'
    ]
    color_balrog = 'black'
    line_style_balrog = [
        (0, (3, 1, 1, 1, 1, 1)),
        (0, (5, 1)),
        "dotted",
        "dashdot",
    ]

    # Create the plot
    fig, axes = plt.subplots(1, 1, figsize=(16, 9))

    for bin_idx in range(num_bins):  # Iterate over tomographic bins
        # Plot bootstrap lines for gaNdalF in the background
        for idx, gandalf_hist in enumerate(gandalf_files):
            if isinstance(gandalf_hist, pd.DataFrame):  # Handle DataFrame
                axes.plot(
                    zmean,
                    gandalf_hist.iloc[bin_idx, :],  # Row corresponding to the current tomographic bin
                    color=color_gandalf[bin_idx],
                    alpha=0.01,
                    lw=8
                )
            elif isinstance(gandalf_hist, np.ndarray):  # Handle numpy array
                axes.plot(
                    zmean,
                    gandalf_hist[bin_idx, :],  # Row corresponding to the current tomographic bin
                    color=color_gandalf[bin_idx],
                    alpha=0.01,
                    lw=8
                )

        # Plot Balrog distribution in the foreground
        if isinstance(balrog_file, pd.DataFrame):
            axes.plot(
                zmean,
                balrog_file.iloc[bin_idx, :],  # Row corresponding to the current tomographic bin
                color=color_balrog,
                lw=3
            )
        elif isinstance(balrog_file, np.ndarray):
            axes.plot(
                zmean,
                balrog_file[bin_idx, :],  # Row corresponding to the current tomographic bin
                color=color_balrog,
                lw=3
            )

        # Plot gaNdalF mean line and shaded ±1σ range
        axes.axvline(gandalf_means[bin_idx], color=color_gandalf[bin_idx], linestyle='-', linewidth=3,
                            label=f'$<z>$ gaNdalF {gandalf_means[bin_idx]:.4f} ± {gandalf_stds[bin_idx]:.3f}')
        axes.axvspan(
            gandalf_means[bin_idx] - gandalf_stds[bin_idx],
            gandalf_means[bin_idx] + gandalf_stds[bin_idx],
            color=color_gandalf[bin_idx], alpha=0.1
        )

        # Plot Balrog mean line and shaded ±0.01 range
        axes.axvline(balrog_means[bin_idx], color=color_balrog, linestyle=line_style_balrog[bin_idx], linewidth=3,
                            label=f'$<z>$ Balrog {balrog_means[bin_idx]:.4f} ± 0.01')
        axes.axvspan(
            balrog_means[bin_idx] - 0.01,
            balrog_means[bin_idx] + 0.01,
            color=color_balrog, alpha=0.1
        )

    # Customize major and minor ticks
    axes.xaxis.set_major_locator(ticker.MultipleLocator(0.1))  # Major ticks every 0.1
    axes.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))  # Minor ticks every 0.05
    # Make minor ticks visible and adjust size
    axes.tick_params(axis='x', which='both', direction='in', length=6)  # Longer major ticks
    axes.tick_params(axis='x', which='minor', direction='out', length=6)  # Shorter minor ticks

    # Customize each subplot
    axes.set_xlim((0, 1.3))
    axes.set_ylim(0, 7)
    # Remove y-tick labels but keep tick marks
    axes.set_yticklabels([])  # Remove numerical labels
    axes.tick_params(axis='y', length=5, direction='in', left=True, right=True)

    # Add "Probability Density" label on the y-axis for each subplot
    axes.set_ylabel(f'Probability')  # , fontsize=font_size_labels
    axes.legend()
    axes.grid()
    axes.set_xlabel('Redshift')  # , fontsize=font_size_labels

    fig.suptitle(title)  # Set the title for the whole figure  , fontsize=font_size_title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout

    if show_plot:
        plt.show()
    if save_plot and save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=300)


def plot_binning_statistics(df_gandalf, df_balrog, conditions, bands, sample_size=10000, show_plot=True,
                            save_plot=False, path_save_plots=""):
    color_gandalf = '#ff8c00'
    color_balrog = '#51a6fb'

    for condition in conditions:
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

        cond_lims = np.percentile(df_balrog[condition], [2, 98])
        standard_levels = [0.393, 0.865, 0.989]

        for idx, out in enumerate(zip(outputs, output_errs, true_outputs)):
            output_ = out[0]
            output_err_ = out[1]
            true_output_ = out[2]

            diff_true = (df_balrog[true_output_] - df_balrog[output_]) / df_balrog[true_output_]

            df_conditional_true = pd.DataFrame({
                condition: df_balrog[condition],
                f"residual band {bands[idx]}": diff_true,
                "dataset": ["Balrog" for _ in range(len(df_balrog[condition]))]
            })
            bin_means_true, bin_edges_mean_true, binnumber_true = binned_statistic(
                df_balrog[condition], diff_true, statistic='median', bins=10, range=cond_lims)
            bin_stds_true, bin_edges_true, binnumber_true = binned_statistic(
                df_balrog[condition], diff_true, statistic=median_abs_deviation, bins=10, range=cond_lims)
            xerr_true = (bin_edges_mean_true[1:] - bin_edges_mean_true[:-1]) / 2
            xmean_true = (bin_edges_mean_true[1:] + bin_edges_mean_true[:-1]) / 2
            lst_axis_con[idx].errorbar(
                xmean_true, bin_means_true, xerr=xerr_true, yerr=bin_stds_true, color=color_balrog, lw=2,
                label='Balrog')

            diff_generated = (df_gandalf[true_output_] - df_gandalf[output_]) / df_gandalf[true_output_]

            df_conditional_generated = pd.DataFrame({
                condition: df_gandalf[condition],
                f"residual band {bands[idx]}": diff_generated,
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
                color=color_gandalf, lw=2, label='gaNdalF')

            m, s = np.median(diff_generated), median_abs_deviation(diff_generated)
            range_ = [m - 4 * s, m + 4 * s]

            df_conditional_true_sampled = df_conditional_true.sample(n=sample_size, random_state=42)
            df_conditional_generated_sampled = df_conditional_generated.sample(n=sample_size, random_state=42)

            sns.kdeplot(
                data=df_conditional_true_sampled,
                x=condition,
                y=f"residual band {bands[idx]}",
                fill=True,
                thresh=0,
                alpha=.4,
                levels=standard_levels,  # 10
                color=color_balrog,
                legend="Balrog",
                ax=lst_axis_con[idx]
            )
            sns.kdeplot(
                data=df_conditional_generated_sampled,
                x=condition,
                y=f"residual band {bands[idx]}",
                fill=False,
                thresh=0,
                levels=standard_levels,  # 10
                alpha=.8,
                color=color_gandalf,
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
        if save_plot is True:
            save_name = f"{path_save_plots}/cond_stat_binning_{condition}.png"
            plt.savefig(save_name, dpi=300)
        if show_plot is True:
            plt.show()
        plt.clf()
        plt.close()


def plot_binning_statistics_test2(df_gandalf, df_balrog, conditions, bands, sample_size=10000, show_plot=True,
                                  save_plot=False, path_save_plots=""):
    color_gandalf = '#ff8c00'
    color_balrog = '#51a6fb'
    standard_levels = [0.393, 0.865, 0.989]

    for idx, band in enumerate(bands):
        # Create a figure for the band
        fig, axes = plt.subplots(nrows=1, ncols=len(conditions), figsize=(6 * len(conditions), 6), sharey=True)
        fig.suptitle(f"Binning Statistics and KDE for Band {band.upper()}", fontsize=16)

        if len(conditions) == 1:
            axes = [axes]  # Ensure axes is iterable for single condition

        for ax, condition in zip(axes, conditions):
            output = f'unsheared/mag_{band}'
            true_output = f'BDF_MAG_DERED_CALIB_{band.upper()}'
            output_err = f'unsheared/mag_err_{band}'

            # Calculate normalized residuals
            diff_true = (df_balrog[true_output] - df_balrog[output]) / (df_balrog[true_output] + 1e-6)
            diff_generated = (df_gandalf[true_output] - df_gandalf[output]) / (df_gandalf[true_output] + 1e-6)

            # Downsample data for faster plotting
            df_conditional_true = pd.DataFrame({condition: df_balrog[condition], f"residual band {band}": diff_true})
            df_conditional_generated = pd.DataFrame({condition: df_gandalf[condition], f"residual band {band}": diff_generated})

            sampled_true = df_conditional_true.sample(n=min(sample_size, len(df_conditional_true)), random_state=42)
            sampled_generated = df_conditional_generated.sample(n=min(sample_size, len(df_conditional_generated)), random_state=42)

            # Binning statistics
            cond_lims = np.percentile(df_balrog[condition], [2, 98])
            bin_means_true, bin_edges_mean_true, binnumber_true = binned_statistic(
                sampled_true[condition], sampled_true[f"residual band {band}"], statistic='median', bins=10, range=cond_lims)
            bin_stds_true, _, _ = binned_statistic(
                sampled_true[condition], sampled_true[f"residual band {band}"], statistic=median_abs_deviation, bins=10, range=cond_lims)

            bin_means_generated, bin_edges_mean_generated, _ = binned_statistic(
                sampled_generated[condition], sampled_generated[f"residual band {band}"], statistic='median', bins=10, range=cond_lims)
            bin_stds_generated, _, _ = binned_statistic(
                sampled_generated[condition], sampled_generated[f"residual band {band}"], statistic=median_abs_deviation, bins=10, range=cond_lims)

            xerr_true = (bin_edges_mean_true[1:] - bin_edges_mean_true[:-1]) / 2
            xmean_true = (bin_edges_mean_true[1:] + bin_edges_mean_true[:-1]) / 2
            xerr_generated = (bin_edges_mean_generated[1:] - bin_edges_mean_generated[:-1]) / 2
            xmean_generated = (bin_edges_mean_generated[1:] + bin_edges_mean_generated[:-1]) / 2

            # KDE plot
            sns.kdeplot(
                data=sampled_true,
                x=condition,
                y=f"residual band {band}",
                fill=True,
                thresh=0,
                alpha=.4,
                levels=standard_levels,
                color=color_balrog,
                ax=ax,
                label="Balrog KDE"
            )
            sns.kdeplot(
                data=sampled_generated,
                x=condition,
                y=f"residual band {band}",
                fill=False,
                thresh=0,
                levels=standard_levels,
                color=color_gandalf,
                ax=ax,
                label="gaNdalF KDE"
            )

            # Plot binned statistics
            ax.errorbar(
                xmean_true, bin_means_true, xerr=xerr_true, yerr=bin_stds_true, fmt='o',
                color=color_balrog, label="Balrog Binning"
            )
            ax.errorbar(
                xmean_generated, bin_means_generated, xerr=xerr_generated, yerr=bin_stds_generated, fmt='o',
                color=color_gandalf, label="gaNdalF Binning"
            )

            # Formatting
            ax.set_xlim(cond_lims)
            ax.set_xlabel(condition, fontsize=12)
            ax.set_ylabel(f"Residual Band {band}", fontsize=12)
            ax.legend()

        # Save or show the plot
        if save_plot:
            save_name = f"{path_save_plots}/binning_kde_band_{band}.png"
            plt.savefig(save_name, dpi=300)
        if show_plot:
            plt.show()
        plt.close()


def plot_binning_statistics_comparison(
        df_gandalf, df_balrog, sample_size=10000, show_plot=True, save_plot=False, path_save_plots=""
):
    #import numpy as np
    #import matplotlib.pyplot as plt
    #import seaborn as sns
    #from scipy.stats import binned_statistic, median_abs_deviation

    print("start plotting")

    color_gandalf = '#ff8c00'
    color_balrog = '#51a6fb'
    standard_levels = [0.393, 0.865, 0.989]

    # Define the grid layout
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 18), sharex=False, sharey=False)

    # Define the conditions and their corresponding bands, grouped by column
    conditions_bands = {
        "R": [("FWHM_WMEAN_R", "r"), ("AIRMASS_WMEAN_R", "r"), ("MAGLIM_R", "r"), ("EBV_SFD98", "r")],
        "I": [("FWHM_WMEAN_I", "i"), ("AIRMASS_WMEAN_I", "i"), ("MAGLIM_I", "i"), ("EBV_SFD98", "i")],
        "Z": [("FWHM_WMEAN_Z", "z"), ("AIRMASS_WMEAN_Z", "z"), ("MAGLIM_Z", "z"), ("EBV_SFD98", "z")],
    }

    for col_idx, (band, conditions) in enumerate(conditions_bands.items()):
        for row_idx, (condition, band_letter) in enumerate(conditions):
            ax = axes[row_idx, col_idx]  # Correctly index the subplot based on row and column
            print(f"Plotting {band_letter.upper()} - {condition}")

            output = f'unsheared/mag_{band_letter}'
            output_err = f'unsheared/mag_err_{band_letter}'

            # Calculate residuals normalized by Balrog uncertainty
            residual = (df_balrog[output] - df_gandalf[output]) / df_balrog[output_err]
            residual = residual[np.isfinite(residual)]  # Keep only finite values

            # Downsample data for faster plotting
            df_conditional = pd.DataFrame({condition: df_balrog[condition], f"Residual {band_letter}": residual})
            sampled = df_conditional.sample(n=min(sample_size, len(df_conditional)), random_state=42)
            # Filter out non-finite values
            finite_mask = sampled[condition].replace([np.inf, -np.inf], np.nan).notna()
            filtered_sampled = sampled[finite_mask]

            # Binning statistics
            cond_lims = np.percentile(df_balrog[condition], [2, 98])
            bin_means, bin_edges, binnumber = binned_statistic(
                filtered_sampled[condition], filtered_sampled[f"Residual {band_letter}"], statistic='median', bins=10, range=cond_lims)
            bin_stds, _, _ = binned_statistic(
                filtered_sampled[condition], filtered_sampled[f"Residual {band_letter}"], statistic=median_abs_deviation, bins=10,
                range=cond_lims)

            xerr = (bin_edges[1:] - bin_edges[:-1]) / 2
            xmean = (bin_edges[1:] + bin_edges[:-1]) / 2

            # KDE plot
            sns.kdeplot(
                data=filtered_sampled,
                x=condition,
                y=f"Residual {band_letter}",
                fill=True,
                thresh=0,
                alpha=.4,
                levels=standard_levels,
                color=color_balrog,
                ax=ax
            )

            # Plot binned statistics
            ax.errorbar(
                xmean, bin_means, xerr=xerr, yerr=bin_stds, fmt='o',
                color=color_balrog, label="Balrog Residuals"
            )

            if len(residual) == 0:
                print(f"Empty residual array for condition: {condition}, band: {band_letter}. Using default range.")
                range_ = [-0.02, 0.02]  # Default range
            else:
                m, s = np.median(residual), median_abs_deviation(residual)
                range_ = [m - 4 * s, m + 4 * s]

            # Check if range_ is valid
            if np.any(np.isnan(range_)) or np.any(np.isinf(range_)):
                print(f"Invalid range for condition: {condition}, band: {band_letter}. Using default.")
                range_ = [-0.02, 0.02]  # Default range or some sensible fallback

            # Formatting
            ax.set_xlim(cond_lims)
            ax.set_ylim(range_)
            ax.set_xlabel(condition, fontsize=8)
            ax.set_ylabel(f"Residual Band {band_letter}", fontsize=8)
            ax.set_title(f"{band_letter.upper()} - {condition}", fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=6)

    # Adjust layout and save/show the plot
    fig.tight_layout()
    if save_plot:
        save_name = f"{path_save_plots}/comparison_binning_kde.png"
        plt.savefig(save_name, dpi=300)
    if show_plot:
        plt.show()
    plt.close()


def plot_binning_statistics_properties(
        df_gandalf, df_balrog, sample_size=10000, show_plot=True, save_plot=False, path_save_plots=""
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import binned_statistic, median_abs_deviation

    print("start plotting")

    color_gandalf = '#ff8c00'
    color_balrog = '#51a6fb'
    standard_levels = [0.393, 0.865, 0.989]

    # Define the grid layout
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 18), sharex=False, sharey=False)

    # Define conditions and properties to analyze
    conditions = ["FWHM_WMEAN_R", "AIRMASS_WMEAN_R", "MAGLIM_R", "EBV_SFD98"]
    properties = ["unsheared/snr", "unsheared/T", "unsheared/weight"]

    # Iterate over properties and conditions
    for col_idx, property in enumerate(properties):
        for row_idx, condition in enumerate(conditions):
            ax = axes[row_idx, col_idx]
            print(f"Plotting {property} vs. {condition}")

            # Calculate residuals for the property
            residual = (df_balrog[property] - df_gandalf[property]) / df_balrog[property]
            residual = residual[np.isfinite(residual)]  # Keep only finite values

            # Downsample for faster plotting
            df_conditional = pd.DataFrame({condition: df_balrog[condition], f"Residual {property}": residual})
            sampled = df_conditional.sample(n=min(sample_size, len(df_conditional)), random_state=42)
            # Filter out non-finite values
            finite_mask = sampled[condition].replace([np.inf, -np.inf], np.nan).notna()
            filtered_sampled = sampled[finite_mask]

            # Binning statistics
            cond_lims = np.percentile(df_balrog[condition], [2, 98])
            bin_means, bin_edges, binnumber = binned_statistic(
                filtered_sampled[condition], filtered_sampled[f"Residual {property}"], statistic='median', bins=10, range=cond_lims)
            bin_stds, _, _ = binned_statistic(
                filtered_sampled[condition], filtered_sampled[f"Residual {property}"], statistic=median_abs_deviation, bins=10,
                range=cond_lims)

            xerr = (bin_edges[1:] - bin_edges[:-1]) / 2
            xmean = (bin_edges[1:] + bin_edges[:-1]) / 2

            # KDE plot
            sns.kdeplot(
                data=filtered_sampled,
                x=condition,
                y=f"Residual {property}",
                fill=True,
                thresh=0,
                alpha=.4,
                levels=standard_levels,
                color=color_balrog,
                ax=ax
            )

            # Plot binned statistics
            ax.errorbar(
                xmean, bin_means, xerr=xerr, yerr=bin_stds, fmt='o',
                color=color_balrog, label="Balrog Residuals"
            )
            band_letter = 'add letter'
            # Formatting
            if len(residual) == 0:
                print(f"Empty residual array for condition: {condition}, band: {band_letter}. Using default range.")
                range_ = [-0.02, 0.02]  # Default range
            else:
                m, s = np.median(residual), median_abs_deviation(residual)
                range_ = [m - 4 * s, m + 4 * s]

            # Check if range_ is valid
            if np.any(np.isnan(range_)) or np.any(np.isinf(range_)):
                print(f"Invalid range for condition: {condition}, band: {band_letter}. Using default.")
                range_ = [-0.02, 0.02]  # Default range or some sensible fallback

            ax.set_xlim(cond_lims)
            ax.set_ylim(range_)
            ax.set_xlabel(condition, fontsize=8)
            ax.set_ylabel(f"Residual {property}", fontsize=8)
            ax.set_title(f"{property} vs. {condition}", fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)

            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=6)

    # Adjust layout and save/show the plot
    fig.tight_layout()
    if save_plot:
        save_name = f"{path_save_plots}/properties_vs_conditions.png"
        plt.savefig(save_name, dpi=300)
    if show_plot:
        plt.show()
    plt.close()


def plot_binning_statistics_combined(df_gandalf, df_balrog, sample_size=10000, standard_levels=3, title="",
                                     save_plot=False, show_plot=True, save_name="", plot_scatter=False):
    print("Start plotting")

    import matplotlib as mpl
    # Use LaTeX fonts in matplotlib
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern Roman']
    mpl.rcParams['axes.labelsize'] = 24
    mpl.rcParams['font.size'] = 28
    mpl.rcParams['legend.fontsize'] = 24
    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['ytick.labelsize'] = 20

    # Define colors and levels
    color_gandalf = 'darkgreen'
    color_balrog = 'purple'
    color_residual = 'black'

    dict_conditions = {
        "r": {
            "FWHM_WMEAN": {"Short_Name": "FWHM", "Limits": [(0.75, 1.25), (-0.04, 0.04)]},
            "AIRMASS_WMEAN": {"Short_Name": "AIRMASS", "Limits": [(0.95, 1.45), (-0.04, 0.04)]},
            "MAGLIM": {"Short_Name": "MAGLIM", "Limits": [(23.25, 24.5), (-0.04, 0.04)]},
            "EBV_SFD98": {"Short_Name": "EBV_SFD98", "Limits": [(-0.005, 0.1), (-0.04, 0.05)]},
        },
        "i": {
            "FWHM_WMEAN": {"Short_Name": "FWHM", "Limits": [(0.72, 1.1), (-0.04, 0.04)]},
            "AIRMASS_WMEAN": {"Short_Name": "AIRMASS", "Limits": [(0.95, 1.45), (-0.04, 0.04)]},
            "MAGLIM": {"Short_Name": "MAGLIM", "Limits": [(22.75, 24), (-0.04, 0.04)]},
            "EBV_SFD98": {"Short_Name": "EBV_SFD98", "Limits": [(-0.005, 0.1), (-0.04, 0.05)]},
        },
        "z": {
            "FWHM_WMEAN": {"Short_Name": "FWHM", "Limits": [(0.68, 1.12), (-0.04, 0.05)]},
            "AIRMASS_WMEAN": {"Short_Name": "AIRMASS", "Limits": [(0.95, 1.45), (-0.04, 0.05)]},
            "MAGLIM": {"Short_Name": "MAGLIM", "Limits": [(22, 23.25), (-0.04, 0.05)]},
            "EBV_SFD98": {"Short_Name": "EBV_SFD98", "Limits": [(-0.005, 0.1), (-0.05, 0.06)]},
        }
    }

    for band_idx, band in enumerate(dict_conditions.keys()):
        fig = plt.figure(figsize=(18, 24))  # Adjust height as needed
        main_gs = GridSpec(4, 1, figure=fig)  # Rows: conditions, Columns: bands main_gs = GridSpec(len(conditions), len(bands), figure=fig)  # Rows: conditions, Columns: bands

        for condition_idx, condition_base in enumerate(dict_conditions[band].keys()):
            # Construct condition and label strings
            if condition_base != "EBV_SFD98":
                condition = f"{condition_base}_{band.upper()}"
                label = f"{dict_conditions[band][condition_base]['Short_Name']} {band.upper()}"
            else:
                condition = condition_base  # EBV_SFD98 does not vary by band
                label = dict_conditions[band][condition_base]['Short_Name']

            print(f"Plotting {band.upper()} - {condition}, {label}")

            # Create a nested GridSpec within the main GridSpec cell
            inner_gs = GridSpecFromSubplotSpec(2, 1,  # 2 rows, 1 column
                                               subplot_spec=main_gs[condition_idx],  # band_idx
                                               height_ratios=[3, 1],  # Adjust as needed
                                               hspace=0.05)  # No space between distribution and error plots

            # Subplots for main plot and error bar plot
            ax_main = fig.add_subplot(inner_gs[0])
            ax_err = fig.add_subplot(inner_gs[1], sharex=ax_main)

            # Residual calculations
            output = f'unsheared/mag_{band}'
            true_output = f'BDF_MAG_DERED_CALIB_{band.upper()}'
            output_err = f'unsheared/mag_err_{band}'

            residual = (df_balrog[output] - df_gandalf[output]) / df_balrog[output_err]
            residual = residual[np.isfinite(residual)]

            residual_gandalf = (df_gandalf[true_output] - df_gandalf[output]) / df_gandalf[true_output]
            residual_balrog = (df_balrog[true_output] - df_balrog[output]) / df_balrog[true_output]

            # Prepare dataframes
            df_conditional = pd.DataFrame({condition: df_balrog[condition], "residual": residual})
            df_conditional_gandalf = pd.DataFrame({condition: df_gandalf[condition], "residual": residual_gandalf})
            df_conditional_balrog = pd.DataFrame({condition: df_balrog[condition], "residual": residual_balrog})

            # Downsample data
            if sample_size is not None:
                sampled_generated = df_conditional.sample(n=min(sample_size, len(df_conditional)), random_state=42)
                sampled_generated_gandalf = df_conditional_gandalf.sample(
                    n=min(sample_size, len(df_conditional_gandalf)), random_state=42)
                sampled_generated_balrog = df_conditional_balrog.sample(
                    n=min(sample_size, len(df_conditional_balrog)), random_state=42)
            else:
                sampled_generated = df_conditional
                sampled_generated_gandalf = df_conditional_gandalf
                sampled_generated_balrog = df_conditional_balrog

            # KDE plot on ax_main
            if plot_scatter is True:
                # First, plot the data points
                ax_main.scatter(sampled_generated_gandalf[condition], sampled_generated_gandalf["residual"],
                                color=color_gandalf, s=1, alpha=0.5)
                ax_main.scatter(sampled_generated_balrog[condition], sampled_generated_balrog["residual"],
                                color=color_balrog, s=1, alpha=0.5)

            # Then plot the KDE contours
            sns.kdeplot(
                data=sampled_generated_gandalf, x=condition, y="residual", fill=False, alpha=0.8,
                levels=standard_levels,
                color=color_gandalf,
                ax=ax_main,  # , bw_method="silverman"
                linewidths=3
            )
            sns.kdeplot(
                data=sampled_generated_balrog, x=condition, y="residual", fill=False, alpha=0.8,
                levels=standard_levels,
                color=color_balrog,
                ax=ax_main,  # , bw_method="silverman"
                linewidths=3
            )

            # Add dashed lines at y=0 and x=mean(condition)
            mean_condition = sampled_generated[condition].mean()
            ax_main.axhline(0, color='black')

            line_res_balrog = ax_main.axhline(
                residual_balrog.mean(), linestyle='--', color=color_balrog, linewidth=3,
                label=f"$<$res. Balrog$>$ = {residual_balrog.mean():.3f}")
            line_res_gandalf = ax_main.axhline(
                residual_gandalf.mean(), linestyle='--', color=color_gandalf, linewidth=3,
                label=f"$<$res. gaNdalF$>$ = {residual_gandalf.mean():.3f}")

            ax_main.axvline(mean_condition, color='black')
            ax_err.axhline(0, color='black')
            ax_err.axvline(mean_condition, color='black')

            cond_lims = np.percentile(df_balrog[condition], [0.01, 99.9])

            # Binned statistics for error bar plot
            bin_means, bin_edges, _ = binned_statistic(
                sampled_generated[condition], sampled_generated["residual"], statistic='median', bins=10,
                range=cond_lims)
            bin_stds, _, _ = binned_statistic(
                sampled_generated[condition], sampled_generated["residual"], statistic=median_abs_deviation,
                bins=10, range=cond_lims)
            xmean = (bin_edges[1:] + bin_edges[:-1]) / 2

            # Error bar plot on ax_err
            ax_err.errorbar(
                xmean,
                bin_means,
                yerr=bin_stds,
                fmt='o',  # Change '.' to 'o' for larger circular markers
                color=color_residual,
                label="gaNdalF",
                linewidth=2,  # Controls the error bar line width
                markersize=10,  # Controls the marker size (increase this for bigger dots)
                capsize=8,  # Adds caps to error bars
                capthick=4  # Adjusts the cap thickness
            )
            bin_nan_median = np.nanmedian(bin_means)
            ax_err.axhline(bin_nan_median, linestyle='--', color=color_residual,
                           label=f"$<$abs. res.$>$ = {bin_nan_median:.3f}", linewidth=3)  # bin_means.mean()

            # Adjust axis limits
            if len(residual) > 0:
                m, s = np.median(residual), median_abs_deviation(residual)
                y_range = [m - 4 * s, m + 4 * s]
            else:
                y_range = [-0.02, 0.02]

            ax_main.set_ylim(dict_conditions[band][condition_base]['Limits'][1])
            ax_main.set_xlim(dict_conditions[band][condition_base]['Limits'][0])
            ax_err.set_ylim(y_range)
            # ax_err.set_xlim(cond_lims)
            ax_err.set_xlim(dict_conditions[band][condition_base]['Limits'][0])
            handles_err = [ax_err.axhline(bin_nan_median, linestyle='--',
                                          color=color_residual,
                                          label=f"$<$abs. res.$>$ = {bin_nan_median:.3f}")
                           ]
            ax_err.legend(handles=handles_err, loc="upper right", ncol=1, frameon=True)  # bbox_to_anchor=(0.99, 0.97),

            # Formatting
            # ax_main.set_title(f"{band} - {label}", fontsize=font_size_labels)
            ax_main.set_ylabel(f"Normalized {band.lower()} Band \n Magnitude Difference")  # , fontsize=font_size_labels
            ax_err.set_xlabel(label)  # , fontsize=font_size_labels
            ax_err.set_ylabel(f"Median \n Residual")
            ax_main.set_xlabel('')
            handles_main = [
                line_res_balrog,
                line_res_gandalf
            ]
            ax_main.legend(handles=handles_main, loc="upper right", ncol=1, frameon=True)  # bbox_to_anchor=(0.99, 0.97),

            ax_main.grid(True)
            ax_err.grid(True)

            # Remove x tick labels from ax_main
            plt.setp(ax_main.get_xticklabels(), visible=False)
            ax_main.tick_params(axis='x', which='both', length=0)

        # Create custom legend handles (outside loop, only once per figure)
        handles_fig = [
            mpatches.Patch(color=color_gandalf, label='gaNdalF'),
            mpatches.Patch(color=color_balrog, label='Balrog'),
            mpatches.Patch(color=color_residual, label='Residual'),
        ]

        # Move legend to the top right of the figure
        fig.legend(handles=handles_fig, loc="upper right", bbox_to_anchor=(0.98, 0.96), ncol=1, frameon=True)  # Ensure frame is visible

        plt.suptitle(title)  # Figure title
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust space to fit the external legend

        # Save or show plot
        if save_plot:
            outname = f"{save_name}_{band}.pdf"
            fig.savefig(outname, dpi=300, bbox_inches='tight')  # Ensure legend is included
        if show_plot:
            plt.show()


def make_gif(frame_folder, name_save_folder, fps=10):
    filenames = natsorted(os.listdir(frame_folder))
    images_data = []
    for filename in filenames:
        image = imageio.imread(f"{frame_folder}/{filename}")
        images_data.append(image)
    imageio.mimwrite(uri=f"{name_save_folder}", ims=images_data, format='.gif', duration=int(1000*1/fps))

    
def plot_multivariate_clf(df_balrog_detected, df_gandalf_detected, df_balrog_not_detected, df_gandalf_not_detected,
                          columns, show_plot, save_plot, save_name, train_plot=False, sample_size=5000, x_range=(18, 26),
                          title='Histogram'):
    import matplotlib as mpl
    # Use LaTeX fonts in matplotlib
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern Roman']
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['font.size'] = 24
    mpl.rcParams['legend.fontsize'] = 20
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16

    print("Sample data...")
    if sample_size is None:
        df_gandalf_detected_sample = df_gandalf_detected
        df_balrog_detected_sample = df_balrog_detected
        df_gandalf_not_detected_sample = df_gandalf_not_detected
        df_balrog_not_detected_sample = df_balrog_not_detected
    else:
        df_gandalf_detected_sample = df_gandalf_detected.sample(n=sample_size, random_state=42)
        df_balrog_detected_sample = df_balrog_detected.sample(n=sample_size, random_state=42)
        df_gandalf_not_detected_sample = df_gandalf_not_detected.sample(n=sample_size, random_state=42)
        df_balrog_not_detected_sample = df_balrog_not_detected.sample(n=sample_size, random_state=42)
    print("Data sampled!")

    plot_linewidth = 2.5

    # Calculate the number of rows and columns
    num_cols = 4
    num_rows = 4
    if train_plot is True:
        num_cols = 1
        num_rows = 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

    color_balrog_detected = 'purple'  # '#51a6fb'
    color_gandalf_detected = 'darkgreen'  # '#ff8c00'
    color_balrog_not_detected = "darkviolet"  # 'coral'
    color_gandalf_not_detected = "green"  # 'blueviolet'

    for i, col in enumerate(columns.keys()):
        df_detected = pd.DataFrame({
            f"{col}": df_gandalf_detected_sample[col].to_list() + df_balrog_detected_sample[col].to_list(),
            f"BDF_MAG_DERED_CALIB_I": df_gandalf_detected_sample["BDF_MAG_DERED_CALIB_I"].to_list() + df_balrog_detected_sample["BDF_MAG_DERED_CALIB_I"].to_list(),
            f"Catalog": ["gaNdalF" for _ in range(len(df_gandalf_detected_sample[col]))] + ["Balrog" for _ in range(len(df_balrog_detected_sample[col]))],
        })

        df_not_detected = pd.DataFrame({
            f"{col}": df_gandalf_not_detected_sample[col].to_list() + df_balrog_not_detected_sample[col].to_list(),
            f"BDF_MAG_DERED_CALIB_I": df_gandalf_not_detected_sample["BDF_MAG_DERED_CALIB_I"].to_list() + df_balrog_not_detected_sample["BDF_MAG_DERED_CALIB_I"].to_list(),
            f"Catalog": ["gaNdalF" for _ in range(len(df_gandalf_not_detected_sample[col]))] + ["Balrog" for _ in range(len(df_balrog_not_detected_sample[col]))],
        })

        pos = columns[col]["position"]

        if train_plot is True:
            ax = axes
        else:
            ax = axes[pos[0], pos[1]]



        # Set the plot ranges
        y_range = columns[col]["range"]
        label = columns[col]["label"]

        try:
            sns.kdeplot(
                data=df_detected,
                x=f"BDF_MAG_DERED_CALIB_I",
                y=f"{col}",
                hue="Catalog",
                ax=ax,
                levels=3,  #[0.393, 0.865, 0.989],  # normalized_density_levels_gandalf_detected,  # [0.393, 0.865, 0.989]
                fill=False,
                palette={"gaNdalF": color_gandalf_detected, "Balrog": color_balrog_detected},
                linewidths=plot_linewidth,
                alpha=0.5,
                legend=False
            )

        except Exception as e:
            print(f"An error occurred with plotting seaborn gandalf detected: {e}")

        # Plot contours
        try:
            sns.kdeplot(
                data=df_not_detected,
                x=f"BDF_MAG_DERED_CALIB_I",
                y=f"{col}",
                hue="Catalog",
                ax=ax,
                levels=3,
                fill=True,
                palette={"gaNdalF": color_gandalf_not_detected, "Balrog": color_balrog_not_detected},
                alpha=0.5,
                linewidths=plot_linewidth,
                legend=False
            )

        except Exception as e:
            print(f"An error occurred with plotting seaborn gandalf not detected: {e}")

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_ylabel(label) # , fontsize=font_size_label

        # Add axis labels only to the bottom row subplots
        if train_plot is True:
            ax.set_xlabel('BDF Mag I')  # , fontsize=font_size_label
        else:
            if i >= len(axes) - num_cols:
                ax.set_xlabel('BDF Mag I') # , fontsize=font_size_label

        # Remove any unused subplots
    if train_plot is False:
        fig.delaxes(axes[0, 3])
        fig.delaxes(axes[1, 3])

    # Customize layout and legend
    legend_elements = [
        Line2D([0], [0], color=color_gandalf_detected, lw=plot_linewidth, alpha=0.5, linestyle='-', label='gaNdalF detected'),
        Line2D([0], [0], color=color_balrog_detected, lw=plot_linewidth, alpha=0.5, linestyle='-', label='Balrog detected'),
        mpatches.Patch(color=color_gandalf_not_detected, alpha=0.5, label='gaNdalF non-detected'),
        mpatches.Patch(color=color_balrog_not_detected, alpha=0.5, label='Balrog non-detected')
    ]

    fig.legend(
        handles=legend_elements,
        loc='upper right',
        #fontsize=font_size_legend,
        bbox_to_anchor=(1.0, 0.76)
    )

    plt.suptitle(title)  # , fontsize=font_size_title
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close(fig)


def plot_balrog_histogram_with_error(
    df_gandalf, df_balrog, columns, labels, ranges, binwidths,
    title, show_plot, save_plot, save_name
):
    import matplotlib as mpl
    # Use LaTeX fonts in matplotlib
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern Roman']
    mpl.rcParams['axes.labelsize'] = 24
    mpl.rcParams['font.size'] = 28
    mpl.rcParams['legend.fontsize'] = 24
    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['ytick.labelsize'] = 20

    # Define colors and levels
    color_gandalf = 'darkgreen'
    color_balrog = 'purple'

    # Determine subplot grid
    ncols = 3
    nrows = (len(columns) + ncols - 1) // ncols

    # Create main figure and GridSpec
    fig = plt.figure(figsize=(16, 5 * nrows))  # Adjust vertical size as needed
    main_gs = GridSpec(nrows, ncols, figure=fig)

    for idx, col in enumerate(columns):
        row_idx = idx // ncols
        col_idx = idx % ncols

        # Create a nested GridSpec for each subplot with no vertical space between hist and error
        inner_gs = GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=main_gs[row_idx, col_idx],
            height_ratios=[3,1],
            hspace=0
        )

        ax_hist = fig.add_subplot(inner_gs[0])
        ax_error = fig.add_subplot(inner_gs[1], sharex=ax_hist)

        binwidth = binwidths[idx] if binwidths[idx] is not None else 0.2

        # Set range for histogram
        if ranges[idx] is not None:
            range_min, range_max = ranges[idx]
        else:
            range_min = min(df_gandalf[col].min(), df_balrog[col].min())
            range_max = max(df_gandalf[col].max(), df_balrog[col].max())

        # Create common bins
        bins = np.arange(range_min, range_max + binwidth, binwidth)

        # Plot histograms
        sns.histplot(
            data=df_gandalf,
            x=col,
            ax=ax_hist,
            bins=bins,
            element="step",
            stat="count",
            color=color_gandalf,
            log_scale=(False, True),
            fill=False,
            label="gaNdalF"
        )
        sns.histplot(
            data=df_balrog,
            x=col,
            ax=ax_hist,
            bins=bins,
            element="step",
            stat="count",
            color=color_balrog,
            log_scale=(False, True),
            fill=False,
            label="Balrog"
        )

        # Compute counts
        counts_gandalf, _ = np.histogram(df_gandalf[col], bins=bins)
        counts_balrog, _ = np.histogram(df_balrog[col], bins=bins)

        epsilons = 1e-10
        # Calculate percent difference and uncertainty
        counts_gandalf = counts_gandalf.astype(float)# + epsilons
        counts_balrog = counts_balrog.astype(float)# + epsilons

        with np.errstate(divide='ignore', invalid='ignore'):
            percent_error = 100 * (counts_balrog - counts_gandalf) / counts_balrog
            sigma_E = 100 * np.sqrt(counts_balrog + counts_gandalf) / counts_balrog

        # Handle division by zero
        percent_error[counts_gandalf == 0] = np.nan
        sigma_E[counts_gandalf == 0] = np.nan

        # Bin centers
        bin_centers = bins[:-1] + binwidth / 2

        # Plot error bars
        ax_error.errorbar(
            bin_centers, percent_error, yerr=sigma_E,
            fmt='o', color='black', ecolor='black', capsize=2, markersize=2, clip_on=True
        )

        # Calculate Median, ignoring NaNs
        median_percent_error = np.nanmedian(percent_error)
        ax_error.axhline(
            median_percent_error,
            color='black',
            linestyle='--',
            label=rf'Median \% Error = {median_percent_error:.3f}'
        )

        # Set limits
        ax_hist.set_xlim(range_min, range_max)
        ax_error.set_xlim(range_min, range_max)

        # Formatting
        ax_error.set_ylabel('% Error')  # , fontsize=font_size_labels
        ax_error.set_xlabel(labels[idx])  # , fontsize=font_size_labels
        ax_hist.set_ylabel('Counts')  # , fontsize=font_size_labels

        handles_err = [ax_error.axhline(median_percent_error, color='black', linestyle='--',
                                        label = rf'Med. \% Err. = {median_percent_error:.3f}')
                       ]
        ax_error.legend(handles=handles_err, loc="upper right", ncol=1, frameon=True, fontsize=14)  # bbox_to_anchor=(0.99, 0.97),

        # Hide x-axis labels on the top histogram
        plt.setp(ax_hist.get_xticklabels(), visible=False)
        ax_hist.tick_params(
            axis='x',
            labelbottom=False
        )

        # Add grid
        ax_hist.grid(True)
        ax_error.grid(True)

        # Set symmetric y-axis ticks on both sides
        ax_hist.tick_params(
            axis='y',
            direction='in',  # Ticks pointing inside
            left=True,  # Ticks on the left
            right=True  # Ticks on the right
        )
        ax_error.tick_params(
            axis='y',
            direction='in',  # Ticks pointing inside
            left=True,  # Ticks on the left
            right=True  # Ticks on the right
        )

        # Adjust error plot y-limits if desired
        ax_error.set_ylim(-15, 15)

    # Create custom legend handles (outside loop, only once per figure)
    handles_fig = [
        mpatches.Patch(color=color_gandalf, label='gaNdalF'),
        mpatches.Patch(color=color_balrog, label='Balrog')
    ]

    # Move legend to the top right of the figure
    fig.legend(handles=handles_fig, loc="upper right", bbox_to_anchor=(0.98, 0.96), ncol=1,
               frameon=True)  # Ensure frame is visible
    # Adjust layout and add title
    plt.suptitle(title, y=0.99)  # fontsize=font_size_title,
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Show or save plot
    if save_plot:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()

    # Clear the figure
    plt.clf()
    plt.close(fig)


def plot_trans_norm_compare(data_frame, data_frame_yj, data_frame_yj_scaled, column, ranges, bins, title,
                            show_plot, save_plot, save_name):
    import matplotlib as mpl
    # Use LaTeX fonts in matplotlib
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern Roman']
    mpl.rcParams['axes.labelsize'] = 24
    mpl.rcParams['font.size'] = 28
    mpl.rcParams['legend.fontsize'] = 24
    mpl.rcParams['xtick.labelsize'] = 20
    mpl.rcParams['ytick.labelsize'] = 20

    # Create main figure
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 9))

    sns.histplot(data_frame[column], kde=True, ax=axes[0], stat="density", bins=bins)
    axes[0].set_title("Original Data")
    axes[0].set_xlabel(column)
    axes[0].set_ylabel("Density")
    axes[0].grid(True)  # Add grid lines

    # Yeo-Johnson transformed data
    sns.histplot(data_frame_yj[column], kde=True, ax=axes[1], stat="density", bins=bins)
    axes[1].set_title("Yeo-Johnson \n Transformed Data")
    axes[1].set_xlabel(f"Transformed \n {column}")
    axes[1].set_ylabel("")  # Explicitly remove y-label
    axes[1].grid(True)  # Add grid lines

    # MaxAbsScaler normalized data
    sns.histplot(data_frame_yj_scaled[column], kde=True, ax=axes[2], stat="density", bins=bins)
    axes[2].set_title("MaxAbsScaler \n Normalized Data")
    axes[2].set_xlabel(f"Transformed and Normalized \n {column}")
    axes[2].set_ylabel("")  # Explicitly remove y-label
    axes[2].grid(True)  # Add grid lines

    axes[0].set_xlim(ranges[0])
    axes[1].set_xlim(ranges[1])
    axes[2].set_xlim(ranges[2])

    plt.suptitle(title, y=0.99)  # fontsize=font_size_title,

    plt.tight_layout()

    if show_plot:
        plt.show()
    if save_plot and save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()


def plot_reliability_diagram(y_true, y_prob, n_bins=10, title="Reliability Diagram", show_hist=True, show_plot=False, save_plot=True, save_name="plot.pdf"):
    """
    Plots a reliability diagram (calibration curve) for binary classifier probabilities.

    Parameters:
    - y_true: array-like, shape (n_samples,) – True binary labels (0 or 1)
    - y_prob: array-like, shape (n_samples,) – Predicted probabilities (between 0 and 1)
    - n_bins: Number of bins to divide the probability range into (default: 10)
    - title: Title of the plot (default: "Reliability Diagram")
    - show_hist: Whether to show a histogram of predicted probabilities (default: True)
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    plt.xlabel("Predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)

    if show_hist:
        plt.twinx()
        plt.hist(y_prob, range=(0, 1), bins=n_bins, alpha=0.3, edgecolor='black')
        plt.ylabel("Number of predictions")

    plt.tight_layout()

    if show_plot:
        plt.show()
    if save_plot and save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def plot_combined_reliability_diagram(
        y_true,
        y_prob_uncal,
        y_prob_cal,
        n_bins=15,
        title="Reliability Diagram: Calibrated vs. Uncalibrated",
        show_hist=True,
        score_uncal="",
        score_cal="",
        ece_uncal="",
        ece_cal="",
        show_plot=False,
        save_plot=True,
        save_name="plot.pdf"
):
    """
    Plots a combined reliability diagram for both calibrated and uncalibrated models.

    Parameters:
    - y_true: True binary labels (0 or 1)
    - y_prob_uncal: Uncalibrated model predictions
    - y_prob_cal: Calibrated model predictions
    - n_bins: Number of bins (default: 15)
    - title: Plot title
    - show_hist: Whether to show histograms (default: True)
    """
    # Calibration curves
    prob_true_uncal, prob_pred_uncal = calibration_curve(y_true, y_prob_uncal, n_bins=n_bins, strategy='uniform')
    prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_prob_cal, n_bins=n_bins, strategy='uniform')

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot reliability curves
    ax1.plot(prob_pred_uncal, prob_true_uncal, marker='o', color='C0', label=f'Uncalibrated: brier={score_uncal:.6f}, ece={ece_uncal:.6f}')
    ax1.plot(prob_pred_cal, prob_true_cal, marker='o', color='C1', label=f'Calibrated: brier={score_cal:.6f}, ece={ece_cal:.6f}')
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')

    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True)

    # Histogram overlay
    if show_hist:
        ax2 = ax1.twinx()
        ax2.hist(y_prob_uncal, range=(0, 1), bins=n_bins, alpha=0.3, color='C0', edgecolor='black', label='Uncalibrated')
        ax2.hist(y_prob_cal, range=(0, 1), bins=n_bins, alpha=0.3, color='C1', edgecolor='black', label='Calibrated')
        ax2.set_ylabel("Number of predictions")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels() if show_hist else ([], [])
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="lower right")

    fig.suptitle(title)
    fig.tight_layout()

    if show_plot:
        plt.show()
    if save_plot and save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close()


def plot_compare_mean_z_bootstrap(
        data_frame,
        n_bins=25,
        title="Reliability Diagram: Calibrated vs. Uncalibrated",
        show_plot=False,
        save_plot=True,
        save_name="plot.pdf"
):
    # Melt the DataFrame
    df_mean = data_frame[['Mean Bin 1', 'Mean Bin 2', 'Mean Bin 3', 'Mean Bin 4']].melt(
        var_name='Bins', value_name='Mean'
    )
    df_mean['Bins'] = df_mean['Bins'].str.replace('Mean ', '')  # "Bin 1", "Bin 2", ...

    # Erstelle Subplots: 2x2 Layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, sharey=False)
    axes = axes.flatten()

    # Plot je Bin
    for i, bin_label in enumerate(sorted(df_mean['Bins'].unique())):
        ax = axes[i]
        subset = df_mean[df_mean['Bins'] == bin_label]

        # Histogram
        sns.histplot(subset['Mean'], bins=n_bins, ax=ax, kde=False)

        # Mean und Median
        mean_val = subset['Mean'].mean()
        median_val = subset['Mean'].median()
        ax.axvline(mean_val, color='blue', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='orange', linestyle='--', label=f'Median: {median_val:.2f}')

        ax.set_title(bin_label)
        ax.legend()

    fig.suptitle(title, fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Show or save plot
    if save_plot:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()

    # Clear the figure
    plt.clf()
    plt.close(fig)


def plot_features(cfg, plot_log, df_gandalf, df_balrog, columns, title_prefix, epoch, today, savename):
    n_features = len(columns)
    ncols = min(n_features, 3)
    nrows = int(np.ceil(n_features / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    for i, k in enumerate(columns):
        plot_log.log_info_stream(f"{k}: gandalf NaNs={df_gandalf[k].isna().sum()}, balrog NaNs={df_balrog[k].isna().sum()}")
        plot_log.log_info_stream(f"{k}: gandalf infs={np.isinf(df_gandalf[k]).sum()}, balrog infs={np.isinf(df_balrog[k]).sum()}")
        plot_log.log_info_stream(f"{k}: gandalf unique={df_gandalf[k].nunique()}, balrog unique={df_balrog[k].nunique()}")

        x1 = df_gandalf[k].replace([np.inf, -np.inf], np.nan).dropna()
        x2 = df_balrog[k].replace([np.inf, -np.inf], np.nan).dropna()
        plot_log.log_info_stream(f"{k}: min={x1.min()}, max={x1.max()}, finite={np.isfinite(x1).all()}, len={len(x1)}")

        V_MIN, V_MAX = -1e5, 1e5

        # Clippen pro Feature, du kannst das für jede Variable auch individuell machen!
        x1_clip = x1[np.isfinite(x1)]
        x2_clip = x2[np.isfinite(x2)]
        x1_clip = x1_clip[(x1_clip >= V_MIN) & (x1_clip <= V_MAX)]
        x2_clip = x2_clip[(x2_clip >= V_MIN) & (x2_clip <= V_MAX)]

        if len(x1_clip) == 0 or len(x2_clip) == 0:
            plot_log.log_info_stream(f"Skip plotting {k}: no data after clipping")
            continue

        if np.isclose(x1_clip.max(), x1_clip.min()) or not np.isfinite([x1_clip.min(), x1_clip.max()]).all():
            plot_log.log_info_stream(f"Skip plotting {k}: invalid range in gandalf ({x1_clip.min()}..{x1_clip.max()})")
            continue
        if np.isclose(x2_clip.max(), x2_clip.min()) or not np.isfinite([x2_clip.min(), x2_clip.max()]).all():
            plot_log.log_info_stream(f"Skip plotting {k}: invalid range in balrog ({x2_clip.min()}..{x2_clip.max()})")
            continue

        sns.histplot(x=x1_clip, bins=100, ax=axes[i], label="gandalf")
        sns.histplot(x=x2_clip, bins=100, ax=axes[i], label="balrog")
        axes[i].set_yscale("log")
        axes[i].set_title(f"{epoch} {title_prefix} {k}")
        axes[i].set_xlabel(k)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    plt.legend()
    plt.savefig(savename, bbox_inches='tight', dpi=300)
    img_tensor = plot_to_tensor()
    plt.clf()
    plt.close(fig)
    return img_tensor



def plot_single_feature_dist(df, columns, title_prefix, save_name, epoch=None):
    n_features = len(columns)
    ncols = min(n_features, 3)
    nrows = int(np.ceil(n_features / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    for i, k in enumerate(columns):
        x = df[k].replace([np.inf, -np.inf], np.nan).dropna()
        x_clip = x[np.isfinite(x)]
        V_MIN, V_MAX = -1e5, 1e5
        x_clip = x_clip[(x_clip >= V_MIN) & (x_clip <= V_MAX)]
        if len(x_clip) == 0:
            continue
        if np.isclose(x_clip.max(), x_clip.min()) or not np.isfinite([x_clip.min(), x_clip.max()]).all():
            continue
        sns.histplot(x=x_clip, bins=100, ax=axes[i])
        axes[i].set_yscale("log")
        axes[i].set_title(f"{epoch if epoch is not None else ''} {title_prefix} {k}")
        axes[i].set_xlabel(k)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    img_tensor = plot_to_tensor()
    plt.clf()
    plt.close(fig)
    return img_tensor