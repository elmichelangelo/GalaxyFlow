import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
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


# def plot_compare_corner(data_frame_generated, data_frame_true, columns, labels, title, epoch, dict_delta, ranges=None,
#                         show_plot=False, save_plot=False, save_name=None):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from chainconsumer import ChainConsumer
#     from matplotlib.lines import Line2D
#
#     if epoch == 1:
#         for label in labels:
#             dict_delta[f"delta mean {label}"] = []
#             dict_delta[f"delta median {label}"] = []
#             dict_delta[f"delta q16 {label}"] = []
#             dict_delta[f"delta q84 {label}"] = []
#
#     arr_generated = data_frame_generated[columns].values
#     arr_true = data_frame_true[columns].values
#
#     # Quantiles for arr_generated
#     quantiles_generated = np.quantile(arr_generated, q=[0.16, 0.84], axis=0)
#
#     # Quantiles for arr_true
#     quantiles_true = np.quantile(arr_true, q=[0.16, 0.84], axis=0)
#
#     delta_names = ["mean", "median", "q16", "q84"]
#
#     # Calculate deltas
#     delta_mean = np.mean(arr_generated, axis=0) - np.mean(arr_true, axis=0)
#     delta_median = np.median(arr_generated, axis=0) - np.median(arr_true, axis=0)
#     delta_q16 = quantiles_generated[0, :] - quantiles_true[0, :]
#     delta_q84 = quantiles_generated[1, :] - quantiles_true[1, :]
#
#     # Update dict_delta
#     if dict_delta is not None:
#         for i, label in enumerate(labels):
#             dict_delta[f"delta mean {label}"].append(delta_mean[i])
#             dict_delta[f"delta median {label}"].append(delta_median[i])
#             dict_delta[f"delta q16 {label}"].append(delta_q16[i])
#             dict_delta[f"delta q84 {label}"].append(delta_q84[i])
#
#     # Initialize ChainConsumer
#     c = ChainConsumer()
#     c.add_chain(arr_generated, parameters=labels, name='gaNdalF', color='#ff8c00')
#     c.add_chain(arr_true, parameters=labels, name='Balrog', color='#51a6fb')
#
#     # Prepare parameter limits (extents) if ranges are provided
#     if ranges is not None:
#         extents = ranges
#     else:
#         extents = None
#
#     # Configure ChainConsumer
#     c.configure(
#         smooth=0.8,
#         shade=True,
#         shade_alpha=0.5,
#         bar_shade=True,
#         linewidths=1.2,
#         contour_labels=None,
#         plot_contour=True,
#         plot_point=False,
#         kde=True,
#         sigma2d=True,
#     )
#
#     # Plot the corner plot and retrieve fig and axes
#     fig, axes = c.plotter.plot(
#         figsize=(16, 12),
#         extents=extents,
#     )
#
#     ndim = arr_generated.shape[1]
#
#     # Adjust the figure to make space for delta plots
#     fig.subplots_adjust(top=0.9, bottom=0.1 + 0.05 * len(delta_names))
#
#     # Add title
#     if epoch is not None:
#         fig.suptitle(f'{title}, epoch {epoch}', fontsize=20)
#     else:
#         fig.suptitle(f'{title}', fontsize=20)
#
#     # Create legend elements
#     legend_elements = [
#         Line2D([0], [0], color='#ff8c00', lw=4, label='gaNdalF'),
#         Line2D([0], [0], color='#51a6fb', lw=4, label='Balrog')
#     ]
#
#     if dict_delta is not None:
#         delta_legend_elements = []
#         epochs = list(range(1, epoch + 1))
#         delta_fig_height = 1.5 * len(delta_names)
#         delta_fig, delta_axes = plt.subplots(len(delta_names), 1, figsize=(16, delta_fig_height), sharex=True)
#
#         for idx, (delta_name, delta_ax) in enumerate(zip(delta_names, delta_axes)):
#             for i, label in enumerate(labels):
#                 line, = delta_ax.plot(epochs, dict_delta[f"delta {delta_name} {label}"], '-o', label=f"{label}")
#                 if idx == 0:
#                     delta_legend_elements.append(line)
#
#             delta_ax.axhline(y=0, color='gray', linestyle='--')
#             delta_ax.set_ylim(-0.05, 0.05)
#
#             if idx == len(delta_names) - 1:
#                 delta_ax.set_xlabel('Epoch')
#             else:
#                 delta_ax.set_xticklabels([])
#
#             delta_ax.set_ylabel(f'Delta {delta_name}')
#
#         # Adjust layout
#         delta_fig.tight_layout()
#
#         # Combine the ChainConsumer plot and delta plots
#         from matplotlib.transforms import Bbox
#         delta_bbox = Bbox.from_bounds(0.125, 0.05, 0.775, 0.15 * len(delta_names))
#         fig.add_axes(delta_axes[0].get_position().translated(0, -delta_bbox.height))
#
#         # Remove the delta_fig since we've added its axes to the main fig
#         plt.close(delta_fig)
#
#         # Add legend
#         fig.legend(handles=legend_elements + delta_legend_elements, loc='upper right', fontsize=12)
#     else:
#         fig.legend(handles=legend_elements, loc='upper right', fontsize=16)
#
#     # Optionally save or show the plot
#     if save_plot:
#         plt.savefig(save_name, dpi=300)
#     if show_plot:
#         plt.show()
#     plt.close(fig)
#
#     return dict_delta


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
    fig, axs = plt.subplots(5, 3, figsize=(12, 24))  # Adjust subplots as needed
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


# def plot_multivariate_classifier(df_balrog, df_gandalf, columns, labels, ranges, show_plot, save_plot, save_name,
#                                  title='Corner Plot'):
#     import matplotlib.patches as mpatches
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     # Prepare the dataframes
#     df_gandalf_detected = df_gandalf[df_gandalf["detected"] == 1]
#     df_gandalf_not_detected = df_gandalf[df_gandalf["detected"] == 0]
#     df_balrog_detected = df_balrog[df_balrog["detected"] == 1]
#     df_balrog_not_detected = df_balrog[df_balrog["detected"] == 0]
#
#     num_cols = len(columns)
#
#     fig, axes = plt.subplots(num_cols, num_cols, figsize=(3 * num_cols, 3 * num_cols))
#
#     for i in range(num_cols):
#         for j in range(num_cols):
#             ax = axes[i, j]
#             if i < j:
#                 # Turn off plots in the upper triangle
#                 ax.axis('off')
#                 continue
#             elif i == j:
#                 # Optionally, we can turn off the diagonal subplots
#                 ax.axis('off')
#                 continue
#             else:
#                 col_i = columns[i]
#                 col_j = columns[j]
#                 # Gandalf detected KDE
#                 sns.kdeplot(
#                     x=df_gandalf_detected[col_j],
#                     y=df_gandalf_detected[col_i],
#                     fill=False,
#                     thresh=0,
#                     levels=5,
#                     cmap='Oranges',
#                     alpha=0.5,
#                     ax=ax
#                 )
#                 # Balrog detected KDE
#                 sns.kdeplot(
#                     x=df_balrog_detected[col_j],
#                     y=df_balrog_detected[col_i],
#                     fill=True,
#                     thresh=0,
#                     levels=5,
#                     cmap='Blues',
#                     alpha=0.5,
#                     ax=ax
#                 )
#                 # Gandalf not detected KDE
#                 sns.kdeplot(
#                     x=df_gandalf_not_detected[col_j],
#                     y=df_gandalf_detected[col_i],
#                     fill=False,
#                     thresh=0,
#                     levels=5,
#                     cmap='Oranges',  #'Purples',
#                     alpha=0.2,
#                     ax=ax
#                 )
#                 # Balrog not detected KDE
#                 sns.kdeplot(
#                     x=df_balrog_not_detected[col_j],
#                     y=df_balrog_not_detected[col_i],
#                     fill=True,
#                     thresh=0,
#                     levels=5,
#                     cmap='Blues',  # 'Greens',
#                     alpha=0.2,
#                     ax=ax
#                 )
#                 # Set axis limits
#                 xlim = ranges[j]
#                 ylim = ranges[i]
#                 ax.set_xlim(xlim[0], xlim[1])
#                 ax.set_ylim(ylim[0], ylim[1])
#
#                 # Set labels
#                 if i == num_cols - 1:
#                     ax.set_xlabel(labels[j], fontsize=10)
#                 else:
#                     ax.set_xlabel('')
#                 if j == 0:
#                     ax.set_ylabel(labels[i], fontsize=10)
#                 else:
#                     ax.set_ylabel('')
#                 # Adjust tick labels
#                 if i != num_cols - 1:
#                     ax.set_xticklabels([])
#                 if j != 0:
#                     ax.set_yticklabels([])
#
#         # Turn off plots in the upper triangle and diagonal
#     for i in range(num_cols):
#         for j in range(num_cols):
#             if i <= j:
#                 axes[i, j].axis('off')
#
#     # Customize layout and legend
#     legend_elements = [
#         mpatches.Patch(color='orange', label='Gandalf Detected'),
#         mpatches.Patch(color='purple', label='Gandalf Not Detected'),
#         mpatches.Patch(color='blue', label='Balrog Detected'),
#         mpatches.Patch(color='green', label='Balrog Not Detected')
#     ]
#
#     fig.legend(handles=legend_elements, loc='upper right', fontsize=12, bbox_to_anchor=(0.98, 0.95))
#
#     plt.suptitle(title, fontsize=18)
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     if show_plot:
#         plt.show()
#     if save_plot:
#         plt.savefig(save_name, dpi=300)
#     plt.clf()
#     plt.close(fig)


def plot_multivariate_classifier(df_balrog, df_gandalf, columns, labels, ranges, show_plot, save_plot, save_name,
                                 title='Histogram'):
    import matplotlib.patches as mpatches
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    import corner

    # Set the start time
    start_time = time.time()

    num_cols = int(np.round(np.sqrt(len(columns))))

    # Prepare the dataframes
    df_gandalf_detected = df_gandalf[df_gandalf["detected"] == 1]
    df_gandalf_not_detected = df_gandalf[df_gandalf["detected"] == 0]
    df_balrog_detected = df_balrog[df_balrog["detected"] == 1]
    df_balrog_not_detected = df_balrog[df_balrog["detected"] == 0]

    # Create the grid of subplots
    fig, axes = plt.subplots(num_cols, num_cols, figsize=(15, 15))

    subplot_idx_x = num_cols - 1
    subplot_idx_y = 0

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

    def compute_detection_probabilities(df, bins, feature_column):
        """Compute detection probabilities for a given feature and set of bins."""
        detected_count, _ = np.histogram(df[df["detected"] == 1][feature_column], bins=bins)
        total_count, _ = np.histogram(df[feature_column], bins=bins)
        detection_prob = detected_count / total_count
        return detection_prob

    # Define the bin edges for BDF_MAG_DERED_CALIB_I (adjust range and bin size as needed)
    bins = np.linspace(21, 26, 10)  # 10 bins from 21 to 26 in i-mag

    # Compute detection probabilities for both Gandalf and Balrog
    detection_prob_gandalf = compute_detection_probabilities(df_gandalf, bins, "BDF_MAG_DERED_CALIB_I")
    detection_prob_balrog = compute_detection_probabilities(df_balrog, bins, "BDF_MAG_DERED_CALIB_I")

    # Compute the absolute difference between the detection probabilities
    detection_prob_difference = np.abs(detection_prob_gandalf - detection_prob_balrog)
    print("Detection Probability Difference (absolute):", detection_prob_difference)

    for i, col in enumerate(columns):
        try:
            ax = axes[subplot_idx_x, subplot_idx_y]
        except TypeError:
            ax = axes
        get_elapsed_time(start_time)
        print(f"Plotting column: {col} ({i + 1}/{len(columns)})")

        # Set the plot ranges
        x_range = (21, 26)
        y_range = ranges[i]

        # Gandalf detected KDE
        corner.hist2d(
            x=df_gandalf_detected["BDF_MAG_DERED_CALIB_I"].values,
            y=df_gandalf_detected[col].values,
            ax=ax,
            bins=50,
            range=[x_range, y_range],
            levels=(0.68, 0.95),
            color='orange',
            smooth=1.0,
            plot_datapoints=False,
            fill_contours=False,
            plot_density=True,
            plot_contours=True,
            contour_kwargs={'colors': 'orange', 'alpha': 0.5}
        )
        get_elapsed_time(start_time)

        # Balrog detected KDE
        corner.hist2d(
            x=df_balrog_detected["BDF_MAG_DERED_CALIB_I"].values,
            y=df_balrog_detected[col].values,
            ax=ax,
            bins=50,
            range=[x_range, y_range],
            levels=(0.68, 0.95),
            color='blue',
            smooth=1.0,
            plot_datapoints=False,
            fill_contours=True,
            plot_density=True,
            plot_contours=True,
            contourf_kwargs={'colors': 'blue', 'alpha': 0.5}
        )
        get_elapsed_time(start_time)

        # Gandalf not detected KDE
        corner.hist2d(
            x=df_gandalf_not_detected["BDF_MAG_DERED_CALIB_I"].values,
            y=df_gandalf_not_detected[col].values,
            ax=ax,
            bins=50,
            range=[x_range, y_range],
            levels=(0.68, 0.95),
            color='red',
            smooth=1.0,
            plot_datapoints=True,
            fill_contours=False,
            plot_density=True,
            plot_contours=True,
            contour_kwargs={'colors': 'red', 'alpha': 0.2}
        )
        get_elapsed_time(start_time)

        # Balrog not detected KDE
        corner.hist2d(
            x=df_balrog_not_detected["BDF_MAG_DERED_CALIB_I"].values,
            y=df_balrog_not_detected[col].values,
            ax=ax,
            bins=50,
            range=[x_range, y_range],
            levels=(0.68, 0.95),
            color='purple',
            smooth=1.0,
            plot_datapoints=True,
            fill_contours=False,
            plot_density=True,
            plot_contours=True,
            contourf_kwargs={'colors': 'purple', 'alpha': 0.2}
        )

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_ylabel(labels[i], fontsize=10)

        # Add axis labels for edge subplots
        if subplot_idx_x == num_cols - 1:
            ax.set_xlabel('BDF Mag I', fontsize=10)

        subplot_idx_x -= 1
        if subplot_idx_x < 0:
            subplot_idx_x = num_cols - 1
            subplot_idx_y += 1
        if subplot_idx_y > num_cols - 1:
            break

    # Remove any unused subplots
    total_plots = num_cols * num_cols
    used_plots = len(columns)
    for idx in range(used_plots, total_plots):
        x = idx % num_cols
        y = idx // num_cols
        fig.delaxes(axes[x, y])

    # Customize layout and legend
    legend_elements = [
        mpatches.Patch(color='orange', alpha=0.5, label='Gandalf Detected'),
        mpatches.Patch(color='red', alpha=0.2, label='Gandalf Not Detected'),
        mpatches.Patch(color='blue', alpha=0.5, label='Balrog Detected'),
        mpatches.Patch(color='purple', alpha=0.2, label='Balrog Not Detected')
    ]

    fig.legend(handles=legend_elements, loc='upper right', fontsize=16, bbox_to_anchor=(0.98, 0.76))

    plt.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(save_name, dpi=300)
    plt.clf()
    plt.close(fig)


# def plot_multivariate_classifier(df_balrog, df_gandalf, columns, labels, ranges, show_plot, save_plot, save_name, title='Histogram'):
#     import matplotlib.patches as mpatches
#     import time
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import corner
#
#     # Set the start time
#     start_time = time.time()
#
#     # Prepare the dataframes
#     df_gandalf_detected = df_gandalf[df_gandalf["detected"] == 1]
#     df_gandalf_not_detected = df_gandalf[df_gandalf["detected"] == 0]
#     df_balrog_detected = df_balrog[df_balrog["detected"] == 1]
#     df_balrog_not_detected = df_balrog[df_balrog["detected"] == 0]
#
#     def get_elapsed_time(start_time):
#         # Calculate elapsed time
#         elapsed_time = time.time() - start_time
#
#         # Convert to days, hours, minutes, and seconds
#         days = elapsed_time // (24 * 3600)
#         elapsed_time = elapsed_time % (24 * 3600)
#         hours = elapsed_time // 3600
#         elapsed_time %= 3600
#         minutes = elapsed_time // 60
#         seconds = elapsed_time % 60
#
#         # Print the elapsed time
#         print(f"Elapsed time: {int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s")
#
#     def compute_detection_probabilities(df, bins, feature_column):
#         """Compute detection probabilities for a given feature and set of bins."""
#         detected_count, _ = np.histogram(df[df["detected"] == 1][feature_column], bins=bins)
#         total_count, _ = np.histogram(df[feature_column], bins=bins)
#         detection_prob = detected_count / total_count
#         return detection_prob
#
#     # Define the bin edges for BDF_MAG_DERED_CALIB_I (adjust range and bin size as needed)
#     bins = np.linspace(21, 26, 10)  # 10 bins from 21 to 26 in i-mag
#
#     # Compute detection probabilities for both Gandalf and Balrog
#     detection_prob_gandalf = compute_detection_probabilities(df_gandalf, bins, "BDF_MAG_DERED_CALIB_I")
#     detection_prob_balrog = compute_detection_probabilities(df_balrog, bins, "BDF_MAG_DERED_CALIB_I")
#
#     # Compute the absolute difference between the detection probabilities
#     detection_prob_difference = np.abs(detection_prob_gandalf - detection_prob_balrog)
#     print("Detection Probability Difference (absolute):", detection_prob_difference)
#
#     # Prepare data for corner plots
#     def prepare_data(df_list, column_list):
#         return [df[column_list].values for df in df_list]
#
#     # Lists of dataframes and their labels
#     data_dfs = [
#         df_gandalf_detected,
#         df_gandalf_not_detected,
#         df_balrog_detected,
#         df_balrog_not_detected
#     ]
#
#     data_labels = [
#         'Gandalf Detected',
#         'Gandalf Not Detected',
#         'Balrog Detected',
#         'Balrog Not Detected'
#     ]
#
#     colors = ['orange', 'red', 'blue', 'purple']
#     alphas = [0.5, 0.2, 0.5, 0.2]
#
#     # Prepare the figure
#     fig = plt.figure(figsize=(15, 15))
#
#     # Collect the data for corner plots
#     datasets = []
#     for df in data_dfs:
#         data = df[['BDF_MAG_DERED_CALIB_I'] + columns].values
#         datasets.append(data)
#
#     # Define the ranges for corner plots
#     plot_ranges = [(21, 26)] + ranges
#
#     # Plot each dataset on the same corner plot
#     for idx, data in enumerate(datasets):
#         get_elapsed_time(start_time)
#         print(f"Processing {data_labels[idx]}")
#
#         corner.corner(
#             data,
#             labels=['BDF_MAG_DERED_CALIB_I'] + labels,
#             range=plot_ranges,
#             color=colors[idx],
#             bins=50,
#             smooth=1.0,
#             fig=fig,
#             plot_contours=True,
#             plot_density=True,
#             fill_contours=True if alphas[idx] > 0 else False,
#             contourf_kwargs={'alpha': alphas[idx]},
#             hist_kwargs={'alpha': alphas[idx]},
#             levels=(0.68, 0.95),
#             label_kwargs={'fontsize': 12},
#             show_titles=False,
#             title_kwargs={'fontsize': 12},
#             data_kwargs={'alpha': alphas[idx]},
#             use_math_text=True
#         )
#
#     # Customize the legend
#     legend_elements = [
#         mpatches.Patch(color='orange', alpha=0.5, label='Gandalf Detected'),
#         mpatches.Patch(color='red', alpha=0.2, label='Gandalf Not Detected'),
#         mpatches.Patch(color='blue', alpha=0.5, label='Balrog Detected'),
#         mpatches.Patch(color='purple', alpha=0.2, label='Balrog Not Detected')
#     ]
#     fig.legend(handles=legend_elements, loc='upper right', fontsize=16, bbox_to_anchor=(0.98, 0.9))
#
#     plt.suptitle(title, fontsize=18)
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#
#     if show_plot:
#         plt.show()
#     if save_plot:
#         plt.savefig(save_name, dpi=300)
#     plt.clf()
#     plt.close(fig)


# def plot_multivariate_classifier(df_balrog, df_gandalf, columns, labels, ranges, show_plot, save_plot, save_name, title='Histogram'):
#     import matplotlib.patches as mpatches
#     import time
#     import numpy as np
#
#     # Set the start time
#     start_time = time.time()
#
#     num_cols = int(np.round(np.sqrt(len(columns))))
#
#     # Prepare the dataframes (assuming you already have df_gandalf and df_balrog)
#     df_gandalf_detected = df_gandalf[df_gandalf["detected"] == 1]
#     df_gandalf_not_detected = df_gandalf[df_gandalf["detected"] == 0]
#     df_balrog_detected = df_balrog[df_balrog["detected"] == 1]
#     df_balrog_not_detected = df_balrog[df_balrog["detected"] == 0]
#
#     # Create the FacetGrid or a pair grid setup
#     fig, axes = plt.subplots(num_cols, num_cols, figsize=(15, 15))
#
#     subplot_idx_x = num_cols - 1
#     subplot_idx_y = 0
#
#     def get_elapsed_time(start_time):
#         # Calculate elapsed time
#         elapsed_time = time.time() - start_time
#
#         # Convert to days, hours, minutes, and seconds
#         days = elapsed_time // (24 * 3600)
#         elapsed_time = elapsed_time % (24 * 3600)
#         hours = elapsed_time // 3600
#         elapsed_time %= 3600
#         minutes = elapsed_time // 60
#         seconds = elapsed_time % 60
#
#         # Print the elapsed time
#         print(f"Elapsed time: {int(days)}d {int(hours)}h {int(minutes)}m {int(seconds)}s")
#
#     def compute_detection_probabilities(df, bins, feature_column):
#         """Compute detection probabilities for a given feature and set of bins."""
#         detected_count, _ = np.histogram(df[df["detected"] == 1][feature_column], bins=bins)
#         total_count, _ = np.histogram(df[feature_column], bins=bins)
#         detection_prob = detected_count / total_count
#         return detection_prob
#
#     # Define the bin edges for BDF_MAG_DERED_CALIB_I (adjust range and bin size as needed)
#     bins = np.linspace(21, 26, 10)  # 10 bins from 21 to 26 in i-mag
#
#     # Compute detection probabilities for both gaNdalF and Balrog
#     detection_prob_gandalf = compute_detection_probabilities(df_gandalf, bins, "BDF_MAG_DERED_CALIB_I")
#     detection_prob_balrog = compute_detection_probabilities(df_balrog, bins, "BDF_MAG_DERED_CALIB_I")
#
#     # Compute the absolute difference between the detection probabilities
#     detection_prob_difference = np.abs(detection_prob_gandalf - detection_prob_balrog)
#     print("Detection Probability Difference (absolute):", detection_prob_difference)
#
#     for i, col in enumerate(columns):
#         try:
#             ax = axes[subplot_idx_x, subplot_idx_y]
#         except TypeError:
#             ax = axes
#         get_elapsed_time(start_time)
#         print(f"Plot gandalf detected column: {col} out off {len(columns)-(i+1)}/{len(columns)}")
#         # Gandalf detected KDE
#         sns.kdeplot(
#             x=df_gandalf_detected["BDF_MAG_DERED_CALIB_I"],
#             y=df_gandalf_detected[col],
#             fill=False,
#             thresh=0,
#             levels=5,
#             cmap='Oranges',
#             alpha=0.5,
#             ax=ax
#         )
#         get_elapsed_time(start_time)
#         print(f"Plot balrog detected column: {col} out off {len(columns) - (i + 1)}/{len(columns)}")
#         # Balrog detected KDE
#         sns.kdeplot(
#             x=df_balrog_detected["BDF_MAG_DERED_CALIB_I"],
#             y=df_balrog_detected[col],
#             fill=True,
#             thresh=0,
#             levels=5,
#             cmap='Blues',
#             alpha=0.5,
#             ax=ax
#         )
#         get_elapsed_time(start_time)
#         print(f"Plot gandalf not detected column: {col} out off {len(columns) - (i + 1)}/{len(columns)}")
#         # Gandalf not detected KDE
#         sns.kdeplot(
#             x=df_gandalf_not_detected["BDF_MAG_DERED_CALIB_I"],
#             y=df_gandalf_not_detected[col],
#             fill=False,
#             thresh=0,
#             levels=5,
#             cmap='Reds',
#             alpha=0.2,
#             ax=ax
#         )
#         get_elapsed_time(start_time)
#         print(f"Plot balrog not detected column: {col} out off {len(columns) - (i + 1)}/{len(columns)}")
#         # Balrog not detected KDE
#         sns.kdeplot(
#             x=df_balrog_not_detected["BDF_MAG_DERED_CALIB_I"],
#             y=df_balrog_not_detected[col],
#             fill=True,
#             thresh=0,
#             levels=5,
#             cmap='Purples',
#             alpha=0.2,
#             ax=ax
#         )
#
#         ax.set_xlim(21, 26)
#         ax.set_ylim(ranges[i][0], ranges[i][1])
#         ax.set_ylabel(labels[i], fontsize=10)
#
#         # Add axis labels for edge subplots
#         if subplot_idx_x == num_cols-1:
#             ax.set_xlabel('BDF Mag I', fontsize=10)
#         try:
#             ax.set_ylim(ranges[i][0], ranges[i][1])
#         except:
#             pass
#
#         subplot_idx_x -= 1
#         if subplot_idx_x < 0:
#             subplot_idx_x = num_cols - 1
#             subplot_idx_y += 1
#         if subplot_idx_y > num_cols - 1:
#             break
#
#     fig.delaxes(axes[0, 3])
#     fig.delaxes(axes[1, 3])
#
#     # Customize layout and legend
#     legend_elements = [
#         mpatches.Patch(color='orange', alpha=0.5, label='Gandalf Detected'),
#         mpatches.Patch(color='red', alpha=0.2, label='Gandalf Not Detected'),
#         mpatches.Patch(color='blue', alpha=0.5, label='Balrog Detected'),
#         mpatches.Patch(color='purple', alpha=0.2, label='Balrog Not Detected')
#     ]
#
#     fig.legend(handles=legend_elements, loc='upper right', fontsize=16, bbox_to_anchor=(0.98, 0.76))
#
#     plt.suptitle(title, fontsize=18)
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     if show_plot is True:
#         plt.show()
#     if save_plot is True:
#         plt.savefig(save_name, dpi=300)
#     plt.clf()
#     plt.close(fig)


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


def plot_balrog_histogram_with_error(
    df_gandalf, df_balrog, columns, labels, ranges, binwidths,
    title, show_plot, save_plot, save_name
):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.gridspec as gridspec

    color_gandalf = '#ff8c00'
    color_balrog = '#51a6fb'

    # Number of columns and rows for subplots
    ncols = 3
    nrows = (len(columns) + ncols - 1) // ncols

    # Define height ratios with extra space between groups
    height_ratios = []
    total_rows = 0
    for idx in range(nrows):
        height_ratios.extend([3, 1])    # Histogram and error plot heights
        total_rows += 2
        if idx != nrows - 1:
            height_ratios.append(0.5)   # Spacer row between groups
            total_rows += 1

    fig = plt.figure(figsize=(4 * ncols, 2 * nrows))

    # Create GridSpec with specified height ratios
    gs = gridspec.GridSpec(total_rows, ncols, figure=fig, height_ratios=height_ratios)

    for idx, col in enumerate(columns):
        col_idx = idx % ncols
        group_idx = idx // ncols

        # Compute row indices in GridSpec
        hist_row = group_idx * 3        # Each group consists of 2 plots + spacer
        error_row = hist_row + 1

        # Create axes without sharing x-axis
        ax_hist = fig.add_subplot(gs[hist_row, col_idx])
        ax_error = fig.add_subplot(gs[error_row, col_idx])

        binwidth = binwidths[idx] if binwidths[idx] is not None else 0.2

        # Set range for histogram
        if ranges[idx] is not None:
            range_min, range_max = ranges[idx]
        else:
            range_min = min(df_gandalf[col].min(), df_balrog[col].min())
            range_max = max(df_gandalf[col].max(), df_balrog[col].max())

        # Create common bins
        bins = np.arange(range_min, range_max + binwidth, binwidth)

        # Plot histograms with sns.histplot and specify binrange
        sns.histplot(
            data=df_gandalf,
            x=col,
            ax=ax_hist,
            bins=bins,
            binrange=(range_min, range_max),
            element="step",
            stat="count",
            color=color_gandalf,
            log_scale=(False, True),
            fill=False,
            label="Gandalf"
        )
        sns.histplot(
            data=df_balrog,
            x=col,
            ax=ax_hist,
            bins=bins,
            binrange=(range_min, range_max),
            element="step",
            stat="count",
            color=color_balrog,
            log_scale=(False, True),
            fill=False,
            label="Balrog"
        )

        # Extract counts from histograms, specifying the range
        counts_gandalf, _ = np.histogram(df_gandalf[col], bins=bins, range=(range_min, range_max))
        counts_balrog, _ = np.histogram(df_balrog[col], bins=bins, range=(range_min, range_max))

        # Calculate percent difference and uncertainty
        counts_gandalf = counts_gandalf.astype(float)
        counts_balrog = counts_balrog.astype(float)

        with np.errstate(divide='ignore', invalid='ignore'):
            percent_error = 100 * (counts_balrog - counts_gandalf) / counts_gandalf
            sigma_E = 100 * np.sqrt(counts_balrog + counts_gandalf) / counts_gandalf

        # Handle division by zero
        percent_error[counts_gandalf == 0] = np.nan
        sigma_E[counts_gandalf == 0] = np.nan

        # Bin centers for error plot
        bin_centers = bins[:-1] + binwidth / 2

        # Plot error diagram with percent differences and error bars
        ax_error.errorbar(
            bin_centers, percent_error, yerr=sigma_E,
            fmt='o', color='black', ecolor='black', capsize=2, markersize=2, clip_on=True
        )
        ax_error.axhline(0, color='red', linestyle='--')

        # Set x-limits and remove margins
        ax_hist.set_xlim(range_min, range_max)
        ax_error.set_xlim(range_min, range_max)
        ax_hist.margins(x=0)
        ax_error.margins(x=0)

        ax_error.set_ylabel('% Error')
        ax_error.set_xlabel(labels[idx])

        # Ensure x-axis labels and tick labels are visible on ax_error
        ax_error.tick_params(labelbottom=True)
        plt.setp(ax_error.get_xticklabels(), visible=True)

        # Hide x-axis labels and tick labels on histograms
        ax_hist.set_xlabel('')
        ax_hist.tick_params(labelbottom=False)
        plt.setp(ax_hist.get_xticklabels(), visible=False)

        # Add gridlines
        ax_hist.grid(True)
        ax_error.grid(True)

        ax_hist.set_ylabel('Counts')
        ax_hist.legend()

        # Optionally adjust y-axis limits for better visualization
        max_percent_error = np.nanmax(np.abs(percent_error + sigma_E))
        if not np.isnan(max_percent_error):
            ax_error.set_ylim(-15, 15)
            # ax_error.set_ylim(-max_percent_error * 1.1, max_percent_error * 1.1)

    # Adjust overall layout
    plt.subplots_adjust(wspace=0.3, hspace=0.3)  # Adjust horizontal and vertical space

    plt.suptitle(title)

    # Show or save plot
    if save_plot:
        plt.savefig(save_name, dpi=300)
    if show_plot:
        plt.show()

    # Clear the figure to free memory
    plt.clf()
    plt.close(fig)


def plot_number_density_fluctuation(df_balrog, df_gandalf, columns, labels, ranges, title, show_plot, save_plot,
                                    save_name):
    from scipy.ndimage import gaussian_filter1d

    # Create subsets for detected and not detected objects
    df_balrog_detected = df_balrog[df_balrog["detected"] == 1]
    df_gandalf_detected = df_gandalf[df_gandalf["detected"] == 1]
    df_balrog_not_detected = df_balrog[df_balrog["detected"] == 0]
    df_gandalf_not_detected = df_gandalf[df_gandalf["detected"] == 0]

    ncols = 4
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for idx, col in enumerate(columns):
        ax = axes[idx]
        bin_range = ranges[idx]

        if bin_range is not None:
            range_min, range_max = bin_range
        else:
            range_min = min(df_balrog[col].min(), df_gandalf[col].min())
            range_max = max(df_balrog[col].max(), df_gandalf[col].max())

        num_bins = 20
        bins = np.linspace(range_min, range_max, num_bins + 1)

        counts_balrog_detected, _ = np.histogram(df_balrog_detected[col], bins=bins)
        counts_gandalf_detected, _ = np.histogram(df_gandalf_detected[col], bins=bins)
        counts_balrog_not_detected, _ = np.histogram(df_balrog_not_detected[col], bins=bins)
        counts_gandalf_not_detected, _ = np.histogram(df_gandalf_not_detected[col], bins=bins)

        # Compute mean counts
        mean_counts_balrog_detected = np.mean(counts_balrog_detected)
        mean_counts_gandalf_detected = np.mean(counts_gandalf_detected)
        mean_counts_balrog_not_detected = np.mean(counts_balrog_not_detected)
        mean_counts_gandalf_not_detected = np.mean(counts_gandalf_not_detected)

        epsilon = 1e-10
        mean_counts_balrog_detected += epsilon
        mean_counts_gandalf_detected += epsilon
        mean_counts_balrog_not_detected += epsilon
        mean_counts_gandalf_not_detected += epsilon

        # Compute fluctuations
        fluctuation_balrog_detected = counts_balrog_detected / mean_counts_balrog_detected
        fluctuation_gandalf_detected = counts_gandalf_detected / mean_counts_gandalf_detected
        fluctuation_balrog_not_detected = counts_balrog_not_detected / mean_counts_balrog_not_detected
        fluctuation_gandalf_not_detected = counts_gandalf_not_detected / mean_counts_gandalf_not_detected

        # Apply smoothing to fluctuations
        smoothed_fluctuation_balrog = gaussian_filter1d(fluctuation_balrog_detected, sigma=1)
        smoothed_fluctuation_gandalf = gaussian_filter1d(fluctuation_gandalf_detected, sigma=1)
        smoothed_fluctuation_not_balrog = gaussian_filter1d(fluctuation_balrog_not_detected, sigma=1)
        smoothed_fluctuation_not_gandalf = gaussian_filter1d(fluctuation_gandalf_not_detected, sigma=1)

        # Plot histograms in the background
        ax_bar = ax.twinx()
        ax_bar.hist(df_balrog_detected[col], bins=bins, color='gold', alpha=0.5, label='Balrog detected', density=True)
        ax_bar.hist(df_balrog_not_detected[col], bins=bins, color='lightcoral', alpha=0.5, label='Balrog not detected', density=True)
        ax_bar.set_yticks([])

        # Plot fluctuations on top
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.plot(bin_centers, smoothed_fluctuation_balrog, marker='o', linestyle='-', color='#51a6fb', label='Balrog detected')
        ax.plot(bin_centers, smoothed_fluctuation_gandalf, marker='s', linestyle='-', color='#ff8c00', label='Gandalf detected')
        ax.plot(bin_centers, smoothed_fluctuation_not_balrog, marker='+', linestyle='-', color='coral', label='Balrog not detected')
        ax.plot(bin_centers, smoothed_fluctuation_not_gandalf, marker='*', linestyle='-', color='blueviolet', label='Gandalf not detected')

        # Plot a reference line at N/<N> = 1
        ax.axhline(1, color='red', linestyle='--')

        ax.set_xlabel(labels[idx])
        ax.set_ylabel(r'$N / \langle N \rangle$')
        ax.grid(True)
        ax.set_xlim(range_min, range_max)

    # Hide the unused (16th) plot in the top-right corner
    if len(columns) < len(axes):
        fig.delaxes(axes[len(columns)])  # Hide the top-right plot

    # Create a global legend above the last unused plot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.9, 0.1), bbox_transform=plt.gcf().transFigure)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make space for the legend at the top
    plt.suptitle(title)
    if save_plot:
        plt.savefig(save_name, dpi=300)
    if show_plot:
        plt.show()
    plt.close(fig)


# def plot_number_density_fluctuation(df_balrog, df_gandalf, columns, labels, ranges, title, show_plot, save_plot, save_name):
#     """
#     Plots the number density fluctuation (N/<N>) for detected and not detected objects
#     in df_balrog and df_gandalf for specified columns, with histograms in the background.
#     Also computes and plots the percentage difference between gaNdalF and Balrog.
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
#
#     # Create subsets for detected and not detected objects
#     df_balrog_detected = df_balrog[df_balrog["detected"] == 1]
#     df_balrog_not_detected = df_balrog[df_balrog["detected"] == 0]
#     df_gandalf_detected = df_gandalf[df_gandalf["detected"] == 1]
#     df_gandalf_not_detected = df_gandalf[df_gandalf["detected"] == 0]
#
#     ncols = 3
#     nrows = (len(columns) + ncols - 1) // ncols
#
#     fig, axes = plt.subplots(
#         nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows)
#     )
#     axes = axes.flatten()
#
#     # Define colors
#     color_balrog_detected = '#51a6fb'       # Blue
#     color_balrog_not_detected = '#9933ff'   # Light Blue
#     color_gandalf_detected = '#ff8c00'      # Orange
#     color_gandalf_not_detected = '#cc3300'  # Light Orange
#
#     for idx, col in enumerate(columns):
#         ax = axes[idx]
#         bin_range = ranges[idx]
#         label = labels[idx]
#
#         # Set bin edges
#         if bin_range is not None:
#             range_min, range_max = bin_range
#         else:
#             range_min = min(
#                 df_balrog[col].min(), df_gandalf[col].min()
#             )
#             range_max = max(
#                 df_balrog[col].max(), df_gandalf[col].max()
#             )
#
#         # Adjusted number of bins
#         num_bins = 20  # Reduced from 50 to 20
#
#         bins = np.linspace(range_min, range_max, num_bins + 1)
#
#         # Compute histograms for detected and not detected objects
#         counts_balrog_detected, _ = np.histogram(
#             df_balrog_detected[col], bins=bins
#         )
#         counts_balrog_not_detected, _ = np.histogram(
#             df_balrog_not_detected[col], bins=bins
#         )
#
#         counts_gandalf_detected, _ = np.histogram(
#             df_gandalf_detected[col], bins=bins
#         )
#         counts_gandalf_not_detected, _ = np.histogram(
#             df_gandalf_not_detected[col], bins=bins
#         )
#
#         # Convert counts to float for accurate division
#         counts_balrog_detected = counts_balrog_detected.astype(float)
#         counts_balrog_not_detected = counts_balrog_not_detected.astype(float)
#         counts_gandalf_detected = counts_gandalf_detected.astype(float)
#         counts_gandalf_not_detected = counts_gandalf_not_detected.astype(float)
#
#         # Compute mean counts over all bins
#         mean_counts_balrog_detected = np.mean(counts_balrog_detected)
#         mean_counts_balrog_not_detected = np.mean(counts_balrog_not_detected)
#         mean_counts_gandalf_detected = np.mean(counts_gandalf_detected)
#         mean_counts_gandalf_not_detected = np.mean(counts_gandalf_not_detected)
#
#         # Add epsilon to prevent division by zero
#         epsilon = 1e-10
#         mean_counts_balrog_detected += epsilon
#         mean_counts_balrog_not_detected += epsilon
#         mean_counts_gandalf_detected += epsilon
#         mean_counts_gandalf_not_detected += epsilon
#
#         # Compute fluctuations centered around zero
#         fluctuation_balrog_detected = (counts_balrog_detected - mean_counts_balrog_detected) / mean_counts_balrog_detected
#         fluctuation_balrog_not_detected = (counts_balrog_not_detected - mean_counts_balrog_not_detected) / mean_counts_balrog_not_detected
#
#         fluctuation_gandalf_detected = (counts_gandalf_detected - mean_counts_gandalf_detected) / mean_counts_gandalf_detected
#         fluctuation_gandalf_not_detected = (counts_gandalf_not_detected - mean_counts_gandalf_not_detected) / mean_counts_gandalf_not_detected
#
#         # Handle invalid values
#         fluctuation_balrog_detected[~np.isfinite(fluctuation_balrog_detected)] = np.nan
#         fluctuation_balrog_not_detected[~np.isfinite(fluctuation_balrog_not_detected)] = np.nan
#         fluctuation_gandalf_detected[~np.isfinite(fluctuation_gandalf_detected)] = np.nan
#         fluctuation_gandalf_not_detected[~np.isfinite(fluctuation_gandalf_not_detected)] = np.nan
#
#         # Calculate percentage difference
#         with np.errstate(divide='ignore', invalid='ignore'):
#             percentage_diff_detected = 100 * (counts_gandalf_detected - counts_balrog_detected) / counts_balrog_detected
#             percentage_diff_not_detected = 100 * (
#                         counts_gandalf_not_detected - counts_balrog_not_detected) / counts_balrog_not_detected
#
#         # Handle division by zero and invalid values
#         percentage_diff_detected[~np.isfinite(percentage_diff_detected)] = np.nan
#         percentage_diff_not_detected[~np.isfinite(percentage_diff_not_detected)] = np.nan
#
#         # Bin centers
#         bin_centers = (bins[:-1] + bins[1:]) / 2
#
#         # Plot histograms in the background
#         ax_bar = ax.twinx()  # Create a twin y-axis sharing the same x-axis
#
#         # Calculate maximum counts for normalization
#         max_count = max(
#             counts_balrog_detected.max(),
#             counts_balrog_not_detected.max(),
#             counts_gandalf_detected.max(),
#             counts_gandalf_not_detected.max()
#         )
#
#         # Normalize counts for background plotting
#         norm_counts_balrog_detected = counts_balrog_detected / max_count
#         norm_counts_balrog_not_detected = counts_balrog_not_detected / max_count
#         norm_counts_gandalf_detected = counts_gandalf_detected / max_count
#         norm_counts_gandalf_not_detected = counts_gandalf_not_detected / max_count
#
#         bar_width = (bins[1] - bins[0]) / 2  # Adjusted bar width
#
#         # Plot the histograms as background
#         ax_bar.bar(
#             bin_centers - bar_width / 2, norm_counts_balrog_detected, width=bar_width / 2,
#             color=color_balrog_detected, alpha=0.5, label='Balrog Detected', align='edge'
#         )
#         ax_bar.bar(
#             bin_centers - bar_width / 2, norm_counts_balrog_not_detected, width=bar_width / 2,
#             color=color_balrog_not_detected, alpha=0.2, label='Balrog Not Detected', align='edge'
#         )
#
#         ax_bar.bar(
#             bin_centers + bar_width / 2, norm_counts_gandalf_detected, width= -bar_width / 2,
#             color=color_gandalf_detected, alpha=0.5, label='Gandalf Detected', align='edge'
#         )
#         ax_bar.bar(
#             bin_centers + bar_width / 2, norm_counts_gandalf_not_detected, width= -bar_width / 2,
#             color=color_gandalf_not_detected, alpha=0.2, label='Gandalf Not Detected', align='edge'
#         )
#
#         ax_bar.set_yticks([])  # Hide y-axis ticks for the background histogram
#         ax_bar.set_zorder(1)   # Set background histogram behind the fluctuation plot
#         ax.set_zorder(2)       # Ensure the fluctuation plot is on top
#         ax.patch.set_visible(False)  # Make the ax background transparent
#
#         # Plot fluctuations on top
#         ax.plot(
#             bin_centers, fluctuation_balrog_detected,
#             marker='o', linestyle='-', color=color_balrog_detected, alpha=1.0, label='Balrog Detected'
#         )
#         ax.plot(
#             bin_centers, fluctuation_balrog_not_detected,
#             marker='o', linestyle='--', color=color_balrog_not_detected, alpha=1.0, label='Balrog Not Detected'
#         )
#         ax.plot(
#             bin_centers, fluctuation_gandalf_detected,
#             marker='s', linestyle='-', color=color_gandalf_detected, alpha=1.0, label='Gandalf Detected'
#         )
#         ax.plot(
#             bin_centers, fluctuation_gandalf_not_detected,
#             marker='s', linestyle='--', color=color_gandalf_not_detected, alpha=1.0, label='Gandalf Not Detected'
#         )
#
#         ax.axhline(0, color='red', linestyle='--')  # Zero line
#         ax.set_xlabel(label)
#         ax.set_ylabel(r'$(N - \langle N \rangle) / \langle N \rangle$')
#         ax.grid(True)
#         ax.set_xlim(range_min, range_max)
#         ax.legend(loc='upper left', fontsize='small', ncol=2)
#
#         # Plot percentage differences on the secondary y-axis
#         # ax_bar.plot(
#         #     bin_centers, percentage_diff_detected,
#         #     marker='o', linestyle='-', color='green', alpha=0.7, label='% Diff Detected'
#         # )
#         # ax_bar.plot(
#         #     bin_centers, percentage_diff_not_detected,
#         #     marker='s', linestyle='--', color='red', alpha=0.7, label='% Diff Not Detected'
#         # )
#
#         # ax.axhline(1, color='red', linestyle='--')
#         # ax.set_xlabel(label)
#         # ax.set_ylabel(r'$N / \langle N \rangle$')
#         # ax.grid(True)
#         # ax.set_xlim(range_min, range_max)
#         # ax.legend(loc='upper left', fontsize='small', ncol=2)
#
#         ax_bar.set_ylabel('% Difference', color='green')
#         ax_bar.grid(False)  # Ensure the grid is only on the main axis
#
#     # Remove any unused axes
#     total_plots = nrows * ncols
#     if len(columns) < total_plots:
#         for idx in range(len(columns), total_plots):
#             fig.delaxes(axes[idx])
#
#     plt.suptitle(title)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
#
#     # Show or save plot based on arguments
#     if save_plot:
#         plt.savefig(save_name, dpi=300)
#     if show_plot:
#         plt.show()
#
#     plt.close(fig)



def make_gif(frame_folder, name_save_folder, fps=10):
    filenames = natsorted(os.listdir(frame_folder))
    images_data = []
    for filename in filenames:
        image = imageio.imread(f"{frame_folder}/{filename}")
        images_data.append(image)
    imageio.mimwrite(uri=f"{name_save_folder}", ims=images_data, format='.gif', duration=int(1000*1/fps))
