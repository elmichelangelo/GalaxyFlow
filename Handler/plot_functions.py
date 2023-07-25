import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio
import os
from natsort import natsorted
from chainconsumer import ChainConsumer
from scipy.stats import gaussian_kde


def plot_chain(data_frame, plot_name, max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12, columns=None,
               parameter=None, extends=None):
    """

    :param extends: extents={
                "mag r": (17.5, 26),
                "mag i": (17.5, 26),
                "mag z": (17.5, 26),
                "snr": (-11, 55),
                "size ratio": (-1.5, 4),
                "T": (-1, 2.5)
            }
    :param label_font_size:
    :param tick_font_size:
    :param shade_alpha:
    :param max_ticks:
    :param plot_name: "generated observed properties: chat*"
    :param data_frame:
    :param columns: Mutable list, default values are columns = [
            "unsheared/mag_r",
            "unsheared/mag_i",
            "unsheared/mag_z",
            "unsheared/snr",
            "unsheared/size_ratio",
            "unsheared/T"
        ]
    :param parameter: Mutable list, default values are parameter = [
                "mag r",
                "mag i",
                "mag z",
                "snr",              # signal-noise      Range: min=0.3795, max=38924.4662
                "size ratio",       # T/psfrec_T        Range: min=-0.8636, max=4346136.5645
                "T"                 # T=<x^2>+<y^2>     Range: min=-0.6693, max=1430981.5103
            ]
    :return:
    """
    df_plot = pd.DataFrame({})

    if columns is None:
        columns = [
            "unsheared/mag_r",
            "unsheared/mag_i",
            "unsheared/mag_z",
            "unsheared/snr",
            "unsheared/size_ratio",
            "unsheared/T"
        ]

    if parameter is None:
        parameter = [
                "mag r",
                "mag i",
                "mag z",
                "snr",              # signal-noise      Range: min=0.3795, max=38924.4662
                "size ratio",       # T/psfrec_T        Range: min=-0.8636, max=4346136.5645
                "T"                 # T=<x^2>+<y^2>     Range: min=-0.6693, max=1430981.5103
            ]

    for col in columns:
        df_plot[col] = np.array(data_frame[col])

    chain = ChainConsumer()
    chain.add_chain(df_plot.to_numpy(), parameters=parameter, name=plot_name)
    chain.configure(
        max_ticks=max_ticks,
        shade_alpha=shade_alpha,
        tick_font_size=tick_font_size,
        label_font_size=label_font_size
    )
    # if extends is not None:
    chain.plotter.plot(
        figsize="page",
        extents=extends
    )
    plt.show()
    plt.clf()


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
    statistical_figure, ((stat_ax1, stat_ax2), (stat_ax3, stat_ax4)) = plt.subplots(nrows=2, ncols=2)
    statistical_figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)
    statistical_figure.suptitle(f"Epoch: {epoch}", fontsize=16)

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

    df_training_loss_per_epoch.plot(
        figsize=(16, 9),
        alpha=0.5,
        marker=".",
        grid=True,
        # yticks=(0, 0.25, 0.5, 0.69, 1.0, 5.0),
        ax=stat_ax2)

    stat_ax2.set_xlabel("epoch", fontsize=10, loc='right')
    stat_ax2.set_ylabel("loss", fontsize=12, loc='top')
    stat_ax2.set_title(f"Loss per epoch")

    # Create plot
    df_valid_loss_per_batch.plot(
        figsize=(16, 9),
        alpha=0.5,
        marker=".",
        grid=True,
        # yticks=(0, 0.25, 0.5, 0.69, 1.0, 5.0),
        ax=stat_ax3)

    stat_ax3.set_xlabel("batch", fontsize=10, loc='right')
    stat_ax3.set_ylabel("loss", fontsize=12, loc='top')
    stat_ax3.set_title(f"Loss per batch")

    df_valid_loss_per_epoch.plot(
        figsize=(16, 9),
        alpha=0.5,
        marker=".",
        grid=True,
        # yticks=(0, 0.25, 0.5, 0.69, 1.0, 5.0),
        ax=stat_ax4)

    stat_ax4.set_xlabel("epoch", fontsize=10, loc='right')
    stat_ax4.set_ylabel("loss", fontsize=12, loc='top')
    stat_ax4.set_title(f"Loss per epoch")

    if show_plot is True:
        statistical_figure.show()
    if save_plot is True:
        statistical_figure.savefig(f"{save_name}", dpi=200)

    # Clear and close open figure to avoid memory overload
    statistical_figure.clf()
    plt.close(statistical_figure)
    plt.clf()


def color_color_plot(data_frame_generated, data_frame_true, colors, show_plot, save_name, extents=None):
    """"""
    df_generated_measured = pd.DataFrame({})
    df_true_measured = pd.DataFrame({})
    for color in colors:
        df_generated_measured[f"{color[0]}-{color[1]}"] = \
            np.array(data_frame_generated[f"unsheared/lupt_{color[0]}"]) - np.array(
                data_frame_generated[f"unsheared/lupt_{color[1]}"])
        df_true_measured[f"{color[0]}-{color[1]}"] = \
            np.array(data_frame_true[f"unsheared/lupt_{color[0]}"]) - np.array(
                data_frame_true[f"unsheared/lupt_{color[1]}"])

    arr_true = df_true_measured.to_numpy()
    arr_generated = df_generated_measured.to_numpy()
    parameter = [
        "unsheared/lupt r-i",
        "unsheared/lupt i-z"
    ]
    chainchat = ChainConsumer()
    chainchat.add_chain(arr_true, parameters=parameter, name="true observed properties: chat")
    chainchat.add_chain(arr_generated, parameters=parameter,
                        name="generated observed properties: chat*")
    chainchat.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
    chainchat.plotter.plot(
        filename=save_name,
        figsize="page",
        extents=extents
    )
    if show_plot is True:
        plt.show()
    plt.clf()
    plt.close()


def residual_plot(data_frame_generated, data_frame_true, bands, plot_title, show_plot, save_plot, save_name):
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
        "dataset": ["balrog" for _ in range(len(data_frame_true[f"unsheared/lupt_r"]))]
    })
    df_hist_generated = pd.DataFrame({
        "dataset": ["generated" for _ in range(len(data_frame_generated[f"unsheared/lupt_r"]))]
    })
    for band in bands:
        df_hist_balrog[f"BDF_LUPT_DERED_CALIB - unsheared/lupt {band}"] = data_frame_true[
                                                                              f"BDF_LUPT_DERED_CALIB_{band.upper()}"] - \
                                                                          data_frame_true[f"unsheared/lupt_{band}"]
        df_hist_generated[f"BDF_LUPT_DERED_CALIB - unsheared/lupt {band}"] = data_frame_generated[
                                                                                 f"BDF_LUPT_DERED_CALIB_{band.upper()}"] - \
                                                                             data_frame_generated[
                                                                                 f"unsheared/lupt_{band}"]

    for idx, band in enumerate(bands):
        sns.histplot(
            data=df_hist_balrog,
            x=f"BDF_LUPT_DERED_CALIB - unsheared/lupt {band}",
            ax=lst_axis_res[idx],
            element="step",
            stat="density",
            color="dodgerblue",
            bins=50,
            label="balrog"
        )
        sns.histplot(
            data=df_hist_generated,
            x=f"BDF_LUPT_DERED_CALIB - unsheared/lupt {band}",
            ax=lst_axis_res[idx],
            element="step",
            stat="density",
            color="darkorange",
            fill=False,
            bins=50,
            label="generated"
        )
        lst_axis_res[idx].axvline(
            x=df_hist_balrog[f"BDF_LUPT_DERED_CALIB - unsheared/lupt {band}"].median(),
            color='dodgerblue',
            ls='--',
            lw=1.5,
            label="Mean balrog"
        )
        lst_axis_res[idx].axvline(
            x=df_hist_generated[f"BDF_LUPT_DERED_CALIB - unsheared/lupt {band}"].median(),
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
    plt.clf()
    plt.close()


def plot_chain_compare(data_frame_generated, data_frame_true, epoch, show_plot, save_name, columns=None, parameter=None,
                       extends=None, max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12):
    """"""
    if columns is None:
        columns = [
            "unsheared/mag_r",
            "unsheared/mag_i",
            "unsheared/mag_z",
            "unsheared/snr",
            "unsheared/size_ratio",
            "unsheared/T"
        ]

    if parameter is None:
        parameter = [
            "mag r",
            "mag i",
            "mag z",
            "snr",  # signal-noise      Range: min=0.3795, max=38924.4662
            "size ratio",  # T/psfrec_T        Range: min=-0.8636, max=4346136.5645
            "T"  # T=<x^2>+<y^2>     Range: min=-0.6693, max=1430981.5103
        ]

    df_plot_generated = data_frame_generated[columns]
    df_plot_true = data_frame_true[columns]

    chainchat = ChainConsumer()
    chainchat.add_chain(df_plot_true.to_numpy(), parameters=parameter, name="balrog observed properties: chat")
    chainchat.add_chain(df_plot_generated.to_numpy(), parameters=parameter, name="generated observed properties: chat*")
    chainchat.configure(
        max_ticks=max_ticks,
        shade_alpha=shade_alpha,
        tick_font_size=tick_font_size,
        label_font_size=label_font_size
    )
    try:
        chainchat.plotter.plot(
            filename=save_name,
            figsize="page",
            extents=extends
        )
    except:
        print("chain error at epoch", epoch + 1)
    if show_plot is True:
        plt.show()
    plt.clf()


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
    plt.clf()
    plt.close()

    return lists_to_plot


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


def make_gif(frame_folder, name_save_folder, fps=10):
    filenames = natsorted(os.listdir(frame_folder))
    images_data = []
    for filename in filenames:
        image = imageio.imread(f"{frame_folder}/{filename}")
        images_data.append(image)
    imageio.mimwrite(uri=f"{name_save_folder}", ims=images_data, format='.gif', fps=fps)
