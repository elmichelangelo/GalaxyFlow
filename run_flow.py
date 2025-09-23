import torch
import torch.optim as optim
from Handler import fnn, get_os, unsheared_shear_cuts, unsheared_mag_cut, LoggerHandler, plot_features, plot_binning_statistics_combined, plot_balrog_histogram_with_error, plot_compare_corner, calc_color
from gandalf import gaNdalF
import argparse
import logging
import yaml
from datetime import datetime
import pandas as pd
import sys
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os


def init_network(num_outputs, num_input, nb, nh, act, nl, lr, device):
    modules = []
    for _ in range(nb):
        modules += [
            fnn.MADE(num_inputs=num_outputs, num_hidden=nh, num_cond_inputs=num_input, act=act,
                     num_layers=nl),
            fnn.BatchNormFlow(num_outputs),
            fnn.Reverse(num_outputs)
        ]
    model = fnn.FlowSequential(*modules)
    model = model.to(dtype=torch.float64)
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.orthogonal_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.fill_(0)

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, )
    return model, optimizer


def load_config_and_parser(system_path):
    if get_os() == "Mac":
        print("load MAC config-file")
        config_file_name = "MAC_run_flow.cfg"
    elif get_os() == "Linux":
        print("load LMU config-file")
        config_file_name = "LMU_run_flow.cfg"
    else:
        print("Undefined operating system")
        sys.exit()

    parser = argparse.ArgumentParser(description='Start gaNdalF')
    parser.add_argument(
        '--config_filename',
        "-cf",
        type=str,
        nargs=1,
        required=False,
        default=config_file_name,
        help='Name of config file. If not given default.cfg will be used'
    )
    args = parser.parse_args()

    if isinstance(args.config_filename, list):
        args.config_filename = args.config_filename[0]

    path_config_file = f"{system_path}/conf/{args.config_filename}"
    with open(path_config_file, 'r') as fp:
        print(f"open {path_config_file}")
        config = yaml.safe_load(fp)

    now = datetime.now()
    config['RUN_DATE'] = now.strftime('%Y-%m-%d_%H-%M')
    return config, path_config_file


def main(cfg, logger):
    flow_model = gaNdalF(logger, cfg)
    df_gandalf, df_balrog = flow_model.run_flow()

    cfg["PATH_PLOTS"] = f'{cfg["PATH_PLOTS"]}/{cfg["RUN_DATE"]}_FLOW_PLOTS'

    df_gandalf = calc_color(
        data_frame=df_gandalf,
        colors=cfg['COLORS_FLOW'],
        column_name=f"unsheared/mag"
    )
    df_balrog = calc_color(
        data_frame=df_balrog,
        colors=cfg['COLORS_FLOW'],
        column_name=f"unsheared/mag"
    )

    plot_flow(
        cfg=cfg,
        logger=logger,
        df_gandalf=df_gandalf,
        df_balrog=df_balrog
    )

    df_gandalf = unsheared_shear_cuts(df_gandalf)
    df_gandalf = unsheared_mag_cut(df_gandalf)
    df_balrog = unsheared_shear_cuts(df_balrog)
    df_balrog = unsheared_mag_cut(df_balrog)

    plot_flow(
        cfg=cfg,
        logger=logger,
        df_gandalf=df_gandalf,
        df_balrog=df_balrog,
        prefix="_cut"
    )


def plot_flow(cfg, logger, df_gandalf, df_balrog, prefix=""):
    if cfg["SAVE_PLOT"] is True:
        os.makedirs(cfg["PATH_PLOTS"], exist_ok=True)

    plot_features(
        cfg=cfg,
        plot_log=logger,
        df_gandalf=df_gandalf,
        df_balrog=df_balrog,
        columns=cfg["OUTPUT_COLS"],
        title_prefix=f"Feature Plot",
        savename=f"{cfg['PATH_PLOTS']}/feature_plot{prefix}.pdf"
    )

    plot_binning_statistics_combined(
        df_gandalf=df_gandalf,
        df_balrog=df_balrog,
        sample_size=10000,
        plot_scatter=False,
        show_plot=cfg["SHOW_PLOT"],
        save_plot=cfg["SAVE_PLOT"],
        title="gaNdalF vs. Balrog: Measured Photometric Property Distribution Comparison",
        save_name=f"{cfg['PATH_PLOTS']}/binning_statistics_combined{prefix}",
    )

    plot_compare_corner(
        data_frame_generated=df_gandalf,
        data_frame_true=df_balrog,
        dict_delta=None,
        epoch=None,
        title=f"chain plot",
        show_plot=cfg['SHOW_PLOT'],
        save_plot=cfg['SAVE_PLOT'],
        save_name=f"{cfg[f'PATH_PLOTS']}/chainplot{prefix}.pdf",
        columns=cfg["OUTPUT_COLS"],
        labels=[
            f"mag_r",
            f"mag_i",
            f"mag_z",
            f"log10(mag_err_r)",
            f"log10(mag_err_i)",
            f"log10(mag_err_z)",
            "e_1",
            "e_1",
            "snr",
            "size_ratio",
            "T",
        ],
        ranges=None,  # [(15, 30), (15, 30), (15, 30), (-15, 75), (-1.5, 4), (-1.5, 2), (-8, 8), (-8, 8)]
    )

    plot_compare_corner(
        data_frame_generated=df_gandalf,
        data_frame_true=df_balrog,
        dict_delta=None,
        epoch=None,
        title=f"color-color plot",
        columns=["r-i", "i-z"],
        labels=["r-i", "i-z"],
        show_plot=cfg['SHOW_PLOT'],
        save_plot=cfg['SAVE_PLOT'],
        save_name=f"{cfg[f'PATH_PLOTS']}/color_color{prefix}.pdf",
        ranges=[(-4, 4), (-4, 4)]
    )

    plot_balrog_histogram_with_error(
        df_gandalf=df_gandalf,
        df_balrog=df_balrog,
        columns=cfg["HIST_PLOT_COLS"],
        labels=[
            "mag r-i",
            "mag i-z",
            "mag r",
            "mag i",
            "mag z",
            "log10(mag err r)",
            "log10(mag err i)",
            "log10(mag err z)",
            "e_1",
            "e_2",
            "snr",
            "size ratio",
            "T"
        ],
        ranges=[
            [-0.5, 1.5],  # mag r-i
            [-0.5, 1.5],  # mag i-z
            [18, 24.5],  # mag r
            [18, 24.5],  # mag i
            [18, 24.5],  # mag z
            None,  # mag err r
            None,  # mag err i
            None,  # mag err z
            None,  # e_1
            None,  # e_2
            [2, 100],  # snr
            [-0.5, 5],  # size ratio
            [0, 3.5]  # T
        ],
        binwidths=[
            0.08,  # mag r-i
            0.08,  # mag i-z
            None,  # mag r
            None,  # mag i
            None,  # mag z
            None,  # mag err r
            None,  # mag err i
            None,  # mag err z
            None,  # e_1
            None,  # e_2
            2,  # snr
            0.2,  # size ratio
            0.2,  # T
        ],
        title=r"gaNdalF vs. Balrog: Property Distribution Comparison",
        show_plot=cfg["SHOW_PLOT"],
        save_plot=cfg["SAVE_PLOT"],
        save_name=f"{cfg[f'PATH_PLOTS']}/hist_plot{prefix}.pdf"
    )


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    sys.path.append(os.path.dirname(__file__))
    cfg, path_cfg_file = load_config_and_parser(system_path=os.path.abspath(sys.path[-1]))

    log_lvl = logging.INFO
    if cfg["LOGGING_LEVEL"] == "DEBUG":
        log_lvl = logging.DEBUG
    elif cfg["LOGGING_LEVEL"] == "ERROR":
        log_lvl = logging.ERROR

    run_flow_logger = LoggerHandler(
        logger_dict={"logger_name": "train flow logger",
                     "info_logger": cfg['INFO_LOGGER'],
                     "error_logger": cfg['ERROR_LOGGER'],
                     "debug_logger": cfg['DEBUG_LOGGER'],
                     "stream_logger": cfg['STREAM_LOGGER'],
                     "stream_logging_level": log_lvl},
        log_folder_path=f"{cfg['PATH_OUTPUT']}/"
    )

    main(cfg=cfg, logger=run_flow_logger)
    exit()
    model, _ = init_network(
        num_outputs=11,
        num_input=20,
        nb=8,
        nh=128,
        act="tanh",
        nl=2,
        lr=0.00266,
        device=torch.device('cpu')
    )

    model.load_state_dict(torch.load(
        "/Users/P.Gebhardt/Development/PhD/output/gaNdalF/study_use_log10/trial_9e6ec15842/Save_NN/best_model_state_e_100_lr_0.00026627876555723886_bs_4096_run_2025-09-22_15-55.pt",
        map_location="cpu",
        weights_only=True
    ))

    model.eval()

    scalers = joblib.load(f"/Volumes/elmichelangelo_external_ssd_1/Data/transformers/20250922_StandardScalers_nf.pkl")

    with open(f"/Volumes/elmichelangelo_external_ssd_1/Data/20250922_balrog_test_1050720_nf.pkl", 'rb') as file:
        df_data = pd.read_pickle(file)
        df_test = df_data.sample(n=1200000, replace=True).reset_index(drop=True)

    for band in ["r", "i", "z"]:
        df_test[f"unsheared/mag_err_{band}"] = np.log10(df_test[f"unsheared/mag_err_{band}"])

    for col in NF_COLUMNS_OF_INTEREST:
        scaler = scalers[col]
        mean = scaler.mean_[0]
        scale = scaler.scale_[0]
        df_test[col] = (df_test[col] - mean) / scale

    input_data = torch.tensor(df_test[INPUT_COLS].values, dtype=torch.float64)
    output_data = torch.tensor(df_test[OUTPUT_COLS].values, dtype=torch.float64)

    input_data = input_data.to(torch.device('cpu'))

    model.num_inputs = 11
    with torch.no_grad():
        arr_gandalf_output = model.sample(len(input_data), cond_inputs=input_data).detach()

    output_data_np = arr_gandalf_output.cpu().numpy()

    input_data_np_true = input_data.cpu().numpy()
    output_data_np_true = output_data.cpu().numpy()
    arr_all_true = np.concatenate([input_data_np_true, output_data_np_true], axis=1)
    arr_all = np.concatenate([input_data_np_true, output_data_np], axis=1)

    df_output_true = pd.DataFrame(arr_all_true, columns=list(INPUT_COLS) + list(OUTPUT_COLS))
    df_output_true = df_output_true[NF_COLUMNS_OF_INTEREST]

    df_output_gandalf = pd.DataFrame(arr_all, columns=list(INPUT_COLS) + list(OUTPUT_COLS))
    df_output_gandalf = df_output_gandalf[NF_COLUMNS_OF_INTEREST]

    for col in NF_COLUMNS_OF_INTEREST:
        scaler = scalers[col]
        mean = scaler.mean_[0]
        scale = scaler.scale_[0]
        df_output_true[col] = (df_output_true[col] * scale) + mean
        df_output_gandalf[col] = (df_output_gandalf[col] * scale) + mean


    n_features = len(OUTPUT_COLS)
    ncols = min(n_features, 3)
    nrows = int(np.ceil(n_features / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    for i, k in enumerate(OUTPUT_COLS):
        sns.histplot(x=df_output_gandalf[k], bins=100, ax=axes[i], label="gandalf")
        sns.histplot(x=df_output_true[k], bins=100, ax=axes[i], label="balrog")
        axes[i].set_yscale("log")
        if k in ["unsheared/mag_err_r", "unsheared/mag_err_i", "unsheared/mag_err_z"]:
            axes[i].set_title(f"log10({k})")
            axes[i].set_xlabel(f"log10({k})")
        else:
            axes[i].set_title(k)
            axes[i].set_xlabel(k)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"Before applying mag and shear cuts", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.legend()
    plt.savefig(f"/Users/P.Gebhardt/Development/PhD/output/gaNdalF/before_cuts.pdf", bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close(fig)




    import corner

    arr_generated = df_output_gandalf[OUTPUT_COLS].values
    arr_true = df_output_true[OUTPUT_COLS].values

    ndim = arr_generated.shape[1]

    fig, axes = plt.subplots(ndim, ndim, figsize=(16, 9))

    # Plot gandalf
    corner.corner(
        arr_generated,
        fig=fig,
        # bins=100,
        range=None,
        color='#ff8c00',
        smooth=.8,
        smooth1d=.8,
        labels=[
            f"mag_r",
            f"mag_i",
            f"mag_z",
            f"log10(mag_err_r)",
            f"log10(mag_err_i)",
            f"log10(mag_err_z)",
            "e_1",
            "e_1",
            "snr",
            "size_ratio",
            "T",
        ],
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
        hist_kwargs={'alpha': 1},
        scale_hist=True,
        quantiles=[0.16, 0.5, 0.84],
        levels=[0.393, 0.865, 0.989],  # , 0.989
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
        range=None,
        color='#51a6fb',
        smooth=.8,
        smooth1d=.8,
        labels=[
            f"mag_r",
            f"mag_i",
            f"mag_z",
            f"log10(mag_err_r)",
            f"log10(mag_err_i)",
            f"log10(mag_err_z)",
            "e_1",
            "e_1",
            "snr",
            "size_ratio",
            "T",
        ],
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
        hist_kwargs={'alpha': 1},
        scale_hist=True,
        quantiles=[0.16, 0.5, 0.84],
        levels=[0.393, 0.865, 0.989],  # , 0.989
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

    fig.suptitle(f'Before applying mag and shear cuts', fontsize=20)
    fig.legend(handles=legend_elements, loc='upper right', fontsize=16)
    plt.savefig(f"/Users/P.Gebhardt/Development/PhD/output/gaNdalF/corner_before_cuts.pdf", dpi=300)
    plt.clf()
    plt.close(fig)








    df_output_gandalf = unsheared_shear_cuts(df_output_gandalf)
    df_output_gandalf = unsheared_mag_cut(df_output_gandalf)

    df_output_true = unsheared_shear_cuts(df_output_true)
    df_output_true = unsheared_mag_cut(df_output_true)








    n_features = len(OUTPUT_COLS)
    ncols = min(n_features, 3)
    nrows = int(np.ceil(n_features / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    for i, k in enumerate(OUTPUT_COLS):
        sns.histplot(x=df_output_gandalf[k], bins=100, ax=axes[i], label="gandalf")
        sns.histplot(x=df_output_true[k], bins=100, ax=axes[i], label="balrog")
        axes[i].set_yscale("log")
        if k in ["unsheared/mag_err_r", "unsheared/mag_err_i", "unsheared/mag_err_z"]:
            axes[i].set_title(f"log10({k})")
            axes[i].set_xlabel(f"log10({k})")
        else:
            axes[i].set_title(k)
            axes[i].set_xlabel(k)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"After applying mag and shear cuts", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.legend()
    plt.savefig(f"/Users/P.Gebhardt/Development/PhD/output/gaNdalF/feature_hist_after_cuts.pdf", bbox_inches='tight', dpi=300)
    plt.clf()
    plt.close(fig)





    import corner

    arr_generated = df_output_gandalf[OUTPUT_COLS].values
    arr_true = df_output_true[OUTPUT_COLS].values

    ndim = arr_generated.shape[1]

    fig, axes = plt.subplots(ndim, ndim, figsize=(16, 9))

    # Plot gandalf
    corner.corner(
        arr_generated,
        fig=fig,
        # bins=100,
        range=None,
        color='#ff8c00',
        smooth=.8,
        smooth1d=.8,
        labels=[
            f"mag_r",
            f"mag_i",
            f"mag_z",
            f"log10(mag_err_r)",
            f"log10(mag_err_i)",
            f"log10(mag_err_z)",
            "e_1",
            "e_1",
            "snr",
            "size_ratio",
            "T",
        ],
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
        hist_kwargs={'alpha': 1},
        scale_hist=True,
        quantiles=[0.16, 0.5, 0.84],
        levels=[0.393, 0.865, 0.989],  # , 0.989
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
        range=None,
        color='#51a6fb',
        smooth=.8,
        smooth1d=.8,
        labels=[
            f"mag_r",
            f"mag_i",
            f"mag_z",
            f"log10(mag_err_r)",
            f"log10(mag_err_i)",
            f"log10(mag_err_z)",
            "e_1",
            "e_1",
            "snr",
            "size_ratio",
            "T",
        ],
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
        hist_kwargs={'alpha': 1},
        scale_hist=True,
        quantiles=[0.16, 0.5, 0.84],
        levels=[0.393, 0.865, 0.989],  # , 0.989
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

    fig.suptitle(f'After applying mag and shear cuts', fontsize=20)
    fig.legend(handles=legend_elements, loc='upper right', fontsize=16)
    plt.savefig(f"/Users/P.Gebhardt/Development/PhD/output/gaNdalF/corner_after_cuts.pdf", dpi=300)
    plt.clf()
    plt.close(fig)
