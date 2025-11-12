import torch
import torch.optim as optim
from Handler import (
    fnn,
    get_os,
    unsheared_shear_cuts,
    unsheared_mag_cut,
    LoggerHandler,
    plot_features,
    plot_binning_statistics_combined,
    plot_balrog_histogram_with_error,
    plot_features_single,
    plot_balrog_histogram_with_error_and_detection,
    plot_compare_corner,
    calc_color
)
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


def load_config_and_parser(system_path):
    now = datetime.now()
    if get_os() == "Mac":
        print("load MAC config-file")
        config_file_name_classifier = "MAC_run_classifier.cfg"
        config_file_name_flow = "MAC_run_flow.cfg"
    elif get_os() == "Linux":
        print("load LMU config-file")
        config_file_name_classifier = "LMU_run_classifier.cfg"
        config_file_name_flow = "LMU_run_flow.cfg"
    else:
        print("Undefined operating system")
        sys.exit()

    parser_classifier = argparse.ArgumentParser(description='Start gaNdalF')
    parser_classifier.add_argument(
        '--config_filename',
        "-cf",
        type=str,
        nargs=1,
        required=False,
        default=config_file_name_classifier,
        help='Name of config file. If not given default.cfg will be used'
    )
    args_classifier = parser_classifier.parse_args()
    if isinstance(args_classifier.config_filename, list):
        args_classifier.config_filename = args_classifier.config_filename[0]
    with open(f"{system_path}/conf/{args_classifier.config_filename}", 'r') as fp:
        print(f"open {f'{system_path}/conf/{args_classifier.config_filename}'}")
        config_classifier = yaml.safe_load(fp)
    config_classifier['RUN_DATE'] = now.strftime('%Y-%m-%d_%H-%M')

    parser_flow = argparse.ArgumentParser(description='Start gaNdalF')
    parser_flow.add_argument(
        '--config_filename',
        "-cf",
        type=str,
        nargs=1,
        required=False,
        default=config_file_name_flow,
        help='Name of config file. If not given default.cfg will be used'
    )
    args_flow = parser_flow.parse_args()
    if isinstance(args_flow.config_filename, list):
        args_flow.config_filename = args_flow.config_filename[0]
    with open(f"{system_path}/conf/{args_flow.config_filename}", 'r') as fp:
        print(f"open {f'{system_path}/conf/{args_flow.config_filename}'}")
        config_flow = yaml.safe_load(fp)
    config_flow['RUN_DATE'] = now.strftime('%Y-%m-%d_%H-%M')

    return config_classifier, config_flow


def main(classifier_cfg, flow_cfg, logger):
    flow_model = gaNdalF(logger, classifier_cfg=classifier_cfg, flow_cfg=flow_cfg)
    df_gandalf, df_balrog = flow_model.run_flow()

    if flow_cfg["CHECK_INPUT_PLOT"] is True:
        os.makedirs(flow_cfg['PATH_PLOTS'], exist_ok=True)
        plot_features_single(
            cfg=flow_cfg,
            df_gandalf=df_gandalf,
            columns=flow_cfg["INPUT_COLS"],
            title_prefix=f"Flow Input w/o Classifier",
            savename=f"{flow_cfg['PATH_PLOTS']}/feature_input_flow_w_o_classifier.pdf"
        )

    flow_cfg["PATH_PLOTS"] = f'{flow_cfg["PATH_PLOTS"]}/{flow_cfg["RUN_DATE"]}_FLOW_PLOTS'

    df_gandalf = calc_color(
        data_frame=df_gandalf,
        colors=flow_cfg['COLORS_FLOW'],
        column_name=f"unsheared/mag"
    )
    df_balrog = calc_color(
        data_frame=df_balrog,
        colors=flow_cfg['COLORS_FLOW'],
        column_name=f"unsheared/mag"
    )

    plot_flow(
        cfg=flow_cfg,
        logger=logger,
        df_gandalf=df_gandalf,
        df_balrog=df_balrog
    )

    df_gandalf = unsheared_shear_cuts(df_gandalf)
    df_gandalf = unsheared_mag_cut(df_gandalf)
    df_balrog = unsheared_shear_cuts(df_balrog)
    df_balrog = unsheared_mag_cut(df_balrog)

    plot_flow(
        cfg=flow_cfg,
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
            "e_2",
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

    if prefix == "":
        lst_ranges = [
            [-1.5, 4],  # mag r-i
            [-1.5, 4],  # mag i-z
            [15, 38],  # mag r
            [18, 38],  # mag i
            [15, 38],  # mag z
            None,  # mag err r
            None,  # mag err i
            None,  # mag err z
            [-1.2, 1.2],  # e_1
            [-1.2, 1.2],  # e_2
            [1, 2000],  # snr
            [0.5, 200],  # size ratio
            [0, 100]  # T
        ]
        lst_binwidths = [
            0.25,  # mag r-i
            0.25,  # mag i-z
            0.25,  # mag r
            0.25,  # mag i
            0.25,  # mag z
            0.25,  # mag err r
            0.25,  # mag err i
            0.25,  # mag err z
            0.1,  # e_1
            0.1,  # e_2
            50,  # snr
            1.5,  # size ratio
            0.5,  # T
        ]
    else:
        lst_ranges = [
            [-1.5, 4],  # mag r-i
            [-4, 1.5],  # mag i-z
            [15, 26],  # mag r
            [18, 23.5],  # mag i
            [15, 26],  # mag z
            None,  # mag err r
            None,  # mag err i
            None,  # mag err z
            [-1.2, 1.2],  # e_1
            [-1.2, 1.2],  # e_2
            [10, 1000],  # snr
            [0.5, 30],  # size ratio
            [0, 10]  # T
        ]
        lst_binwidths = [
            0.25,  # mag r-i
            0.25,  # mag i-z
            0.25,  # mag r
            0.25,  # mag i
            0.25,  # mag z
            0.25,  # mag err r
            0.25,  # mag err i
            0.25,  # mag err z
            0.1,  # e_1
            0.1,  # e_2
            50,  # snr
            1.5,  # size ratio
            0.5,  # T
        ]

    plot_balrog_histogram_with_error_and_detection(
        df_gandalf=df_gandalf,
        df_balrog=df_balrog,
        columns=cfg["HIST_PLOT_COLS"],
        labels=cfg["HIST_PLOT_LABELS"],
        ranges=lst_ranges,
        binwidths=lst_binwidths,
        title=r"gaNdalF vs. Balrog: Property Distribution Comparison",
        show_plot=cfg["SHOW_PLOT"],
        save_plot=cfg["SAVE_PLOT"],
        save_name=f"{cfg[f'PATH_PLOTS']}/hist_plot{prefix}.pdf"
    )


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    sys.path.append(os.path.dirname(__file__))
    classifier_cfg, flow_cfg = load_config_and_parser(system_path=os.path.abspath(sys.path[-1]))

    log_lvl = logging.INFO
    if flow_cfg["LOGGING_LEVEL"] == "DEBUG":
        log_lvl = logging.DEBUG
    elif flow_cfg["LOGGING_LEVEL"] == "ERROR":
        log_lvl = logging.ERROR

    run_flow_logger = LoggerHandler(
        logger_dict={"logger_name": "train flow logger",
                     "info_logger": flow_cfg['INFO_LOGGER'],
                     "error_logger": flow_cfg['ERROR_LOGGER'],
                     "debug_logger": flow_cfg['DEBUG_LOGGER'],
                     "stream_logger": flow_cfg['STREAM_LOGGER'],
                     "stream_logging_level": log_lvl},
        log_folder_path=f"{flow_cfg['PATH_OUTPUT']}/"
    )

    main(classifier_cfg=classifier_cfg, flow_cfg=flow_cfg, logger=run_flow_logger)
