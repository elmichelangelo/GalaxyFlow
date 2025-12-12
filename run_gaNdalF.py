from datetime import datetime
from Handler import (
    fnn,
    apply_cuts,
    get_os,
    unsheared_shear_cuts,
    unsheared_mag_cut,
    binary_cut,
    LoggerHandler,
    calc_color,
    mag2flux,
    flux2mag,
    fluxerr2magerr,
    plot_roc_curve_gandalf,
    plot_features_single,
    plot_balrog_histogram_with_error_and_detection,
    plot_rate_ratio_by_mag,
    plot_confusion_matrix,
    plot_features,
    plot_binning_statistics_combined,
    plot_balrog_histogram_with_error,
    plot_features_compare,
    plot_compare_corner,
    compute_injection_counts,
    run_flow_cuts,
    plot_balrog_histogram_with_error_and_detection_true_balrog
)
from gandalf import gaNdalF
import argparse
import matplotlib.pyplot as plt
import sys
import yaml
import os
import gc
import pandas as pd
import logging
import numpy as np

sys.path.append(os.path.dirname(__file__))
plt.rcParams["figure.figsize"] = (16, 9)


def load_config_and_parser(system_path):
    now = datetime.now()
    if get_os() == "Mac":
        print("load MAC config-file")
        config_file_name = "MAC_run_gandalf.cfg"
    elif get_os() == "Linux":
        print("load LMU config-file")
        config_file_name = "LMU_run_gandalf.cfg"
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
    with open(f"{system_path}/conf/{args.config_filename}", 'r') as fp:
        print(f"open {f'{system_path}/conf/{args.config_filename}'}")
        config = yaml.safe_load(fp)
    config['RUN_DATE'] = now.strftime('%Y-%m-%d_%H-%M')

    return config


def plot_classifier(cfg, df_gandalf):
    if cfg["SAVE_PLOT"] is True:
        os.makedirs(cfg["PATH_PLOTS"], exist_ok=True)

    plot_confusion_matrix(
        df_gandalf=df_gandalf,
        # df_balrog=df_balrog,
        y_true_col="mcal_galaxy",
        y_pred_col="sampled mcal_galaxy",
        show_plot=cfg['SHOW_PLOT'],
        save_plot=cfg['SAVE_PLOT'],
        save_name=f"{cfg['PATH_PLOTS']}/{cfg['RUN_NUMBER']}thr_confusion_matrix.png",
        title=f"Confusion Matrix"
    )

    plot_confusion_matrix(
        df_gandalf=df_gandalf,
        # df_balrog=df_balrog,
        y_true_col="mcal_galaxy",
        y_pred_col="sampled mcal_galaxy",
        show_plot=cfg['SHOW_PLOT'],
        save_plot=cfg['SAVE_PLOT'],
        save_name=f"{cfg['PATH_PLOTS']}/{cfg['RUN_NUMBER']}smpl_confusion_matrix.png",
        title=f"Confusion Matrix"
    )

    plot_roc_curve_gandalf(
        df_gandalf=df_gandalf,
        # df_balrog=df_balrog,
        y_true_col="mcal_galaxy",
        y_pred_col="sampled mcal_galaxy",
        show_plot=cfg['SHOW_PLOT'], save_plot=cfg['SAVE_PLOT'],
        save_name=f"{cfg['PATH_PLOTS']}/{cfg['RUN_NUMBER']}thr_roc_curve.pdf",
        title=f"Receiver Operating Characteristic (ROC) Curve threshold",
    )

    plot_roc_curve_gandalf(
        df_gandalf=df_gandalf,
        # df_balrog=df_balrog,
        y_true_col="mcal_galaxy",
        y_pred_col="sampled mcal_galaxy",
        show_plot=cfg['SHOW_PLOT'], save_plot=cfg['SAVE_PLOT'],
        save_name=f"{cfg['PATH_PLOTS']}/{cfg['RUN_NUMBER']}smpl_roc_curve.pdf",
        title=f"Receiver Operating Characteristic (ROC) Curve sample",
    )

    # plot_rate_ratio_by_mag(
    #     mag=df_gandalf["BDF_MAG_DERED_CALIB_Z"].to_numpy(float),
    #     y_true=df_gandalf["mcal_galaxy"].to_numpy().astype(int).ravel(),
    #     probs_raw=df_gandalf["sampled mcal_galaxy"].to_numpy(float),
    #     calibrated=False,
    #     bin_width=1,
    #     mag_label="BDF_MAG_DERED_CALIB_Z",
    #     show_density_ratio=False,
    #     save_name=f"{cfg['PATH_PLOTS']}/{cfg['RUN_NUMBER']}thr_rate_ratio_raw_only.pdf"
    # )
    #
    # plot_rate_ratio_by_mag(
    #     mag=df_gandalf["BDF_MAG_DERED_CALIB_Z"].to_numpy(float),
    #     y_true=df_gandalf["mcal_galaxy"].to_numpy().astype(int).ravel(),
    #     probs_raw=df_gandalf["sampled mcal_galaxy"].to_numpy(float),
    #     calibrated=False,
    #     bin_width=1,
    #     mag_label="BDF_MAG_DERED_CALIB_Z",
    #     show_density_ratio=False,
    #     save_name=f"{cfg['PATH_PLOTS']}/{cfg['RUN_NUMBER']}smpl_rate_ratio_raw_only.pdf"
    # )


def plot_flow(cfg, logger, df_gandalf, df_balrog, prefix=""):
    if cfg["SAVE_PLOT"] is True:
        os.makedirs(cfg["PATH_PLOTS"], exist_ok=True)

    # plot_features(
    #     cfg=cfg,
    #     plot_log=logger,
    #     df_gandalf=df_gandalf,
    #     df_balrog=df_balrog,
    #     columns=cfg["OUTPUT_COLS"],
    #     title_prefix=f"Feature Plot",
    #     savename=f"{cfg['PATH_PLOTS']}/feature_plot{prefix}.pdf"
    # )
    #
    # plot_binning_statistics_combined(
    #     df_gandalf=df_gandalf,
    #     df_balrog=df_balrog,
    #     sample_size=10000,
    #     plot_scatter=False,
    #     show_plot=cfg["SHOW_PLOT"],
    #     save_plot=cfg["SAVE_PLOT"],
    #     title="gaNdalF vs. Balrog: Measured Photometric Property Distribution Comparison",
    #     save_name=f"{cfg['PATH_PLOTS']}/binning_statistics_combined{prefix}",
    # )

    if prefix != "":
        plot_compare_corner(
            data_frame_generated=df_gandalf,
            data_frame_true=df_balrog,
            dict_delta=None,
            epoch=None,
            title=f"chain plot",
            show_plot=cfg['SHOW_PLOT'],
            save_plot=cfg['SAVE_PLOT'],
            save_name=f"{cfg[f'PATH_PLOTS']}/{cfg['RUN_NUMBER']}chainplot{prefix}.pdf",
            columns=cfg["OUTPUT_COLS_NF"],
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
            save_name=f"{cfg[f'PATH_PLOTS']}/{cfg['RUN_NUMBER']}color_color{prefix}.pdf",
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
        save_name=f"{cfg[f'PATH_PLOTS']}/{cfg['RUN_NUMBER']}hist_plot{prefix}.pdf",
        detection_name="mcal_galaxy"
    )


# def test_plot(df_gandalf, cfg, logger, title=f"Feature Plot"):
#     plot_features(
#         cfg=cfg,
#         plot_log=logger,
#         df_gandalf=df_gandalf,
#         df_balrog=df_gandalf,
#         columns=cfg["OUTPUT_COLS"],
#         title_prefix=title,
#         savename=f"{cfg['PATH_PLOTS']}/feature_plot.pdf"
#     )


def add_confusion_label(df, pred_col="detected", true_col="true_detected"):
    df = df.copy()
    conditions = [
        (df[pred_col] == 1) & (df[true_col] == 1),
        (df[pred_col] == 0) & (df[true_col] == 0),
        (df[pred_col] == 1) & (df[true_col] == 0),
        (df[pred_col] == 0) & (df[true_col] == 1),
    ]
    labels = ["TP", "TN", "FP", "FN"]
    df["confusion"] = np.select(conditions, labels, default="unknown")
    return df

def save_catalogs(cfg, df, filename, key: str = "catalog", mode: str = "w", format: str = "table", complib: str = "blosc", complevel: int = 9,):
    """"""
    os.makedirs(cfg["PATH_CATALOGS"], exist_ok=True)

    df_to_save = df.copy()
    df_to_save.columns = [str(c) for c in df_to_save.columns]

    if ".h5" in filename:
        df_to_save.to_hdf(
            f"{cfg['PATH_CATALOGS']}/{filename}",
            key=key,
            mode=mode,
            format=format,
            complib=complib,
            complevel=complevel,
        )
    else:
        df_to_save.to_pickle(
            f"{cfg['PATH_CATALOGS']}/{filename}",
        )

def plot_compare_true_wide_histogram(cfg,
                                     df_gandalf_selected: pd.DataFrame | None = None,
                                     df_balrog_selected: pd.DataFrame | None = None):
    df_missing_columns = pd.read_pickle(f"{cfg['PATH_DATA']}/{cfg['FILENAME_FULL_GANDALF_CATALOG']}")
    df_missing_columns = df_missing_columns[["bal_id", "unsheared/extended_class_sof", "unsheared/flags_gold", "match_flag_1.5_asec", "flags_foreground", "flags_badregions", "flags_footprint", "unsheared/dec", "unsheared/ra"]]
    df_full_balrog = pd.read_pickle(f"{cfg['PATH_DATA']}/{cfg['FILENAME_FULL_GANDALF_TRAINING_SET']}")
    df_merged_balrog = df_full_balrog.merge(df_missing_columns, how="left", on="bal_id")

    for band in ["r", "i", "z"]:
        df_merged_balrog[f"unsheared/flux_{band}"] = mag2flux(df_merged_balrog[f"unsheared/mag_{band}"])

    df_merged_balrog["r-i"] = df_merged_balrog["unsheared/mag_r"] - df_merged_balrog["unsheared/mag_i"]
    df_merged_balrog["i-z"] = df_merged_balrog["unsheared/mag_i"] - df_merged_balrog["unsheared/mag_z"]
    df_merged_balrog = df_merged_balrog[df_merged_balrog["detected"] == 1]

    df_merged_balrog = apply_cuts(cfg=cfg, data_frame=df_merged_balrog)
    df_merged_balrog = df_merged_balrog[cfg["HIST_PLOT_COLS"]+["mcal_galaxy"]]

    df_true_balrog = pd.read_hdf(f"{cfg['PATH_DATA']}/{cfg['FILENAME_TRUE_BALROG_CATALOG']}", key="df/")
    for band in ["r", "i", "z"]:
        df_true_balrog[f"unsheared/mag_{band}"] = flux2mag(df_true_balrog[f"unsheared/flux_{band}"])
        df_true_balrog[f"unsheared/mag_err_{band}"] = np.log10(
            fluxerr2magerr(df_true_balrog[f"unsheared/flux_{band}"], df_true_balrog[f"unsheared/flux_err_{band}"]))
        df_merged_balrog[f"unsheared/mag_err_{band}"] = np.log10(df_merged_balrog[f"unsheared/mag_err_{band}"])
    df_true_balrog["r-i"] = df_true_balrog["unsheared/mag_r"] - df_true_balrog["unsheared/mag_i"]
    df_true_balrog["i-z"] = df_true_balrog["unsheared/mag_i"] - df_true_balrog["unsheared/mag_z"]
    df_true_balrog = df_true_balrog[cfg["HIST_PLOT_COLS"]]

    if df_balrog_selected is None:
        df_balrog_selected = df_true_balrog
        df_balrog_selected['mcal_galaxy'] = np.ones(len(df_balrog_selected))
        df_balrog_selected['detected'] = np.ones(len(df_balrog_selected))
    else:
        df_true_balrog["mcal_galaxy"] = np.ones(len(df_true_balrog))
    if df_gandalf_selected is None:
        df_gandalf_selected = df_merged_balrog
        df_gandalf_selected['mcal_galaxy'] = np.ones(len(df_gandalf_selected))
        df_gandalf_selected['sampled mcal_galaxy'] = np.ones(len(df_gandalf_selected))

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

    plot_balrog_histogram_with_error_and_detection_true_balrog(
        df_gandalf=df_gandalf_selected,
        df_balrog=df_balrog_selected,
        true_balrog=df_true_balrog,
        complete_balrog=df_merged_balrog,
        columns=cfg["HIST_PLOT_COLS"],
        labels=cfg["HIST_PLOT_LABELS"],
        ranges=lst_ranges,
        binwidths=lst_binwidths,
        title=r"gaNdalF vs. Balrog: Property Distribution Comparison",
        show_plot=cfg["SHOW_PLOT"],
        save_plot=cfg["SAVE_PLOT"],
        save_name=f"{cfg[f'PATH_PLOTS']}/{cfg['RUN_NUMBER']}hist_plot_three.pdf",
        detection_name="mcal_galaxy"
    )


def add_true_id(cfg, df):
    df_information = pd.read_pickle(f"{cfg['PATH_DATA']}/{cfg['FILENAME_FULL_GANDALF_CATALOG']}")
    df_information = df_information[["bal_id", "ID"]].copy()
    df_information.rename(columns={"ID": "true_id"}, inplace=True)
    return df.merge(df_information, how="left", on="bal_id")

def init_output_paths(cfg):
    cfg["PATH_OUTPUT"] = f'{cfg["PATH_OUTPUT"]}/{cfg["RUN_DATE"]}_RUN_GANDALF'
    os.makedirs(cfg["PATH_OUTPUT"], exist_ok=True)

    cfg["PATH_PLOTS"] = f'{cfg["PATH_OUTPUT"]}/Plots'
    os.makedirs(cfg["PATH_PLOTS"], exist_ok=True)

    cfg["PATH_LOGS"] = f'{cfg["PATH_OUTPUT"]}/Logs'
    os.makedirs(cfg["PATH_LOGS"], exist_ok=True)

    cfg["PATH_CATALOGS"] = f'{cfg["PATH_OUTPUT"]}/Catalogs'
    os.makedirs(cfg["PATH_CATALOGS"], exist_ok=True)


def main(cfg, logger):
    """"""
    # Init gaNdalF model
    model = gaNdalF(logger, cfg=cfg)

    # plot_compare_true_wide_histogram(
    #     cfg=cfg,
    #     df_gandalf_selected=None,
    #     df_balrog_selected=None
    # )

    # Init Classifier model from gaNdalF
    model.init_classifier()

    # Scale input data for gaNdalF run
    logger.log_info_stream(f"Scale classifier data")
    model.classifier_galaxies.scale_data(
        cfg_key_cols_interest="COLUMNS_OF_INTEREST_CF",
        cfg_key_filename_scaler="FILENAME_STANDARD_SCALER_CF",
    )

    # Run gaNdalF classifier
    df_gandalf, df_balrog = model.run_classifier()

    # add new gaNdalF ID
    logger.log_info_stream(f"Add gandalf_id")
    df_gandalf["gandalf_id"] = np.arange(len(df_gandalf))

    logger.log_info_stream(f"Add true_id")
    df_gandalf = add_true_id(
        cfg=cfg,
        df=df_gandalf
    )

    logger.log_info_stream(f"compute injection counts")
    df_gandalf = compute_injection_counts(
        det_catalog=df_gandalf.copy(),
        id_col="true_id",
        count_col="injection_counts"
    )

    df_gandalf_injection_counts = df_gandalf[["gandalf_id", "bal_id", "injection_counts", "true_id"]].copy()

    logger.log_info_stream(f"Inverse scale classifier data")
    df_gandalf = model.classifier_galaxies.inverse_scale_data(df_gandalf)
    df_balrog = model.classifier_galaxies.inverse_scale_data(df_balrog)

    if cfg["CHECK_INPUT_PLOT"] is True:
        logger.log_info_stream(f"Plot input columns")
        os.makedirs(cfg['PATH_PLOTS'], exist_ok=True)
        try:
            plot_features_single(
                cfg=cfg,
                df_gandalf=df_gandalf,
                columns=cfg["INPUT_COLS"],
                title_prefix=f"Classifier Input Columns",
                savename=f"{cfg['PATH_PLOTS']}/{cfg['RUN_NUMBER']}feature_input_classifier.pdf"
            )
            plot_classifier(
                cfg=cfg,
                df_gandalf=df_gandalf,
            )
        except Exception as e:
            logger.log_error(f"Error plot classifier: {e}")

    if cfg["SAVE_CLF_DATA"] is True:
        logger.log_info_stream(f"Save classifier data")
        save_catalogs(
            cfg=cfg,
            df=df_gandalf,
            filename=f"{cfg['RUN_DATE']}_{cfg['RUN_NUMBER']}gandalf_Classified.pkl")
        save_catalogs(
            cfg=cfg,
            df=df_balrog,
            filename=f"{cfg['RUN_DATE']}_{cfg['RUN_NUMBER']}balrog_Classified.pkl")

    model.init_flow(
        data_frame=df_gandalf,
    )

    logger.log_info_stream(f"Apply log10 flow data")
    df_gandalf = model.flow_galaxies.apply_log10(
        data_frame=df_gandalf,
        cfg_key_log10_cols="LOG10_COLS"
    )

    logger.log_info_stream(f"Apply scale flow data")
    df_gandalf = model.flow_galaxies.scale_data(
        data_frame=df_gandalf,
        cfg_key_cols_interest="COLUMNS_OF_INTEREST_NF",
        cfg_key_filename_scaler="FILENAME_STANDARD_SCALER_NF"
    )

    logger.log_info_stream(f"Run flow")
    df_gandalf, df_balrog = model.run_flow(data_frame=df_gandalf)

    logger.log_info_stream(f"Add color")
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

    for band in ["r", "i", "z"]:
        df_gandalf[f"unsheared/flux_{band}"] = mag2flux(df_gandalf[f"unsheared/mag_{band}"])
        df_balrog[f"unsheared/flux_{band}"] = mag2flux(df_balrog[f"unsheared/mag_{band}"])

    logger.log_info_stream(f"Only selected")
    df_balrog_selected = df_balrog[df_balrog["mcal_galaxy"]==1]
    df_gandalf_selected = df_gandalf[df_gandalf["sampled mcal_galaxy"]==1]

    logger.log_info_stream(f"Plot flow before mcal cuts")

    if cfg["PLOT_FLOW"] is True:
        try:
            plot_flow(
                cfg=cfg,
                logger=logger,
                df_gandalf=df_gandalf_selected,
                df_balrog=df_balrog_selected,
                prefix=""
            )
        except Exception as e:
            logger.log_error(f"Error plot flow before cut: {e}")

    logger.log_info_stream(f"Apply mcal cuts to gandalf data")
    df_gandalf_selected = unsheared_shear_cuts(df_gandalf_selected)
    df_gandalf_selected = unsheared_mag_cut(df_gandalf_selected)
    df_gandalf_selected = binary_cut(df_gandalf_selected)

    logger.log_info_stream(f"Apply mcal cuts to balrog data")
    df_balrog_selected = unsheared_shear_cuts(df_balrog_selected)
    df_balrog_selected = unsheared_mag_cut(df_balrog_selected)
    df_balrog_selected = binary_cut(df_balrog_selected)

    logger.log_info_stream(f"Plot flow after mcal cuts")
    #
    # plot_compare_true_wide_histogram(
    #     cfg=cfg,
    #     df_gandalf_selected=df_gandalf_selected,
    #     df_balrog_selected=df_balrog_selected
    # )

    if cfg["PLOT_FLOW"] is True:
        try:
            plot_flow(
                cfg=cfg,
                logger=logger,
                df_gandalf=df_gandalf_selected,
                df_balrog=df_balrog_selected,
                prefix="_cut"
            )
        except Exception as e:
            logger.log_error(f"Error plot flow after cut: {e}")

    logger.log_info_stream(f"Merge flow catalog")
    df_merged = df_gandalf_selected.merge(df_gandalf_injection_counts, how="left", on="gandalf_id")

    logger.log_info_stream(f"Save flow catalog")
    save_catalogs(
        cfg=cfg,
        df=df_merged,
        filename=f"{cfg['RUN_DATE']}_{cfg['RUN_NUMBER']}gandalf_Emulated_Classified_{cfg['NUMBER_SAMPLES']}_{len(df_merged)}.h5")

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    sys.path.append(os.path.dirname(__file__))
    config = load_config_and_parser(system_path=os.path.abspath(sys.path[-1]))

    log_lvl = logging.INFO
    if config["LOGGING_LEVEL"] == "DEBUG":
        log_lvl = logging.DEBUG

    init_output_paths(cfg=config)
    run_logger = LoggerHandler(
        logger_dict={"logger_name": "train flow logger",
                     "info_logger": config['INFO_LOGGER'],
                     "error_logger": config['ERROR_LOGGER'],
                     "debug_logger": config['DEBUG_LOGGER'],
                     "stream_logger": config['STREAM_LOGGER'],
                     "stream_logging_level": log_lvl},
        log_folder_path=f"{config['PATH_LOGS']}/"
    )

    if config["BOOTSTRAP"] is True:
        for actual_boot in range(config["TOTAL_BOOTSTRAP"]):
            config['RUN_NUMBER'] = f"{actual_boot+1}_"
            config['BERNOULLI_SEED'] = config['BERNOULLI_SEED'] + actual_boot + 1
            main(cfg=config, logger=run_logger)
    else:
        config['RUN_NUMBER'] = ""
        main(cfg=config, logger=run_logger)
