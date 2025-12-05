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


def plot_classifier(cfg, df_gandalf, df_balrog):
    if cfg["SAVE_PLOT"] is True:
        os.makedirs(cfg["PATH_PLOTS"], exist_ok=True)

    plot_confusion_matrix(
        df_gandalf=df_gandalf,
        # df_balrog=df_balrog,
        y_true_col="true detected",
        y_pred_col="threshold detected",
        show_plot=cfg['SHOW_PLOT'],
        save_plot=cfg['SAVE_PLOT'],
        save_name=f"{cfg['PATH_PLOTS']}/thr_confusion_matrix.png",
        title=f"Confusion Matrix"
    )

    plot_confusion_matrix(
        df_gandalf=df_gandalf,
        # df_balrog=df_balrog,
        y_true_col="true detected",
        y_pred_col="sampled detected",
        show_plot=cfg['SHOW_PLOT'],
        save_plot=cfg['SAVE_PLOT'],
        save_name=f"{cfg['PATH_PLOTS']}/smpl_confusion_matrix.png",
        title=f"Confusion Matrix"
    )

    plot_roc_curve_gandalf(
        df_gandalf=df_gandalf,
        # df_balrog=df_balrog,
        y_true_col="true detected",
        y_pred_col="threshold detected",
        show_plot=cfg['SHOW_PLOT'], save_plot=cfg['SAVE_PLOT'],
        save_name=f"{cfg['PATH_PLOTS']}/thr_roc_curve.pdf",
        title=f"Receiver Operating Characteristic (ROC) Curve threshold",
    )

    plot_roc_curve_gandalf(
        df_gandalf=df_gandalf,
        # df_balrog=df_balrog,
        y_true_col="true detected",
        y_pred_col="sampled detected",
        show_plot=cfg['SHOW_PLOT'], save_plot=cfg['SAVE_PLOT'],
        save_name=f"{cfg['PATH_PLOTS']}/smpl_roc_curve.pdf",
        title=f"Receiver Operating Characteristic (ROC) Curve sample",
    )

    plot_rate_ratio_by_mag(
        mag=df_gandalf["BDF_MAG_DERED_CALIB_Z"].to_numpy(float),
        y_true=df_gandalf["true detected"].to_numpy().astype(int).ravel(),
        probs_raw=df_gandalf["threshold detected"].to_numpy(float),
        calibrated=False,
        bin_width=1,
        mag_label="BDF_MAG_DERED_CALIB_Z",
        show_density_ratio=False,
        save_name=f"{cfg['PATH_PLOTS']}/thr_rate_ratio_raw_only.pdf"
    )

    plot_rate_ratio_by_mag(
        mag=df_gandalf["BDF_MAG_DERED_CALIB_Z"].to_numpy(float),
        y_true=df_gandalf["true detected"].to_numpy().astype(int).ravel(),
        probs_raw=df_gandalf["sampled detected"].to_numpy(float),
        calibrated=False,
        bin_width=1,
        mag_label="BDF_MAG_DERED_CALIB_Z",
        show_density_ratio=False,
        save_name=f"{cfg['PATH_PLOTS']}/smpl_rate_ratio_raw_only.pdf"
    )


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
        save_name=f"{cfg[f'PATH_PLOTS']}/hist_plot{prefix}.pdf",
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
    os.makedirs(cfg["PATH_OUTPUT"], exist_ok=True)

    df_to_save = df.copy()
    df_to_save.columns = [str(c) for c in df_to_save.columns]

    if ".h5" in filename:
        df_to_save.to_hdf(
            f"{cfg['PATH_OUTPUT']}/{filename}",
            key=key,
            mode=mode,
            format=format,
            complib=complib,
            complevel=complevel,
        )
    else:
        df_to_save.to_pickle(
            f"{cfg['PATH_OUTPUT']}/{filename}",
        )



def main(classifier_cfg, flow_cfg, logger):
    """"""
    model = gaNdalF(logger, classifier_cfg=classifier_cfg, flow_cfg=flow_cfg)

    df_gandalf, df_balrog = model.run_classifier()

    df_merged_balrog = pd.read_pickle("/Volumes/elmichelangelo_external_ssd_1/Data/20250927_balrog_complete_26303386.pkl")
    df_information = df_merged_balrog[["bal_id", "ID"]].copy()
    df_merged_balrog["r-i"] = df_merged_balrog["unsheared/mag_r"] - df_merged_balrog["unsheared/mag_i"]
    df_merged_balrog["i-z"] = df_merged_balrog["unsheared/mag_i"] - df_merged_balrog["unsheared/mag_z"]
    df_merged_balrog = df_merged_balrog[df_merged_balrog["detected"]==1]

    df_merged_balrog = apply_cuts(cfg=flow_cfg, data_frame=df_merged_balrog)

    df_merged_balrog = df_merged_balrog[flow_cfg["HIST_PLOT_COLS"]]

    true_balrog_dataset = "/Volumes/elmichelangelo_external_ssd_1/Data/y3_balrog2_v1.2_merged_select2_bstarcut_matchflag1.5asec_snr_SR_corrected_uppersizecuts.h5"
    df_true_balrog = pd.read_hdf(true_balrog_dataset, key="df/")
    for band in ["r", "i", "z"]:
        df_true_balrog[f"unsheared/mag_{band}"] = flux2mag(df_true_balrog[f"unsheared/flux_{band}"])
        df_true_balrog[f"unsheared/mag_err_{band}"] = np.log10(fluxerr2magerr(df_true_balrog[f"unsheared/flux_{band}"], df_true_balrog[f"unsheared/flux_err_{band}"]))
        df_merged_balrog[f"unsheared/mag_err_{band}"] = np.log10(df_merged_balrog[f"unsheared/mag_err_{band}"])
    df_true_balrog["r-i"] = df_true_balrog["unsheared/mag_r"] - df_true_balrog["unsheared/mag_i"]
    df_true_balrog["i-z"] = df_true_balrog["unsheared/mag_i"] - df_true_balrog["unsheared/mag_z"]
    df_true_balrog = df_true_balrog[flow_cfg["HIST_PLOT_COLS"]]

    df_information.rename(columns={"ID": "true_id"}, inplace=True)

    df_gandalf = df_gandalf.merge(df_information, how="left", on="bal_id")

    df_gandalf = compute_injection_counts(
        det_catalog=df_gandalf.copy(),
        id_col="true_id",
        count_col="injection_counts"
    )

    del df_information

    classifier_cfg["PATH_PLOTS"] = f'{classifier_cfg["PATH_PLOTS"]}/{classifier_cfg["RUN_DATE"]}_RUN_PLOTS'
    flow_cfg["PATH_PLOTS"] = classifier_cfg["PATH_PLOTS"]

    df_gandalf = model.classifier_galaxies.inverse_scale_data(df_gandalf)
    df_balrog = model.classifier_galaxies.inverse_scale_data(df_balrog)

    if classifier_cfg["CHECK_INPUT_PLOT"] is True:
        flow_cfg['PATH_PLOTS'] = classifier_cfg['PATH_PLOTS']
        os.makedirs(classifier_cfg['PATH_PLOTS'], exist_ok=True)
        plot_features_single(
            cfg=classifier_cfg,
            df_gandalf=df_gandalf,
            columns=classifier_cfg["INPUT_COLS"],
            title_prefix=f"Classifier Input Columns",
            savename=f"{classifier_cfg['PATH_PLOTS']}/feature_input_classifier.pdf"
        )

    # df = pd.read_pickle("/Volumes/elmichelangelo_external_ssd_1/Data/20250927_balrog_complete_26303386.pkl")
    # df_information = df[["bal_id", "ID", "injection_counts"]]

    # df_out_gandalf = df_gandalf.merge(df_information, how="left", on="bal_id")
    # df_out_gandalf.rename(columns={"ID": "true_id"}, inplace=True)
    #
    # df_out_balrog = df_balrog.merge(df_information, how="left", on="bal_id")
    # df_out_balrog.rename(columns={"ID": "true_id"}, inplace=True)
    #
    # df_out_gandalf = compute_injection_counts(
    #     det_catalog=df_out_gandalf,
    #     id_col="true_id",
    #     count_col="injection_counts_m"
    # )
    #
    # df_out_balrog = compute_injection_counts(
    #     det_catalog=df_out_balrog,
    #     id_col="true_id",
    #     count_col="injection_counts_m"
    # )
    #
    # plt.scatter(df_out_balrog["injection_counts_m"], df_out_gandalf["injection_counts_m"])
    # plt.show()

    # df_gandalf_detected = df_gandalf[df_gandalf["sampled mcal_galaxy"]==1]
    # df_balrog_detected = df_balrog[df_balrog["mcal_galaxy"]==1]

    # df_out_gandalf_detected = df_gandalf_detected.merge(df_information, how="left", on="bal_id")
    # df_out_gandalf_detected.rename(columns={"ID": "true_id"}, inplace=True)
    #
    # df_out_balrog_detetced = df_balrog_detected.merge(df_information, how="left", on="bal_id")
    # df_out_balrog_detetced.rename(columns={"ID": "true_id"}, inplace=True)
    #
    # df_out_gandalf_detected = compute_injection_counts(
    #     det_catalog=df_out_gandalf_detected,
    #     id_col="true_id",
    #     count_col="detection_counts_n"
    # )
    #
    # df_out_balrog_detetced = compute_injection_counts(
    #     det_catalog=df_out_balrog_detetced,
    #     id_col="true_id",
    #     count_col="detection_counts_n"
    # )
    #
    # df_check = df_out_gandalf_detected[["true_id", "bal_id", "detection_counts_n"]].copy()
    # df_check.rename(columns={"detection_counts_n": "detection_counts_n_gandalf"}, inplace=True)
    #
    # counts_balrog = (
    #     df_out_balrog_detetced[["true_id", "detection_counts_n"]]
    #     .dropna(subset=["true_id"])
    #     .drop_duplicates("true_id")  # eine Zeile pro true_id
    #     .set_index("true_id")["detection_counts_n"]  # Series: index=true_id, value=count
    # )
    #
    # df_check["detection_counts_n_balrog"] = df_check["true_id"].map(counts_balrog)
    #
    # df_check = df_check.drop_duplicates("true_id")
    #
    # x = df_check["detection_counts_n_balrog"]
    # y = df_check["detection_counts_n_gandalf"]
    #
    # lo = int(min(x.min(), y.min()))
    # hi = int(max(x.max(), y.max()))
    #
    # plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)
    # plt.gca().set_aspect("equal", "box")
    # plt.scatter(df_check["detection_counts_n_balrog"], df_check["detection_counts_n_gandalf"])
    # plt.title("Compare Detection Counts: n vs. n'")
    # plt.xlim(0, hi)
    # plt.ylim(0, hi)
    # plt.xlabel("detection counts balrog: n")
    # plt.ylabel("detection counts gandalf: n'")
    # plt.savefig(f"{flow_cfg['PATH_PLOTS']}/n_vs_ndash.png", dpi=150, bbox_inches='tight')
    # plt.clf()
    # plt.close()
    #
    # plt.scatter(df_check["detection_counts_n_balrog"], (df_check["detection_counts_n_gandalf"]-df_check["detection_counts_n_balrog"])/np.sqrt(df_check["detection_counts_n_balrog"]), alpha=0.3)
    # plt.title("Compare Detection Counts: n vs. n'")
    # plt.ylabel("(n'-n)/sqrt(n)")
    # plt.xlabel("n")
    # plt.savefig(f"{flow_cfg['PATH_PLOTS']}/compare_detection_counts.png", dpi=150, bbox_inches='tight')
    # plt.clf()
    # plt.close()

    save_catalogs(
        cfg=classifier_cfg,
        df=df_gandalf,
        filename=f"{classifier_cfg['RUN_DATE']}_gandalf_Classified.pkl")

    save_catalogs(
        cfg=classifier_cfg,
        df=df_balrog,
        filename=f"{classifier_cfg['RUN_DATE']}_balrog_Classified.pkl")


    # plot_classifier(
    #     cfg=classifier_cfg,
    #     df_gandalf=df_gandalf,
    #     df_balrog=df_balrog,
    # )

    # df_gandalf = df_gandalf[df_gandalf["BDF_MAG_DERED_CALIB_R"]<32]
    # df_gandalf = df_gandalf[df_gandalf["BDF_MAG_DERED_CALIB_I"]<32]
    # df_gandalf = df_gandalf[df_gandalf["BDF_MAG_DERED_CALIB_Z"]<32]

    # df_gandalf = add_confusion_label(df=df_gandalf, pred_col="sampled detected", true_col="true detected")

    # df_gandalf2 = model.flow_data.copy()
    #
    # df_gandalf2 = model.flow_galaxies.inverse_scale_data(df_gandalf2)
    #
    # if flow_cfg["CHECK_INPUT_PLOT"] is True:
    #     os.makedirs(flow_cfg['PATH_PLOTS'], exist_ok=True)
    #     plot_features_compare(
    #         cfg=flow_cfg,
    #         df_gandalf=df_gandalf,
    #         df_gandalf2=df_gandalf2,
    #         columns=flow_cfg["INPUT_COLS"],
    #         title_prefix=f"Flow Input w/ compare",
    #         savename=f"{flow_cfg['PATH_PLOTS']}/feature_input_flow_w_classifier.pdf"
    #     )

    # df_gandalf = run_flow_cuts(cfg=flow_cfg, data_frame=df_gandalf)

    # if flow_cfg["CHECK_INPUT_PLOT"] is True:
    #     plot_features_compare(
    #         cfg=flow_cfg,
    #         df_gandalf=df_gandalf,
    #         df_gandalf2=df_gandalf2,
    #         columns=flow_cfg["INPUT_COLS"],
    #         title_prefix=f"Flow Input w/ compare new cuts",
    #         savename=f"{flow_cfg['PATH_PLOTS']}/feature_input_flow_w_classifier_cut.pdf"
    #     )

    df_gandalf_injection_counts = df_gandalf[["bal_id", "injection_counts", "true_id"]].copy()

    df_gandalf = model.flow_galaxies.apply_log10(df_gandalf)
    df_gandalf = model.flow_galaxies.scale_data(df_gandalf)
    df_gandalf, df_balrog = model.run_flow(data_frame=df_gandalf)

    # df_out_gandalf_flow = df_gandalf.merge(df_information, how="left", on="bal_id")
    # df_out_gandalf_flow.rename(columns={"ID": "true_id"}, inplace=True)
    #
    # df_out_balrog_flow = df_balrog.merge(df_information, how="left", on="bal_id")
    # df_out_balrog_flow.rename(columns={"ID": "true_id"}, inplace=True)

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

    for band in ["r", "i", "z"]:
        df_gandalf[f"unsheared/flux_{band}"] = mag2flux(df_gandalf[f"unsheared/mag_{band}"])
        df_balrog[f"unsheared/flux_{band}"] = mag2flux(df_balrog[f"unsheared/mag_{band}"])

    df_balrog_selected = df_balrog[df_balrog["mcal_galaxy"]==1]
    df_gandalf_selected = df_gandalf[df_gandalf["sampled mcal_galaxy"]==1]

    plot_flow(
        cfg=flow_cfg,
        logger=logger,
        df_gandalf=df_gandalf_selected,
        df_balrog=df_balrog_selected,
        prefix=""
    )

    df_gandalf_selected = unsheared_shear_cuts(df_gandalf_selected)
    df_gandalf_selected = unsheared_mag_cut(df_gandalf_selected)
    df_gandalf_selected = binary_cut(df_gandalf_selected)

    df_balrog_selected = unsheared_shear_cuts(df_balrog_selected)
    df_balrog_selected = unsheared_mag_cut(df_balrog_selected)
    df_balrog_selected = binary_cut(df_balrog_selected)

    # df_out_gandalf_flow = df_gandalf.merge(df_information, how="left", on="bal_id")
    # df_out_gandalf_flow.rename(columns={"ID": "true_id"}, inplace=True)
    #
    # df_out_balrog_flow = df_balrog.merge(df_information, how="left", on="bal_id")
    # df_out_balrog_flow.rename(columns={"ID": "true_id"}, inplace=True)
    #
    # df_out_gandalf_flow = compute_injection_counts(
    #     det_catalog=df_out_gandalf_flow,
    #     id_col="true_id",
    #     count_col="detection_counts_k"
    # )
    #
    # df_out_balrog_flow = compute_injection_counts(
    #     det_catalog=df_out_balrog_flow,
    #     id_col="true_id",
    #     count_col="detection_counts_k"
    # )
    #
    # df_check_flow = df_out_gandalf_flow[["true_id", "bal_id", "detection_counts_k"]].copy()
    # df_check_flow.rename(columns={"detection_counts_k": "detection_counts_k_gandalf"}, inplace=True)
    #
    # counts_balrog_flow = (
    #     df_out_balrog_flow[["true_id", "detection_counts_k"]]
    #     .dropna(subset=["true_id"])
    #     .drop_duplicates("true_id")  # eine Zeile pro true_id
    #     .set_index("true_id")["detection_counts_k"]  # Series: index=true_id, value=count
    # )
    #
    # df_check_flow["detection_counts_k_balrog"] = df_check_flow["true_id"].map(counts_balrog_flow)
    #
    # df_check_flow = df_check_flow.drop_duplicates("true_id")
    #
    # x = df_check_flow["detection_counts_k_balrog"]
    # y = df_check_flow["detection_counts_k_gandalf"]
    #
    # lo = int(min(x.min(), y.min()))
    # hi = int(max(x.max(), y.max()))
    #
    # plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)
    # plt.gca().set_aspect("equal", "box")
    # plt.scatter(df_check_flow["detection_counts_k_balrog"], df_check_flow["detection_counts_k_gandalf"])
    # plt.title("Compare Selection Counts w/o mag cut w/ mcal cuts: k vs. k'")
    # plt.xlim(0, hi)
    # plt.ylim(0, hi)
    # plt.xlabel("selection counts balrog: k")
    # plt.ylabel("selection counts gandalf: k'")
    # plt.savefig(f"{flow_cfg['PATH_PLOTS']}/k_vs_kdash_w_mag_cuts_w_mcal_cuts.png", dpi=150, bbox_inches='tight')
    # plt.clf()
    # plt.close()
    #
    # plt.scatter(df_check_flow["detection_counts_k_balrog"], (df_check_flow["detection_counts_k_gandalf"] - df_check_flow["detection_counts_k_balrog"]) / np.sqrt(df_check_flow["detection_counts_k_balrog"]), alpha=0.3)
    # plt.title("Compare Selection Counts w/o mag cut w/ mcal cuts: k vs. k'")
    # plt.ylabel("(k'-k)/sqrt(k)")
    # plt.xlabel("k")
    # plt.savefig(f"{flow_cfg['PATH_PLOTS']}/compare_selection_counts.png", dpi=150, bbox_inches='tight')
    # plt.clf()
    # plt.close()
    #
    # lst_ranges = [
    #     [-1.5, 4],  # mag r-i
    #     [-4, 1.5],  # mag i-z
    #     [15, 26],  # mag r
    #     [18, 23.5],  # mag i
    #     [15, 26],  # mag z
    #     None,  # mag err r
    #     None,  # mag err i
    #     None,  # mag err z
    #     [-1.2, 1.2],  # e_1
    #     [-1.2, 1.2],  # e_2
    #     [10, 1000],  # snr
    #     [0.5, 50],  # size ratio
    #     [0, 10]  # T
    # ]
    # lst_binwidths = [
    #     0.25,  # mag r-i
    #     0.25,  # mag i-z
    #     0.25,  # mag r
    #     0.25,  # mag i
    #     0.25,  # mag z
    #     0.25,  # mag err r
    #     0.25,  # mag err i
    #     0.25,  # mag err z
    #     0.1,  # e_1
    #     0.1,  # e_2
    #     50,  # snr
    #     1.5,  # size ratio
    #     0.5,  # T
    # ]
    # os.makedirs(flow_cfg["PATH_PLOTS"], exist_ok=True)

    # plot_balrog_histogram_with_error_and_detection(
    #     df_gandalf=df_gandalf,
    #     df_balrog=df_balrog,
    #     columns=flow_cfg["HIST_PLOT_COLS"],
    #     labels=flow_cfg["HIST_PLOT_LABELS"],
    #     ranges=lst_ranges,
    #     binwidths=lst_binwidths,
    #     title=r"MCAL selected galaxies gaNdalF vs. Balrog",
    #     show_plot=flow_cfg["SHOW_PLOT"],
    #     save_plot=flow_cfg["SAVE_PLOT"],
    #     save_name=f"{flow_cfg[f'PATH_PLOTS']}/selection_hist_plot_with_selection_cf.pdf",
    #     detection_name="selected",
    # )

    # df_non_detected = df_gandalf[df_gandalf["true detected"] == 0]
    # df_only_detected = df_gandalf[df_gandalf["true detected"] == 1]
    #
    # plot_balrog_histogram_with_error_and_detection(
    #     df_gandalf=df_non_detected,
    #     df_balrog=df_balrog,
    #     columns=flow_cfg["HIST_PLOT_COLS"],
    #     labels=flow_cfg["HIST_PLOT_LABELS"],
    #     ranges=lst_ranges,
    #     binwidths=lst_binwidths,
    #     title=r"gaNdalF vs. Balrog: non detected",
    #     show_plot=flow_cfg["SHOW_PLOT"],
    #     save_plot=flow_cfg["SAVE_PLOT"],
    #     save_name=f"{flow_cfg[f'PATH_PLOTS']}/hist_plot_non_detected.pdf"
    # )
    #
    # plot_balrog_histogram_with_error_and_detection(
    #     df_gandalf=df_only_detected,
    #     df_balrog=df_balrog,
    #     columns=flow_cfg["HIST_PLOT_COLS"],
    #     labels=flow_cfg["HIST_PLOT_LABELS"],
    #     ranges=lst_ranges,
    #     binwidths=lst_binwidths,
    #     title=r"gaNdalF vs. Balrog: only detected",
    #     show_plot=flow_cfg["SHOW_PLOT"],
    #     save_plot=flow_cfg["SAVE_PLOT"],
    #     save_name=f"{flow_cfg[f'PATH_PLOTS']}/hist_plot_only_detected.pdf"
    # )

    # lst_galaxy_properties = [
    #     "BDF_MAG_DERED_CALIB_R",
    #     "BDF_MAG_DERED_CALIB_I",
    #     "BDF_MAG_DERED_CALIB_Z",
    #     "BDF_MAG_ERR_DERED_CALIB_R",
    #     "BDF_MAG_ERR_DERED_CALIB_I",
    #     "BDF_MAG_ERR_DERED_CALIB_Z",
    #     'Color BDF MAG R-I',
    #     'Color BDF MAG I-Z',
    #     "BDF_T",
    #     "BDF_G"
    # ]
    # lst_galaxy_probs_labels = [
    #     "true mag r",
    #     "true mag i",
    #     "true mag z",
    #     "true mag err r",
    #     "true mag err i",
    #     "true mag err z",
    #     'Color BDF MAG R-I',
    #     'Color BDF MAG I-Z',
    #     "BDF_T",
    #     "BDF_G"
    #     ]
    #
    # plot_balrog_histogram_with_error_and_detection(
    #     df_gandalf=df_non_detected,
    #     df_balrog=df_balrog,
    #     columns=lst_galaxy_properties,
    #     labels=lst_galaxy_probs_labels,
    #     ranges=[None for _ in range(len(lst_galaxy_properties))],
    #     binwidths=[None for _ in range(len(lst_galaxy_properties))],
    #     title=r"gaNdalF vs. Balrog: non detected input",
    #     show_plot=flow_cfg["SHOW_PLOT"],
    #     save_plot=flow_cfg["SAVE_PLOT"],
    #     save_name=f"{flow_cfg[f'PATH_PLOTS']}/hist_plot_non_detected_input_galaxy_properties.pdf"
    # )
    #
    # plot_balrog_histogram_with_error_and_detection(
    #     df_gandalf=df_gandalf,
    #     df_balrog=df_balrog,
    #     columns=lst_galaxy_properties,
    #     labels=lst_galaxy_probs_labels,
    #     ranges=[None for _ in range(len(lst_galaxy_properties))],
    #     binwidths=[None for _ in range(len(lst_galaxy_properties))],
    #     # binwidths=[
    #     #     0.25,  # mag r
    #     #     0.25,  # mag i
    #     #     0.25,  # mag z
    #     #     0.25,  # mag err r
    #     #     0.25,  # mag err i
    #     #     0.25,  # mag err z
    #     #     0.25,  # color r-i
    #     #     0.25,  # color i-z
    #     #     0.25,  # bdft
    #     #     0.25,  # bdfg
    #     #     0.25,  # fwhm r
    #     #     0.25,  # fwhm i
    #     #     0.25,  # fwhm z
    #     #     0.25,  # airmass r
    #     #     0.25,  # airmass i
    #     #     0.25,  # airmass z
    #     #     0.25,  # maglim r
    #     #     0.25,  # maglim i
    #     #     0.25,  # maglim z
    #     #     0.25,  # ebv
    #     # ],
    #     title=r"gaNdalF vs. Balrog: detected and non detected input",
    #     show_plot=flow_cfg["SHOW_PLOT"],
    #     save_plot=flow_cfg["SAVE_PLOT"],
    #     save_name=f"{flow_cfg[f'PATH_PLOTS']}/hist_plot_all_input_galaxy_properties.pdf"
    # )
    #
    # lst_observing_conditions = [
    #     "FWHM_WMEAN_R",
    #     "FWHM_WMEAN_I",
    #     "FWHM_WMEAN_Z",
    #     "AIRMASS_WMEAN_R",
    #     "AIRMASS_WMEAN_I",
    #     "AIRMASS_WMEAN_Z",
    #     "MAGLIM_R",
    #     "MAGLIM_I",
    #     "MAGLIM_Z",
    #     "EBV_SFD98"
    # ]
    #
    # lst_observing_cond_labels = [
    #     "FWHM R",
    #     "FWHM I",
    #     "FWHM Z",
    #     "AIRMASS R",
    #     "AIRMASS I",
    #     "AIRMASS Z",
    #     "MAGLIM_R",
    #     "MAGLIM_I",
    #     "MAGLIM_Z",
    #     "EBV_SFD98"
    # ]
    #
    # plot_balrog_histogram_with_error_and_detection(
    #     df_gandalf=df_non_detected,
    #     df_balrog=df_balrog,
    #     columns=lst_observing_conditions,
    #     labels=lst_observing_cond_labels,
    #     ranges=[None for _ in range(len(lst_observing_conditions))],
    #     binwidths=[None for _ in range(len(lst_observing_conditions))],
    #     title=r"gaNdalF vs. Balrog: non detected input",
    #     show_plot=flow_cfg["SHOW_PLOT"],
    #     save_plot=flow_cfg["SAVE_PLOT"],
    #     save_name=f"{flow_cfg[f'PATH_PLOTS']}/hist_plot_non_detected_input_obs_cond.pdf"
    # )
    #
    # plot_balrog_histogram_with_error_and_detection(
    #     df_gandalf=df_gandalf,
    #     df_balrog=df_balrog,
    #     columns=lst_observing_conditions,
    #     labels=lst_observing_cond_labels,
    #     ranges=[None for _ in range(len(lst_observing_conditions))],
    #     binwidths=[None for _ in range(len(lst_observing_conditions))],
    #     title=r"gaNdalF vs. Balrog: detected and non detected input",
    #     show_plot=flow_cfg["SHOW_PLOT"],
    #     save_plot=flow_cfg["SAVE_PLOT"],
    #     save_name=f"{flow_cfg[f'PATH_PLOTS']}/hist_plot_all_input_obs_cond.pdf"
    # )

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
        columns=flow_cfg["HIST_PLOT_COLS"],
        labels=flow_cfg["HIST_PLOT_LABELS"],
        ranges=lst_ranges,
        binwidths=lst_binwidths,
        title=r"gaNdalF vs. Balrog: Property Distribution Comparison",
        show_plot=flow_cfg["SHOW_PLOT"],
        save_plot=flow_cfg["SAVE_PLOT"],
        save_name=f"{flow_cfg[f'PATH_PLOTS']}/hist_plot_three.pdf",
        detection_name="mcal_galaxy"
    )

    plot_flow(
        cfg=flow_cfg,
        logger=logger,
        df_gandalf=df_gandalf_selected,
        df_balrog=df_balrog_selected,
        prefix="_cut"
    )

    exit()

    df_merged = df_gandalf_selected.merge(df_gandalf_injection_counts, how="left", on="bal_id")

    save_catalogs(
        cfg=classifier_cfg,
        df=df_merged,
        filename=f"{classifier_cfg['RUN_DATE']}_gandalf_Emulated_Classified_{classifier_cfg['NUMBER_SAMPLES']}_{len(df_merged)}.h5")


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    sys.path.append(os.path.dirname(__file__))
    classifier_cfg, flow_cfg = load_config_and_parser(system_path=os.path.abspath(sys.path[-1]))

    log_lvl = logging.INFO
    if classifier_cfg["LOGGING_LEVEL"] == "DEBUG":
        log_lvl = logging.DEBUG
    elif classifier_cfg["LOGGING_LEVEL"] == "ERROR":
        log_lvl = logging.ERROR

    # true_balrog_dataset = "/Volumes/elmichelangelo_external_ssd_1/Data/y3_balrog2_v1.2_merged_select2_bstarcut_matchflag1.5asec_snr_SR_corrected_uppersizecuts.h5"
    # store = pd.HDFStore(true_balrog_dataset, mode="r")
    # print(store.keys())  # zeigt alle verfügbaren Keys, z.B. ['/data', '/table', ...]
    # store.close()
    # df = pd.read_hdf(true_balrog_dataset, key="/df")
    # print(df.head())
    #
    # for band in ["r", "i", "z"]:
    #     df[f"unsheared/mag_{band}"] = flux2mag(df[f"unsheared/flux_{band}"])
    #     df[f"unsheared/mag_err_{band}"] = np.log10(fluxerr2magerr(df[f"unsheared/flux_{band}"], df[f"unsheared/flux_err_{band}"]))
    #
    # df["r-i"] = df[f"unsheared/mag_r"] - df[f"unsheared/mag_i"]
    # df["i-z"] = df[f"unsheared/mag_i"] - df[f"unsheared/mag_z"]
    #
    # df["detected"] = np.ones(len(df))
    # df["sampled detected"] = np.ones(len(df))
    #
    # lst_ranges = [
    #     [-1.5, 4],  # mag r-i
    #     [-4, 1.5],  # mag i-z
    #     [15, 26],  # mag r
    #     [18, 23.5],  # mag i
    #     [15, 26],  # mag z
    #     None,  # mag err r
    #     None,  # mag err i
    #     None,  # mag err z
    #     [-1.2, 1.2],  # e_1
    #     [-1.2, 1.2],  # e_2
    #     [10, 1000],  # snr
    #     [0.5, 30],  # size ratio
    #     [0, 10]  # T
    # ]
    # lst_binwidths = [
    #     0.25,  # mag r-i
    #     0.25,  # mag i-z
    #     0.25,  # mag r
    #     0.25,  # mag i
    #     0.25,  # mag z
    #     0.25,  # mag err r
    #     0.25,  # mag err i
    #     0.25,  # mag err z
    #     0.1,  # e_1
    #     0.1,  # e_2
    #     50,  # snr
    #     1.5,  # size ratio
    #     0.5,  # T
    # ]
    #
    # plot_balrog_histogram_with_error_and_detection(
    #     df_gandalf=df,
    #     df_balrog=df,
    #     columns=flow_cfg["HIST_PLOT_COLS"],
    #     labels=flow_cfg["HIST_PLOT_LABELS"],
    #     ranges=lst_ranges,
    #     binwidths=lst_binwidths,
    #     title=r"Balrog: Property Distribution",
    #     show_plot=flow_cfg["SHOW_PLOT"],
    #     save_plot=flow_cfg["SAVE_PLOT"],
    #     save_name=f"{flow_cfg[f'PATH_PLOTS']}/hist_plot_true_balrog.pdf"
    # )
    #
    # exit()
    run_logger = LoggerHandler(
        logger_dict={"logger_name": "train flow logger",
                     "info_logger": classifier_cfg['INFO_LOGGER'],
                     "error_logger": classifier_cfg['ERROR_LOGGER'],
                     "debug_logger": classifier_cfg['DEBUG_LOGGER'],
                     "stream_logger": classifier_cfg['STREAM_LOGGER'],
                     "stream_logging_level": log_lvl},
        log_folder_path=f"{classifier_cfg['PATH_OUTPUT']}/"
    )

    main(classifier_cfg=classifier_cfg, flow_cfg=flow_cfg, logger=run_logger)
