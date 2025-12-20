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

from gandalf_calibration_model.calibration_benchmark import (
    collect_classifier_outputs, run_calibration_suite, run_covariate_sweep,
    IdentityCalibrator, BetaCalibrator, IsotonicCalibrator,
    MagAwarePlattCalibrator, MagBinnedWrapper
)

from gandalf_calibration_model.calibration_diagnostics import (
    fit_and_predict_on_test, per_bin_metrics,
    plot_reliability, plot_reliability_slices, add_delta_to_uncal, summarize_delta
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
        config_file_name = "MAC_get_calibration_model.cfg"
    elif get_os() == "Linux":
        print("load LMU config-file")
        config_file_name = "LMU_get_calibration_model.cfg"
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
    with open(f"{system_path}/../conf/{args.config_filename}", 'r') as fp:
        print(f"open {f'{system_path}/../conf/{args.config_filename}'}")
        config = yaml.safe_load(fp)
    config['RUN_DATE'] = now.strftime('%Y-%m-%d_%H-%M')

    return config


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
    # Init gaNdalF model
    model = gaNdalF(logger, cfg=cfg)

    # Init Classifier model from gaNdalF
    model.init_classifier()

    # Scale input data for gaNdalF run
    logger.log_info_stream(f"Scale classifier data")

    # calibration test ############################################
    mag_col = model.cfg.get("MAG_COL", "BDF_MAG_DERED_CALIB_I")
    cols_unscaled = list(dict.fromkeys([mag_col] + model.cfg["INPUT_COLS"]))
    df_unscaled = model.classifier_data[cols_unscaled].copy()
    ###############################################################

    model.classifier_galaxies.scale_data(
        cfg_key_cols_interest="COLUMNS_OF_INTEREST_CF",
        cfg_key_filename_scaler="FILENAME_STANDARD_SCALER_CF",
    )

    # calibration test ############################################
    outs = collect_classifier_outputs(model, cov_cols=model.cfg["INPUT_COLS"], batch_size=131072, mag_col=mag_col)

    mag_unscaled = df_unscaled.loc[:, mag_col]
    if isinstance(mag_unscaled, pd.DataFrame):  # happens if mag_col is duplicated
        mag_unscaled = mag_unscaled.iloc[:, 0]

    outs["mag_unscaled"] = mag_unscaled.to_numpy(dtype=float)
    outs["cov_unscaled"] = df_unscaled[model.cfg["INPUT_COLS"]].to_numpy(float)

    df_suite = run_calibration_suite(
        y=outs["y"],
        p_raw=outs["p_raw"],
        logits=outs["logits"],
        mag=outs["mag"],
        seed=41,
        test_size=0.5,
        max_fit=500_000
    )

    df_cov = run_covariate_sweep(
        y=outs["y"],
        p_raw=outs["p_raw"],
        logits=outs["logits"],
        cov=outs["cov"],
        cov_cols=outs["cov_cols"],
        seed=41,
        test_size=0.5,
        max_fit=500_000
    )

    calibrator_factories = {
        "uncalibrated": lambda: IdentityCalibrator(),
        "beta": lambda: BetaCalibrator(),
        "isotonic": lambda: IsotonicCalibrator(),
        "magaware_platt": lambda: MagAwarePlattCalibrator(),
        "magbinned_beta": lambda: MagBinnedWrapper(lambda: BetaCalibrator(), n_mag_bins=10, name="magbinned_beta"),
        "magbinned_isotonic": lambda: MagBinnedWrapper(lambda: IsotonicCalibrator(), n_mag_bins=10, name="magbinned_isotonic"),
    }

    df_test, fitted = fit_and_predict_on_test(
        y=outs["y"], p_raw=outs["p_raw"], logits=outs["logits"], mag=outs["mag"],
        cov=outs.get("cov", None), cov_cols=outs.get("cov_cols", None),
        seed=41, test_size=0.5, max_fit=500_000,
        calibrator_factories=calibrator_factories
    )

    df_test["mag_unscaled"] = outs["mag_unscaled"][df_test.index]

    methods = list(calibrator_factories.keys())

    # --- (A) Reliability global + Mag-Slices ---
    os.makedirs(cfg["PATH_PLOTS"], exist_ok=True)
    few_methods = ["isotonic", "magbinned_isotonic"]
    plot_reliability(df_test, few_methods, title="Reliability (global)", n_bins=20,
                     outpath=f"{cfg["PATH_PLOTS"]}/reliability_global.pdf")

    # --- (B) Per-bin tables: magnitude & observing conditions ---
    # Wichtig: var-Namen müssen Spalten in df_test sein (cov_cols werden hinzugefügt).
    vars_to_check = [
        "mag",  # das ist outs["mag"] (standardmäßig BDF_MAG_DERED_CALIB_I wenn du mag_col so setzt)
        "FWHM_WMEAN_I",
        "AIRMASS_WMEAN_I",
        "MAGLIM_I",
        "EBV_SFD98",
    ]

    all_tables = []
    for v in vars_to_check:
        t = per_bin_metrics(df_test, var=v, methods=methods, n_bins=10, binning="quantile", ece_bins=20)
        all_tables.append(t)

    df_bins = pd.concat(all_tables, ignore_index=True)
    df_bins.to_csv("calibration_per_bin_metrics.csv", index=False)

    df_bins_delta = add_delta_to_uncal(df_bins, baseline="uncalibrated")
    df_bins_delta.to_csv("calibration_per_bin_metrics_delta.csv", index=False)

    df_summary = summarize_delta(df_bins_delta)
    print("\n=== Weighted mean deltas vs uncalibrated (per var) ===")
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))
    df_summary.to_csv("calibration_delta_summary.csv", index=False)

    # Quick view: welche Methode verbessert Brier am meisten (negativ = besser) pro var/bin
    best_delta = (
        df_bins_delta.sort_values(["var", "bin", "delta_brier"])
        .groupby(["var", "bin"], as_index=False)
        .first()[["var", "bin", "bin_lo", "bin_hi", "n", "method", "delta_brier", "delta_ece", "delta_logloss"]]
    )
    print("\n=== Best improvement vs uncalibrated per bin (delta_brier) ===")
    print(best_delta.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))

    # kurze Konsolen-Ansicht: best method by Brier in each bin for each var
    best = (
        df_bins.sort_values(["var", "bin", "brier"])
        .groupby(["var", "bin"], as_index=False)
        .first()[["var", "bin", "bin_lo", "bin_hi", "n", "method", "brier", "ece", "logloss"]]
    )
    print("\n=== Best-by-Brier per bin (quick view) ===")
    print(best.to_string(index=False, float_format=lambda x: f"{x:0.6f}"))

    ###############################################################

    sys.exit()


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    sys.path.append(os.path.dirname(__file__))
    config = load_config_and_parser(system_path=os.path.abspath(sys.path[-1]))

    log_lvl = logging.INFO
    if config["LOGGING_LEVEL"] == "DEBUG":
        log_lvl = logging.DEBUG

    # init_output_paths(cfg=config)
    run_logger = LoggerHandler(
        logger_dict={"logger_name": "train flow logger",
                     "info_logger": config['INFO_LOGGER'],
                     "error_logger": config['ERROR_LOGGER'],
                     "debug_logger": config['DEBUG_LOGGER'],
                     "stream_logger": config['STREAM_LOGGER'],
                     "stream_logging_level": log_lvl},
        log_folder_path=f"{config['PATH_LOGS']}/"
    )

    main(cfg=config, logger=run_logger)