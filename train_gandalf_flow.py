import copy
import random
from datetime import datetime
from Handler import get_os, LoggerHandler
from gandalf_flow import gaNdalFFlow
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import sys
import yaml
import os
import logging
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.stopper import TrialPlateauStopper
from functools import partial
from ray.air import session
import csv
from filelock import FileLock
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

plt.rcParams["figure.figsize"] = (16, 9)

def _read_trained_combinations_no_lock(csv_file):
    runs = []
    if not os.path.exists(csv_file):
        return runs
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            runs.append(row)
    return runs

def _write_all_runs_no_lock(csv_file, runs):
    header = ["trial number", "batch size", "learning rate", "number hidden", "number blocks", "status"]
    with open(csv_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in runs:
            writer.writerow(row)

def add_or_update_run(csv_file, bs, lr, nh, nb, status):
    lr_str = str(lr).replace('.', ',')
    combo = (str(bs), lr_str, str(nh), str(nb))
    lock = FileLock(csv_file + ".lock")
    with lock:
        runs = _read_trained_combinations_no_lock(csv_file)
        updated = False
        for row in runs:
            if (row["batch size"], row["learning rate"], row["number hidden"], row["number blocks"]) == combo:
                row["status"] = status
                updated = True
                break
        if not updated:
            trial_number = str(len(runs) + 1)  # Laufende Nummer ab 1
            runs.append({
                "trial number": trial_number,
                "batch size": str(bs),
                "learning rate": lr_str,
                "number hidden": str(nh),
                "number blocks": str(nb),
                "status": status
            })
        _write_all_runs_no_lock(csv_file, runs)

def check_if_run_exists(csv_file, bs, lr, nh, nb):
    lr_str = str(lr).replace('.', ',')
    combo = (str(bs), lr_str, str(nh), str(nb))
    lock = FileLock(csv_file + ".lock")
    with lock:
        runs = _read_trained_combinations_no_lock(csv_file)
        for row in runs:
            if (row["batch size"], row["learning rate"], row["number hidden"], row["number blocks"]) == combo:
                return row["status"]
    return None


def main(
        cfg,
        learning_rate,
        number_hidden,
        number_blocks,
        batch_size,
        logger):
    """"""

    train_flow = gaNdalFFlow(
        cfg=cfg,
        learning_rate=learning_rate,
        number_hidden=number_hidden,
        number_blocks=number_blocks,
        batch_size=batch_size,
        train_flow_logger=logger
    )
    train_flow.run_training()
    val_loss = train_flow.run_training()
    return val_loss

def train_tune(tune_config, base_config):
    config = dict(base_config)  # vollständige Kopie
    config.update(tune_config)  # überschreibe mit Tunables
    csv_file = os.path.join(config["PATH_OUTPUT_CSV"], "trained_params.csv")

    bs = int(config['batch_size'])
    lr = float(config['learning_rate'])
    nh = int(config['number_hidden'])
    nb = int(config['number_blocks'])
    combo = (bs, lr, nh, nb)

    existing_status = check_if_run_exists(csv_file, bs, lr, nh, nb)
    if existing_status == "started":
        print(f"Trial SKIP: {bs}, {lr}, {nh}, {nb} already started!", flush=True)
        session.report({"loss": 1e10, "epoch": 0, "skipped": True})
        return
    if existing_status == "finished":
        print(f"Trial SKIP: {bs}, {lr}, {nh}, {nb} already finished!", flush=True)
        session.report({"loss": 1e10, "epoch": 0, "skipped": True})
        return

    add_or_update_run(csv_file, bs, lr, nh, nb, "started")

    config['PATH_OUTPUT'] = os.path.join(
        config['PATH_OUTPUT_BASE'],
        f"flow_training_{config['RUN_DATE']}",
        f"bs_{config['batch_size']}_lr_{config['learning_rate']}_nh_{config['number_hidden']}_nb_{config['number_blocks']}"
    )
    config['PATH_OUTPUT_CATALOGS'] = os.path.join(
        config['PATH_OUTPUT_CATALOGS_BASE'],
        f"flow_training_{config['RUN_DATE']}",
        f"bs_{config['batch_size']}_lr_{config['learning_rate']}_nh_{config['number_hidden']}_nb_{config['number_blocks']}"
    )

    os.makedirs(config['PATH_OUTPUT'], exist_ok=True)
    os.makedirs(config['PATH_OUTPUT_CATALOGS'], exist_ok=True)

    logger = LoggerHandler(
        logger_dict={"logger_name": "train flow logger",
                     "info_logger": config['INFO_LOGGER'],
                     "error_logger": config['ERROR_LOGGER'],
                     "debug_logger": config['DEBUG_LOGGER'],
                     "stream_logger": config['STREAM_LOGGER'],
                     "stream_logging_level": logging.INFO},
        log_folder_path=f"{config['PATH_OUTPUT']}/"
    )
    train_flow = gaNdalFFlow(
        cfg=config,
        learning_rate=config["learning_rate"],
        number_hidden=config["number_hidden"],
        number_blocks=config["number_blocks"],
        batch_size=config["batch_size"],
        train_flow_logger=logger
    )

    for epoch, validation_loss in train_flow.run_training():
        session.report({"loss": validation_loss, "epoch": epoch+1})

    add_or_update_run(csv_file, bs, lr, nh, nb, "finished")

def load_config_and_parser(system_path):
    if get_os() == "Mac":
        print("load MAC config-file")
        config_file_name = "MAC_train_flow.cfg"
    elif get_os() == "Linux":
        print("load LMU config-file")
        config_file_name = "LMU_train_flow.cfg"
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

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    sys.path.append(os.path.dirname(__file__))
    cfg, path_cfg_file = load_config_and_parser(system_path=os.path.abspath(sys.path[-1]))
    GLOBAL_BASE_CONFIG = copy.deepcopy(cfg)  # <-- diese Variable

    # Pfade merken (damit Ray pro Trial eigene anlegt)
    cfg['PATH_OUTPUT_BASE'] = cfg['PATH_OUTPUT']
    cfg['PATH_OUTPUT_CATALOGS_BASE'] = cfg['PATH_OUTPUT_CATALOGS']

    cfg['PATH_OUTPUT'] = f"{cfg['PATH_OUTPUT']}/flow_training_{cfg['RUN_DATE']}"
    cfg['PATH_OUTPUT_CATALOGS'] = f"{cfg['PATH_OUTPUT_CATALOGS']}/flow_training_{cfg['RUN_DATE']}"

    os.makedirs(cfg['PATH_OUTPUT'], exist_ok=True)
    os.makedirs(cfg['PATH_OUTPUT_CATALOGS'], exist_ok=True)

    log_lvl = logging.INFO
    if cfg["LOGGING_LEVEL"] == "DEBUG":
        log_lvl = logging.DEBUG
    elif cfg["LOGGING_LEVEL"] == "ERROR":
        log_lvl = logging.ERROR

    train_flow_logger = LoggerHandler(
        logger_dict={"logger_name": "train flow logger",
                     "info_logger": cfg['INFO_LOGGER'],
                     "error_logger": cfg['ERROR_LOGGER'],
                     "debug_logger": cfg['DEBUG_LOGGER'],
                     "stream_logger": cfg['STREAM_LOGGER'],
                     "stream_logging_level": log_lvl},
        log_folder_path=f"{cfg['PATH_OUTPUT']}/"
    )

    batch_size = cfg["BATCH_SIZE_FLOW"]
    number_hidden = cfg["NUMBER_HIDDEN"]
    number_blocks = cfg["NUMBER_BLOCKS"]
    learning_rate = cfg["LEARNING_RATE_FLOW"]

    if not isinstance(batch_size, list):
        batch_size = [batch_size, batch_size, batch_size]
    if not isinstance(number_hidden, list):
        number_hidden = [number_hidden]
    if not isinstance(number_blocks, list):
        number_blocks = [number_blocks]
    if not isinstance(learning_rate, list):
        learning_rate = [learning_rate, learning_rate]

    search_space = {
        "batch_size": tune.qloguniform(batch_size[0], batch_size[1], batch_size[2]),
        "learning_rate": tune.loguniform(learning_rate[0], learning_rate[1]),
        "number_hidden": tune.choice(number_hidden),
        "number_blocks": tune.choice(number_blocks),
        "INFO_LOGGER": cfg["INFO_LOGGER"],
        "ERROR_LOGGER": cfg["ERROR_LOGGER"],
        "DEBUG_LOGGER": cfg["DEBUG_LOGGER"],
        "STREAM_LOGGER": cfg["STREAM_LOGGER"],
        "LOGGING_LEVEL": cfg["LOGGING_LEVEL"],
        "PATH_TRANSFORMERS": cfg["PATH_TRANSFORMERS"],
        "FILENAME_STANDARD_SCALER": cfg["FILENAME_STANDARD_SCALER"],
        "PATH_OUTPUT_BASE": cfg["PATH_OUTPUT_BASE"],
        "PATH_OUTPUT_CATALOGS_BASE": cfg["PATH_OUTPUT_CATALOGS_BASE"],
        "RUN_DATE": cfg['RUN_DATE']
    }

    reporter = CLIReporter(
        parameter_columns=["learning_rate", "number_hidden", "number_blocks", "batch_size"],
        metric_columns=["loss"]
    )

    resources = {"cpu": cfg["RESOURCE_CPU"], "gpu": cfg["RESOURCE_GPU"]}

    optuna_search = OptunaSearch(
        metric="loss",
        mode="min"
    )

    asha = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=cfg["EPOCHS_FLOW"],
        grace_period=10
    )

    # stopper = TrialPlateauStopper(
    #     metric="loss",
    #     mode="min",
    #     std=0.0,
    #     num_results=10,
    #     grace_period=15,
    #     metric_threshold=1e-4
    # )

    analysis = tune.run(
        partial(train_tune, base_config=GLOBAL_BASE_CONFIG),
        config=search_space,
        search_alg=optuna_search,
        scheduler=asha,
        num_samples=60,
        max_concurrent_trials=3,
        resources_per_trial=resources,
        progress_reporter=reporter,
        storage_path=f"{cfg['PATH_OUTPUT_BASE']}/{cfg['RUN_DATE']}_ray_results/",
        name="gandalf_tune_optuna"
    )

    # analysis = tune.run(
    #     partial(train_tune, base_config=GLOBAL_BASE_CONFIG),
    #     config=search_space,
    #     resources_per_trial=resources,
    #     progress_reporter=reporter,
    #     stop=stopper,
    #     max_concurrent_trials=3,
    #     metric="loss",
    #     mode="min",
    #     storage_path=f"{cfg['PATH_OUTPUT_BASE']}/{cfg['RUN_DATE']}_ray_results/",
    #     name="gandalf_tune",
    # )

    print("Best config found:")
    print(analysis.get_best_config(metric="loss", mode="min"))

    # cfg['PATH_OUTPUT'] = f"{cfg['PATH_OUTPUT']}/flow_training_{cfg['RUN_DATE']}"
    # if not os.path.exists(cfg['PATH_OUTPUT_CATALOGS']):
    #     os.mkdir(cfg['PATH_OUTPUT_CATALOGS'])
    # cfg['PATH_OUTPUT_CATALOGS'] = f"{cfg['PATH_OUTPUT_CATALOGS']}/flow_training_{cfg['RUN_DATE']}"
    # if not os.path.exists(cfg['PATH_OUTPUT']):
    #     os.mkdir(cfg['PATH_OUTPUT'])
    # if not os.path.exists(cfg['PATH_OUTPUT_CATALOGS']):
    #     os.mkdir(cfg['PATH_OUTPUT_CATALOGS'])
    #
    # log_lvl = logging.INFO
    # if cfg["LOGGING_LEVEL"] == "DEBUG":
    #     log_lvl = logging.DEBUG
    # elif cfg["LOGGING_LEVEL"] == "ERROR":
    #     log_lvl = logging.ERROR
    # train_flow_logger = LoggerHandler(
    #     logger_dict={"logger_name": "train flow logger",
    #                  "info_logger": cfg['INFO_LOGGER'],
    #                  "error_logger": cfg['ERROR_LOGGER'],
    #                  "debug_logger": cfg['DEBUG_LOGGER'],
    #                  "stream_logger": cfg['STREAM_LOGGER'],
    #                  "stream_logging_level": log_lvl},
    #     log_folder_path=f"{cfg['PATH_OUTPUT']}/"
    # )
    #
    # batch_size = cfg["BATCH_SIZE_FLOW"]
    # number_hidden = cfg["NUMBER_HIDDEN"]
    # number_blocks = cfg["NUMBER_BLOCKS"]
    # learning_rate = cfg["LEARNING_RATE_FLOW"]
    #
    # if not isinstance(batch_size, list):
    #     batch_size = [batch_size]
    # if not isinstance(number_hidden, list):
    #     number_hidden = [number_hidden]
    # if not isinstance(number_blocks, list):
    #     number_blocks = [number_blocks]
    # if not isinstance(learning_rate, list):
    #     learning_rate = [learning_rate]
    #
    # for lr in learning_rate:
    #     for nh in number_hidden:
    #         for nb in number_blocks:
    #             for bs in batch_size:
    #                 train_flow_logger.log_info_stream(f"Batch size {bs}")
    #                 train_flow_logger.log_info_stream(f"Number blocks {nb}")
    #                 train_flow_logger.log_info_stream(f"Number hidden {nh}")
    #                 train_flow_logger.log_info_stream(f"Learning rate {lr}")
    #                 main(
    #                     cfg=cfg,
    #                     learning_rate=lr,
    #                     number_hidden=nh,
    #                     number_blocks=nb,
    #                     batch_size=bs,
    #                     logger=train_flow_logger
    #                 )
