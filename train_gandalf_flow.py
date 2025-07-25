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

plt.rcParams["figure.figsize"] = (16, 9)


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
    # final_loss = train_flow.run_training()
    # # Melde Ergebnis an Ray Tune
    # tune.report(loss=final_loss)

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
        batch_size = [batch_size]
    if not isinstance(number_hidden, list):
        number_hidden = [number_hidden]
    if not isinstance(number_blocks, list):
        number_blocks = [number_blocks]
    if not isinstance(learning_rate, list):
        learning_rate = [learning_rate]

    # ------ RAY TUNE BLOCK: MODERNES TUNING ------
    # Suchraum für Ray Tune
    search_space = {
        "learning_rate": tune.grid_search(learning_rate),
        "number_hidden": tune.grid_search(number_hidden),
        "number_blocks": tune.grid_search(number_blocks),
        "batch_size": tune.grid_search(batch_size),
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

    stopper = TrialPlateauStopper(
        metric="loss",
        mode="min",
        std=0.0,
        num_results=10,
        grace_period=15,
        metric_threshold=1e-4
    )

    # Ray Tune-Run!
    analysis = tune.run(
        partial(train_tune, base_config=GLOBAL_BASE_CONFIG),
        config=search_space,
        resources_per_trial=resources,
        progress_reporter=reporter,
        stop=stopper,
        max_concurrent_trials=3,
        metric="loss",
        mode="min",
        storage_path=f"{cfg['PATH_OUTPUT_BASE']}/{cfg['RUN_DATE']}_ray_results/",
        name="gandalf_tune",
    )

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
