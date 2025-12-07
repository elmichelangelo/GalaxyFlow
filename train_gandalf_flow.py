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
import hashlib, json, shutil
try:
    from ray.train import Checkpoint, RunConfig, CheckpointConfig
except (ImportError, AttributeError):
    try:
        from ray.air import Checkpoint, RunConfig, CheckpointConfig
    except (ImportError, AttributeError):
        Checkpoint = None

plt.rcParams["figure.figsize"] = (16, 9)

def main(
        cfg,
        learning_rate,
        number_hidden,
        number_blocks,
        number_layers,
        batch_size):
    """"""

    train_flow = gaNdalFFlow(
        cfg=cfg,
        learning_rate=learning_rate,
        number_hidden=number_hidden,
        number_blocks=number_blocks,
        number_layers=number_layers,
        batch_size=batch_size
    )
    last_val = train_flow.run_training()
    return last_val

def train_tune(tune_config, base_config):
    config = dict(base_config)
    config.update(tune_config)

    ray_trial_dir = session.get_trial_dir()

    config['PATH_OUTPUT'] = ray_trial_dir
    config['PATH_OUTPUT_CKPT'] = ray_trial_dir

    config["TRIAL_ID"] = hparam_hash(
        config["batch_size"], config["learning_rate"],
        config["number_hidden"], config["number_blocks"], config["number_layers"],
        # arch=config.get("FLOW_ARCH", "rqs"),
        # bins=config.get("RQS_NUM_BINS"), tb=config.get("RQS_TAIL_BOUND"),
        # conv=config.get("RQS_USE_INV1X1")
    )

    train_flow = gaNdalFFlow(
        cfg=config,
        learning_rate=config["learning_rate"],
        number_hidden=config["number_hidden"],
        number_blocks=config["number_blocks"],
        number_layers=config["number_layers"],
        batch_size=config["batch_size"]
    )

    ckpt_src = os.path.join(config['PATH_OUTPUT'], "last.ckpt.pt")
    ray_ckpt_dir = os.path.join(config['PATH_OUTPUT_CKPT'], "ray_ckpt")

    best_so_far = [float("inf")]

    def reporter(epoch, train_loss, val_loss):
        payload = {
            "epoch": int(epoch + 1),
            "loss": float(val_loss),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
        }
        improved = val_loss < best_so_far[0] - 1e-6
        if improved and os.path.exists(ckpt_src) and Checkpoint is not None:
            best_so_far[0] = val_loss
            os.makedirs(ray_ckpt_dir, exist_ok=True)
            shutil.copy2(ckpt_src, os.path.join(ray_ckpt_dir, "last.ckpt.pt"))
            session.report(payload, checkpoint=Checkpoint.from_directory(ray_ckpt_dir))
        else:
            session.report(payload)

    train_flow.run_training(on_epoch_end=reporter)

    final_payload = {
        "epoch": int(train_flow.current_epoch + 1),
        "loss": float(train_flow.lst_valid_loss_per_epoch[-1]) if train_flow.lst_valid_loss_per_epoch else None,
        "train_loss": float(train_flow.lst_train_loss_per_epoch[-1]) if train_flow.lst_train_loss_per_epoch else None,
        "val_loss": float(train_flow.lst_valid_loss_per_epoch[-1]) if train_flow.lst_valid_loss_per_epoch else None,
    }
    session.report(final_payload)

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

def hparam_hash(bs, lr, nh, nb, nl, arch="rqs", bins=None, tb=None, conv=None):
    payload = {
        # "arch": arch,
        "bs": int(bs),
        "lr": float(lr),
        "nh": int(nh),
        "nb": int(nb),
        "nl": int(nl)
    }
    # if arch == "rqs":
    #     payload.update({"bins": bins, "tb": tb, "conv": int(bool(conv))})
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:10]

def my_trial_name_creator(trial):
    cfg = trial.config
    return "trial_" + hparam_hash(
        cfg["batch_size"], cfg["learning_rate"],
        cfg["number_hidden"], cfg["number_blocks"], cfg["number_layers"],
        #arch=cfg.get("FLOW_ARCH","rqs"),
        #bins=cfg.get("RQS_NUM_BINS"),
        #tb=cfg.get("RQS_TAIL_BOUND"),
        #conv=cfg.get("RQS_USE_INV1X1")
    )

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    sys.path.append(os.path.dirname(__file__))
    # sys.path.append("/dss/dsshome1/04/di97tac/development/GalaxyFlow/")
    path = os.path.abspath(sys.path[-1])
    cfg, path_cfg_file = load_config_and_parser(system_path=path)

    cfg['PATH_OUTPUT_BASE'] = cfg['PATH_OUTPUT']
    cfg['PATH_OUTPUT_CATALOGS_BASE'] = cfg['PATH_OUTPUT_CATALOGS']
    os.makedirs(cfg['PATH_OUTPUT_BASE'], exist_ok=True)
    os.makedirs(cfg['PATH_OUTPUT_CATALOGS_BASE'], exist_ok=True)

    GLOBAL_BASE_CONFIG = copy.deepcopy(cfg)

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

    if cfg['HPARAM_SEARCH'] is False:
        batch_size = cfg["BATCH_SIZE_SINGLE_FLOW"]
        number_hidden = cfg["NUMBER_SINGLE_HIDDEN"]
        number_blocks = cfg["NUMBER_SINGLE_BLOCKS"]
        number_layers = cfg["NUMBER_SINGLE_LAYERS"]
        learning_rate = cfg["LEARNING_RATE_SINGLE_FLOW"]

        main(
            cfg=cfg,
            learning_rate=learning_rate,
            number_hidden=number_hidden,
            number_blocks=number_blocks,
            number_layers=number_layers,
            batch_size=batch_size,
        )
    else:
        batch_size = cfg["BATCH_SIZE_FLOW"]
        number_hidden = cfg["NUMBER_HIDDEN"]
        number_blocks = cfg["NUMBER_BLOCKS"]
        number_layers = cfg["NUMBER_LAYERS"]
        learning_rate = cfg["LEARNING_RATE_FLOW"]

        if not isinstance(batch_size, list):
            batch_size = [batch_size]
        if not isinstance(number_hidden, list):
            number_hidden = [number_hidden]
        if not isinstance(number_blocks, list):
            number_blocks = [number_blocks]
        if not isinstance(number_layers, list):
            number_layers = [number_layers]
        if not isinstance(learning_rate, list):
            learning_rate = [learning_rate, learning_rate]

        search_space = {
            "batch_size": tune.choice(batch_size),
            "learning_rate": tune.loguniform(learning_rate[0], learning_rate[1]),  # tune.loguniform(learning_rate[0], learning_rate[1]),
            "number_hidden": tune.choice(number_hidden),
            "number_blocks": tune.choice(number_blocks),
            "number_layers": tune.choice(number_layers),

            # bestehend …
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
            parameter_columns=["learning_rate", "number_hidden", "number_blocks", "number_layers", "batch_size"],
            metric_columns=["loss", "train_loss", "val_loss", "epoch"]
        )

        resources = {"cpu": cfg["RESOURCE_CPU"], "gpu": cfg["RESOURCE_GPU"]}

        optuna_search = OptunaSearch(
            metric="loss",
            mode="min"
        )

        asha = ASHAScheduler(
            metric="loss",
            time_attr="epoch",
            mode="min",
            max_t=cfg["EPOCHS_FLOW"],
            grace_period=1,
            reduction_factor=4,
            stop_last_trials=False
        )

        plateau = TrialPlateauStopper(
            metric="loss",
            mode="min",
            num_results=5,
            grace_period=10,
            std=1e-4
        )
        ray.init(local_mode=cfg["DEBUG_MODE"], num_cpus=cfg["RESOURCE_CPU"], include_dashboard=cfg["INCLUDE_DASHBOARD"])
        analysis = tune.run(
            partial(train_tune, base_config=GLOBAL_BASE_CONFIG),
            config=search_space,
            search_alg=optuna_search,
            scheduler=asha,
            stop=None,  # plateau,
            num_samples=cfg['OPTUNA_RUNS'],
            max_concurrent_trials=cfg['MAX_TRAILS'],
            resources_per_trial=resources,
            progress_reporter=reporter,
            storage_path=cfg['PATH_OUTPUT_BASE'],
            resume="AUTO",
            name=f"study_{cfg['RUN_ID']}",
            trial_name_creator=my_trial_name_creator,
            trial_dirname_creator=my_trial_name_creator,
            keep_checkpoints_num=1,
            checkpoint_score_attr="min-loss",
        )

        train_flow_logger.log_info_stream("Best config found:")
        train_flow_logger.log_info_stream(analysis.get_best_config(metric="loss", mode="min"))
