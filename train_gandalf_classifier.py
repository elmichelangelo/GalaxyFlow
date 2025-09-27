import copy
from datetime import datetime
from Handler import get_os
from Handler.logger import LoggerHandler
from gandalf_classifier.gaNdalF_deep_classifier import gaNdalFClassifier
import argparse
import matplotlib.pyplot as plt
import sys
import yaml
import os
import warnings
from torch import nn
from torch.utils.data import DataLoader
from gandalf_galaxie_dataset import DESGalaxies
import random
import logging
import pandas as pd
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.stopper import TrialPlateauStopper
from functools import partial
from ray.air import session
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

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.append(os.path.dirname(__file__))
plt.rcParams["figure.figsize"] = (16, 9)


def load_config_and_parser(system_path):
    if get_os() == "Mac":
        print("load MAC config-file")
        config_file_name = "MAC_train_classifier.cfg"
    elif get_os() == "Linux":
        print("load LMU config-file")
        config_file_name = "LMU_train_classifier.cfg"
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


def main(cfg, galaxies, iteration, lgr):
    """"""

    now = datetime.now()
    cfg['RUN_DATE'] = now.strftime('%Y%m%d_%H%M')
    cfg['PATH_OUTPUT'] = f"{cfg['PATH_OUTPUT']}/{cfg['RUN_DATE']}_classifier_training/"
    os.makedirs(cfg['PATH_OUTPUT'], exist_ok=True)


def train_tune_classifier(tune_config, base_config):
    config = dict(base_config)
    config.update(tune_config)

    ray_trial_dir = session.get_trial_dir()

    config['PATH_OUTPUT'] = ray_trial_dir
    config['PATH_OUTPUT_CKPT'] = ray_trial_dir

    config['TRIAL_ID'] = hparam_hash(
        int(config['batch_size']),
        float(config['learning_rate']),
        list(config['hidden_sizes']),
        float(config['dropout_prob']),
        int(bool(config['batch_norm']))
    )

    train_cf = gaNdalFClassifier(
        cfg=config,
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        hidden_sizes=config["hidden_sizes"],
        dropout_prob=config["dropout_prob"],
        batch_norm=config["batch_norm"],
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

    train_cf.run_training(on_epoch_end=reporter)

    final_payload = {
        "epoch": int(train_cf.current_epoch + 1),
        "loss": float(train_cf.lst_valid_loss_per_epoch[-1]) if train_cf.lst_valid_loss_per_epoch else None,
        "train_loss": float(
            train_cf.lst_train_loss_per_epoch[-1]) if train_cf.lst_train_loss_per_epoch else None,
        "val_loss": float(train_cf.lst_valid_loss_per_epoch[-1]) if train_cf.lst_valid_loss_per_epoch else None,
    }
    session.report(final_payload)


def hparam_hash(bs, lr, hs, dp, bn):
    """
    """
    payload = {
        "bs": int(bs),
        "lr": float(lr),
        "hs": "-".join(map(str, hs)),
        "dp": int(dp),
        "bn": int(bn)
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def my_trial_name_creator(trial):
    bs = int(trial.config["batch_size"])
    lr = float(trial.config["learning_rate"])
    hs = trial.config["hidden_sizes"]
    dp = float(trial.config["dropout_prob"])
    bn = int(bool(trial.config["batch_norm"]))
    return f"trial_{hparam_hash(bs, lr, hs, dp, bn)}"


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    path = os.path.abspath(sys.path[-1])
    cfg, path_cfg_file = load_config_and_parser(system_path=os.path.abspath(sys.path[-1]))

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

    # Initialize the logger
    train_classifier_logger = LoggerHandler(
        logger_dict={"logger_name": "train classifier",
                     "info_logger": cfg['INFO_LOGGER'],
                     "error_logger": cfg['ERROR_LOGGER'],
                     "debug_logger": cfg['DEBUG_LOGGER'],
                     "stream_logger": cfg['STREAM_LOGGER'],
                     "stream_logging_level": log_lvl},
        log_folder_path=f"{cfg['PATH_OUTPUT']}/Logs/"
    )

    # Write status to logger
    train_classifier_logger.log_info_stream("Start train classifier")

    if cfg['HPARAM_SEARCH'] is False:
        pass
    else:
        batch_size = cfg["BATCH_SIZE"]
        hidden_sizes = cfg["HIDDEN_SIZES"]
        learning_rate = cfg["LEARNING_RATE"]
        dropout_prob = cfg["DROPOUT_PROB"]
        batch_norm = cfg["BATCH_NORM"]

        if not isinstance(batch_size, list):
            batch_size = [batch_size]
        if not isinstance(hidden_sizes, list):
            hidden_sizes = [hidden_sizes]
        if not isinstance(dropout_prob, list):
            dropout_prob = [dropout_prob]
        if not isinstance(learning_rate, list):
            learning_rate = [learning_rate, learning_rate]
        if not isinstance(batch_norm, list):
            batch_norm = [batch_norm]

        search_space = {
            "batch_size": tune.choice(batch_size),
            "learning_rate": tune.loguniform(learning_rate[0], learning_rate[1]),
            "hidden_sizes": tune.choice(hidden_sizes),
            "dropout_prob": tune.choice(dropout_prob),
            "batch_norm": tune.choice(batch_norm),
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
            parameter_columns=["batch_size", "learning_rate", "hidden_sizes", "dropout_prob", "batch_norm"],
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
            max_t=cfg["EPOCHS"],
            grace_period=10,
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

        analysis = tune.run(
            partial(train_tune_classifier, base_config=GLOBAL_BASE_CONFIG),
            config=search_space,
            search_alg=optuna_search,
            scheduler=asha,
            stop=plateau,
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

        train_classifier_logger.log_info_stream("Best config found:")
        train_classifier_logger.log_info_stream(analysis.get_best_config(metric="loss", mode="min"))