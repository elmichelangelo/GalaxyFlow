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

    train_detector = gaNdalFClassifier(
        cfg=cfg,
        galaxies=galaxies,
        iteration=iteration,
        classifier_logger=lgr
    )

    train_detector.run_training()

    if cfg['SAVE_NN_CLASSF'] is True:
        train_detector.save_model()


def train_tune_classifier(config, cfg, galaxies, performance_logger):
    now = datetime.now()
    cfg['RUN_DATE'] = now.strftime('%Y-%m-%d_%H-%M-%S')
    cfg['PATH_OUTPUT'] = os.path.join(cfg['PATH_OUTPUT'], f"raytune_run_{cfg['RUN_DATE']}")

    # Apply hyperparameters from Ray Tune
    cfg["ACTIVATIONS"] = [lambda: getattr(nn, config["activation"])()]
    cfg["POSSIBLE_NUM_LAYERS"] = [config["num_layers"]]
    cfg["POSSIBLE_HIDDEN_SIZES"] = config["hidden_sizes"]
    cfg["LEARNING_RATE_CLASSF"] = [config["lr"]]
    cfg["BATCH_SIZE_CLASSF"] = [config["batch_size"]]
    cfg["USE_BATCHNORM_CLASSF"] = [config["batch_norm"]]
    cfg["DROPOUT_PROB_CLASSF"] = [config["dropout"]]

    model = gaNdalFClassifier(cfg=cfg, galaxies=galaxies, iteration=0, classifier_logger=performance_logger)
    model.run_training()

    # Use any score you want to optimize here
    acc = model.classifier_logger.handlers[0].stream.getvalue().splitlines()[-1]
    tune.report(calibrated_accuracy=model.best_validation_acc)


def hparam_hash(bs, lr, nh, nb, nl):
    """
    """
    payload = {
        "bs": int(bs),
        "lr": float(lr),
        "nh": int(nh),
        "nb": int(nb),
        "nl": int(nl)
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def my_trial_name_creator(trial):
    bs = int(trial.config["batch_size"])
    lr = float(trial.config["learning_rate"])
    nh = int(trial.config["number_hidden"])
    nb = int(trial.config["number_blocks"])
    nl = int(trial.config["number_layers"])
    return f"trial_{hparam_hash(bs, lr, nh, nb, nl)}"


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
        cfg["ACTIVATIONS"] = [lambda: nn.LeakyReLU(0.2)]

        galaxies = DESGalaxies(
            cfg=cfg,
            dataset_logger=train_classifier_logger
        )

        start = datetime.now()
        base_path = cfg['PATH_OUTPUT']
        for i in range(cfg["ITERATIONS"]):
            main(
                cfg=cfg,
                galaxies=galaxies,
                iteration=i+1,
                lgr=train_classifier_logger
            )
            cfg['PATH_OUTPUT'] = base_path
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
            "learning_rate": tune.loguniform(learning_rate[0], learning_rate[1]),
            "number_hidden": tune.choice(number_hidden),
            "number_blocks": tune.choice(number_blocks),
            "number_layers": tune.choice(number_layers),
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