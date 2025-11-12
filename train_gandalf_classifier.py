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
from ray.tune.tune_config import TuneConfig
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
    import numpy as _np
    import shutil, os

    config = dict(base_config)
    config.update(tune_config)

    ray_trial_dir = session.get_trial_dir()
    config['PATH_OUTPUT'] = ray_trial_dir
    config['PATH_OUTPUT_CKPT'] = ray_trial_dir
    config['TRIAL_ID'] = hparam_hash_from_config(config)

    train_cf = gaNdalFClassifier(
        cfg=config,
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        hidden_sizes=config.get("hidden_sizes", [config.get("RES_WIDTH", 256)]),
        dropout_prob=config.get("dropout_prob", 0.0),
        batch_norm=config.get("batch_norm", True),
    )

    ckpt_src = os.path.join(config['PATH_OUTPUT'], "last.ckpt.pt")
    ray_ckpt_dir = os.path.join(config['PATH_OUTPUT_CKPT'], "ray_ckpt")
    best_so_far = [float("inf")]

    def _to_py_scalar(v):
        # sicheres Casting für Ray (keine numpy dtypes)
        if isinstance(v, (_np.floating, _np.integer)):
            return float(v)
        return v

    def reporter(**kwargs):
        """
        Nimmt alle vom Trainer gelieferten Metriken entgegen (epoch, train_loss, val_loss,
        accuracy, f1, brier, nll, ece, mre, gre, val_objective, ...), reportet sie an Ray
        und speichert bei Verbesserung ein Checkpoint.
        """
        # epoch & val_objective für Schedulers/Stopper
        epoch = int(kwargs.get("epoch", 0))
        val_obj = kwargs.get("val_objective", None)

        # Payload sauber in Python-Scalars casten
        payload = {}
        for k, v in kwargs.items():
            if v is None:
                payload[k] = None
            elif k == "epoch":
                payload[k] = int(v)
            else:
                payload[k] = _to_py_scalar(v)

        improved = (val_obj is not None) and (float(val_obj) < best_so_far[0] - 1e-12)

        if improved and os.path.exists(ckpt_src) and Checkpoint is not None:
            best_so_far[0] = float(val_obj)
            os.makedirs(ray_ckpt_dir, exist_ok=True)
            shutil.copy2(ckpt_src, os.path.join(ray_ckpt_dir, "last.ckpt.pt"))
            session.report(payload, checkpoint=Checkpoint.from_directory(ray_ckpt_dir))
        else:
            session.report(payload)

    # Training starten – ruft am Ende jeder Epoche unseren flexiblen Reporter
    train_cf.run_training(on_epoch_end=reporter)

    # Abschlussreport (falls der Scheduler das letzte Resultat noch sehen soll)
    extra = getattr(train_cf, "last_val_extra", {}) or {}
    final_payload = {
        "epoch": int(getattr(train_cf, "current_epoch", -1) + 1),
        "train_loss": float(train_cf.lst_train_loss_per_epoch[-1]) if train_cf.lst_train_loss_per_epoch else None,
        "val_loss": float(train_cf.lst_valid_loss_per_epoch[-1]) if train_cf.lst_valid_loss_per_epoch else None,
        "accuracy": float(train_cf.lst_valid_acc_per_epoch[-1]) if train_cf.lst_valid_acc_per_epoch else None,
        "f1": float(train_cf.lst_valid_f1_per_epoch[-1]) if train_cf.lst_valid_f1_per_epoch else None,
        "val_objective": float(extra.get("val_objective")) if "val_objective" in extra else None,
        "brier": float(extra.get("brier")) if "brier" in extra else None,
        "nll": float(extra.get("nll")) if "nll" in extra else None,
        "ece": float(extra.get("ece")) if "ece" in extra else None,
        "mre": float(extra.get("mre")) if "mre" in extra else None,
        "gre": float(extra.get("gre")) if "gre" in extra else None,
    }
    session.report(final_payload)


def hparam_hash_from_config(config):
    payload = {"bs": int(config["batch_size"]), "lr": float(config["learning_rate"])}

    if str(config.get("ARCH", "mlp")).lower() == "resmlp":
        payload.update({
            "arch": "resmlp",
            "rw": int(config["RES_WIDTH"]),
            "rd": int(config["RES_DEPTH"]),
            "rdo": float(config["RES_DROPOUT"]),
            "sched": str(config.get("SCHEDULER", None))
        })
    else:
        payload.update({
            "arch": "mlp",
            "hs": "-".join(map(str, config["hidden_sizes"])),
            "dp": float(config["dropout_prob"]),
            "bn": int(bool(config["batch_norm"])),
            "sched": str(config.get("SCHEDULER", None))
        })

    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def my_trial_name_creator(trial):
    cfg = trial.config
    h = hparam_hash_from_config(cfg)
    return f"trial_{h}"


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

    os.makedirs(f"{cfg['PATH_OUTPUT']}/Logs/", exist_ok=True)
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
        # arch = str(GLOBAL_BASE_CONFIG.get("ARCH", "mlp")).lower()

        # if arch == "resmlp":
        #     res_width_choices = GLOBAL_BASE_CONFIG.get("RES_WIDTH", [256, 384, 512])
        #     res_depth_choices = GLOBAL_BASE_CONFIG.get("RES_DEPTH", [3, 4, 6])
        #     res_drop_choices = GLOBAL_BASE_CONFIG.get("RES_DROPOUT", [0.1, 0.2, 0.3])
        #
        #     search_space = {
        #         "batch_size": tune.choice(cfg["BATCH_SIZE"]),  # z.B. [1024, 2048, 4096, 8192]
        #         "learning_rate": tune.choice(cfg["LEARNING_RATE"]),  # tune.loguniform(cfg["LEARNING_RATE"][0], cfg["LEARNING_RATE"][1]),
        #         "RES_WIDTH": tune.choice(res_width_choices),
        #         "RES_DEPTH": tune.choice(res_depth_choices),
        #         "RES_DROPOUT": tune.choice(res_drop_choices),
        #
        #         # feste Schalter (optional in die Suche aufnehmen, wenn gewünscht)
        #         "ARCH": "resmlp",
        #         "SCHEDULER": cfg.get("SCHEDULER", None),
        #
        #         # Rest wie gehabt (nur keine HS/DP mehr variieren)
        #         "batch_norm": cfg.get("BATCH_NORM", True),  # wird im ResMLP eh ignoriert (du nutzt LayerNorm)
        #         "INFO_LOGGER": cfg["INFO_LOGGER"],
        #         "ERROR_LOGGER": cfg["ERROR_LOGGER"],
        #         "DEBUG_LOGGER": cfg["DEBUG_LOGGER"],
        #         "STREAM_LOGGER": cfg["STREAM_LOGGER"],
        #         "LOGGING_LEVEL": cfg["LOGGING_LEVEL"],
        #         "PATH_TRANSFORMERS": cfg["PATH_TRANSFORMERS"],
        #         "FILENAME_STANDARD_SCALER": cfg["FILENAME_STANDARD_SCALER"],
        #         "PATH_OUTPUT_BASE": cfg["PATH_OUTPUT_BASE"],
        #         "PATH_OUTPUT_CATALOGS_BASE": cfg["PATH_OUTPUT_CATALOGS_BASE"],
        #         "RUN_DATE": cfg['RUN_DATE'],
        #     }
        # else:
        hidden_sizes = cfg["HIDDEN_SIZES"]
        hidden_sizes = [tuple(h) if isinstance(h, list) else h for h in hidden_sizes]
        dropout_prob = cfg["DROPOUT_PROB"]
        batch_norm = cfg["BATCH_NORM"]
        lr_lo, lr_hi = cfg["LEARNING_RATE"]

        search_space = {
            "batch_size": tune.choice(cfg["BATCH_SIZE"]),
            "learning_rate": tune.loguniform(lr_lo, lr_hi),  # tune.loguniform(lr_lo, lr_hi),
            "hidden_sizes": tune.choice(hidden_sizes),
            "dropout_prob": tune.choice(dropout_prob),
            "batch_norm": tune.choice(batch_norm),

            # "ARCH": cfg["ARCH"],
            # "SCHEDULER": cfg.get("SCHEDULER", None),

            "INFO_LOGGER": cfg["INFO_LOGGER"],
            "ERROR_LOGGER": cfg["ERROR_LOGGER"],
            "DEBUG_LOGGER": cfg["DEBUG_LOGGER"],
            "STREAM_LOGGER": cfg["STREAM_LOGGER"],
            "LOGGING_LEVEL": cfg["LOGGING_LEVEL"],
            "PATH_TRANSFORMERS": cfg["PATH_TRANSFORMERS"],
            "FILENAME_STANDARD_SCALER": cfg["FILENAME_STANDARD_SCALER"],
            "PATH_OUTPUT_BASE": cfg["PATH_OUTPUT_BASE"],
            "PATH_OUTPUT_CATALOGS_BASE": cfg["PATH_OUTPUT_CATALOGS_BASE"],
            "RUN_DATE": cfg['RUN_DATE'],
        }

        # if arch == "resmlp":
        #     param_cols = {
        #         "batch_size": "bs",
        #         "learning_rate": "lr",
        #         "RES_WIDTH": "rw",
        #         "RES_DEPTH": "rd",
        #         "RES_DROPOUT": "rdo",
        #     }
        # else:
        param_cols = {
            "batch_size": "bs",
            "learning_rate": "lr",
            "hidden_sizes": "hs",
            "dropout_prob": "dp",
            "batch_norm": "bn",
        }

        reporter = CLIReporter(
            parameter_columns=param_cols,
            metric_columns={
                "val_objective": "Obj",
                "brier": "Brier",
                "ece": "ECE",
                "mre": "MRE",
                "gre": "GRE",
                "nll": "NLL",
                "f1": "F1",
                "accuracy": "Acc",
                "train_loss": "Train",
                "val_loss": "Val",
                "epoch": "Ep",
            },
        )

        resources = {"cpu": cfg["RESOURCE_CPU"], "gpu": cfg["RESOURCE_GPU"]}

        optuna_search = OptunaSearch(metric="val_objective", mode="min")

        asha = ASHAScheduler(
            metric="val_objective",
            time_attr="epoch",
            mode="min",
            max_t=cfg["EPOCHS"],
            grace_period=max(20, cfg["EPOCHS"]//3),
            reduction_factor=3,
            stop_last_trials=True
        )

        plateau = TrialPlateauStopper(
            metric="val_objective",
            mode="min",
            num_results=20,
            grace_period=max(20, cfg["EPOCHS"]//3),
            std=5e-3
        )

        ray.init(local_mode=cfg["DEBUG_MODE"], num_cpus=cfg["RESOURCE_CPU"], include_dashboard=cfg["INCLUDE_DASHBOARD"])
        analysis = tune.run(
            partial(train_tune_classifier, base_config=GLOBAL_BASE_CONFIG),
            config=search_space,
            search_alg=optuna_search,
            scheduler=asha,
            stop=None, #plateau,
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
            # checkpoint_score_attr="max-accuracy",
        )

        train_classifier_logger.log_info_stream("Best config found:")
        train_classifier_logger.log_info_stream(analysis.get_best_config(metric="val_objective", mode="min"))