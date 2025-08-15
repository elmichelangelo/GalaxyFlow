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
    from ray.train import Checkpoint
except (ImportError, AttributeError):
    try:
        from ray.air import Checkpoint
    except (ImportError, AttributeError):
        Checkpoint = None

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
    header = ["trial number", "batch size", "learning rate", "number hidden", "number blocks", "patience", "status"]
    with open(csv_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in runs:
            writer.writerow(row)

def add_or_update_run(csv_file, bs, lr, nh, nb, nl, pa, status):
    lr_str = str(lr).replace('.', ',')
    combo = (str(bs), lr_str, str(nh), str(nb), str(nl), str(pa))
    lock = FileLock(csv_file + ".lock")
    with lock:
        runs = _read_trained_combinations_no_lock(csv_file)
        updated = False
        for row in runs:
            if (row["batch size"], row["learning rate"], row["number hidden"], row["number blocks"], row["number layers"], row["patience"]) == combo:
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
                "number layers": str(nl),
                "patience": str(pa),
                "status": status
            })
        _write_all_runs_no_lock(csv_file, runs)

def check_if_run_exists(csv_file, bs, lr, nh, nb, nl, pa):
    lr_str = str(lr).replace('.', ',')
    combo = (str(bs), lr_str, str(nh), str(nb), str(nl), str(pa))
    lock = FileLock(csv_file + ".lock")
    with lock:
        runs = _read_trained_combinations_no_lock(csv_file)
        for row in runs:
            if (row["batch size"], row["learning rate"], row["number hidden"], row["number blocks"], row["number layers"], row["patience"]) == combo:
                return row["status"]
    return None


def main(
        cfg,
        learning_rate,
        number_hidden,
        number_blocks,
        number_layers,
        batch_size,
        patience,
        logger):
    """"""

    train_flow = gaNdalFFlow(
        cfg=cfg,
        learning_rate=learning_rate,
        number_hidden=number_hidden,
        number_blocks=number_blocks,
        number_layers=number_layers,
        batch_size=batch_size,
        patience=patience,
        train_flow_logger=logger
    )
    train_flow.run_training()
    val_loss = train_flow.run_training()
    return val_loss

def train_tune(tune_config, base_config):
    config = dict(base_config)
    config.update(tune_config)

    config['batch_size'] = int(config['batch_size'])
    bs = config['batch_size']
    lr = float(config['learning_rate'])
    nh = int(config['number_hidden'])
    nb = int(config['number_blocks'])
    nl = int(config['number_layers'])
    pa = int(config['patience'])

    trial_id = hparam_hash(bs, lr, nh, nb, nl, pa)
    study_root_out = os.path.join(config['PATH_OUTPUT_BASE'], f"study_{config['RUN_ID']}")
    study_root_cat = os.path.join(config['PATH_OUTPUT_CATALOGS_BASE'], f"study_{config['RUN_ID']}")
    os.makedirs(study_root_out, exist_ok=True)
    os.makedirs(study_root_cat, exist_ok=True)

    # Die Trial-Roots sind jetzt deterministisch
    config['PATH_OUTPUT'] = os.path.join(study_root_out, f"trial_{trial_id}")
    config['PATH_OUTPUT_CATALOGS'] = os.path.join(study_root_cat, f"trial_{trial_id}")
    os.makedirs(config['PATH_OUTPUT'], exist_ok=True)
    os.makedirs(config['PATH_OUTPUT_CATALOGS'], exist_ok=True)

    if cfg['GRID_SEARCH'] is True:
        csv_file = os.path.join(config["PATH_OUTPUT_CSV"], "trained_params.csv")

        existing_status = check_if_run_exists(csv_file, bs, lr, nh, nb, nl, pa)
        if existing_status == "started":
            print(f"Trial SKIP: {bs}, {lr}, {nh}, {nb}, {nl}, {pa} already started!", flush=True)
            session.report({"loss": 1e10, "epoch": 1, "skipped": True})
            return
        if existing_status == "finished":
            print(f"Trial SKIP: {bs}, {lr}, {nh}, {nb}, {nl}, {pa} already finished!", flush=True)
            session.report({"loss": 1e10, "epoch": 1, "skipped": True})
            return

        add_or_update_run(csv_file, bs, lr, nh, nb, nl, pa, "started")

    # config['PATH_OUTPUT'] = os.path.join(
    #     config['PATH_OUTPUT_BASE'],
    #     f"flow_training_{config['RUN_DATE']}",
    #     f"bs_{config['batch_size']}_lr_{config['learning_rate']}_nh_{config['number_hidden']}_nb_{config['number_blocks']}_nl_{config['number_layers']}_pa_{config['patience']}"
    # )
    # config['PATH_OUTPUT_CATALOGS'] = os.path.join(
    #     config['PATH_OUTPUT_CATALOGS_BASE'],
    #     f"flow_training_{config['RUN_DATE']}",
    #     f"bs_{config['batch_size']}_lr_{config['learning_rate']}_nh_{config['number_hidden']}_nb_{config['number_blocks']}_nl_{config['number_layers']}_pa_{config['patience']}"
    # )

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
        number_layers=config["number_layers"],
        batch_size=config["batch_size"],
        patience=config["patience"],
        train_flow_logger=logger
    )

    for epoch, validation_loss in train_flow.run_training():
        # Ray will ein Verzeichnis. Wir spiegeln die letzte .pt-Datei rein.
        ckpt_src = os.path.join(config['PATH_OUTPUT_SUBFOLDER'], "last.ckpt.pt")
        if os.path.exists(ckpt_src):
            ray_ckpt_dir = os.path.join(config['PATH_OUTPUT'], "ray_ckpt")
            os.makedirs(ray_ckpt_dir, exist_ok=True)
            shutil.copy2(ckpt_src, os.path.join(ray_ckpt_dir, "last.ckpt.pt"))
            if Checkpoint is not None:
                session.report({"loss": validation_loss, "epoch": epoch + 1},
                               checkpoint=Checkpoint.from_directory(ray_ckpt_dir))
            else:
                session.report({"loss": validation_loss, "epoch": epoch + 1})
        else:
            session.report({"loss": validation_loss, "epoch": epoch + 1})

    if cfg['GRID_SEARCH'] is True:
        add_or_update_run(csv_file, bs, lr, nh, nb, nl, pa, "finished")

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

def get_or_create_run_id(cfg):
    """
    Hält eine Kampagne stabil. Reihenfolge:
    1) ENV RUN_ID (wenn gesetzt),
    2) Datei <PATH_OUTPUT_BASE>/RUN_ID.txt,
    3) neu erzeugen und persistieren.
    """
    base = cfg['PATH_OUTPUT_BASE']
    os.makedirs(base, exist_ok=True)
    marker = os.path.join(base, "RUN_ID.txt")

    run_id = os.environ.get("RUN_ID")
    if run_id:
        with open(marker, "w") as f:
            f.write(run_id.strip())
        return run_id.strip()

    if os.path.exists(marker):
        return open(marker).read().strip()

    # fallback: einmalig erzeugen
    run_id = datetime.now().strftime('%Y-%m-%d')  # z.B. "2025-08-10"
    with open(marker, "w") as f:
        f.write(run_id)
    return run_id

def hparam_hash(bs, lr, nh, nb, nl, pa):
    """
    Deterministischer Hash nur aus den HPO-Parametern.
    Kürzen auf 10 Zeichen für kürzere Ordnernamen.
    """
    payload = {
        "bs": int(bs),
        "lr": float(lr),
        "nh": int(nh),
        "nb": int(nb),
        "nl": int(nl),
        "pa": int(pa),
    }
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:10]

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    sys.path.append(os.path.dirname(__file__))
    cfg, path_cfg_file = load_config_and_parser(system_path=os.path.abspath(sys.path[-1]))
    # Basispfade festhalten
    cfg['PATH_OUTPUT_BASE'] = cfg['PATH_OUTPUT']
    cfg['PATH_OUTPUT_CATALOGS_BASE'] = cfg['PATH_OUTPUT_CATALOGS']

    # Kampagnen-ID stabilisieren (ENV RUN_ID überschreibt, sonst Datei/neu)
    cfg['RUN_ID'] = get_or_create_run_id(cfg)

    storage_dir = os.path.join(cfg['PATH_OUTPUT_BASE'], f"{cfg['RUN_ID']}_ray_results")
    # Keine flow_training_DATUM-Verzeichnisse mehr hier erzwingen – das macht jetzt die Trial-Hash-Logik unten.
    cfg['PATH_OUTPUT'] = cfg['PATH_OUTPUT_BASE']
    cfg['PATH_OUTPUT_CATALOGS'] = cfg['PATH_OUTPUT_CATALOGS_BASE']

    GLOBAL_BASE_CONFIG = copy.deepcopy(cfg)

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
    number_layers = cfg["NUMBER_LAYERS"]
    learning_rate = cfg["LEARNING_RATE_FLOW"]
    patience = cfg["PATIENCE"]

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
    if not isinstance(patience, list):
        patience = [patience, patience]

    search_space = {
        "batch_size": tune.choice(batch_size),
        "learning_rate": tune.loguniform(learning_rate[0], learning_rate[1]),
        "number_hidden": tune.choice(number_hidden),
        "number_blocks": tune.choice(number_blocks),
        "number_layers": tune.choice(number_layers),
        "patience": tune.randint(patience[0], patience[1]),
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
        metric_columns=["loss"]
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
        grace_period=2
    )

    analysis = tune.run(
        partial(train_tune, base_config=GLOBAL_BASE_CONFIG),
        config=search_space,
        search_alg=optuna_search,
        scheduler=asha,
        num_samples=cfg['OPTUNA_RUNS'],
        max_concurrent_trials=cfg['MAX_TRAILS'],
        resources_per_trial=resources,
        progress_reporter=reporter,
        storage_path=storage_dir,
        name="gandalf_tune_optuna",
        resume="AUTO",
    )

    print("Best config found:")
    print(analysis.get_best_config(metric="loss", mode="min"))
