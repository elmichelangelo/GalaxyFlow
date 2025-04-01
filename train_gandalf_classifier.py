from datetime import datetime
from Handler import get_os
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
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.append(os.path.dirname(__file__))
plt.rcParams["figure.figsize"] = (16, 9)


def main(cfg, galaxies, iteration, performance_logger):
    """"""

    train_detector = gaNdalFClassifier(
        cfg=cfg,
        galaxies=galaxies,
        iteration=iteration,
        performance_logger=performance_logger
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
    cfg["YJ_TRANSFORMATION"] = [config["yj_transform"]]
    cfg["MAXABS_SCALER"] = [config["maxabs_scaler"]]
    cfg["USE_BATCHNORM_CLASSF"] = [config["batch_norm"]]
    cfg["DROPOUT_PROB_CLASSF"] = [config["dropout"]]

    model = gaNdalFClassifier(cfg=cfg, galaxies=galaxies, iteration=0, performance_logger=performance_logger)
    model.run_training()

    # Use any score you want to optimize here
    acc = model.performance_logger.handlers[0].stream.getvalue().splitlines()[-1]
    tune.report(calibrated_accuracy=model.best_validation_acc)


if __name__ == '__main__':
    path = os.path.abspath(sys.path[-1])
    if get_os() == "Mac":
        print("load mac config-file")
        config_file_name = "mac.cfg"
    elif get_os() == "Windows":
        print("load windows config-file")
        config_file_name = "windows.cfg"
    elif get_os() == "Linux":
        print("load linux config-file")
        config_file_name = "linux.cfg"
    else:
        print("load default config-file")
        config_file_name = "default.cfg"

    folder_prefix_name = "MAG"

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
    parser.add_argument(
        '--folder_prefix_name',
        "-fpn",
        type=str,
        nargs=1,
        required=False,
        default=folder_prefix_name,
        help='Some prefix for saving folder'
    )
    args = parser.parse_args()

    if isinstance(args.config_filename, list):
        args.config_filename = args.config_filename[0]

    if isinstance(args.folder_prefix_name, list):
        args.folder_prefix_name = args.folder_prefix_name[0]

    with open(f"{path}/conf/{args.config_filename}", 'r') as fp:
        cfg = yaml.safe_load(fp)

    now = datetime.now()
    cfg['RUN_DATE'] = now.strftime('%Y-%m-%d_%H-%M')
    cfg['PATH_OUTPUT'] = f"{cfg['PATH_OUTPUT']}/classifier_training_{cfg['RUN_DATE']}_{args.folder_prefix_name}"
    cfg["LUM_TYPE_CLASSF"] = args.folder_prefix_name
    if not os.path.exists(cfg['PATH_OUTPUT']):
        os.mkdir(cfg['PATH_OUTPUT'])

    cfg["ACTIVATIONS"] = [nn.ReLU, lambda: nn.LeakyReLU(0.2)]

    galaxies = DESGalaxies(
        cfg=cfg,
        kind="classifier_training"
    )

    performance_logger = logging.getLogger("PerformanceLogger")
    performance_logger.setLevel(logging.INFO)
    performance_handler = logging.FileHandler(f"{cfg['PATH_OUTPUT']}/time_info_{cfg['RUN_DATE']}.log", mode='a')
    performance_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(message)s'))
    performance_logger.addHandler(performance_handler)

    # performance_logger.info("############################### CPU-Test gestartet ######################################")
    # cfg["DEVICE_CLASSF"] = "cpu"
    # start = datetime.now()
    # for iteration in range(cfg["ITERATIONS"]):
    #     main(
    #         cfg=cfg,
    #         galaxies=galaxies,
    #         iteration=iteration,
    #         performance_logger=performance_logger
    #     )
    # elapsed_cpu = (datetime.now() - start).total_seconds()
    # performance_logger.info(f"CPU-Test beendet: {elapsed_cpu} Seconds")
    performance_logger.info("############################### GPU-Test gestartet ######################################")
    if get_os() == "Mac":
        cfg["DEVICE_CLASSF"] = "mps"
    else:
        cfg["DEVICE_CLASSF"] = "cuda"
    start = datetime.now()
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32, 64]),
        "activation": tune.choice(["ReLU", "LeakyReLU"]),
        "num_layers": tune.choice([1, 2, 3]),
        "hidden_sizes": tune.choice([[32, 64], [64, 128], [128, 256]]),
        "yj_transform": tune.choice([True, False]),
        "maxabs_scaler": tune.choice([True, False]),
        "batch_norm": tune.choice([True, False]),
        "dropout": tune.uniform(0.0, 0.5),
    }

    scheduler = ASHAScheduler(
        max_t=cfg["EPOCHS_CLASSF"],
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter(
        metric_columns=["calibrated_accuracy", "training_iteration"]
    )

    tune.run(
        tune.with_parameters(train_tune_classifier, cfg=cfg, galaxies=galaxies, performance_logger=performance_logger),
        # resources_per_trial={"cpu": 2, "gpu": 1},
        resources_per_trial={"cpu": 1},  # , "gpu": 1},
        config=search_space,
        num_samples=20,  # increase if you're running on cluster
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_gandalf_classifier"
    )
    # for iteration in range(cfg["ITERATIONS"]):
    #     main(
    #         cfg=cfg,
    #         galaxies=galaxies,
    #         iteration=iteration,
    #         performance_logger=performance_logger
    #     )
    elapsed_gpu = (datetime.now() - start).total_seconds()
    performance_logger.info(f"GPU-Test beendet: {elapsed_gpu} Seconds")

