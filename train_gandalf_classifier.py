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
# from ray import tune
# from ray.tune.schedulers import ASHAScheduler
# from ray.tune import CLIReporter

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.append(os.path.dirname(__file__))
plt.rcParams["figure.figsize"] = (16, 9)


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


# def train_tune_classifier(config, cfg, galaxies, performance_logger):
#     now = datetime.now()
#     cfg['RUN_DATE'] = now.strftime('%Y-%m-%d_%H-%M-%S')
#     cfg['PATH_OUTPUT'] = os.path.join(cfg['PATH_OUTPUT'], f"raytune_run_{cfg['RUN_DATE']}")
#
#     # Apply hyperparameters from Ray Tune
#     cfg["ACTIVATIONS"] = [lambda: getattr(nn, config["activation"])()]
#     cfg["POSSIBLE_NUM_LAYERS"] = [config["num_layers"]]
#     cfg["POSSIBLE_HIDDEN_SIZES"] = config["hidden_sizes"]
#     cfg["LEARNING_RATE_CLASSF"] = [config["lr"]]
#     cfg["BATCH_SIZE_CLASSF"] = [config["batch_size"]]
#     cfg["USE_BATCHNORM_CLASSF"] = [config["batch_norm"]]
#     cfg["DROPOUT_PROB_CLASSF"] = [config["dropout"]]
#
#     model = gaNdalFClassifier(cfg=cfg, galaxies=galaxies, iteration=0, classifier_logger=performance_logger)
#     model.run_training()
#
#     # Use any score you want to optimize here
#     acc = model.classifier_logger.handlers[0].stream.getvalue().splitlines()[-1]
#     tune.report(calibrated_accuracy=model.best_validation_acc)


if __name__ == '__main__':
    path = os.path.abspath(sys.path[-1])
    config_file_name = "MAC_train_classifier.cfg"

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

    with open(f"{path}/conf/{args.config_filename}", 'r') as fp:
        cfg = yaml.safe_load(fp)

    # now = datetime.now()
    os.makedirs(os.path.join(cfg['PATH_OUTPUT'], "Logs"), exist_ok=True)

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

    cfg["ACTIVATIONS"] = [lambda: nn.LeakyReLU(0.2)]  # nn.ReLU,

    galaxies = DESGalaxies(
        cfg=cfg,
        dataset_logger=train_classifier_logger
    )

    if get_os() == "Mac":
        cfg["DEVICE_CLASSF"] = "mps"
    else:
        # cfg["DEVICE_CLASSF"] = "cuda"
        cfg["DEVICE_CLASSF"] = "cpu"
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
    # search_space = {
    #     "lr": tune.loguniform(1e-7, 1e-1),
    #     "batch_size": tune.choice([16, 32, 64, 512, 1024, 2048, 131072]),
    #     "activation": tune.choice(["ReLU", "LeakyReLU"]),
    #     "num_layers": tune.choice([1, 2, 3, 4]),
    #     "hidden_sizes": tune.choice([[32, 64], [64, 128], [128, 256]]),  # ([[32, 64], [64, 128], [128, 256]]),
    #     "batch_norm": tune.choice([True, False]),
    #     "dropout": tune.uniform(0.0, 0.8),
    # }
    #
    # scheduler = ASHAScheduler(
    #     metric="calibrated_accuracy",
    #     mode="max",
    #     max_t=cfg["EPOCHS"],
    #     grace_period=1,
    #     reduction_factor=2
    # )
    #
    # reporter = CLIReporter(
    #     metric_columns=["calibrated_accuracy", "training_iteration"]
    # )
    #
    # tune.run(
    #     tune.with_parameters(train_tune_classifier, cfg=cfg, galaxies=galaxies, performance_logger=train_classifier_logger),
    #     # resources_per_trial={"cpu": 2, "gpu": 1},
    #     resources_per_trial={"cpu": 1},  # , "gpu": 1},
    #     config=search_space,
    #     num_samples=20,  # increase if you're running on cluster
    #     scheduler=scheduler,
    #     progress_reporter=reporter,
    #     name="tune_gandalf_classifier"
    # )
    # for iteration in range(cfg["ITERATIONS"]):
    #     main(
    #         cfg=cfg,
    #         galaxies=galaxies,
    #         iteration=iteration,
    #         performance_logger=performance_logger
    #     )
    # elapsed_gpu = (datetime.now() - start).total_seconds()
    # train_classifier_logger.info(f"GPU-Test beendet: {elapsed_gpu} Seconds")

