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

    now = datetime.now()
    cfg['RUN_DATE'] = now.strftime('%Y-%m-%d_%H-%M')
    cfg['PATH_OUTPUT'] = f"{cfg['PATH_OUTPUT']}/classifier_training_{cfg['RUN_DATE']}"
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
    for iteration in range(cfg["ITERATIONS"]):
        main(
            cfg=cfg,
            galaxies=galaxies,
            iteration=iteration,
            performance_logger=performance_logger
        )
    elapsed_gpu = (datetime.now() - start).total_seconds()
    performance_logger.info(f"GPU-Test beendet: {elapsed_gpu} Seconds")

