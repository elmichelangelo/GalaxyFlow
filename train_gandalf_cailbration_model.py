from datetime import datetime
from Handler import get_os
from gandalf_calibration_model.gaNdalF_calibration_model import gaNdalFCalibModel
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
from gandalf import gaNdalF

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.append(os.path.dirname(__file__))
plt.rcParams["figure.figsize"] = (16, 9)


def main(cfg, performance_logger):
    """"""
    gandalf_calib_model = gaNdalFCalibModel(cfg=cfg, performance_logger=performance_logger)
    gandalf_calib_model.run_training_calib_model()


if __name__ == '__main__':
    path = os.path.abspath(sys.path[-1])
    if get_os() == "Mac":
        print("load mac config-file")
        config_file_name = "mac_train_calib_model.cfg"
    elif get_os() == "Windows":
        print("load windows config-file")
        config_file_name = "windows_train_calib_model.cfg"
    elif get_os() == "Linux":
        print("load linux config-file")
        config_file_name = "linux_train_calib_model.cfg"
    else:
        print("load default config-file")
        config_file_name = "default.cfg"

    folder_prefix_name = "MAG"

    parser = argparse.ArgumentParser(description='Start train calibration model')
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
    cfg['PATH_OUTPUT'] = f"{cfg['PATH_OUTPUT']}/calibration_model_training_{cfg['RUN_DATE']}"
    if not os.path.exists(cfg['PATH_OUTPUT']):
        os.mkdir(cfg['PATH_OUTPUT'])

    performance_logger = logging.getLogger("PerformanceLogger")
    performance_logger.setLevel(logging.INFO)
    performance_handler = logging.FileHandler(f"{cfg['PATH_OUTPUT']}/time_info_{cfg['RUN_DATE']}.log", mode='a')
    performance_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(message)s'))
    performance_logger.addHandler(performance_handler)

    performance_logger.info("############################### GPU-Test gestartet ######################################")
    if get_os() == "Mac":
        cfg["DEVICE_CLASSF"] = "mps"
    else:
        cfg["DEVICE_CLASSF"] = "cuda"
    start = datetime.now()

    main(cfg=cfg, performance_logger=performance_logger)

    elapsed_gpu = (datetime.now() - start).total_seconds()
    performance_logger.info(f"GPU-Test beendet: {elapsed_gpu} Seconds")

