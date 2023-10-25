from datetime import datetime
from Handler.helper_functions import get_os
from detected_classifier.train_detector import TrainDet
import argparse
import matplotlib.pyplot as plt
import sys
import yaml
import os
sys.path.append(os.path.dirname(__file__))
plt.rcParams["figure.figsize"] = (16, 9)


def main(
        cfg,
        epochs,
        batch_size,
        lr
):
    """"""

    train_detector = TrainDet(
        cfg=cfg,
        bs=batch_size,
        lr=lr
    )

    train_detector.run_training()


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

    with open(f"{path}/files/conf/{args.config_filename}", 'r') as fp:
        cfg = yaml.safe_load(fp)

    now = datetime.now()
    cfg['RUN_DATE_CLASSF'] = now.strftime('%Y-%m-%d_%H-%M')
    cfg['PATH_OUTPUT_CLASSF'] = f"{cfg['PATH_OUTPUT_CLASSF']}/run_{cfg['RUN_DATE_CLASSF']}"
    if not os.path.exists(cfg['PATH_OUTPUT_CLASSF']):
        os.mkdir(cfg['PATH_OUTPUT_CLASSF'])

    for lr in cfg['LEARNING_RATE_CLASSF']:
        for bs in cfg['BATCH_SIZE_CLASSF']:
            main(
                cfg=cfg,
                epochs=150,
                batch_size=bs,
                lr=lr
            )
