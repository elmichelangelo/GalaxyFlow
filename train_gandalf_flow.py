from datetime import datetime
from Handler import get_os
from gandalf_flow import gaNdalFFlow
import argparse
import matplotlib.pyplot as plt
import sys
import yaml
import os

plt.rcParams["figure.figsize"] = (16, 9)


def main(
        cfg,
        learning_rate,
        weight_decay,
        number_hidden,
        number_blocks,
        batch_size):
    """"""

    train_flow = gaNdalFFlow(
        cfg=cfg,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        number_hidden=number_hidden,
        number_blocks=number_blocks,
        batch_size=batch_size
    )
    train_flow.run_training()


if __name__ == '__main__':
    sys.path.append(os.path.dirname(__file__))
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
    prefix_mock = ""

    if cfg['TRAIN_ON_MOCK_FLOW'] is True:
        prefix_mock = "_mock"

    cfg['RUN_DATE'] = now.strftime('%Y-%m-%d_%H-%M')
    cfg['PATH_OUTPUT'] = f"{cfg['PATH_OUTPUT']}/flow_training_{cfg['RUN_DATE']}{prefix_mock}"
    if not os.path.exists(cfg['PATH_OUTPUT_CATALOGS']):
        os.mkdir(cfg['PATH_OUTPUT_CATALOGS'])
    cfg['PATH_OUTPUT_CATALOGS'] = f"{cfg['PATH_OUTPUT_CATALOGS']}/flow_training_{cfg['RUN_DATE']}{prefix_mock}"
    if not os.path.exists(cfg['PATH_OUTPUT']):
        os.mkdir(cfg['PATH_OUTPUT'])
    if not os.path.exists(cfg['PATH_OUTPUT_CATALOGS']):
        os.mkdir(cfg['PATH_OUTPUT_CATALOGS'])

    batch_size = cfg["BATCH_SIZE_FLOW"]
    number_hidden = cfg["NUMBER_HIDDEN"]
    number_blocks = cfg["NUMBER_BLOCKS"]
    learning_rate = cfg["LEARNING_RATE_FLOW"]
    weight_decay = cfg["WEIGHT_DECAY"]

    if not isinstance(batch_size, list):
        batch_size = [batch_size]
    if not isinstance(number_hidden, list):
        number_hidden = [number_hidden]
    if not isinstance(number_blocks, list):
        number_blocks = [number_blocks]
    if not isinstance(learning_rate, list):
        learning_rate = [learning_rate]
    if not isinstance(weight_decay, list):
        weight_decay = [weight_decay]

    for lr in learning_rate:
        for wd in weight_decay:
            for nh in number_hidden:
                for nb in number_blocks:
                    for bs in batch_size:
                        main(
                            cfg=cfg,
                            learning_rate=lr,
                            weight_decay=wd,
                            number_hidden=nh,
                            number_blocks=nb,
                            batch_size=bs
                        )
