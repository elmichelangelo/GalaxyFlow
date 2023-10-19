
from datetime import datetime
import torch.cuda
from Handler.helper_functions import get_os, string_to_tuple
from galaxyflow.training import TrainFlow
import argparse
import matplotlib.pyplot as plt
import sys
import yaml
import os
sys.path.append(os.path.dirname(__file__))
plt.rcParams["figure.figsize"] = (16, 9)


def main(
        cfg,
        learning_rate,
        weight_decay,
        number_hidden,
        number_blocks,
        batch_size):
    """"""

    train_flow = TrainFlow(
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
        print(f"OS Error: {get_os()}")

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
        '--mode',
        "-m",
        type=str,
        nargs=1,
        required=False,
        help='Mode of gaNdalF'
    )
    args = parser.parse_args()

    if isinstance(args.mode, list):
        args.mode = args.mode[0]

    if isinstance(args.config_filename, list):
        args.config_filename = args.config_filename[0]

    with open(f"{path}/files/conf/{args.config_filename}", 'r') as fp:
        cfg = yaml.safe_load(fp)

    if args.mode is None:
        args.mode = cfg["MODE"]
        mode = args.mode
    else:
        mode = args.mode
        cfg["MODE"] = mode

    now = datetime.now()
    cfg['RUN_DATE'] = now.strftime('%Y-%m-%d_%H-%M')
    path_output = f"{cfg['PATH_OUTPUT']}/run_{cfg['RUN_DATE']}"
    cfg['PATH_OUTPUT'] = path_output
    if not os.path.exists(cfg['PATH_OUTPUT']):
        os.mkdir(cfg['PATH_OUTPUT'])

    batch_size = cfg["BATCH_SIZE"]
    scaler = cfg["SCALER"]
    number_hidden = cfg["NUMBER_HIDDEN"]
    number_blocks = cfg["NUMBER_BLOCKS"]
    learning_rate = cfg["LEARNING_RATE"]
    weight_decay = cfg["WEIGHT_DECAY"]

    if not isinstance(batch_size, list):
        batch_size = [batch_size]
    if not isinstance(scaler, list):
        scaler = [scaler]
    if not isinstance(number_hidden, list):
        number_hidden = [number_hidden]
    if not isinstance(number_blocks, list):
        number_blocks = [number_blocks]
    if not isinstance(learning_rate, list):
        learning_rate = [learning_rate]
    if not isinstance(weight_decay, list):
        weight_decay = [weight_decay]

    output_cols = cfg[f"OUTPUT_COLS_{cfg['LUM_TYPE']}"]
    input_cols = cfg[f"INPUT_COLS_{cfg['LUM_TYPE']}"]
    path_data = cfg["PATH_DATA"]
    path_output = cfg["PATH_OUTPUT"]

    for lr in learning_rate:
        for wd in weight_decay:
            for nh in number_hidden:
                for nb in number_blocks:
                    for bs in batch_size:
                        for sc in scaler:

                            main(
                                cfg=cfg,
                                learning_rate=lr,
                                weight_decay=wd,
                                number_hidden=nh,
                                number_blocks=nb,
                                batch_size=bs
                            )
