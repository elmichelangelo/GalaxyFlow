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

plt.rcParams["figure.figsize"] = (16, 9)


def main(
        cfg,
        learning_rate,
        weight_decay,
        number_hidden,
        number_blocks,
        batch_size,
        logger):
    """"""

    train_flow = gaNdalFFlow(
        cfg=cfg,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        number_hidden=number_hidden,
        number_blocks=number_blocks,
        batch_size=batch_size,
        train_flow_logger=logger
    )
    train_flow.run_training()

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

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    sys.path.append(os.path.dirname(__file__))
    cfg, path_cfg_file = load_config_and_parser(system_path=os.path.abspath(sys.path[-1]))

    cfg['PATH_OUTPUT'] = f"{cfg['PATH_OUTPUT']}/flow_training_{cfg['RUN_DATE']}"
    if not os.path.exists(cfg['PATH_OUTPUT_CATALOGS']):
        os.mkdir(cfg['PATH_OUTPUT_CATALOGS'])
    cfg['PATH_OUTPUT_CATALOGS'] = f"{cfg['PATH_OUTPUT_CATALOGS']}/flow_training_{cfg['RUN_DATE']}"
    if not os.path.exists(cfg['PATH_OUTPUT']):
        os.mkdir(cfg['PATH_OUTPUT'])
    if not os.path.exists(cfg['PATH_OUTPUT_CATALOGS']):
        os.mkdir(cfg['PATH_OUTPUT_CATALOGS'])

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
                        train_flow_logger.log_info_stream(f"Batch size {bs}")
                        train_flow_logger.log_info_stream(f"Number blocks {nb}")
                        train_flow_logger.log_info_stream(f"Number hidden {nh}")
                        train_flow_logger.log_info_stream(f"Weight decay {wd}")
                        train_flow_logger.log_info_stream(f"Learning rate {lr}")
                        main(
                            cfg=cfg,
                            learning_rate=lr,
                            weight_decay=wd,
                            number_hidden=nh,
                            number_blocks=nb,
                            batch_size=bs,
                            logger=train_flow_logger
                        )
