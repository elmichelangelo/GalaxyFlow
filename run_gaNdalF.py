from datetime import datetime
from Handler import get_os
from gandalf import gaNdalF
import argparse
import matplotlib.pyplot as plt
import sys
import yaml
import os
sys.path.append(os.path.dirname(__file__))
plt.rcParams["figure.figsize"] = (16, 9)


def main(cfg):
    """"""

    cfg = make_dirs(cfg)

    for i in range(1):
        print(f"Run {i + 1} of 10")
        cfg['RUN_NUMBER'] = i+1
        gandalf = gaNdalF(cfg=cfg)
        gandalf.run()


def make_dirs(cfg):
    """"""
    cfg['PATH_PLOTS_FOLDER'] = {}
    cfg['PATH_OUTPUT'] = f"{cfg['PATH_OUTPUT']}/gandalf_run_{cfg['RUN_DATE']}"
    cfg['PATH_PLOTS'] = f"{cfg['PATH_OUTPUT']}/{cfg['FOLDER_PLOTS']}"
    cfg['PATH_CATALOGS'] = f"{cfg['PATH_OUTPUT']}/{cfg['FOLDER_CATALOGS']}"
    if not os.path.exists(cfg['PATH_OUTPUT']):
        os.mkdir(cfg['PATH_OUTPUT'])
    if not os.path.exists(cfg['PATH_PLOTS']):
        os.mkdir(cfg['PATH_PLOTS'])
    if not os.path.exists(cfg['PATH_CATALOGS']):
        os.mkdir(cfg['PATH_CATALOGS'])
    for plot in cfg['PLOTS_RUN']:
        cfg[f'PATH_PLOTS_FOLDER'][plot.upper()] = f"{cfg['PATH_PLOTS']}/{plot}"
        if not os.path.exists(cfg[f'PATH_PLOTS_FOLDER'][plot.upper()]):
            os.mkdir(cfg[f'PATH_PLOTS_FOLDER'][plot.upper()])
    return cfg


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
    main(cfg)
