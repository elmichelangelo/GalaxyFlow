from datetime import datetime
from Handler.helper_functions import get_os
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
    gandalf = gaNdalF(cfg=cfg)

    gandalf.run()


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
    cfg['RUN_DATE_RUN'] = now.strftime('%Y-%m-%d_%H-%M')
    cfg['PATH_OUTPUT_RUN'] = f"{cfg['PATH_OUTPUT_RUN']}/run_{cfg['RUN_DATE_RUN']}"
    if not os.path.exists(cfg['PATH_OUTPUT_RUN']):
        os.mkdir(cfg['PATH_OUTPUT_RUN'])

    main(cfg)