from datetime import datetime
from Handler import *
from gandalf import gaNdalF
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import yaml
import os
import gc
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(__file__))
plt.rcParams["figure.figsize"] = (16, 9)


def load_tmp_data(cfg, file_name):
    """"""
    if os.path.exists(f"{cfg['PATH_CATALOGS']}/{file_name}"):
        data_frame = pd.read_pickle(f"{cfg['PATH_CATALOGS']}/{file_name}")
        os.remove(f"{cfg['PATH_CATALOGS']}/{file_name}")
    else:
        data_frame = pd.DataFrame()
    if isinstance(data_frame, dict):
        data_frame = pd.DataFrame(data_frame)
    return data_frame


def main(cfg):
    """"""
    cfg = make_dirs(cfg)
    total_number_of_samples = -100
    run_number = 1
    nm_emu = ""
    nm_cls = ""

    if cfg["EMULATE_GALAXIES"] is True:
        nm_emu = "_Emulated"
    if cfg["CLASSF_GALAXIES"] is True:
        nm_cls = "Classified_"

    # if cfg['NUMBER_SAMPLES'] == -1:
    #     cfg['NUMBER_SAMPLES'] = 3031255  # int(3254744 * 0.15)

    cfg['FILENAME_GANDALF_CATALOG'] = f"{cfg['RUN_DATE']}_gandalf{nm_emu}_{nm_cls}{cfg['NUMBER_SAMPLES']}_{cfg['DATASET_TYPE']}"
    bootsrap = False
    while total_number_of_samples < cfg['NUMBER_SAMPLES']:
        if cfg['NUMBER_SAMPLES'] == -1:
            bootsrap = True
        print(f"Run {run_number}")
        cfg['RUN_NUMBER'] = run_number
        gandalf = gaNdalF(cfg=cfg)

        df_balrog = gandalf.galaxies.run_dataset

        print(f"Length sample dataset: {len(df_balrog)}")

        if cfg['CLASSF_GALAXIES']:
            df_balrog, df_gandalf = gandalf.run_classifier(data_frame=df_balrog)
        else:
            df_gandalf = df_balrog.copy()

        df_balrog_detected = df_balrog[df_balrog["detected"] == 1].copy()
        df_gandalf_detected = df_gandalf[df_gandalf["detected"] == 1].copy()

        if cfg['PLOT_RUN']:
            gandalf.plot_classf_data(
                df_balrog=df_balrog,
                df_gandalf=df_gandalf,
                df_balrog_detected=df_balrog_detected,
                df_gandalf_detected=df_gandalf_detected
            )

        if cfg['EMULATE_GALAXIES']:
            df_balrog_detected, df_gandalf_detected = gandalf.run_emulator(
                df_balrog_detected,
                df_gandalf_detected
            )
        else:
            df_gandalf = df_gandalf_detected

        df_balrog_cut = df_balrog_detected.copy()
        df_gandalf_cut = df_gandalf_detected.copy()

        # for col in cfg[f'SAMPLE_COLUMNS']:
        #     df_gandalf_cut = sample_columns(df_balrog=df_balrog_cut, df_gandalf=df_gandalf_cut, column_name=col)
        #
        # df_balrog_cut.loc[:, "unsheared/e"] = np.sqrt(df_balrog_cut["unsheared/e_1"] ** 2 + df_balrog_cut["unsheared/e_2"] ** 2)
        #
        # for col in cfg[f'SELECT_COLUMNS']:
        #     df_gandalf_cut = select_columns(df_balrog_cut, df_gandalf_cut, column_name=col)
        #
        if "unsheared/flux_r" not in df_gandalf.keys():
            df_gandalf_cut.loc[:, "unsheared/flux_r"] = mag2flux(df_gandalf_cut["unsheared/mag_r"])

        if "unsheared/flux_r" not in df_balrog.keys():
            df_balrog_cut.loc[:, "unsheared/flux_r"] = mag2flux(df_balrog_cut["unsheared/mag_r"])

        # del df_balrog_detected, df_gandalf_detected
        gc.collect()

        df_balrog_cut = gandalf.apply_cuts(cfg, df_balrog_cut)
        df_gandalf_cut = gandalf.apply_cuts(cfg, df_gandalf_cut)

        if cfg['PLOT_RUN']:
            gandalf.plot_data_flow(
                df_gandalf=df_gandalf_detected,
                df_balrog=df_balrog_detected,
                mcal=''
            )

            gandalf.plot_data_flow(
                df_gandalf=df_gandalf_cut,
                df_balrog=df_balrog_cut,
                mcal='mcal_'
            )

        del df_balrog, df_gandalf
        gc.collect()

        df_gandalf_samples = load_tmp_data(
            cfg=cfg,
            file_name=f"{cfg['FILENAME_GANDALF_CATALOG']}_gandalf_tmp.pkl"
        )
        df_balrog_samples = load_tmp_data(
            cfg=cfg,
            file_name=f"{cfg['FILENAME_GANDALF_CATALOG']}_balrog_tmp.pkl"
        )

        df_gandalf_samples = pd.concat([df_gandalf_samples, df_gandalf_cut], ignore_index=True)
        df_balrog_samples = pd.concat([df_balrog_samples, df_balrog_cut], ignore_index=True)

        total_number_of_samples = len(df_gandalf_samples)
        run_number += 1

        print(f"Actual number of samples: {total_number_of_samples}")

        gandalf.save_data(
            data_frame=df_gandalf_samples,
            file_name=f"{cfg['FILENAME_GANDALF_CATALOG']}_gandalf_tmp.pkl",
            tmp_samples=True
        )

        gandalf.save_data(
            data_frame=df_balrog_samples,
            file_name=f"{cfg['FILENAME_GANDALF_CATALOG']}_balrog_tmp.pkl",
            tmp_samples=True
        )

        del df_gandalf_cut, df_balrog_cut  #, df_gandalf_samples, df_balrog_samples
        gc.collect()

        if bootsrap is True:
            cfg['NUMBER_SAMPLES'] = -1

    cfg['RUN_NUMBER'] = run_number + 1

    df_gandalf_samples = load_tmp_data(
        cfg=cfg,
        file_name=f"{cfg['FILENAME_GANDALF_CATALOG']}_gandalf_tmp.pkl"
    )
    df_balrog_samples = load_tmp_data(
        cfg=cfg,
        file_name=f"{cfg['FILENAME_GANDALF_CATALOG']}_balrog_tmp.pkl"
    )

    print(f"Total number of samples: {total_number_of_samples}")
    print(f"Number of runs: {run_number - 1}")
    print(df_gandalf_samples)

    # df_gandalf_samples = df_gandalf_samples.sample(n=cfg['NUMBER_SAMPLES'], random_state=None, replace=True)
    # df_balrog_samples = df_balrog_samples.sample(n=cfg['NUMBER_SAMPLES'], random_state=None, replace=True)

    # df_gandalf_samples.loc[:, "true_id"] = df_gandalf_samples["ID"].values
    # df_gandalf_samples = df_gandalf_samples[cfg['SOMPZ_COLS']]

    # plot_compare_corner(
    #     data_frame_generated=df_gandalf_samples,
    #     data_frame_true=df_balrog_samples,
    #     dict_delta=None,
    #     epoch=None,
    #     title=f"Observed Properties gaNdalF compared to Balrog",
    #     show_plot=False,
    #     save_plot=True,
    #     save_name=f"{cfg[f'PATH_PLOTS_FOLDER'][f'MCAL_COLOR_COLOR_PLOT']}/mcal_chainplot_slide_new_{cfg['RUN_NUMBER']}.png",
    #     columns=[
    #         f"unsheared/mag_r",
    #         f"unsheared/mag_i",
    #         f"unsheared/mag_z",
    #         "unsheared/snr",
    #         "unsheared/size_ratio"
    #     ],
    #     labels=[
    #         f"mag r",
    #         f"mag i",
    #         f"mag z",
    #         "snr",
    #         "size_ratio"
    #     ],
    #     ranges=[(17, 25), (17, 25), (17, 25), (-2, 300), (0, 6)]
    # )

    gandalf.plot_data_flow(
        df_gandalf=df_gandalf_samples,
        df_balrog=df_balrog_samples,
        mcal='mcal_'
    )

    file_name = f"{cfg['FILENAME_GANDALF_CATALOG']}.h5"
    if cfg["SPATIAL_TEST"] is True:
        file_name = f"{cfg['FILENAME_GANDALF_CATALOG']}_Spatial_{cfg['SPATIAL_NUMBER']}.h5"

    gandalf.save_data(
        data_frame=df_gandalf_samples,
        file_name=file_name,
        tmp_samples=False
    )
    # os.remove(f"{cfg['PATH_CATALOGS']}/{cfg['FILENAME_GANDALF_CATALOG']}_gandalf_tmp.pkl")
    # os.remove(f"{cfg['PATH_CATALOGS']}/{cfg['FILENAME_GANDALF_CATALOG']}_balrog_tmp.pkl")


def make_dirs(cfg):
    """"""
    cfg['PATH_PLOTS_FOLDER'] = {}
    cfg['PATH_OUTPUT'] = f"{cfg['PATH_OUTPUT']}/gandalf_run_{cfg['RUN_DATE']}"
    # if not os.path.exists(cfg['PATH_OUTPUT']):
    #     os.mkdir(cfg['PATH_OUTPUT'])
    # cfg['PATH_OUTPUT'] = f"{cfg['PATH_OUTPUT']}/{cfg['RUN_NUMBER']}"
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
    parser.add_argument(
        '--spatial',
        type=int,
        required=False,
        help='Run number'
    )

    args = parser.parse_args()

    if isinstance(args.config_filename, list):
        args.config_filename = args.config_filename[0]

    with open(f"{path}/conf/{args.config_filename}", 'r') as fp:
        cfg = yaml.safe_load(fp)

    now = datetime.now()
    cfg['RUN_DATE'] = now.strftime('%Y-%m-%d_%H-%M')
    if args.spatial is not None:
        cfg['SPATIAL_NUMBER'] = args.spatial - 1
    else:
        cfg['SPATIAL_NUMBER'] = 0

    # for i in range(1, 101):
    #     with open(f"{path}/conf/{args.config_filename}", 'r') as fp:
    #         cfg = yaml.safe_load(fp)
    #
    #     cfg['RUN_DATE'] = now.strftime('%Y-%m-%d_%H-%M')
    #     cfg['RUN_NUMBER'] = i
    main(cfg)
