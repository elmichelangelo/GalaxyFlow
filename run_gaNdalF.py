from datetime import datetime
from Handler import *
from gandalf import gaNdalF
import argparse
import matplotlib.pyplot as plt
import sys
import yaml
import os
import gc
import pandas as pd
import logging

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
    run_gandalf_logger.log_info_stream(f"start main function")

    gandalf = gaNdalF(gandalf_logger=run_gandalf_logger, cfg=cfg)
    df_balrog = gandalf.galaxies.run_dataset

    run_gandalf_logger.log_info_stream(f"Len of sampled data: {len(df_balrog)}")

    run_gandalf_logger.log_info_stream(f"Length sample dataset: {len(df_balrog)}")
    run_gandalf_logger.log_info_stream("Balrog start")
    run_gandalf_logger.log_info_stream(df_balrog.isna().sum())

    if cfg['CLASSF_GALAXIES']:
        run_gandalf_logger.log_info_stream("running classifier")
        df_gandalf = gandalf.run_classifier(data_frame=df_balrog)
    else:
        df_gandalf = df_balrog.copy()

    run_gandalf_logger.log_info_stream("Balrog after classifier")
    run_gandalf_logger.log_info_stream(df_balrog.isna().sum())

    run_gandalf_logger.log_info_stream("gaNdalF after classifier")
    run_gandalf_logger.log_info_stream(df_gandalf.isna().sum())

    run_gandalf_logger.log_info_stream(cfg["SAVE_CLF_DATA"])

    if cfg["SAVE_CLF_DATA"] is True:
        run_gandalf_logger.log_info_stream(f"{cfg['RUN_DATE']}_balrog_clf_sample.pkl")
        gandalf.save_data(
            data_frame=df_balrog,
            file_name=f"{cfg['RUN_DATE']}_balrog_clf_sample.pkl",
            protocol=5,
            tmp_samples=False
        )

        gandalf.save_data(
            data_frame=df_gandalf,
            file_name=f"{cfg['RUN_DATE']}_gandalf_clf_sample.pkl",
            protocol=5,
            tmp_samples=False
        )

    df_balrog_detected = df_balrog[df_balrog["detected"] == 1].copy()
    df_gandalf_detected = df_gandalf[df_gandalf["detected"] == 1].copy()

    run_gandalf_logger.log_info_stream(f"length of detected objects in gandalf: {len(df_gandalf_detected)}")
    run_gandalf_logger.log_info_stream(f"length of detected objects in balrog: {len(df_balrog_detected)}")

    if cfg['PLOT_RUN']:
        gandalf.plot_classf_data(
            df_balrog=df_balrog,
            df_gandalf=df_gandalf
        )

    if cfg['EMULATE_GALAXIES']:
        df_gandalf_detected = gandalf.run_emulator(
            data_frame=df_gandalf_detected,
        )
    else:
        df_gandalf = df_gandalf_detected

    run_gandalf_logger.log_info_stream("Balrog after normalizing flow")
    run_gandalf_logger.log_info_stream(df_balrog_detected.isna().sum())

    run_gandalf_logger.log_info_stream("gaNdalF after normalizing flow")
    run_gandalf_logger.log_info_stream(df_gandalf_detected.isna().sum())

    if cfg["SAVE_FLW_DATA"] is True:
        print(f"{cfg['RUN_DATE']}_balrog_flw_sample.pkl")
        gandalf.save_data(
            data_frame=df_balrog_detected,
            file_name=f"{cfg['RUN_DATE']}_balrog_flw_sample.pkl",
            protocol=5,
            tmp_samples=False
        )

        gandalf.save_data(
            data_frame=df_gandalf_detected,
            file_name=f"{cfg['RUN_DATE']}_gandalf_flw_sample.pkl",
            protocol=5,
            tmp_samples=False
        )

    if cfg["BOOTSTRAP"] is True:
        df_balrog_cut = df_balrog_detected.copy()
        df_gandalf_cut = df_gandalf_detected.copy()

        df_balrog_cut = gandalf.apply_cuts(df_balrog_cut)
        df_gandalf_cut = gandalf.apply_cuts(df_gandalf_cut)

        # gandalf.save_data(
        #     data_frame=df_balrog_cut,
        #     file_name=f"{cfg['RUN_DATE']}_balrog_{cfg['DATASET_TYPE']}_samples_{len(df_balrog_cut)}.h5",
        #     tmp_samples=False
        # )
        print("Saving .h5 file...")
        gandalf.save_data(
            data_frame=df_gandalf_cut,
            file_name=f"{cfg['RUN_DATE']}_gandalf_{cfg['DATASET_TYPE']}_samples_{len(df_gandalf_cut)}.h5",
            tmp_samples=False
        )
        print(".h5 file saved.")

        # print(f"Number of detected samples: {df_gandalf_detected}")
        # print(f"Number of mcal samples: {df_gandalf_cut}")
        # print(f"Number of total samples: {df_gandalf}")
        total_number_of_samples = cfg['NUMBER_SAMPLES']
        # del gandalf
        df_balrog = df_gandalf = df_balrog_cut = df_gandalf_cut = df_balrog_detected = df_gandalf_detected = None
        gc.collect()
        return

    else:
        if cfg['PLOT_RUN']:
            gandalf.plot_data_flow(
                df_gandalf=df_gandalf_detected,
                df_balrog=df_balrog_detected,
                mcal=''
            )

            # gandalf.plot_data_flow(
            #     df_gandalf=df_gandalf_cut,
            #     df_balrog=df_balrog_cut,
            #     mcal='mcal_'
            # )

        # del df_balrog, df_gandalf
        # gc.collect()
        #
        # df_gandalf_samples = load_tmp_data(
        #     cfg=cfg,
        #     file_name=f"{cfg['FILENAME_GANDALF_RUN']}_gandalf_tmp.pkl"
        # )
        # df_balrog_samples = load_tmp_data(
        #     cfg=cfg,
        #     file_name=f"{cfg['FILENAME_GANDALF_RUN']}_balrog_tmp.pkl"
        # )
        #
        # df_gandalf_samples = pd.concat([df_gandalf_samples, df_gandalf_cut], ignore_index=True)
        # df_balrog_samples = pd.concat([df_balrog_samples, df_balrog_cut], ignore_index=True)
        #
        # total_number_of_samples = len(df_gandalf_samples)
        # run_number += 1
        #
        # print(f"Actual number of samples: {total_number_of_samples}")
        #
        # gandalf.save_data(
        #     data_frame=df_gandalf_samples,
        #     file_name=f"{cfg['FILENAME_GANDALF_RUN']}_gandalf_tmp.pkl",
        #     tmp_samples=True
        # )
        #
        # gandalf.save_data(
        #     data_frame=df_balrog_samples,
        #     file_name=f"{cfg['FILENAME_GANDALF_RUN']}_balrog_tmp.pkl",
        #     tmp_samples=True
        # )
        #
        # del df_gandalf_cut, df_balrog_cut  # , df_gandalf_samples, df_balrog_samples
        # gc.collect()

    sys.exit()

    while total_number_of_samples < cfg['NUMBER_SAMPLES']:
        # cfg['RUN_NUMBER'] = run_number
        gandalf = gaNdalF(cfg=cfg)

        df_balrog = gandalf.galaxies.run_dataset

        print(f"Length sample dataset: {len(df_balrog)}")
        print("Balrog start")
        print("############################################")
        print(df_balrog.isna().sum())
        print("############################################")
        
        if cfg['CLASSF_GALAXIES']:
            print("running classifier")
            df_balrog, df_gandalf = gandalf.run_classifier(data_frame=df_balrog)
        else:
            df_gandalf = df_balrog.copy()
            
        print("Balrog after classifier")
        print("############################################")
        print(df_balrog.isna().sum())
        print("############################################")
        
        print("gaNdalF after classifier")
        print("############################################")
        print(df_gandalf.isna().sum())
        print("############################################")

        print(cfg["SAVE_CLF_DATA"])

        if cfg["SAVE_CLF_DATA"] is True:
            print(f"{cfg['RUN_DATE']}_balrog_clf_{cfg['DATASET_TYPE']}_sample_w_non_calib.pkl")
            gandalf.save_data(
                data_frame=df_balrog,
                file_name=f"{cfg['RUN_DATE']}_balrog_clf_{cfg['DATASET_TYPE']}_sample_w_non_calib.pkl",
                protocol=5,
                tmp_samples=False
            )

            gandalf.save_data(
                data_frame=df_gandalf,
                file_name=f"{cfg['RUN_DATE']}_gandalf_clf_{cfg['DATASET_TYPE']}_sample_w_non_calib.pkl",
                protocol=5,
                tmp_samples=False
            )

        df_balrog_detected = df_balrog[df_balrog["detected"] == 1].copy()
        df_gandalf_detected = df_gandalf[df_gandalf["detected"] == 1].copy()

        # print("Balrog after classifier detected")
        # print("############################################")
        # print(df_balrog_detected.isna().sum())
        # print("############################################")
        #
        # print("gaNdalF after classifier detected")
        # print("############################################")
        # print(df_gandalf_detected.isna().sum())
        # print("############################################")

        if cfg["BOOTSTRAP"] is False:
            if cfg['PLOT_RUN']:
                gandalf.plot_classf_data(
                    df_balrog=df_balrog,
                    df_gandalf=df_gandalf
                )

        if cfg['EMULATE_GALAXIES']:
            df_balrog_detected, df_gandalf_detected = gandalf.run_emulator(
                df_balrog_detected,
                df_gandalf_detected
            )
        else:
            df_gandalf = df_gandalf_detected

        # for col in cfg[f'SAMPLE_COLUMNS']:
        #     df_gandalf_cut = sample_columns(df_balrog=df_balrog_cut, df_gandalf=df_gandalf_cut, column_name=col)
        #
        # df_balrog_cut.loc[:, "unsheared/e"] = np.sqrt(df_balrog_cut["unsheared/e_1"] ** 2 + df_balrog_cut["unsheared/e_2"] ** 2)
        #
        # for col in cfg[f'SELECT_COLUMNS']:
        #     df_gandalf_cut = select_columns(df_balrog_cut, df_gandalf_cut, column_name=col)
        #
        
        if "unsheared/flux_r" not in df_gandalf.keys():
            try:
                df_gandalf_detected.loc[:, "unsheared/flux_r"] = mag2flux(df_gandalf_detected["unsheared/mag_r"])
            except Exception as e:
                print(f"An error occurred: {e}")
                raise  # Re-raise the exception if necessary
        if "unsheared/flux_r" not in df_balrog.keys():
            try:
                df_balrog_detected.loc[:, "unsheared/flux_r"] = mag2flux(df_balrog_detected["unsheared/mag_r"])
            except Exception as e:
                print(f"An error occurred: {e}")
                raise  # Re-raise the exception if necessary

        # del df_balrog_detected, df_gandalf_detected
        # gc.collect()

        if cfg["SAVE_FLW_DATA"] is True:
            print(f"{cfg['RUN_DATE']}_balrog_flw_{cfg['DATASET_TYPE']}_sample.pkl")
            gandalf.save_data(
                data_frame=df_balrog_detected,
                file_name=f"{cfg['RUN_DATE']}_balrog_flw_{cfg['DATASET_TYPE']}_sample_non_calib.pkl",
                protocol=5,
                tmp_samples=False
            )

            gandalf.save_data(
                data_frame=df_gandalf_detected,
                file_name=f"{cfg['RUN_DATE']}_gandalf_flw_{cfg['DATASET_TYPE']}_sample_non_calib.pkl",
                protocol=5,
                tmp_samples=False
            )

        # sys.exit()



        if cfg["BOOTSTRAP"] is True:
            df_balrog_cut = df_balrog_detected.copy()
            df_gandalf_cut = df_gandalf_detected.copy()

            df_balrog_cut = gandalf.apply_cuts(df_balrog_cut)
            df_gandalf_cut = gandalf.apply_cuts(df_gandalf_cut)

            # gandalf.save_data(
            #     data_frame=df_balrog_cut,
            #     file_name=f"{cfg['RUN_DATE']}_balrog_{cfg['DATASET_TYPE']}_samples_{len(df_balrog_cut)}.h5",
            #     tmp_samples=False
            # )
            print("Saving .h5 file...")
            gandalf.save_data(
                data_frame=df_gandalf_cut,
                file_name=f"{cfg['RUN_DATE']}_gandalf_{cfg['DATASET_TYPE']}_samples_{len(df_gandalf_cut)}.h5",
                tmp_samples=False
            )
            print(".h5 file saved.")

            # print(f"Number of detected samples: {df_gandalf_detected}")
            # print(f"Number of mcal samples: {df_gandalf_cut}")
            # print(f"Number of total samples: {df_gandalf}")
            total_number_of_samples = cfg['NUMBER_SAMPLES']
            # del gandalf
            df_balrog = df_gandalf = df_balrog_cut = df_gandalf_cut = df_balrog_detected = df_gandalf_detected = None
            gc.collect()
            return

        else:
            if cfg['PLOT_RUN']:
                gandalf.plot_data_flow(
                    df_gandalf=df_gandalf_detected,
                    df_balrog=df_balrog_detected,
                    mcal=''
                )

                # gandalf.plot_data_flow(
                #     df_gandalf=df_gandalf_cut,
                #     df_balrog=df_balrog_cut,
                #     mcal='mcal_'
                # )

            del df_balrog, df_gandalf
            gc.collect()

            df_gandalf_samples = load_tmp_data(
                cfg=cfg,
                file_name=f"{cfg['FILENAME_GANDALF_RUN']}_gandalf_tmp.pkl"
            )
            df_balrog_samples = load_tmp_data(
                cfg=cfg,
                file_name=f"{cfg['FILENAME_GANDALF_RUN']}_balrog_tmp.pkl"
            )

            df_gandalf_samples = pd.concat([df_gandalf_samples, df_gandalf_cut], ignore_index=True)
            df_balrog_samples = pd.concat([df_balrog_samples, df_balrog_cut], ignore_index=True)

            total_number_of_samples = len(df_gandalf_samples)
            run_number += 1

            print(f"Actual number of samples: {total_number_of_samples}")

            gandalf.save_data(
                data_frame=df_gandalf_samples,
                file_name=f"{cfg['FILENAME_GANDALF_RUN']}_gandalf_tmp.pkl",
                tmp_samples=True
            )

            gandalf.save_data(
                data_frame=df_balrog_samples,
                file_name=f"{cfg['FILENAME_GANDALF_RUN']}_balrog_tmp.pkl",
                tmp_samples=True
            )

            del df_gandalf_cut, df_balrog_cut  #, df_gandalf_samples, df_balrog_samples
            gc.collect()

            # if bootsrap is True:
            #     cfg['NUMBER_SAMPLES'] = -1
    if cfg["BOOTSTRAP"] is False:
        cfg['BOOTSTRAP_NUMBER'] = run_number + 1

        df_gandalf_samples = load_tmp_data(
            cfg=cfg,
            file_name=f"{cfg['FILENAME_GANDALF_RUN']}_gandalf_tmp.pkl"
        )
        df_balrog_samples = load_tmp_data(
            cfg=cfg,
            file_name=f"{cfg['FILENAME_GANDALF_RUN']}_balrog_tmp.pkl"
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

        file_name = f"{cfg['FILENAME_GANDALF_RUN']}.h5"
        if cfg["SPATIAL_TEST"] is True:
            file_name = f"{cfg['FILENAME_GANDALF_RUN']}_Spatial_{cfg['SPATIAL_NUMBER']}.h5"
        print("Saving .h5 file...")
        gandalf.save_data(
            data_frame=df_gandalf_samples,
            file_name=file_name,
            tmp_samples=False
        )
        print(".h5 file saved.")
        # os.remove(f"{cfg['PATH_CATALOGS']}/{cfg['FILENAME_GANDALF_RUN']}_gandalf_tmp.pkl")
        # os.remove(f"{cfg['PATH_CATALOGS']}/{cfg['FILENAME_GANDALF_RUN']}_balrog_tmp.pkl")
    return


def make_dirs(config):
    """"""
    config['PATH_PLOTS_FOLDER'] = {}

    if config["BOOTSTRAP"] is False:
        config['PATH_OUTPUT'] = f"{config['PATH_OUTPUT']}/gandalf_run_{config['RUN_DATE']}"
        os.makedirs(config['PATH_OUTPUT'], exist_ok=True)

        config['PATH_CATALOGS'] = f"{config['PATH_OUTPUT']}/catalogs"
        os.makedirs(config['PATH_CATALOGS'], exist_ok=True)

        config['PATH_PLOTS'] = f"{config['PATH_OUTPUT']}/plots"
        os.makedirs(config['PATH_PLOTS'], exist_ok=True)

        for plot in config['PLOTS_RUN']:
            config[f'PATH_PLOTS_FOLDER'][plot.upper()] = f"{config['PATH_PLOTS']}/{plot}"
            os.makedirs(config['PATH_PLOTS_FOLDER'][plot.upper()], exist_ok=True)
    else:
        if config["BOOTSTRAP_NUMBER"] == 1:
            config['PATH_OUTPUT'] = f"{config['PATH_OUTPUT']}/gandalf_run_{config['RUN_DATE']}"
            os.makedirs(config['PATH_OUTPUT'], exist_ok=True)

            config['PATH_BOOTSTRAP'] = os.path.join(config['PATH_OUTPUT'], f"bootstrap_{config['BOOTSTRAP_NUMBER']}")
            os.makedirs(config['PATH_BOOTSTRAP'], exist_ok=True)
            config['PATH_CATALOGS'] = config['PATH_OUTPUT']

            save_path = os.path.join(config['PATH_BOOTSTRAP'], "saved_config.yaml")
            with open(save_path, "w") as f:
                yaml.dump(config, f)

        if config["BOOTSTRAP_NUMBER"] > 1:
            saved_cfg_path = os.path.join(config['PATH_OUTPUT'], "bootstrap_1", "saved_config.yaml")
            if os.path.exists(saved_cfg_path):
                with open(saved_cfg_path, "r") as f:
                    first_cfg = yaml.safe_load(f)
                    config['RUN_DATE'] = first_cfg['RUN_DATE']
                    config['PATH_OUTPUT'] = first_cfg['PATH_OUTPUT']
                    config['PATH_BOOTSTRAP'] = first_cfg['PATH_BOOTSTRAP']
                    config['PATH_CATALOGS'] = first_cfg['PATH_CATALOGS']

    return config

def load_config_and_parser(system_path):
    if get_os() == "Mac":
        print("load MAC config-file")
        config_file_name = "MAC_run_gandalf.cfg"
    elif get_os() == "Linux":
        print("load LMU config-file")
        config_file_name = "LMU_run_gandalf.cfg"
    else:
        print("Undefined operating system")
        sys.exit()

    parser = argparse.ArgumentParser(description='Start gaNdalF')
    parser.add_argument(
        '--config_filename',
        "-cf",
        type=str,
        required=False,
        default=config_file_name,
        help='Name of config file. If not given default.cfg will be used'
    )
    parser.add_argument(
        '--bootstrap',
        "-boot",
        action='store_true',
        help='Enable bootstrapping'
    )
    parser.add_argument(
        '--BOOTSTRAP_NUMBER',
        "-bootn",
        type=int,
        required=False,
        help='Bootstrapping number'
    )
    parser.add_argument(
        '--TOTAL_BOOTSTRAP',
        "-tboot",
        type=int,
        required=False,
        help='Total Number of Bootstraps'
    )

    args = parser.parse_args()

    path_config_file = f"{system_path}/conf/{args.config_filename}"
    with open(path_config_file, 'r') as fp:
        print(f"open {path_config_file}")
        config = yaml.safe_load(fp)

    now = datetime.now()
    config['BOOTSTRAP'] = False
    config['RUN_DATE'] = now.strftime('%Y-%m-%d_%H-%M')

    # nm_emu = ""
    # nm_cls = ""
    #
    # if config["EMULATE_GALAXIES"] is True:
    #     nm_emu = "_Emulated"
    # if config["CLASSF_GALAXIES"] is True:
    #     nm_cls = "Classified_"

    # config['FILENAME_GANDALF_RUN'] = f"{config['RUN_DATE']}_gandalf{nm_emu}_{nm_cls}{config['NUMBER_SAMPLES']}_{config['DATASET_TYPE']}"

    if isinstance(args.BOOTSTRAP_NUMBER, int):
        config['BOOTSTRAP_NUMBER'] = args.BOOTSTRAP_NUMBER
    else:
        config['BOOTSTRAP_NUMBER'] = 1

    if isinstance(args.TOTAL_BOOTSTRAP, int):
        config['TOTAL_BOOTSTRAP'] = args.TOTAL_BOOTSTRAP
    else:
        config['TOTAL_BOOTSTRAP'] = 1

    return config, path_config_file


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    cfg, path_cfg_file = load_config_and_parser(system_path=os.path.abspath(sys.path[-1]))
    cfg = make_dirs(config=cfg)

    log_lvl = logging.INFO
    if cfg["LOGGING_LEVEL"] == "DEBUG":
        log_lvl = logging.DEBUG
    elif cfg["LOGGING_LEVEL"] == "ERROR":
        log_lvl = logging.ERROR

    run_gandalf_logger = LoggerHandler(
        logger_dict={"logger_name": "run gandalf logger",
                     "info_logger": cfg['INFO_LOGGER'],
                     "error_logger": cfg['ERROR_LOGGER'],
                     "debug_logger": cfg['DEBUG_LOGGER'],
                     "stream_logger": cfg['STREAM_LOGGER'],
                     "stream_logging_level": log_lvl},
        log_folder_path=f"{cfg['PATH_OUTPUT']}/"
    )

    run_gandalf_logger.log_info_stream(f"Start gaNdalF at {cfg['RUN_DATE']}")
    run_gandalf_logger.log_info_stream(f"Used config file: {path_cfg_file}")
    run_gandalf_logger.log_info_stream(f"Bootstrapping: {cfg['BOOTSTRAP']}")
    run_gandalf_logger.log_info_stream(f"Bootstrap Number: {cfg['BOOTSTRAP_NUMBER']}")
    run_gandalf_logger.log_info_stream(f"Binary Classifier: {cfg['FILENAME_NN_CLASSF']}")
    run_gandalf_logger.log_info_stream(f"Normalizing Flow: {cfg['FILENAME_NN_FLOW']}")

    main(cfg=cfg)

    run_gandalf_logger.log_info_stream(f"end main function")
    run_gandalf_logger.log_info_stream(f"Done run gaNdalF!")

    run_gandalf_logger.log_info_stream(f"end main function")
    run_gandalf_logger.log_info_stream(f"Done run gaNdalF!")
    sys.exit()
