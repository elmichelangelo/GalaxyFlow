import ray
from ray import tune
from ray import air
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torch.cuda
from Handler.helper_functions import get_os
from galaxyflow.training import TrainFlow
import argparse
import matplotlib.pyplot as plt
import sys
import yaml
import os
sys.path.append(os.path.dirname(__file__))
plt.rcParams["figure.figsize"] = (16, 9)


def main(
        path_train_data,
        size_training_dataset,
        size_validation_dataset,
        size_test_dataset,
        path_output,
        plot_test,
        show_plot,
        save_plot,
        save_nn,
        learning_rate,
        weight_decay,
        number_hidden,
        number_blocks,
        epochs,
        device,
        activation_function,
        batch_size,
        valid_batch_size,
        selected_scaler,
        reproducible,
        col_output_flow,
        col_label_flow,
        lst_replace_values,
        lst_replace_transform_cols,
        lst_fill_na,
        apply_fill_na,
        apply_cuts,
        run_hyperparameter_tuning=True,
        run=None):
    """"""

    train_flow = TrainFlow(
        path_train_data=path_train_data,
        size_training_dataset=size_training_dataset,
        size_validation_dataset=size_validation_dataset,
        size_test_dataset=size_test_dataset,
        path_output=path_output,
        col_output_flow=col_output_flow,
        col_label_flow=col_label_flow,
        plot_test=plot_test,
        show_plot=show_plot,
        save_plot=save_plot,
        save_nn=save_nn,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        number_hidden=number_hidden,
        number_blocks=number_blocks,
        epochs=epochs,
        device=device,
        activation_function=activation_function,
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        selected_scaler=selected_scaler,
        do_loss_plot=True,
        do_color_color_plot=True,
        do_residual_plot=True,
        do_chain_plot=True,
        do_mean_plot=True,
        do_std_plot=True,
        do_flags_plot=True,
        do_detected_plot=True,
        run_hyperparameter_tuning=run_hyperparameter_tuning,
        lst_replace_values=lst_replace_values,
        lst_replace_transform_cols=lst_replace_transform_cols,
        lst_fill_na=lst_fill_na,
        run=run,
        reproducible=reproducible,
        apply_fill_na=apply_fill_na,
        apply_cuts=apply_cuts
    )

    if run_hyperparameter_tuning is True:
        ray.init()
        search_space = {
            "learning_rate": tune.grid_search(learning_rate),
            "number_hidden": tune.grid_search(number_hidden),
            "number_blocks": tune.grid_search(number_blocks),
            "batch_size": tune.grid_search(batch_size),
            "weight_decay": tune.grid_search(weight_decay),
        }
        trainable_with_resources = tune.with_resources(train_flow.hyperparameter_tuning, {"cpu": 5})
        tuner = tune.Tuner(
            trainable_with_resources,
            run_config=air.RunConfig(local_dir=path_output, name="ray_tune_1"),
            tune_config=tune.TuneConfig(scheduler=ASHAScheduler(metric="loss", mode="min")),
            param_space=search_space
        )
        results = tuner.fit()
        # dfs = {result.log_dir: result.metrics_dataframe for result in results}
        # ax = None
        # for d in dfs.values():
        #     ax = d.loss.plot(ax=ax, legend=False)
        #     ax.set_xlabel("Epochs")
        #     ax.set_ylabel("Loss")
        #     plt.show()
        best_result = results.get_best_result()
        print("Best result config: {}".format(best_result.config))
        print("Best result metrics: {}".format(best_result.metrics))
        ray.shutdown()
    else:
        train_flow.run_training()


if __name__ == '__main__':
    if get_os() == "Mac":
        config_file_name = "mac.cfg"
    elif get_os() == "Windows":
        config_file_name = "windows.cfg"
    else:
        raise "OS Error"

    path = os.path.abspath(sys.path[1])
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
        cfg = yaml.load(fp, Loader=yaml.Loader)

    if args.mode is None:
        args.mode = cfg["MODE"]
        mode = args.mode
    else:
        mode = args.mode
        cfg["MODE"] = mode

    runs = cfg["RUNS"]
    batch_size = cfg["BATCH_SIZE"]
    scaler = cfg["SCALER"]
    number_hidden = cfg["NUMBER_HIDDEN"]
    number_blocks = cfg["NUMBER_BLOCKS"]
    learning_rate = cfg["LEARNING_RATE"]
    weight_decay = cfg["WEIGHT_DECAY"]

    if not isinstance(runs, list):
        runs = [runs]
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

    if mode == "hyperparameter_tuning":
        for sc in scaler:
            main(
                path_train_data=f"{path}/Data/{cfg['DATA_FILE_NAME']}",
                size_training_dataset=cfg["SIZE_TRAINING_DATA"],
                size_validation_dataset=cfg["SIZE_VALIDATION_DATA"],
                size_test_dataset=cfg["SIZE_TEST_DATA"],
                path_output=f"{path}/Output",
                plot_test=cfg["PLOT_TEST"],
                show_plot=cfg["SHOW_PLOT"],
                save_plot=cfg["SAVE_PLOT"],
                save_nn=cfg["SAVE_NN"],
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                number_hidden=number_hidden,
                number_blocks=number_blocks,
                epochs=cfg["EPOCHS"],
                device=cfg["DEVICE"],
                activation_function=cfg["ACTIVATION_FUNCTION"],
                batch_size=batch_size,
                valid_batch_size=cfg["VALIDATION_BATCH_SIZE"],
                selected_scaler=sc,
                run_hyperparameter_tuning=True,
                run=1,
                col_output_flow=cfg["OUTPUT_COLS"],
                col_label_flow=cfg["INPUT_COLS"],
                reproducible=cfg["REPRODUCIBLE"],
                lst_replace_transform_cols=cfg["TRANSFORM_COLS"],
                lst_replace_values=cfg["REPLACE_VALUES"],
                lst_fill_na=cfg["FILL_NA"],
                apply_fill_na=cfg["APPLY_FILL_NA"],
                apply_cuts=cfg["APPLY_CUTS"]
            )


            # main(
            #     path_train_data=f"{path}/Data/{cfg['DATA_FILE_NAME']}",
            #     path_output=f"{path}/Output",
            #     plot_test=cfg["PLOT_TEST"],
            #     show_plot=cfg["SHOW_PLOT"],
            #     save_plot=cfg["SAVE_PLOT"],
            #     save_nn=cfg["SAVE_NN"],
            #     learning_rate=learning_rate,
            #     number_hidden=number_hidden,
            #     number_blocks=number_blocks,
            #     epochs=cfg["EPOCHS"],
            #     device=cfg["DEVICE"],
            #     activation_function=cfg["ACTIVATION_FUNCTION"],
            #     batch_size=batch_size,
            #     valid_batch_size=cfg["VALIDATION_BATCH_SIZE"],
            #     selected_scaler=scaler,
            #     run_hyperparameter_tuning=True,
            #     reproducible=False,
            #     col_output_flow=cfg["OUTPUT_COLS"],
            #     col_label_flow=cfg["INPUT_COLS"],
            #     weight_decay=weight_decay,
            #     run=1
            # )
    elif mode == "train_flow":
        for run in runs:
            for lr in learning_rate:
                for wd in weight_decay:
                    for nh in number_hidden:
                        for nb in number_blocks:
                            for bs in batch_size:
                                for sc in scaler:
                                    main(
                                        path_train_data=f"{path}/Data/{cfg['DATA_FILE_NAME']}",
                                        size_training_dataset=cfg["SIZE_TRAINING_DATA"],
                                        size_validation_dataset=cfg["SIZE_VALIDATION_DATA"],
                                        size_test_dataset=cfg["SIZE_TEST_DATA"],
                                        path_output=f"{path}/Output",
                                        plot_test=cfg["PLOT_TEST"],
                                        show_plot=cfg["SHOW_PLOT"],
                                        save_plot=cfg["SAVE_PLOT"],
                                        save_nn=cfg["SAVE_NN"],
                                        learning_rate=lr,
                                        weight_decay=wd,
                                        number_hidden=nh,
                                        number_blocks=nb,
                                        epochs=cfg["EPOCHS"],
                                        device=cfg["DEVICE"],
                                        activation_function=cfg["ACTIVATION_FUNCTION"],
                                        batch_size=bs,
                                        valid_batch_size=cfg["VALIDATION_BATCH_SIZE"],
                                        selected_scaler=sc,
                                        run_hyperparameter_tuning=False,
                                        run=run,
                                        col_output_flow=cfg["OUTPUT_COLS"],
                                        col_label_flow=cfg["INPUT_COLS"],
                                        reproducible=cfg["REPRODUCIBLE"],
                                        lst_replace_transform_cols=cfg["TRANSFORM_COLS"],
                                        lst_replace_values=cfg["REPLACE_VALUES"],
                                        lst_fill_na=cfg["FILL_NA"],
                                        apply_fill_na=cfg["APPLY_FILL_NA"],
                                        apply_cuts=cfg["APPLY_CUTS"]
                                    )
    else:
        raise TypeError("Wrong Mode!")
