
from datetime import datetime
import torch.cuda
from Handler.helper_functions import get_os
from galaxyflow.training_vikings import TrainFlow
import argparse
import matplotlib.pyplot as plt
import sys
import yaml
import os
sys.path.append(os.path.dirname(__file__))
plt.rcParams["figure.figsize"] = (16, 9)


def main(
        path_training_data,
        size_training_dataset,
        size_validation_dataset,
        size_test_dataset,
        luminosity_type,
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
        lst_yj_transform_cols,
        lst_fill_na,
        apply_fill_na,
        apply_object_cut,
        apply_flag_cut,
        apply_airmass_cut,
        apply_unsheared_mag_cut,
        apply_unsheared_shear_cut,
        plot_load_data,
        run_hyperparameter_tuning=True,
        run=None):
    """"""

    train_flow = TrainFlow(
        path_training_data=path_training_data,
        size_training_dataset=size_training_dataset,
        size_validation_dataset=size_validation_dataset,
        size_test_dataset=size_test_dataset,
        luminosity_type=luminosity_type,
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
        lst_yj_transform_cols=lst_yj_transform_cols,
        lst_fill_na=lst_fill_na,
        run=run,
        reproducible=reproducible,
        apply_fill_na=apply_fill_na,
        apply_object_cut=apply_object_cut,
        apply_flag_cut=apply_flag_cut,
        apply_airmass_cut=apply_airmass_cut,
        apply_unsheared_mag_cut=apply_unsheared_mag_cut,
        apply_unsheared_shear_cut=apply_unsheared_shear_cut,
        plot_load_data=plot_load_data
    )
    train_flow.run_training()


if __name__ == '__main__':
    config_file_name = "mac_viking.cfg"
    path = os.path.abspath(sys.path[0])
    
    parser = argparse.ArgumentParser(description='Start viking')
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
        cfg = yaml.load(fp, Loader=yaml.Loader)

    # if args.mode is None:
    #     args.mode = cfg["MODE"]
    #     mode = args.mode
    # else:
    #     mode = args.mode
    #     cfg["MODE"] = mode

    runs = cfg["RUNS"]
    now = datetime.now()
    run = now.strftime('%Y-%m-%d_%H-%M')
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

    output_cols = cfg[f"OUTPUT_COLS"]
    input_cols = cfg[f"INPUT_COLS"]

    path_data = cfg[f"PATH_DATA"]
    path_output = cfg[f"PATH_OUTPUT"]

    for lr in learning_rate:
        for wd in weight_decay:
            for nh in number_hidden:
                for nb in number_blocks:
                    for bs in batch_size:
                        for sc in scaler:
                            main(
                                path_training_data=f"{path_data}{cfg['DATA_FILE_NAME']}",
                                size_training_dataset=cfg["SIZE_TRAINING_DATA"],
                                size_validation_dataset=cfg["SIZE_VALIDATION_DATA"],
                                size_test_dataset=cfg["SIZE_TEST_DATA"],
                                luminosity_type=cfg["LUM_TYPE"],
                                path_output=path_output,
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
                                col_output_flow=output_cols,
                                col_label_flow=input_cols,
                                reproducible=cfg["REPRODUCIBLE"],
                                lst_yj_transform_cols=cfg["TRANSFORM_COLS"],
                                lst_replace_values=cfg["REPLACE_VALUES"],
                                lst_fill_na=cfg["FILL_NA"],
                                apply_fill_na=cfg["APPLY_FILL_NA"],
                                apply_object_cut=cfg["APPLY_OBJECT_CUT"],
                                apply_flag_cut=cfg["APPLY_FLAG_CUT"],
                                apply_airmass_cut=cfg["APPLY_AIRMASS_CUT"],
                                apply_unsheared_mag_cut=cfg["APPLY_UNSHEARED_MAG_CUT"],
                                apply_unsheared_shear_cut=cfg["APPLY_UNSHEARED_SHEAR_CUT"],
                                plot_load_data=cfg["PLOT_LOAD_DATA"]
                            )
