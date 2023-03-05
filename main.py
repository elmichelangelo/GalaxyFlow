from galaxyflow.training import TrainFlow
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(__file__))

plt.rcParams["figure.figsize"] = (16, 9)


def main(path_train_data,
         path_output,
         col_label_flow,
         col_output_flow,
         plot_test,
         show_plot,
         save_plot,
         save_nn,
         learning_rate,
         number_hidden,
         number_blocks,
         epochs,
         device,
         activation_function,
         batch_size,
         valid_batch_size,
         selected_scaler):
    """"""

    train_flow = TrainFlow(
        path_train_data=path_train_data,
        path_output=path_output,
        col_output_flow=col_output_flow,
        col_label_flow=col_label_flow,
        plot_test=plot_test,
        show_plot=show_plot,
        save_plot=save_plot,
        save_nn=save_nn,
        learning_rate=learning_rate,
        number_hidden=number_hidden,
        number_blocks=number_blocks,
        epochs=epochs,
        device=device,
        activation_function=activation_function,
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        selected_scaler=selected_scaler
    )

    train_flow.run_training()


if __name__ == '__main__':
    path = os.path.abspath(sys.path[1])
    lst_label_flow = [
        "BDF_MAG_DERED_CALIB_R",
        "BDF_MAG_DERED_CALIB_I",
        "BDF_MAG_DERED_CALIB_Z",
        "Color Mag U-G",
        "Color Mag G-R",
        "Color Mag R-I",
        "Color Mag I-Z",
        "Color Mag Z-J",
        "Color Mag J-H",
        "Color Mag H-KS",
        "BDF_T",
        "BDF_G",
        "FWHM_WMEAN_R",
        "FWHM_WMEAN_I",
        "FWHM_WMEAN_Z",
        "AIRMASS_WMEAN_R",
        "AIRMASS_WMEAN_I",
        "AIRMASS_WMEAN_Z",
        "MAGLIM_R",
        "MAGLIM_I",
        "MAGLIM_Z",
        "EBV_SFD98"
    ]

    lst_output_flow = [
        "unsheared/mag_r",
        "unsheared/mag_i",
        "unsheared/mag_z",
        "unsheared/snr",
        "unsheared/size_ratio",
        "unsheared/flags",
        "unsheared/T",
        "detected"
    ]
    main(
        path_train_data=f"{path}/Data/analytical_conditional_balrog_data_MAG_250000.pkl",  # Balrog_2_data_MAG_250000.pkl
        path_output=f"{path}/Output",
        col_output_flow=lst_output_flow,
        col_label_flow=lst_label_flow,
        plot_test=True,
        show_plot=False,
        save_plot=True,
        save_nn=True,
        learning_rate=1E-6,
        number_hidden=64,
        number_blocks=3,
        epochs=150,
        device="cpu",
        activation_function="tanh",
        batch_size=64,
        valid_batch_size=64,
        selected_scaler="MaxAbsScaler"
    )
