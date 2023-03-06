# Use from galaxyflow.training import TrainFlow for DES
from galaxyflow.training_kids import TrainFlow
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(__file__))

plt.rcParams["figure.figsize"] = (16, 9)


def main_des(path_train_data,
         path_output,
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
    from galaxyflow.training import TrainFlow

    col_label_flow = [
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

    col_output_flow = [
        "unsheared/mag_r",
        "unsheared/mag_i",
        "unsheared/mag_z",
        "unsheared/snr",
        "unsheared/size_ratio",
        "unsheared/flags",
        "unsheared/T",
        "detected"
    ]

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


def main_kids(path_train_data,
              path_output,
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
              selected_scaler,
              apply_cuts):
    from galaxyflow.training_kids import TrainFlow

    col_label_flow = [
        "axis_ratio_input",
        "Re_input",
        "sersic_n_input",
        "u_input",
        "g_input",
        "r_input",
        "i_input",
        "Z_input",
        "Y_input",
        "J_input",
        "H_input",
        "Ks_input",
        "InputSeeing_u",
        "InputSeeing_g",
        "InputSeeing_r",
        "InputSeeing_i",
        "InputSeeing_Z",
        "InputSeeing_Y",
        "InputSeeing_J",
        "InputSeeing_H",
        "InputSeeing_Ks",
        "InputBeta_u",
        "InputBeta_g",
        "InputBeta_r",
        "InputBeta_i",
        "InputBeta_Z",
        "InputBeta_Y",
        "InputBeta_J",
        "InputBeta_H",
        "InputBeta_Ks",
        "rmsAW_u",
        "rmsAW_g",
        "rmsAW_r",
        "rmsAW_i",
        "rms_Z",
        "rms_Y",
        "rms_J",
        "rms_H",
        "rms_Ks"
    ]

    # col_output_flow = [
    #     "FLUX_GAAP_u",
    #     "FLUX_GAAP_g",
    #     "FLUX_GAAP_r",
    #     "FLUX_GAAP_i",
    #     "FLUX_GAAP_Z",
    #     "FLUX_GAAP_Y",
    #     "FLUX_GAAP_J",
    #     "FLUX_GAAP_H",
    #     "FLUX_GAAP_Ks",
    #     "FLUXERR_GAAP_u",
    #     "FLUXERR_GAAP_g",
    #     "FLUXERR_GAAP_r",
    #     "FLUXERR_GAAP_i",
    #     "FLUXERR_GAAP_Z",
    #     "FLUXERR_GAAP_Y",
    #     "FLUXERR_GAAP_J",
    #     "FLUXERR_GAAP_H",
    #     "FLUXERR_GAAP_Ks",
    #     "FLUX_AUTO",
    #     "FLUXERR_AUTO",
    # ]

    col_output_flow = [
        "luptize_u",
        "luptize_g",
        "luptize_r",
        "luptize_i",
        "luptize_Z",
        "luptize_Y",
        "luptize_J",
        "luptize_H",
        "luptize_Ks",
        "luptize_err_u",
        "luptize_err_g",
        "luptize_err_r",
        "luptize_err_i",
        "luptize_err_Z",
        "luptize_err_Y",
        "luptize_err_J",
        "luptize_err_H",
        "luptize_err_Ks",
        "FLUX_AUTO",
        "FLUXERR_AUTO",
    ]

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
        selected_scaler=selected_scaler,
        apply_cuts=apply_cuts
    )

    train_flow.run_training()


if __name__ == '__main__':
    path = os.path.abspath(sys.path[1])

    main_kids(
        path_train_data=f"{path}/Data/kids_training_catalog_lup.pkl",
        path_output=f"{path}/Output",
        plot_test=True,
        show_plot=False,
        save_plot=True,
        save_nn=True,
        learning_rate=1E-6,
        number_hidden=64,
        number_blocks=3,
        epochs=2,
        device="cpu",
        activation_function="tanh",
        batch_size=64,
        valid_batch_size=64,
        selected_scaler="MaxAbsScaler",
        apply_cuts=True
    )
