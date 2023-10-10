import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from pzflow import Flow
from pzflow.examples import get_twomoons_data
from Handler.data_loader import load_data
from jax.config import config
from Handler.plot_functions import *
config.update("jax_enable_x64", True)

TRANSFORM_COLS = [
    'unsheared/snr',
    'unsheared/T',
    'unsheared/size_ratio',
    'AIRMASS_WMEAN_R',
                 'AIRMASS_WMEAN_I',
                 'AIRMASS_WMEAN_Z',
                 'FWHM_WMEAN_R',
                 'FWHM_WMEAN_I',
                 'FWHM_WMEAN_Z',
                 'MAGLIM_R',
                 'MAGLIM_I',
                 'MAGLIM_Z',
                 'EBV_SFD98'
                 ]

training_data, validation_data, test_data = load_data(
            path_training_data="/Users/P.Gebhardt/Development/PhD/data/gandalf_training_data_odetec_ncuts_rdef_rnan_250000.pkl",
            path_output="/Users/P.Gebhardt/Development/PhD/output/gaNdalF/",
            luminosity_type='MAG',
            selected_scaler='MaxAbsScaler',
            size_training_dataset=.6,
            size_validation_dataset=.2,
            size_test_dataset=.2,
            reproducible=True,
            run=1,
            lst_yj_transform_cols=TRANSFORM_COLS,
            apply_object_cut=False,
            apply_flag_cut=False,
            apply_airmass_cut=False,
            apply_unsheared_mag_cut=False,
            apply_unsheared_shear_cut=False,
            plot_data=False,
            writer=None
        )

train_data = training_data["data frame training data"]
val_data = validation_data["data frame validation data"]
t_data = test_data["data frame test data"]
scaler = test_data["scaler"]

data_cols = [
        'unsheared/mag_r',
      'unsheared/mag_i',
      'unsheared/mag_z',
      'unsheared/mag_err_r',
      'unsheared/mag_err_i',
      'unsheared/mag_err_z',
      'unsheared/snr',
      'unsheared/size_ratio',
      'unsheared/flags',
      'unsheared/T'
]
conditional_cols = [
                 'BDF_MAG_DERED_CALIB_R',
                 'BDF_MAG_DERED_CALIB_I',
                 'BDF_MAG_DERED_CALIB_Z',
                 'BDF_MAG_ERR_DERED_CALIB_R',
                 'BDF_MAG_ERR_DERED_CALIB_I',
                 'BDF_MAG_ERR_DERED_CALIB_Z',
                 'Color BDF MAG U-G',
                 'Color BDF MAG G-R',
                 'Color BDF MAG R-I',
                 'Color BDF MAG I-Z',
                 'Color BDF MAG Z-J',
                 'Color BDF MAG J-H',
                 'Color BDF MAG H-K',
                 'BDF_T',
                 'BDF_G',
                 'FWHM_WMEAN_R',
                 'FWHM_WMEAN_I',
                 'FWHM_WMEAN_Z',
                 'AIRMASS_WMEAN_R',
                 'AIRMASS_WMEAN_I',
                 'AIRMASS_WMEAN_Z',
                 'MAGLIM_R',
                 'MAGLIM_I',
                 'MAGLIM_Z',
    'EBV_SFD98'
    ]
plt.rcParams["figure.facecolor"] = "white"

# data = get_twomoons_data()

flow = Flow(data_columns=data_cols, conditional_columns=conditional_cols)

train_losses, val_losses = flow.train(train_data, val_data, batch_size=1024, epochs=100, verbose=True)

plt.plot(train_losses, label="Training")
plt.plot(val_losses, label="Validation")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

samples = flow.sample(1, conditions=t_data[conditional_cols], seed=0)

df_generated_scaled = test_data[f"data frame test data"].copy()
df_true_scaled = test_data[f"data frame test data"].copy()

df_generated_output = samples
df_generated_scaled[conditional_cols+data_cols] = df_generated_output
generated_rescaled = scaler.inverse_transform(df_generated_scaled)
df_generated = pd.DataFrame(generated_rescaled, columns=df_generated_scaled.keys())

true = scaler.inverse_transform(df_true_scaled)
df_true = pd.DataFrame(true, columns=df_generated_scaled.keys())

img_grid = plot_chain_compare(
                    data_frame_generated=df_generated,
                    data_frame_true=df_true,
                    epoch=100,
                    show_plot=True,
                    save_name=f'/Users/P.Gebhardt/Development/PhD/output/gaNdalF/chainplot_{100}.png',
                    columns=[
                        f"unsheared/mag_r",
                        f"unsheared/mag_i",
                        f"unsheared/mag_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T"
                    ],
                    parameter=[
                        f"mag_r",
                        f"mag_i",
                        f"mag_z",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    extends={
                        f"mag_r": (15, 30),
                        f"mag_i": (15, 30),
                        f"mag_z": (15, 30),
                        "snr": (-2, 4),
                        "size_ratio": (-3.5, 4),
                        "T": (-1.5, 2)
                    },
                    max_ticks=5,
                    shade_alpha=0.8,
                    tick_font_size=12,
                    label_font_size=12
                )
