import jax.numpy as jnp
import matplotlib.pyplot as plt
from pzflow import Flow
import pandas as pd
import joblib
import numpy as np
import optax

plt.rcParams["figure.facecolor"] = "white"

data_path = "/Volumes/elmichelangelo_external_ssd_1/Data/20250721_balrog_train_7875515_nf.pkl"
transformers_path = "/Volumes/elmichelangelo_external_ssd_1/Data/transformers/20250721_StandardScalers_nf.pkl"


def inplace_transform_nf(df, cols, scalers):

    for col in cols:
        scale = scalers[col].scale_[0]
        mean = scalers[col].mean_[0]
        df[col] = (df[col] - mean) / scale
    return df


df_data = pd.read_pickle(data_path)
scalers_nf = joblib.load(transformers_path)

INPUT_COLS = [
            "BDF_LUPT_DERED_CALIB_R",
            "BDF_LUPT_DERED_CALIB_I",
            "BDF_LUPT_DERED_CALIB_Z",
            "BDF_LUPT_ERR_DERED_CALIB_R",
            "BDF_LUPT_ERR_DERED_CALIB_I",
            "BDF_LUPT_ERR_DERED_CALIB_Z",
            "Color BDF LUPT R-I",
            "Color BDF LUPT I-Z",
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

OUTPUT_COLS = [
            "unsheared/lupt_r",
            "unsheared/lupt_i",
            "unsheared/lupt_z",
            "unsheared/lupt_err_r",
            "unsheared/lupt_err_i",
            "unsheared/lupt_err_z",
            "unsheared/e_1",
            "unsheared/e_2",
            "unsheared/snr",
            "unsheared/size_ratio",
            "unsheared/T",
        ]

NF_COLUMNS_OF_INTEREST = [
            'BDF_LUPT_DERED_CALIB_R',
            'BDF_LUPT_DERED_CALIB_I',
            'BDF_LUPT_DERED_CALIB_Z',
            'BDF_LUPT_ERR_DERED_CALIB_R',
            'BDF_LUPT_ERR_DERED_CALIB_I',
            'BDF_LUPT_ERR_DERED_CALIB_Z',
            'BDF_G',
            'BDF_T',
            'Color BDF LUPT R-I',
            'Color BDF LUPT I-Z',
            'AIRMASS_WMEAN_R',
            'AIRMASS_WMEAN_I',
            'AIRMASS_WMEAN_Z',
            'FWHM_WMEAN_R',
            'FWHM_WMEAN_I',
            'FWHM_WMEAN_Z',
            'MAGLIM_R',
            'MAGLIM_I',
            'MAGLIM_Z',
            'EBV_SFD98',
            'unsheared/snr',
            'unsheared/size_ratio',
            'unsheared/T',
            'unsheared/e_1',
            'unsheared/e_2',
            'unsheared/lupt_r',
            'unsheared/lupt_i',
            'unsheared/lupt_z',
            'unsheared/lupt_err_r',
            'unsheared/lupt_err_i',
            'unsheared/lupt_err_z'
        ]

df_data_transformed = inplace_transform_nf(df_data.copy(), NF_COLUMNS_OF_INTEREST, scalers_nf)

flow = Flow(data_columns=OUTPUT_COLS, conditional_columns=INPUT_COLS)
opt = optax.adam(learning_rate=0.00001)

losses = flow.train(df_data, epochs=20, batch_size=32768, verbose=True, optimizer=opt)

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.show()