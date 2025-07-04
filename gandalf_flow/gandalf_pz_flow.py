import jax.numpy as jnp
import matplotlib.pyplot as plt
from pzflow import Flow
import pandas as pd
import joblib
import numpy as np

plt.rcParams["figure.facecolor"] = "white"

data_path = "/Volumes/elmichelangelo_external_ssd_1/Data/20250628_balrog_train_18412370.pkl"
transformers_path = "/Volumes/elmichelangelo_external_ssd_1/Data/transformers/20250628_MinMaxScalers_nf.pkl"


def inplace_transform_nf(df, cols, scalers, columns_log1p):
    # log1p anwenden, wo nötig
    for col in cols:
        if col in columns_log1p:
            df[col] = np.log1p(df[col])
    # Skalierung anwenden
    for col in cols:
        scale = scalers[col].scale_[0]
        min_ = scalers[col].min_[0]
        df[col] = (df[col] - min_) / scale


df_data = pd.read_pickle(data_path)
scalers_nf = joblib.load(transformers_path)

INPUT_COLS = [
            "BDF_LUPT_DERED_CALIB_R",
            "BDF_LUPT_DERED_CALIB_I",
            "BDF_LUPT_DERED_CALIB_Z",
            "BDF_LUPT_ERR_DERED_CALIB_R",
            "BDF_LUPT_ERR_DERED_CALIB_I",
            "BDF_LUPT_ERR_DERED_CALIB_Z",
            "Color BDF LUPT U-G",
            "Color BDF LUPT G-R",
            "Color BDF LUPT R-I",
            "Color BDF LUPT I-Z",
            "Color BDF LUPT Z-J",
            "Color BDF LUPT J-H",
            "Color BDF LUPT H-K",
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

COLUMNS_LOG1P = [
            'unsheared/snr',
            'unsheared/size_ratio',
            'unsheared/T',
            'unsheared/lupt_err_r',
            'unsheared/lupt_err_i',
            'unsheared/lupt_err_z',
            'BDF_LUPT_ERR_DERED_CALIB_R',
            'BDF_LUPT_ERR_DERED_CALIB_I',
            'BDF_LUPT_ERR_DERED_CALIB_Z'
        ]

inplace_transform_nf(df_data, INPUT_COLS, scalers_nf, COLUMNS_LOG1P)
inplace_transform_nf(df_data, OUTPUT_COLS, scalers_nf, COLUMNS_LOG1P)

flow = Flow(data_columns=OUTPUT_COLS, conditional_columns=INPUT_COLS)

losses = flow.train(df_data, epochs=20, batch_size=32768, verbose=True)

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.show()