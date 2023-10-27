from astropy.table import Table
from helper_functions import luptize_deep_kids
import pandas as pd
import numpy as np
import fitsio
import os
import sys

path = os.path.abspath(sys.path[0])

path_data = f"{path}/../Data/skills_v07D7p1_LF_321_kidsPhotometry_everything_col_flag_shear_m283m283_rot_0.fits"
path_survey_conditions = f"{path}/../Data/noise_selec_combined.csv"
path_save = f"{path}/../Data/kids_training_catalog_lup.pkl"
path_save_small = f"{path}/../Data/kids_training_catalog_lup_small.pkl"

col_kids = [
    "tile_label",
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
]

col_output = [
    "FLUX_GAAP_u",
    "FLUX_GAAP_g",
    "FLUX_GAAP_r",
    "FLUX_GAAP_i",
    "FLUX_GAAP_Z",
    "FLUX_GAAP_Y",
    "FLUX_GAAP_J",
    "FLUX_GAAP_H",
    "FLUX_GAAP_Ks",
    "FLUXERR_GAAP_u",
    "FLUXERR_GAAP_g",
    "FLUXERR_GAAP_r",
    "FLUXERR_GAAP_i",
    "FLUXERR_GAAP_Z",
    "FLUXERR_GAAP_Y",
    "FLUXERR_GAAP_J",
    "FLUXERR_GAAP_H",
    "FLUXERR_GAAP_Ks",
    "FLUX_AUTO",
    "FLUXERR_AUTO",
]

col_survey_cond = [
    "label",
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
    "rms_Ks",
]

kids_data = Table(fitsio.read(path_data))
df_kids_data = pd.DataFrame()
for i, col in enumerate(col_kids+col_output):
    print(i, col)
    df_kids_data[col] = kids_data[col]
df_kids_data = df_kids_data.rename(columns={'tile_label': 'label'})
print(f'Length of deep field catalog: {len(df_kids_data)}')

df_survey_cond_all = pd.read_csv(path_survey_conditions)
df_survey_cond_needed = df_survey_cond_all[col_survey_cond]
print(df_survey_cond_needed)

print('Merging catalogs...')
print('Merging survey catalog and kids on label => df_merged')
df_merged = pd.merge(df_kids_data, df_survey_cond_needed, on='label', how="left")
print(f'Length of merged catalog: {len(df_merged)}')
print(df_merged.isnull().sum())

lst_flux_2_lupt = [
    "FLUX_GAAP_u",
    "FLUX_GAAP_g",
    "FLUX_GAAP_r",
    "FLUX_GAAP_i",
    "FLUX_GAAP_Z",
    "FLUX_GAAP_Y",
    "FLUX_GAAP_J",
    "FLUX_GAAP_H",
    "FLUX_GAAP_Ks"
]

lst_fluxerr_2_lupt = [
    "FLUXERR_GAAP_u",
    "FLUXERR_GAAP_g",
    "FLUXERR_GAAP_r",
    "FLUXERR_GAAP_i",
    "FLUXERR_GAAP_Z",
    "FLUXERR_GAAP_Y",
    "FLUXERR_GAAP_J",
    "FLUXERR_GAAP_H",
    "FLUXERR_GAAP_Ks"
]

lst_bins = [
    "u",
    "g",
    "r",
    "i",
    "Z",
    "Y",
    "J",
    "H",
    "Ks",
]

arr_lup = luptize_deep_kids(np.array(df_merged[lst_flux_2_lupt]), bins=lst_bins)[0]
arr_lup_err = luptize_deep_kids(np.array(df_merged[lst_fluxerr_2_lupt]), bins=lst_bins)[0]

lst_luptize = [
    "luptize_u",
    "luptize_g",
    "luptize_r",
    "luptize_i",
    "luptize_Z",
    "luptize_Y",
    "luptize_J",
    "luptize_H",
    "luptize_Ks",
]

lst_luptize_err = [
    "luptize_err_u",
    "luptize_err_g",
    "luptize_err_r",
    "luptize_err_i",
    "luptize_err_Z",
    "luptize_err_Y",
    "luptize_err_J",
    "luptize_err_H",
    "luptize_err_Ks",
]

for idx_lup, lup in enumerate(lst_luptize):
    df_merged[lup] = arr_lup[:, idx_lup]

for idx_lup_err, lup_err in enumerate(lst_luptize_err):
    df_merged[lup_err] = arr_lup_err[:, idx_lup_err]

df_merged_small = df_merged.sample(n=250000)

df_merged.to_pickle(path_save)
df_merged_small.to_pickle(path_save_small)
