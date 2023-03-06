from astropy.table import Table
import pandas as pd
import fitsio
import os
import sys

path = os.path.abspath(sys.path[0])

path_data = f"{path}/../Data/skills_v07D7p1_LF_321_kidsPhotometry_everything_col_flag_shear_m283m283_rot_0.fits"
path_survey_conditions = f"{path}/../Data/noise_selec_combined.csv"

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
print('Length of deep field catalog: {}'.format(len(df_kids_data)))

df_survey_cond_all = pd.read_csv(path_survey_conditions)
df_survey_cond_needed = df_survey_cond_all[col_survey_cond]
print(df_survey_cond_needed)

print('Merging catalogs...')
print('Merging survey catalog and kids on label => df_merged')
df_merged = pd.merge(df_kids_data, df_survey_cond_needed, on='label', how="left")
print('Length of merged catalog: {}'.format(len(df_merged)))
print(df_merged.isnull().sum())
