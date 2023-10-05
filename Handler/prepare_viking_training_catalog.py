from astropy.table import Table
import fitsio
import pandas as pd


def main():
    """"""
    path_data = "/Users/P.Gebhardt/Development/PhD/data/viking"
    path_output = "/Users/P.Gebhardt/Development/PhD/output/viking/Catalogs"
    path_skills_noise_data = f"{path_data}/noise_selec_combined.csv"
    path_skills_sheared_data = f"{path_data}/skills_v07D7p1_LF_321_kidsPhotometry_everything_col_flag_shear_m283m283_rot_0.fits"

    galaxy_columns = [
        "tile_label",
        "Re_input",
        "e1_input",
        "e2_input",
        "g1_in",
        "g2_in",
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
        "MAG_GAAP_u",
        "MAG_GAAP_g",
        "MAG_GAAP_r",
        "MAG_GAAP_i",
        "MAG_GAAP_Z",
        "MAG_GAAP_Y",
        "MAG_GAAP_J",
        "MAG_GAAP_H",
        "MAG_GAAP_Ks"
    ]
    condition_columns = [
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
        "InputBeta_Ks"
    ]

    df_skills_noise_data = pd.read_csv(path_skills_noise_data, usecols=condition_columns)
    df_skills_sheared_data = Table(fitsio.read(path_skills_sheared_data))[galaxy_columns].to_pandas()
    df_skills_sheared_data = df_skills_sheared_data.rename(columns={'tile_label': 'label'})
    df_merged = pd.merge(df_skills_noise_data, df_skills_sheared_data, on='label', how="right")
    df_merged.to_pickle(f"{path_output}/skills_training_data.pkl")


if __name__ == '__main__':
    main()
