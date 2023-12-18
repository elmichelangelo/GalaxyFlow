import healpy as hp
import os
import sys

import matplotlib.pyplot as plt
from astropy.table import Table
import h5py
import fitsio
import pandas as pd
import numpy as np
import seaborn as sns

def read_catalogs(path_metacal, path_detection, path_deep_field, path_survey, metacal_cols,
                  detection_cols,
                  deep_field_cols, survey_cols, nside, plot_healpix=True, show_plot=True, save_plot=False):
    """"""
    df_mcal = None
    df_detect = None
    df_deep_field = None
    df_survey = None

    if path_metacal is not None:
        metacal_data = h5py.File(path_metacal, 'r')
        df_mcal = pd.DataFrame()
        for i, col in enumerate(metacal_cols + ["unsheared/extended_class_sof", "unsheared/flags_gold"]):
            df_mcal[col] = np.array(metacal_data['catalog/' + col]).byteswap().newbyteorder("<")
        print('Length of mcal catalog: {}'.format(len(df_mcal)))
        print("Apply mcal cuts")
        mcal_cuts = (df_mcal["unsheared/extended_class_sof"] >= 0) & (df_mcal["unsheared/flags_gold"] < 2)
        df_mcal = df_mcal[mcal_cuts]
        df_mcal = df_mcal[metacal_cols]
        print('Length of mcal catalog after applying cuts: {}'.format(len(df_mcal)))
        df_mcal = df_mcal.rename(columns={'unsheared/bal_id': 'bal_id'})
        df_mcal = df_mcal.rename(columns={'unsheared/coadd_object_id': 'COADD_OBJECT_ID'})
        for i, col in enumerate(df_mcal.columns):
            print(i, col)
        print(df_mcal.isnull().sum())

    if path_detection is not None:
        detection_data = Table(fitsio.read(path_detection).byteswap().newbyteorder())  # , columns=detection_cols
        df_detect = pd.DataFrame()
        lst_cut_col = ["match_flag_1.5_asec", "flags_foreground", "flags_badregions", "flags_footprint"]
        for i, col in enumerate(detection_cols + lst_cut_col):
            # print(i, col)
            df_detect[col] = detection_data[col]
        print('Length of detection catalog: {}'.format(len(df_detect)))
        print("Apply detection cuts")
        detect_cuts = (df_detect["match_flag_1.5_asec"] < 2) & \
                      (df_detect["flags_foreground"] == 0) & \
                      (df_detect["flags_badregions"] < 2) & \
                      (df_detect["flags_footprint"] == 1)
        df_detect = df_detect[detect_cuts]
        df_detect = df_detect[detection_cols]
        print('Length of detection catalog after applying cuts: {}'.format(len(df_detect)))
        for i, col in enumerate(df_detect.keys()):
            print(i, col)
        df_detect = df_detect.rename(columns={'true_id': 'ID'})
        df_detect[f"HPIX_{nside}"] = DeclRaToIndex(np.array(df_detect["true_dec"]), np.array(df_detect["true_ra"]), nside)
        print(df_detect.isnull().sum())

        if plot_healpix is True:
            arr_hpix = df_detect[f"HPIX_{nside}"].to_numpy()
            arr_flux = df_detect["ID"].to_numpy()
            npix = hp.nside2npix(nside)
            hpxmap = np.zeros(npix, dtype=np.float)
            for idx, pix in enumerate(arr_hpix):
                hpxmap[pix] = arr_flux[idx]
            hp.mollview(
                hpxmap,
                norm="hist",
                nest=True
            )
            if show_plot is True:
                plt.show()

    if path_deep_field is not None:
        deep_field_data = Table(fitsio.read(path_deep_field, columns=deep_field_cols).byteswap().newbyteorder())
        df_deep_field = pd.DataFrame()
        for i, col in enumerate(deep_field_cols):
            print(i, col)
            df_deep_field[col] = deep_field_data[col]
        print('Length of deep field catalog: {}'.format(len(df_deep_field)))
        print(df_deep_field.isnull().sum())

    if path_survey is not None:
        df_survey = pd.DataFrame()
        for idx, file in enumerate(os.listdir(path_survey)):
            if "fits" in file:
                survey_data = Table(fitsio.read(f"{path_survey}/{file}", columns=survey_cols))  # .byteswap().newbyteorder())
                df_survey_tmp = pd.DataFrame()
                for i, col in enumerate(survey_cols):
                    print(i, col)
                    df_survey_tmp[col] = survey_data[col]
                if idx == 0:
                    df_survey = df_survey_tmp
                else:
                    df_survey = pd.concat([df_survey, df_survey_tmp], ignore_index=True)
                print(df_survey_tmp.shape)
        print('Length of survey catalog: {}'.format(len(df_survey)))
        print(df_survey.isnull().sum())
        print(df_survey.shape)

        if plot_healpix is True:
            arr_hpix = df_survey[f"HPIX_{nside}"].to_numpy()
            arr_flux = df_survey[f"AIRMASS_WMEAN_R"].to_numpy()
            npix = hp.nside2npix(nside)
            hpxmap = np.zeros(npix, dtype=np.float)
            for idx, pix in enumerate(arr_hpix):
                hpxmap[pix] = arr_flux[idx]
            hp.mollview(
                hpxmap,
                norm="hist",
                nest=True)
            if show_plot is True:
                plt.show()
    return df_mcal, df_deep_field, df_detect, df_survey


def merge_catalogs(metacal=None, deep_field=None, detection=None, survey=None):
    """"""
    print('Merging catalogs...')

    print('Merging metacal and detection on bal_id => mcal_detect')
    df_merged = pd.merge(detection, metacal, on='bal_id', how="left")
    print('Length of merged mcal_detect catalog: {}'.format(len(df_merged)))
    print(df_merged.isnull().sum())

    print('Merging mcal_detect and deep field on ID => mcal_detect_df')
    df_merged = pd.merge(df_merged, deep_field, on='ID', how="left")
    print('Length of merged mcal_detect_df catalog: {}'.format(len(df_merged)))
    print(df_merged.isnull().sum())

    print('Merging mcal_detect_df and survey on HPIX_4096 => mcal_detect_df_survey')
    df_merged = pd.merge(df_merged, survey, on='HPIX_4096', how="left")
    print('Length of merged mcal_detect_df_survey catalog: {}'.format(len(df_merged)))
    print(df_merged.isnull().sum())

    print('Cut mcal_detect_df_survey catalog so that AIRMASS_WMEAN_R is not null')
    df_merged = df_merged[pd.notnull(df_merged["AIRMASS_WMEAN_R"])]
    print('Length of merged mcal_detect_df_survey catalog: {}'.format(len(df_merged)))
    print(df_merged.isnull().sum())
    return df_merged


def load_healpix(path2file, hp_show=False, nest=True, partial=False, field=None):
    """
    Function to load fits datasets
    Returns:

    """
    if field is None:
        hp_map = hp.read_map(path2file, nest=nest, partial=partial)
    else:
        hp_map = hp.read_map(path2file, nest=nest, partial=partial, field=field)
    if hp_show is True:
        hp_map_show = hp_map
        if field is not None:
            hp_map_show = hp_map[1]
        hp.mollview(
            hp_map_show,
            norm="hist",
            nest=nest
        )
        hp.graticule()
        plt.show()
    return hp_map


def match_skybrite_2_footprint(path2footprint, path2skybrite, hp_show=False,
                               nest_footprint=True, nest_skybrite=True,
                               partial_footprint=False, partial_skybrite=True, field_footprint=None,
                               field_skybrite=None):
    """
    Main function to run_date
    Returns:

    """

    sky_in_footprint = hp_map_skybrite[:, hp_map_footprint != hp.UNSEEN]
    good_indices = sky_in_footprint[0, :].astype(int)
    return np.column_stack((good_indices, sky_in_footprint[1]))


def IndexToDeclRa(index, NSIDE):
    theta, phi = hp.pixelfunc.pix2ang(NSIDE, index, nest=True)
    return -np.degrees(theta - np.pi / 2.), np.degrees(np.pi * 2. - phi)


def DeclRaToIndex(decl, RA, NSIDE):
    return hp.pixelfunc.ang2pix(NSIDE, np.radians(-decl + 90.), np.radians(360. + RA), nest=True).astype(int)


def write_data_2_file(df_generated_data, save_path, number_of_sources):
    """"""
    df_generated_data.to_pickle(f"{save_path}{number_of_sources}.pkl")


def main(path_metacal, path_detection, path_deep_field, path_survey, path_save, metacal_cols,
         detection_cols, deep_field_cols, survey_cols, nside):
    """"""

    df_mcal, df_deep_field, df_detect, df_survey = read_catalogs(
        path_metacal=path_metacal,
        path_detection=path_detection,
        path_deep_field=path_deep_field,
        path_survey=path_survey,
        metacal_cols=metacal_cols,
        detection_cols=detection_cols,
        deep_field_cols=deep_field_cols,
        survey_cols=survey_cols,
        nside=nside
    )

    if df_mcal is not None and df_detect is not None and df_deep_field is not None and df_survey is not None:
        df_merged = merge_catalogs(metacal=df_mcal, detection=df_detect, deep_field=df_deep_field, survey=df_survey)
        write_data_2_file(df_generated_data=df_merged, save_path=path_save, number_of_sources=len(df_merged))


if __name__ == "__main__":
    path = os.path.abspath(sys.path[0])
    path_data = f"{path}/../Data"

    NSIDE = 4096

    other_metacal_cols = [
        'unsheared/coadd_object_id',
        'unsheared/ra',
        'unsheared/dec',
        'unsheared/snr',
        'unsheared/size_ratio',
        'unsheared/flags',
        'unsheared/bal_id',
        'unsheared/T']

    main(
        path_metacal=f"{path_data}/balrog_mcal_stack-y3v02-0-riz-noNB-mcal_y3-merged_v1.2.h5",  #
        path_detection=f"{path_data}/balrog_detection_catalog_sof_y3-merged_v1.2.fits",  # ,
        path_deep_field=f"{path_data}/deep_field_cat.fits",  # ,
        path_survey=f"{path_data}/sct2",  # survey_conditions.fits
        path_save=f"{path_data}/mcal_detect_df_survey_",
        metacal_cols=other_metacal_cols + ['unsheared/flux_{}'.format(i) for i in 'irz'] + ['unsheared/flux_err_{}'.format(i) for i in 'irz'],
        detection_cols=[
            'bal_id',
            'true_id',
            'true_detected',
            'true_ra',
            'true_dec',
        ],
        deep_field_cols=[
            "ID",
            "RA",
            "DEC",
            "BDF_T",
            "BDF_G_0",
            "BDF_G_1",
            "BDF_FLUX_DERED_CALIB_U",
            "BDF_FLUX_DERED_CALIB_G",
            "BDF_FLUX_DERED_CALIB_R",
            "BDF_FLUX_DERED_CALIB_I",
            "BDF_FLUX_DERED_CALIB_Z",
            "BDF_FLUX_DERED_CALIB_J",
            "BDF_FLUX_DERED_CALIB_H",
            "BDF_FLUX_DERED_CALIB_KS",
            "BDF_FLUX_ERR_DERED_CALIB_R",
            "BDF_FLUX_ERR_DERED_CALIB_I",
            "BDF_FLUX_ERR_DERED_CALIB_Z",
        ],
        survey_cols=[
            f"HPIX_{NSIDE}",
            "AIRMASS_WMEAN_R",
            "AIRMASS_WMEAN_I",
            "AIRMASS_WMEAN_Z",
            "FWHM_WMEAN_R",
            "FWHM_WMEAN_I",
            "FWHM_WMEAN_Z",
            "MAGLIM_R",
            "MAGLIM_I",
            "MAGLIM_Z",
            "EBV_SFD98"
        ],
        nside=NSIDE
    )
