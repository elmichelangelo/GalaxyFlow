import pandas as pd
from Handler import *
import os


def replace_nan(data_frame, cols, default_values):
    """
    Replace all NaN values in a specified column of a DataFrame with a default value.

    Parameters:
    df (pandas.DataFrame): The DataFrame to modify.
    column_name (lst): The name of the column to process.
    default_value (lst): The value to replace NaNs with.

    Returns:
    pandas.DataFrame: The DataFrame with NaNs replaced in the specified column.
    """
    for idx, col in enumerate(cols):
        data_frame[col] = data_frame[col].fillna(default_values[idx])
    return data_frame


def apply_cuts(data_frame, path_master_cat):
    """"""
    data_frame = unsheared_object_cuts(data_frame=data_frame)
    data_frame = flag_cuts(data_frame=data_frame)
    data_frame = unsheared_shear_cuts(data_frame=data_frame)
    data_frame = binary_cut(data_frame=data_frame)
    # if self.cfg['MASK_CUT_FUNCTION'] == "HEALPY":
    data_frame = mask_cut_healpy(
        data_frame=data_frame,
        master=path_master_cat
    )
    # elif self.cfg['MASK_CUT_FUNCTION'] == "ASTROPY":
    #     # Todo there is a bug here, I cutout to many galaxies
    #     data_frame = mask_cut(
    #         data_frame=data_frame,
    #         master=f"{self.cfg['PATH_DATA']}/{self.cfg['FILENAME_MASTER_CAT']}"
    #     )
    # else:
    #     print("No mask cut function defined!!!")
    #     exit()
    data_frame = unsheared_mag_cut(data_frame=data_frame)
    return data_frame


path_data = "/project/ls-gruen/users/patrick.gebhardt/data/gaNdalF_paper_catalogs"
path_data_master_cat = "/project/ls-gruen/users/patrick.gebhardt/data/gaNdalF"
filename_flw_balrog = "2024-10-02_10-37_balrog_flw_Test_sample.pkl"
filename_flw_gandalf = "2024-10-02_10-37_gandalf_flw_Test_sample.pkl"
filename_master_cat = "Y3_mastercat_02_05_21.h5"
path_save_plots = "../../Output/gaNdalF_paper"

df_balrog_flw = pd.read_pickle(f"{path_data}/{filename_flw_balrog}")
df_gandalf_flw = pd.read_pickle(f"{path_data}/{filename_flw_gandalf}")

df_gandalf_flw_cut = apply_cuts(df_gandalf_flw, f"{path_data_master_cat}/{filename_master_cat}")
df_balrog_flw_cut = apply_cuts(df_balrog_flw)

print(f"Length of Balrog objects: {len(df_balrog_flw)}")
print(f"Length of gaNdalF objects: {len(df_gandalf_flw)}")

columns = [
    "unsheared/mag_r",
    "unsheared/mag_i",
    "unsheared/mag_z",
    "Color unsheared MAG r-i",
    "Color unsheared MAG i-z",
    "unsheared/snr",
    "unsheared/size_ratio",
    "unsheared/weight",
    "unsheared/T"
]

lst_col_nan = [
    "unsheared/mag_r",
    "unsheared/mag_i",
    "unsheared/mag_z",
    "unsheared/snr",
    "unsheared/size_ratio",
    "unsheared/weight",
    "unsheared/T",
]

lst_nan = [
    df_balrog_flw["unsheared/mag_r"].max(),
    df_balrog_flw["unsheared/mag_i"].max(),
    df_balrog_flw["unsheared/mag_z"].max(),
    df_balrog_flw["unsheared/snr"].max(),
    df_balrog_flw["unsheared/size_ratio"].max(),
    df_balrog_flw["unsheared/weight"].max(),
    df_balrog_flw["unsheared/T"].max(),
]

df_balrog_flw = df_balrog_flw[columns]
df_gandalf_flw = df_gandalf_flw[columns]

df_gandalf_flw = replace_nan(df_gandalf_flw, lst_col_nan, lst_nan)

df_gandalf_flw["Color unsheared MAG r-i"] = df_gandalf_flw["unsheared/mag_r"] - df_gandalf_flw["unsheared/mag_i"]
df_gandalf_flw["Color unsheared MAG i-z"] = df_gandalf_flw["unsheared/mag_i"] - df_gandalf_flw["unsheared/mag_z"]

df_gandalf_flw_sub = df_gandalf_flw.sample(n=int(1E6))
df_balrog_flw_sub = df_balrog_flw.sample(n=int(1E6))

plot_compare_corner(
    data_frame_generated=df_gandalf_flw,
    data_frame_true=df_balrog_flw,
    dict_delta=None,
    epoch=None,
    title=f"Compare Measured Galaxy Properties Balrog-gaNdalF",
    columns=[
        "Color unsheared MAG r-i",
        "Color unsheared MAG i-z",
        "unsheared/mag_r",
        "unsheared/mag_i",
        "unsheared/mag_z",
        "unsheared/snr",
        "unsheared/size_ratio",
        "unsheared/weight",
        "unsheared/T"
    ],
    labels=[
        "r-i",
        "i-z",
        "mag r",
        "mag i",
        "mag z",
        "snr",
        "size ratio",
        "weight",
        "T"
    ],
    show_plot=False,
    save_plot=True,
    save_name=f"{path_save_plots}/compare_measured_galaxy_properties.png",
    ranges=[
        [-0.5, 1.5],
        [-0.5, 1.5],
        [18, 24.5],
        [18, 24.5],
        [18, 24.5],
        [2, 100],
        [-0.5, 5],
        [10, 80],
        [0, 3.5]
    ]
)

plot_compare_corner(
    data_frame_generated=df_gandalf_flw_cut,
    data_frame_true=df_balrog_flw_cut,
    dict_delta=None,
    epoch=None,
    title=f"Compare MCAL Measured Galaxy Properties Balrog-gaNdalF",
    columns=[
        "Color unsheared MAG r-i",
        "Color unsheared MAG i-z",
        "unsheared/mag_r",
        "unsheared/mag_i",
        "unsheared/mag_z",
        "unsheared/snr",
        "unsheared/size_ratio",
        "unsheared/weight",
        "unsheared/T"
    ],
    labels=[
        "r-i",
        "i-z",
        "mag r",
        "mag i",
        "mag z",
        "snr",
        "size ratio",
        "weight",
        "T"
    ],
    show_plot=False,
    save_plot=True,
    save_name=f"{path_save_plots}/compare_mcal_measured_galaxy_properties.png",
    ranges=[
        [-0.5, 1.5],
        [-0.5, 1.5],
        [18, 24.5],
        [18, 24.5],
        [18, 24.5],
        [2, 100],
        [-0.5, 5],
        [10, 80],
        [0, 3.5]
    ]
)


# plot_compare_seaborn(
#     data_frame_generated=df_gandalf_flw_sub,
#     data_frame_true=df_balrog_flw_sub,
#     dict_delta=None,
#     epoch=None,
#     title=f"Compare Measured Galaxy Properties Balrog-gaNdalF",
#     columns=[
#         "Color unsheared MAG r-i",
#         "Color unsheared MAG i-z",
#         "unsheared/mag_r",
#         "unsheared/mag_i",
#         "unsheared/mag_z",
#         "unsheared/snr",
#         "unsheared/size_ratio",
#         "unsheared/weight",
#         "unsheared/T"
#     ],
#     labels=[
#         "r-i",
#         "i-z",
#         "mag r",
#         "mag i",
#         "mag z",
#         "snr",
#         "size ratio",
#         "weight",
#         "T"
#     ],
#     show_plot=False,
#     save_plot=True,
#     save_name=f"{path_save_plots}/compare_measured_galaxy_properties.png",
#     ranges=[
#         [-0.5, 1.5],  # r-i
#         [-0.5, 1.5],  # i-z
#         [18, 24.5],  # mag r
#         [18, 24.5],  # mag i
#         [18, 24.5],  # mag z
#         [2, 100],  # snr
#         [-0.5, 5],  # size ratio
#         [10, 80],  # weight
#         [0, 3.5]  # T
#     ]
# )
#
# plot_compare_seaborn(
#     data_frame_generated=df_gandalf_flw_sub_cut,
#     data_frame_true=df_balrog_flw_sub_cut,
#     dict_delta=None,
#     epoch=None,
#     title=f"Compare MCAL Measured Galaxy Properties Balrog-gaNdalF",
#     columns=[
#         "Color unsheared MAG r-i",
#         "Color unsheared MAG i-z",
#         "unsheared/mag_r",
#         "unsheared/mag_i",
#         "unsheared/mag_z",
#         "unsheared/snr",
#         "unsheared/size_ratio",
#         "unsheared/weight",
#         "unsheared/T"
#     ],
#     labels=[
#         "r-i",
#         "i-z",
#         "mag r",
#         "mag i",
#         "mag z",
#         "snr",
#         "size ratio",
#         "weight",
#         "T"
#     ],
#     show_plot=False,
#     save_plot=True,
#     save_name=f"{path_save_plots}/compare_mcal_measured_galaxy_properties.png",
#     ranges=[
#         [-0.5, 1.5],  # r-i
#         [-0.5, 1.5],  # i-z
#         [18, 24.5],  # mag r
#         [18, 24.5],  # mag i
#         [18, 24.5],  # mag z
#         [2, 100],  # snr
#         [-0.5, 5],  # size ratio
#         [10, 80],  # weight
#         [0, 3.5]  # T
#     ]
# )