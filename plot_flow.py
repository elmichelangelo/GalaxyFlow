import pandas as pd
from Handler import *
import os

path_data = "/project/ls-gruen/users/patrick.gebhardt/data/gaNdalF_paper_catalogs"
path_data_master_cat = "/project/ls-gruen/users/patrick.gebhardt/data/gaNdalF"
filename_flw_balrog = "2024-10-02_10-37_balrog_flw_Test_sample.pkl"
filename_flw_gandalf = "2024-10-02_10-37_gandalf_flw_Test_sample.pkl"
filename_master_cat = "Y3_mastercat_02_05_21.h5"
path_save_plots = "../../Output/gaNdalF_paper"

df_balrog_flw = pd.read_pickle(f"{path_data}/{filename_flw_balrog}")
df_gandalf_flw = pd.read_pickle(f"{path_data}/{filename_flw_gandalf}")

print(f"Length of Balrog objects: {len(df_balrog_flw)}")
print(f"Length of gaNdalF objects: {len(df_gandalf_flw)}")

plot_compare_seaborn(
    data_frame_generated=df_gandalf_flw,
    data_frame_true=df_balrog_flw,
    dict_delta=None,
    epoch=None,
    title=f"color-color plot",
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
    save_name=f"{path_save_plots}/color_color.png",
    ranges=[
        [-0.5, 1.5],  # r-i
        [-0.5, 1.5],  # i-z
        [18, 24.5],  # mag r
        [18, 24.5],  # mag i
        [18, 24.5],  # mag z
        [2, 100],  # snr
        [-0.5, 5],  # size ratio
        [10, 80],  # weight
        [0, 3.5]  # T
    ]
)