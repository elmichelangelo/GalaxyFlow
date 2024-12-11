# def calc_kullback_leibler(path_data, filename_clf_balrog, filename_clf_gandalf, filename_flw_balrog,
#                           filename_flw_gandalf, path_master_cat, cf_columns, flow_columns):
#     import scipy
#
#     df_balrog_clf = pd.read_pickle(f"{path_data}/{filename_clf_balrog}")
#     df_gandalf_clf = pd.read_pickle(f"{path_data}/{filename_clf_gandalf}")
#
#     df_balrog_clf_detected = df_balrog_clf[df_balrog_clf["detected"] == 1][cf_columns].dropna()
#     df_gandalf_clf_detected = df_gandalf_clf[df_gandalf_clf["detected"] == 1][cf_columns].dropna()
#
#     # Initialize dictionaries to store KL divergence values
#     kl_detected = {}
#     percent_diff_detected = {}
#
#     # Compute KL divergence for each feature
#     for column in cf_columns:
#         data1 = df_balrog_clf_detected[column]
#         data2 = df_gandalf_clf_detected[column]
#
#         # Determine the range and bins
#         min_value = min(data1.min(), data2.min())
#         max_value = max(data1.max(), data2.max())
#         bins = np.linspace(min_value, max_value, 50)  # Adjust the number of bins as needed
#
#         # Compute histograms (probability distributions)
#         hist1, _ = np.histogram(data1, bins=bins, density=True)
#         hist2, _ = np.histogram(data2, bins=bins, density=True)
#
#         # Add a small constant to avoid zeros (which cause issues in KL divergence)
#         hist1 += 1e-10
#         hist2 += 1e-10
#
#         # Normalize the histograms to make them probability distributions
#         hist1 /= hist1.sum()
#         hist2 /= hist2.sum()
#
#         # Compute KL divergence
#         kl_value = scipy.stats.entropy(hist1, hist2)
#         kl_detected[column] = kl_value
#
#         # Calculate percent difference as normalized KL divergence
#         percent_diff_detected[column] = (kl_value / (kl_value + 1)) * 100  # Normalize
#
#     print("KL Divergence and Percent Difference for Detected Objects:")
#     for column in cf_columns:
#         print(f"{column}: KL Divergence = {kl_detected[column]:.4f}, Percent Difference = {percent_diff_detected[column]:.2f}%")
#
#     # Repeat the same process for Not Detected Objects
#     df_balrog_clf_not_detected = df_balrog_clf[df_balrog_clf["detected"] == 0][cf_columns].dropna()
#     df_gandalf_clf_not_detected = df_gandalf_clf[df_gandalf_clf["detected"] == 0][cf_columns].dropna()
#
#     kl_not_detected = {}
#     percent_diff_not_detected = {}
#
#     for column in cf_columns:
#         data1 = df_balrog_clf_not_detected[column]
#         data2 = df_gandalf_clf_not_detected[column]
#
#         min_value = min(data1.min(), data2.min())
#         max_value = max(data1.max(), data2.max())
#         bins = np.linspace(min_value, max_value, 50)
#
#         hist1, _ = np.histogram(data1, bins=bins, density=True)
#         hist2, _ = np.histogram(data2, bins=bins, density=True)
#
#         hist1 += 1e-10
#         hist2 += 1e-10
#
#         hist1 /= hist1.sum()
#         hist2 /= hist2.sum()
#
#         kl_value = scipy.stats.entropy(hist1, hist2)
#         kl_not_detected[column] = kl_value
#         percent_diff_not_detected[column] = (kl_value / (kl_value + 1)) * 100
#
#     print("KL Divergence and Percent Difference for Not Detected Objects:")
#     for column in cf_columns:
#         print(f"{column}: KL Divergence = {kl_not_detected[column]:.4f}, Percent Difference = {percent_diff_not_detected[column]:.2f}%")
#
#     return kl_detected, kl_not_detected, percent_diff_detected, percent_diff_not_detected


def calc_kullback_leibler(df_balrog, df_gandalf, columns):

    # Initialize dictionaries to store KL divergence values
    dict_kl_divergence = {}
    dict_percent_diff = {}

    # Compute KL divergence for each feature
    for column in columns:
        data_balrog = df_balrog[column]
        data_gandalf = df_gandalf[column]

        # Determine the range and bins
        min_value = min(data_balrog.min(), data_gandalf.min())
        max_value = max(data_balrog.max(), data_gandalf.max())
        bins = np.linspace(min_value, max_value, 50)  # Adjust the number of bins as needed

        # Compute histograms (probability distributions)
        hist1, _ = np.histogram(data_balrog, bins=bins, density=True)
        hist2, _ = np.histogram(data_gandalf, bins=bins, density=True)

        # Add a small constant to avoid zeros (which cause issues in KL divergence)
        hist1 += 1e-10
        hist2 += 1e-10

        # Normalize the histograms to make them probability distributions
        hist1 /= hist1.sum()
        hist2 /= hist2.sum()

        # Compute KL divergence
        kl_value = scipy.stats.entropy(hist1, hist2)
        dict_kl_divergence[column] = kl_value

        # Calculate percent difference as normalized KL divergence
        dict_percent_diff[column] = (kl_value / (kl_value + 1)) * 100  # Normalize

    print("KL Divergence and Percent Difference for Detected Objects:")
    for column in columns:
        print(f"{column}: KL Divergence = {dict_kl_divergence[column]:.4f}, Percent Difference = {dict_percent_diff[column]:.2f}%")

    return dict_kl_divergence, dict_percent_diff


def replace_nan(data_frame, cols, default_values):
    """"""
    for idx, col in enumerate(cols):
        data_frame[col] = data_frame[col].fillna(default_values[idx])
    return data_frame


def apply_cuts(data_frame, path_master_cat):
    """"""
    data_frame = unsheared_object_cuts(data_frame=data_frame)
    data_frame = flag_cuts(data_frame=data_frame)
    data_frame = unsheared_shear_cuts(data_frame=data_frame)
    data_frame = binary_cut(data_frame=data_frame)
    data_frame = mask_cut_healpy(
        data_frame=data_frame,
        master=path_master_cat
    )
    data_frame = unsheared_mag_cut(data_frame=data_frame)
    return data_frame

def apply_deep_cuts(path_master_cat, data_frame):
    """"""
    data_frame = flag_cuts(data_frame=data_frame)
    data_frame = mask_cut(
        data_frame=data_frame,
        master=path_master_cat
    )
    return data_frame

def check_idf_flux(data_frame):
    lst_bins = ["r", "i", "z"]
    for mag_bin in lst_bins:
        flux_name = f"unsheared/flux_{mag_bin}"
        mag_name = f"unsheared/mag_{mag_bin}"
        if flux_name not in data_frame.keys():
            data_frame.loc[:, flux_name] = mag2flux(data_frame[mag_name])
    return data_frame

def plot_classifier(cfg, path_master_cat, path_save_plots):
    """"""
    df_balrog_clf = pd.read_pickle(f"{cfg['PATH_DATA']}/{cfg['FILENAME_CLF_BALROG']}")
    df_gandalf_clf = pd.read_pickle(f"{cfg['PATH_DATA']}/{cfg['FILENAME_CLF_GANDALF']}")

    df_balrog_clf_deep_cut = apply_deep_cuts(
        path_master_cat=path_master_cat,
        data_frame=df_balrog_clf
    )
    df_gandalf_clf_deep_cut = apply_deep_cuts(
        path_master_cat=path_master_cat,
        data_frame=df_gandalf_clf
    )

    # print(f"Length of Balrog detected objects: {len(df_balrog_clf[df_balrog_clf['detected'] == 1])}")
    # print(f"Length of Balrog not detected objects: {len(df_balrog_clf[df_balrog_clf['detected'] == 0])}")
    # print(f"Length of gaNdalF detected objects: {len(df_gandalf_clf[df_gandalf_clf['detected'] == 1])}")
    # print(f"Length of gaNdalF not detected objects: {len(df_gandalf_clf[df_gandalf_clf['detected'] == 0])}")
    # print(f"Length of Balrog detected deep cut objects: {len(df_balrog_clf_deep_cut[df_balrog_clf_deep_cut['detected'] == 1])}")
    # print(f"Length of Balrog not detected deep cut objects: {len(df_balrog_clf_deep_cut[df_balrog_clf_deep_cut['detected'] == 0])}")
    # print(f"Length of gaNdalF detected deep cut objects: {len(df_gandalf_clf_deep_cut[df_gandalf_clf_deep_cut['detected'] == 1])}")
    # print(f"Length of gaNdalF not detected deep cut objects: {len(df_gandalf_clf_deep_cut[df_gandalf_clf_deep_cut['detected'] == 0])}")
    # print(f"Length of Balrog detected objects: {len(df_balrog_clf[df_balrog_clf['detected'] == 1])}")
    # print(f"Length of Balrog not detected objects: {len(df_balrog_clf[df_balrog_clf['detected'] == 0])}")
    # print(f"Length of gaNdalF detected objects: {len(df_gandalf_clf[df_gandalf_clf['detected'] == 1])}")
    # print(f"Length of gaNdalF not detected objects: {len(df_gandalf_clf[df_gandalf_clf['detected'] == 0])}")

    if cfg["PLT_FIG_1"] is True:
        plot_multivariate_clf_2(
            df_balrog_detected=df_balrog_clf_deep_cut[df_balrog_clf_deep_cut['detected'] == 1],
            df_gandalf_detected=df_gandalf_clf_deep_cut[df_gandalf_clf_deep_cut['detected'] == 1],
            df_balrog_not_detected=df_balrog_clf_deep_cut[df_balrog_clf_deep_cut['detected'] == 0],
            df_gandalf_not_detected=df_gandalf_clf_deep_cut[df_gandalf_clf_deep_cut['detected'] == 0],
            columns={
                    # "BDF_MAG_DERED_CALIB_R": {
                    #     "label": "BDF Mag R",
                    #     "range": [18.5, 28],
                    #     "position": [0, 0]
                    # },
                    "BDF_MAG_DERED_CALIB_Z": {
                        "label": "BDF Mag Z",
                        "range": [18, 26],
                        "position": [0, 1]
                    },
                    # "BDF_T": {
                    #     "label": "BDF T",
                    #     "range": [-0.25, 1.8],
                    #     "position": [0, 2]
                    # },
                    # "BDF_G": {
                    #     "label": "BDF G",
                    #     "range": [-0.1, 0.9],
                    #     "position": [1, 0]
                    # },
                    # "FWHM_WMEAN_R": {
                    #     "label": "FWHM R",
                    #     "range": [0.7, 1.3],
                    #     "position": [1, 1]
                    # },
                    # "FWHM_WMEAN_I": {
                    #     "label": "FWHM I",
                    #     "range": [0.7, 1.1],
                    #     "position": [1, 2]
                    # },
                    # "FWHM_WMEAN_Z": {
                    #      "label": "FWHM Z",
                    #      "range": [0.6, 1.16],
                    #      "position": [2, 0]
                    #  },
                    #  "AIRMASS_WMEAN_R": {
                    #      "label": "AIRMASS R",
                    #      "range": [0.95, 1.45],
                    #      "position": [2, 1]
                    #  },
                    #  "AIRMASS_WMEAN_I": {
                    #      "label": "AIRMASS I",
                    #      "range": [1, 1.45],
                    #      "position": [2, 2]
                    #  },
                    #  "AIRMASS_WMEAN_Z": {
                    #      "label": "AIRMASS Z",
                    #      "range": [1, 1.4],
                    #      "position": [2, 3]
                    #  },
                    #  "MAGLIM_R": {
                    #      "label": "MAGLIM R",
                    #      "range": [23, 24.8],
                    #      "position": [3, 0]
                    #  },
                    #  "MAGLIM_I": {
                    #      "label": "MAGLIM I",
                    #      "range": [22.4, 24.0],
                    #      "position": [3, 1]
                    #  },
                    #  "MAGLIM_Z": {
                    #      "label": "MAGLIM Z",
                    #      "range": [21.8, 23.2],
                    #      "position": [3, 2]
                    #  },
                    #  "EBV_SFD98": {
                    #      "label": "EBV SFD98",
                    #      "range": [-0.01, 0.10],
                    #      "position": [3, 3]
                    #  }
                },
            cuts=True,
            grid_size=200,
            thresh=0.02,
            show_plot=cfg["SHOW_PLOT"],
            save_plot= cfg["SAVE_PLOT"],
            save_name=f"{path_save_plots}/classifier_multiv_2.png",
            sample_size=5000,
            x_range=(18, 26),
            title=f"Multivariate Comparison of Detection Distributions in gaNdalF and Balrog"
        )

    if cfg["PLT_FIG_2"] is True:
        plot_number_density_fluctuation(
            df_balrog=df_balrog_clf,
            df_gandalf=df_gandalf_clf,
            columns=[
                "BDF_MAG_DERED_CALIB_R",
                "BDF_MAG_DERED_CALIB_I",
                "BDF_MAG_DERED_CALIB_Z",
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
            ],
            labels=[
                "BDF Mag R",
                "BDF Mag I",
                "BDF Mag Z",
                "BDF T",
                "BDF G",
                "FWHM R",
                "FWHM I",
                "FWHM Z",
                "AIRMASS R",
                "AIRMASS I",
                "AIRMASS Z",
                "MAGLIM R",
                "MAGLIM I",
                "MAGLIM Z",
                "EBV SFD98"
            ],
            ranges=[
                [18, 26],
                [18, 26],
                [18, 26],
                [-1, 1.5],
                [-0.1, 0.8],
                [0.8, 1.2],
                [0.7, 1.1],
                [0.7, 1.0],
                [1, 1.4],
                [1, 1.4],
                [1, 1.4],
                [23.5, 24.5],
                [23, 23.75],
                [22, 23],
                [0, 0.05]
            ],
            show_plot=cfg["SHOW_PLOT"],
            save_plot=cfg["SAVE_PLOT"],
            save_name=f"{path_save_plots}/number_density_fluctuation.png",
            title=f"Number Density Fluctuation Analysis of gaNdalF vs. Balrog Detections"
        )

def plot_flow(path_data, filename_flw_balrog, filename_flw_gandalf, path_master_cat, path_save_plots, columns):
    """"""
    df_balrog_flw = pd.read_pickle(f"{path_data}/{filename_flw_balrog}")
    df_gandalf_flw = pd.read_pickle(f"{path_data}/{filename_flw_gandalf}")

    df_gandalf_flw = replace_nan(
        data_frame=df_gandalf_flw,
        cols=[
            "unsheared/mag_r",
            "unsheared/mag_i",
            "unsheared/mag_z",
            "unsheared/snr",
            "unsheared/size_ratio",
            "unsheared/weight",
            "unsheared/T",
        ],
        default_values=[
            df_balrog_flw["unsheared/mag_r"].max(),
            df_balrog_flw["unsheared/mag_i"].max(),
            df_balrog_flw["unsheared/mag_z"].max(),
            df_balrog_flw["unsheared/snr"].max(),
            df_balrog_flw["unsheared/size_ratio"].max(),
            df_balrog_flw["unsheared/weight"].max(),
            df_balrog_flw["unsheared/T"].max(),
        ]
    )

    df_gandalf_flw["Color unsheared MAG r-i"] = df_gandalf_flw["unsheared/mag_r"] - df_gandalf_flw["unsheared/mag_i"]
    df_gandalf_flw["Color unsheared MAG i-z"] = df_gandalf_flw["unsheared/mag_i"] - df_gandalf_flw["unsheared/mag_z"]

    df_balrog_flw = check_idf_flux(df_balrog_flw)
    df_gandalf_flw = check_idf_flux(df_gandalf_flw)

    df_balrog_flw_cut = apply_cuts(df_balrog_flw, path_master_cat)
    df_gandalf_flw_cut = apply_cuts(df_gandalf_flw, path_master_cat)

    print(f"Length of Balrog objects: {len(df_balrog_flw)}")
    print(f"Length of gaNdalF objects: {len(df_gandalf_flw)}")
    print(f"Length of Balrog objects after mag cut: {len(df_balrog_flw_cut)}")
    print(f"Length of gaNdalF objects after mag cut: {len(df_gandalf_flw_cut)}")

    bands = ['r', 'i', 'z']
    conditions = ['AIRMASS_WMEAN', 'MAGLIM', 'FWHM_WMEAN', 'EBV_SFD98']
    residual_properties = ['mag', 'snr', 'size_ratio', 'T', 'weight']

    create_combined_statistics_plot(
        df_gandalf=df_gandalf_flw_cut,
        df_balrog=df_balrog_flw_cut,
        bands=bands,
        conditions=conditions,
        residual_properties=residual_properties,
        save_plot=True,
        path_save_plots=path_save_plots
    )

    # plot_binning_statistics_combined(
    #     df_gandalf=df_gandalf_flw_cut,
    #     df_balrog=df_balrog_flw_cut,
    #     sample_size=10000,
    #     show_plot=False,
    #     save_plot=True,
    #     path_save_plots=path_save_plots
    # )
    # plot_binning_statistics_comparison(
    #     df_gandalf=df_gandalf_flw_cut,
    #     df_balrog=df_balrog_flw_cut,
    #     sample_size=10000,
    #     show_plot=False,
    #     save_plot=True,
    #     path_save_plots=path_save_plots
    # )
    # plot_binning_statistics_properties(
    #     df_gandalf=df_gandalf_flw_cut,
    #     df_balrog=df_balrog_flw_cut,
    #     sample_size=10000,
    #     show_plot=False,
    #     save_plot=True,
    #     path_save_plots=path_save_plots
    # )
    exit()
    plot_binning_statistics(
        df_gandalf=df_gandalf_flw_cut,
        df_balrog=df_balrog_flw_cut,
        conditions=[
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
        ],
        bands=["r", "i", "z"],
        sample_size=100000,
        show_plot=False,
        save_plot=True,
        path_save_plots=path_save_plots
    )

    df_balrog_flw = df_balrog_flw[columns]
    df_gandalf_flw = df_gandalf_flw[columns]

    df_balrog_flw_cut = df_balrog_flw_cut[columns]
    df_gandalf_flw_cut = df_gandalf_flw_cut[columns]

    exit()

    plot_compare_corner(
        data_frame_generated=df_gandalf_flw,
        data_frame_true=df_balrog_flw,
        dict_delta=None,
        epoch=None,
        title=f"Compare Measured Galaxy Properties Balrog-gaNdalF",
        columns=columns,
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
        save_name=f"{path_save_plots}/compare_measured_galaxy_properties_datapoints.png",
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
        columns=columns,
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
        save_name=f"{path_save_plots}/compare_mcal_measured_galaxy_properties_datapoints.png",
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

    plot_balrog_histogram_with_error(
        df_gandalf=df_gandalf_flw,
        df_balrog=df_balrog_flw,
        columns=columns,
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
        ranges=[
            [-0.5, 1.5],  # mag r-i
            [-0.5, 1.5],  # mag i-z
            [18, 24.5],  # mag r
            [18, 24.5],  # mag i
            [18, 24.5],  # mag z
            [2, 100],  # snr
            [-0.5, 5],  # size ratio
            [10, 80],  # weight
            [0, 3.5]  # T
        ],
        binwidths=[
            0.08,  # mag r-i
            0.08,  # mag i-z
            None,  # mag r
            None,  # mag i
            None,  # mag z
            2,  # snr
            0.2,  # size ratio
            2,  # weight
            0.2  # T
        ],
        title="Compare Histogram",
        show_plot=False,
        save_plot=True,
        save_name=f"{path_save_plots}/hist_plot.png"
    )

    plot_balrog_histogram_with_error(
        df_gandalf=df_gandalf_flw_cut,
        df_balrog=df_balrog_flw_cut,
        columns=columns,
        labels=[
            "mag r-i",
            "mag i-z",
            "mag r",
            "mag i",
            "mag z",
            "snr",
            "size ratio",
            "weight",
            "T"
        ],
        ranges=[
            [-0.5, 1.5],  # mag r-i
            [-0.5, 1.5],  # mag i-z
            [18, 24.5],  # mag r
            [18, 24.5],  # mag i
            [18, 24.5],  # mag z
            [2, 100],  # snr
            [-0.5, 5],  # size ratio
            [10, 80],  # weight
            [0, 3.5]  # T
        ],
        binwidths=[
            0.08,  # mag r-i
            0.08,  # mag i-z
            None,  # mag r
            None,  # mag i
            None,  # mag z
            2,  # snr
            0.2,  # size ratio
            2,  # weight
            0.2  # T
        ],
        title="Compare MCAL Histogram",
        show_plot=False,
        save_plot=True,
        save_name=f"{path_save_plots}/mcal_hist_plot.png"
    )


def load_zmean_and_files(path_zmean_folder, path_data_folder, path_gandalf_mean):
    """
    Load zmean.pkl and all histogram files from the specified folders.

    Args:
        path_zmean_folder (str): Path to the folder containing zmean.pkl.
        path_data_folder (str): Path to the folder containing histogram files.

    Returns:
        array: zmean values.
        list: List of histograms (DataFrames or numpy arrays) for hist_wide_cond_gandalf files.
        array/DataFrame: Histogram for the single hist_wide_cond_balrog file.
    """
    # Load zmean.pkl
    zmean_file = os.path.join(path_zmean_folder, 'zmean.pkl')
    if not os.path.isfile(zmean_file):
        raise ValueError("zmean.pkl file not found in the folder.")

    zmean = pd.read_pickle(zmean_file)
    if isinstance(zmean, (pd.DataFrame, pd.Series)):
        zmean = zmean.values
    elif not isinstance(zmean, np.ndarray):
        raise ValueError("zmean.pkl must contain a numpy array, DataFrame, or Series.")

    gandalf_files = []
    balrog_file = None

    # Load histogram files
    for file_name in os.listdir(path_data_folder):
        file_path = os.path.join(path_data_folder, file_name)

        if file_name.startswith("hist_wide_cond_gandalf") and file_name.endswith(".pkl"):
            loaded_file = pd.read_pickle(file_path)
            gandalf_files.append(loaded_file if isinstance(loaded_file, pd.DataFrame) else np.array(loaded_file))

        elif file_name.startswith("hist_wide_cond_balrog") and file_name.endswith(".pkl") and balrog_file is None:
            loaded_file = pd.read_pickle(file_path)
            balrog_file = loaded_file if isinstance(loaded_file, pd.DataFrame) else np.array(loaded_file)

    # Load gandalf mean redshift
    df_sompz_gandalf = pd.read_csv(path_gandalf_mean)
    gandalf_means = list(df_sompz_gandalf[['Mean Bin 1', 'Mean Bin 2', 'Mean Bin 3', 'Mean Bin 4']].mean())
    gandalf_stds = list(df_sompz_gandalf[['Mean Bin 1', 'Mean Bin 2', 'Mean Bin 3', 'Mean Bin 4']].std())

    # Balrog reference lines
    balrog_means = [0.3255, 0.5086, 0.7470, 0.9320]

    return zmean, gandalf_files, balrog_file, gandalf_means, gandalf_stds, balrog_means

def plot_redshift(path_data_folder, path_zmean_folder, path_gandalf_mean):
    """"""
    zmean, gandalf_files, balrog_file, gandalf_means, gandalf_stds, balrog_means = load_zmean_and_files(
        path_zmean_folder, path_data_folder, path_gandalf_mean)

    plot_tomo_bin_redshift_bootstrap(
        zmean=zmean,
        gandalf_files=gandalf_files,
        balrog_file=balrog_file,
        gandalf_means=gandalf_means,
        gandalf_stds=gandalf_stds,
        balrog_means=balrog_means,
        plot_settings={"plt_figsize": (10, 6)},
        show_plot=False,
        save_plot=True,
        save_name="/home/p/P.Gebhardt/Output/gaNdalF_paper/redshift_per_bin_bootstrap.png"
    )


def main(cfg, path_data, path_master_cat, filename_clf_balrog, filename_clf_gandalf, filename_flw_balrog,
         filename_flw_gandalf, path_redshift_hist_folder, path_zmean_folder, path_gandalf_redshift_mean, path_save_plots, calc_kl_div,
         plt_classf, plt_flow, plt_redshift, flow_columns):
    """"""

    if calc_kl_div is True:
        dict_kl_divergence, dict_percent_diff = calc_kullback_leibler(
            path_data=cfg['PATH_DATA'],
            filename_balrog=cfg['FILENAME_CLF_BALROG'],
            filename_gandalf=cfg['FILENAME_CLF_GANDALF'],
            columns=cfg["CLF_COLUMNS"]
        )

    if plt_classf is True:
        plot_classifier(
            cfg=cfg,
            path_master_cat=path_master_cat,
            path_save_plots=path_save_plots,
        )

    if plt_flow is True:
        plot_flow(
            path_data=path_data,
            filename_flw_balrog=filename_flw_balrog,
            filename_flw_gandalf=filename_flw_gandalf,
            path_master_cat=path_master_cat,
            path_save_plots=path_save_plots,
            columns=flow_columns
        )

    if plt_redshift is True:
        plot_redshift(
            path_data_folder=path_redshift_hist_folder,
            path_zmean_folder=path_zmean_folder,
            path_gandalf_mean=path_gandalf_redshift_mean
        )

if __name__ == '__main__':
    import pandas as pd
    from Handler import *
    import os
    import sys
    import argparse
    import yaml
    import scipy

    path = os.path.abspath(sys.path[0])
    parser = argparse.ArgumentParser(description='Plot gaNdalF')
    config_file_name = "paper_plots_LMU.cfg"
    # config_file_name = "paper_plots_MAC.cfg"

    parser.add_argument(
        '--config_filename',
        "-cf",
        type=str,
        nargs=1,
        required=False,
        default=config_file_name,
        help='Name of config file. If not given default.cfg will be used'
    )

    args = parser.parse_args()

    if isinstance(args.config_filename, list):
        args.config_filename = args.config_filename[0]

    with open(f"{path}/conf/{args.config_filename}", 'r') as fp:
        cfg = yaml.safe_load(fp)


    main(
        cfg=cfg,
        path_data=cfg["PATH_DATA"],
        path_master_cat=cfg["PATH_MASTER_CAT"],
        filename_clf_balrog=cfg["FILENAME_CLF_BALROG"],
        filename_clf_gandalf=cfg["FILENAME_CLF_GANDALF"],
        filename_flw_balrog=cfg["FILENAME_FLW_BALROG"],
        filename_flw_gandalf=cfg["FILENAME_FLW_GANDALF"],
        path_redshift_hist_folder=cfg["PATH_REDSHIFT_HIST_FOLDER"],
        path_zmean_folder=cfg["PATH_ZMEAN_FOLDER"],
        path_gandalf_redshift_mean=cfg["PATH_GANDALF_REDSHIFT_MEAN"],
        path_save_plots=cfg["PATH_SAVE_PLOTS"],
        calc_kl_div=cfg["CALC_KL_DIV"],
        plt_classf=cfg["PLT_CLF"],
        plt_flow=cfg["PLT_FLW"],
        plt_redshift=cfg["PLT_REDSHIFT"],
        flow_columns=cfg["FLW_COLUMNS"]
    )