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

def set_nan_2_default(data_frame, dict_default):
    for col, default_value in dict_default.items():
        if col in data_frame.columns:
            data_frame[col] = data_frame[col].fillna(default_value)


def get_scaler(data_frame):
    """"""
    from sklearn.preprocessing import  MaxAbsScaler
    scaler = MaxAbsScaler()
    scaler.fit(data_frame)
    return scaler

def get_yj_transformer(data_frame, columns):
    """"""
    from sklearn.preprocessing import PowerTransformer
    dict_pt = {}
    for col in columns:
        pt = PowerTransformer(method="yeo-johnson")
        pt.fit(np.array(data_frame[col]).reshape(-1, 1))
        data_frame[col] = pt.transform(np.array(data_frame[col]).reshape(-1, 1))
        dict_pt[f"{col} pt"] = pt
    return data_frame, dict_pt

def calc_kullback_leibler(df_balrog, df_gandalf, columns, print_text):

    # Initialize dictionaries to store KL divergence values
    dict_kl_divergence = {}

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

    print(print_text)
    for column in columns:
        print(f"{column}: KL Divergence = {dict_kl_divergence[column]}")

    return dict_kl_divergence

def calc_wasserstein_distance(df_balrog, df_gandalf, columns, print_text):
    """Compute Wasserstein Distance between two distributions"""
    dict_wasserstein = {}

    for column in columns:
        data_balrog = df_balrog[column]
        data_gandalf = df_gandalf[column]

        # Compute Wasserstein distance
        distance = scipy.stats.wasserstein_distance(data_balrog, data_gandalf)
        dict_wasserstein[column] = distance

    print(print_text)
    for column, value in dict_wasserstein.items():
        print(f"{column}: Wasserstein Distance = {value}")

    return dict_wasserstein


def calc_mean_std_diff(df_balrog, df_gandalf, columns, print_text):
    """Compute Mean and Standard Deviation differences"""
    dict_mean_diff = {}
    dict_std_diff = {}

    for column in columns:
        mean_diff = abs(df_balrog[column].mean() - df_gandalf[column].mean())
        std_diff = abs(df_balrog[column].std() - df_gandalf[column].std())

        dict_mean_diff[column] = mean_diff
        dict_std_diff[column] = std_diff

    print(print_text)
    for column in columns:
        print(f"{column}: Mean Difference = {dict_mean_diff[column]}, Std Dev Difference = {dict_std_diff[column]}")

    return dict_mean_diff, dict_std_diff


def calc_total_variation_distance(df_balrog, df_gandalf, columns, print_text):
    """Compute Total Variation Distance between two distributions"""
    dict_tvd = {}

    for column in columns:
        min_value = min(df_balrog[column].min(), df_gandalf[column].min())
        max_value = max(df_balrog[column].max(), df_gandalf[column].max())
        bins = np.linspace(min_value, max_value, 50)

        hist1, _ = np.histogram(df_balrog[column], bins=bins, density=True)
        hist2, _ = np.histogram(df_gandalf[column], bins=bins, density=True)

        hist1 /= hist1.sum()
        hist2 /= hist2.sum()

        tvd = 0.5 * np.sum(np.abs(hist1 - hist2))
        dict_tvd[column] = tvd

    print(print_text)
    for column, value in dict_tvd.items():
        print(f"{column}: Total Variation Distance = {value}")

    return dict_tvd


def calc_skewness_kurtosis_diff(df_balrog, df_gandalf, columns, print_text):
    """Compute differences in Skewness and Kurtosis"""
    dict_skewness_diff = {}
    dict_kurtosis_diff = {}

    for column in columns:
        skewness_diff = abs(scipy.stats.skew(df_balrog[column]) - scipy.stats.skew(df_gandalf[column]))
        kurtosis_diff = abs(scipy.stats.kurtosis(df_balrog[column]) - scipy.stats.kurtosis(df_gandalf[column]))

        dict_skewness_diff[column] = skewness_diff
        dict_kurtosis_diff[column] = kurtosis_diff

    print(print_text)
    for column in columns:
        print(f"{column}: Skewness Diff = {dict_skewness_diff[column]}, Kurtosis Diff = {dict_kurtosis_diff[column]}")

    return dict_skewness_diff, dict_kurtosis_diff


def calc_correlation(df_balrog, df_gandalf, columns, print_text, sample_size=2_000_000):
    """Compute Pearson and Spearman Correlations"""
    dict_pearson = {}
    dict_spearman = {}

    # Ensure the sample size does not exceed the actual dataset size
    sample_size_balrog = min(sample_size, len(df_balrog))
    sample_size_gandalf = min(sample_size, len(df_gandalf))

    # Randomly sample from both datasets
    df_balrog_sample = df_balrog.sample(n=sample_size_balrog, random_state=42)
    df_gandalf_sample = df_gandalf.sample(n=sample_size_gandalf, random_state=42)

    # Align both sampled dataframes to ensure they have the same indices
    # df_balrog_sample, df_gandalf_sample = df_balrog_sample.align(df_gandalf_sample, join='inner', axis=0)

    for column in columns:
        pearson_corr, _ = scipy.stats.pearsonr(df_balrog_sample[column], df_gandalf_sample[column])
        spearman_corr, _ = scipy.stats.spearmanr(df_balrog_sample[column], df_gandalf_sample[column])

        dict_pearson[column] = pearson_corr
        dict_spearman[column] = spearman_corr

    print(print_text)
    for column in columns:
        print(f"{column}: Pearson Correlation = {dict_pearson[column]}, Spearman Correlation = {dict_spearman[column]}")

    return dict_pearson, dict_spearman


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
    data_frame = data_frame[data_frame['BDF_MAG_DERED_CALIB_R'] < 37]
    data_frame = data_frame[data_frame['BDF_MAG_DERED_CALIB_I'] < 37]
    data_frame = data_frame[data_frame['BDF_MAG_DERED_CALIB_Z'] < 37]
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

    detection_rate_balrog = df_balrog_clf["detected"].mean()
    detection_rate_gandalf = df_gandalf_clf["detected"].mean()

    n_detected_balrog = df_balrog_clf["detected"].sum()
    n_detected_gandalf = df_gandalf_clf["detected"].sum()

    print(f"Balrog detection rate: {detection_rate_balrog:.4f} ({n_detected_balrog} / {len(df_balrog_clf)})")
    print(f"gaNdalF detection rate: {detection_rate_gandalf:.4f} ({n_detected_gandalf} / {len(df_gandalf_clf)})")

    df_balrog_clf_deep_cut = apply_deep_cuts(
        path_master_cat=path_master_cat,
        data_frame=df_balrog_clf
    )
    df_gandalf_clf_deep_cut = apply_deep_cuts(
        path_master_cat=path_master_cat,
        data_frame=df_gandalf_clf
    )

    if cfg["PLT_FIG_1"] is True:
        plot_multivariate_clf(
            df_balrog_detected=df_balrog_clf_deep_cut[df_balrog_clf_deep_cut['detected'] == 1],
            df_gandalf_detected=df_gandalf_clf_deep_cut[df_gandalf_clf_deep_cut['detected'] == 1],
            df_balrog_not_detected=df_balrog_clf_deep_cut[df_balrog_clf_deep_cut['detected'] == 0],
            df_gandalf_not_detected=df_gandalf_clf_deep_cut[df_gandalf_clf_deep_cut['detected'] == 0],
            columns={
                    "BDF_MAG_DERED_CALIB_R": {
                        "label": "BDF Mag R",
                        "range": [17.5, 26.5],
                        "position": [0, 0]
                    },
                    "BDF_MAG_DERED_CALIB_Z": {
                        "label": "BDF Mag Z",
                        "range": [17.5, 26.5],
                        "position": [0, 1]
                    },
                    "BDF_T": {
                        "label": "BDF T",
                        "range": [-2, 3],
                        "position": [0, 2]
                    },
                    "BDF_G": {
                        "label": "BDF G",
                        "range": [-0.1, 0.9],
                        "position": [1, 0]
                    },
                    "FWHM_WMEAN_R": {
                        "label": "FWHM R",
                        "range": [0.7, 1.3],
                        "position": [1, 1]
                    },
                    "FWHM_WMEAN_I": {
                        "label": "FWHM I",
                        "range": [0.7, 1.1],
                        "position": [1, 2]
                    },
                    "FWHM_WMEAN_Z": {
                         "label": "FWHM Z",
                         "range": [0.6, 1.16],
                         "position": [2, 0]
                     },
                     "AIRMASS_WMEAN_R": {
                         "label": "AIRMASS R",
                         "range": [0.95, 1.45],
                         "position": [2, 1]
                     },
                     "AIRMASS_WMEAN_I": {
                         "label": "AIRMASS I",
                         "range": [1, 1.45],
                         "position": [2, 2]
                     },
                     "AIRMASS_WMEAN_Z": {
                         "label": "AIRMASS Z",
                         "range": [1, 1.4],
                         "position": [2, 3]
                     },
                     "MAGLIM_R": {
                         "label": "MAGLIM R",
                         "range": [23, 24.8],
                         "position": [3, 0]
                     },
                     "MAGLIM_I": {
                         "label": "MAGLIM I",
                         "range": [22.4, 24.0],
                         "position": [3, 1]
                     },
                     "MAGLIM_Z": {
                         "label": "MAGLIM Z",
                         "range": [21.8, 23.2],
                         "position": [3, 2]
                     },
                     "EBV_SFD98": {
                         "label": "EBV SFD98",
                         "range": [-0.01, 0.10],
                         "position": [3, 3]
                     }
                },
            show_plot=cfg["SHOW_PLOT"],
            save_plot=cfg["SAVE_PLOT"],
            save_name=f"{path_save_plots}/{cfg['RUN_DATE']}_classifier_multiv.pdf",
            sample_size=100000,  # None,
            x_range=(17.5, 26.5),
            title=f"gaNdalF vs. Balrog: Photometric Property Distribution Comparison"
        )

    if cfg["PLT_FIG_2"] is True:
        plot_number_density_fluctuation(
            df_balrog=df_balrog_clf_deep_cut,
            df_gandalf=df_gandalf_clf_deep_cut,
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
            save_name=f"{path_save_plots}/{cfg['RUN_DATE']}_number_density_fluctuation.pdf",
            title=f"gaNdalF vs. Balrog: Detection Number Density Comparison"
        )

def plot_flow(cfg, path_data, filename_flw_balrog, filename_flw_gandalf, path_master_cat, path_save_plots, columns):
    """"""
    df_gandalf_flw = pd.read_pickle(f"{path_data}/{filename_flw_gandalf}")
    df_balrog_flw = pd.read_pickle(f"{path_data}/{filename_flw_balrog}")

    df_gandalf_flw_wo_nan = replace_nan(
        data_frame=df_gandalf_flw,
        cols=[
            "unsheared/mag_r",
            "unsheared/mag_i",
            "unsheared/mag_z",
            "unsheared/mag_err_r",
            "unsheared/mag_err_i",
            "unsheared/mag_err_z",
            "unsheared/snr",
            "unsheared/size_ratio",
            "unsheared/weight",
            "unsheared/T",
        ],
        default_values=[
            37.5,
            37.5,
            37.5,
            -9999,
            -9999,
            -9999,
            -7070.360705084288,
            -9999,
            -9999,
            -9999,
        ]
    )
    # df_gandalf_flw_wo_nan["Color unsheared MAG r-i"] = df_gandalf_flw_wo_nan["unsheared/mag_r"] - df_gandalf_flw_wo_nan["unsheared/mag_i"]
    # df_gandalf_flw_wo_nan["Color unsheared MAG i-z"] = df_gandalf_flw_wo_nan["unsheared/mag_i"] - df_gandalf_flw_wo_nan["unsheared/mag_z"]
    # df_gandalf_flw_wo_nan = check_idf_flux(df_gandalf_flw_wo_nan)
    # weights_nan = assign_new_weights(
    #     x=df_gandalf_flw_wo_nan["unsheared/snr"],
    #     y=df_gandalf_flw_wo_nan["unsheared/size_ratio"],
    #     path_grid=cfg['PATH_GRID_FILE']
    # )
    # df_gandalf_flw_wo_nan.loc[:, "unsheared/weight"] = weights_nan
    # print("gandalf w/ nans before cuts", df_gandalf_flw_wo_nan["unsheared/mag_err_i"].min(), df_gandalf_flw_wo_nan["unsheared/mag_err_i"].max())
    # df_gandalf_flw_wo_nan = apply_cuts(df_gandalf_flw_wo_nan, path_master_cat)
    # print("gandalf w/ nans after cuts", df_gandalf_flw_wo_nan["unsheared/mag_err_i"].min(), df_gandalf_flw_wo_nan["unsheared/mag_err_i"].max())


    df_gandalf_flw["Color unsheared MAG r-i"] = df_gandalf_flw["unsheared/mag_r"] - df_gandalf_flw["unsheared/mag_i"]
    df_gandalf_flw["Color unsheared MAG i-z"] = df_gandalf_flw["unsheared/mag_i"] - df_gandalf_flw["unsheared/mag_z"]

    df_balrog_flw = check_idf_flux(df_balrog_flw)
    df_gandalf_flw = check_idf_flux(df_gandalf_flw)

    weights = assign_new_weights(
        x=df_gandalf_flw["unsheared/snr"],
        y=df_gandalf_flw["unsheared/size_ratio"],
        path_grid=cfg['PATH_GRID_FILE']
    )
    df_gandalf_flw.loc[:, "unsheared/weight"] = weights
    # print("gandalf w nans before cuts", df_gandalf_flw["unsheared/mag_err_i"].min(), df_gandalf_flw["unsheared/mag_err_i"].max())
    df_balrog_flw_cut = apply_cuts(df_balrog_flw, path_master_cat)
    df_gandalf_flw_cut = apply_cuts(df_gandalf_flw, path_master_cat)
    # print("gandalf w nans after cuts", df_gandalf_flw_cut["unsheared/mag_err_i"].min(), df_gandalf_flw_cut["unsheared/mag_err_i"].max())

    exit()

    print(f"Length of Balrog objects: {len(df_balrog_flw)}")
    print(f"Length of gaNdalF objects: {len(df_gandalf_flw)}")
    print(f"Length of Balrog objects after mag cut: {len(df_balrog_flw_cut)}")
    print(f"Length of gaNdalF objects after mag cut: {len(df_gandalf_flw_cut)}")

    if cfg["PLT_FIG_3"] is True:
        plot_binning_statistics_combined(
            df_gandalf=df_gandalf_flw_cut,
            df_balrog=df_balrog_flw_cut,
            sample_size=10000,
            plot_scatter=False,
            show_plot=cfg["SHOW_PLOT"],
            save_plot=cfg["SAVE_PLOT"],
            title="gaNdalF vs. Balrog: Measured Photometric Property Distribution Comparison",
            save_name=f"{path_save_plots}/{cfg['RUN_DATE']}_binning_statistics_combined",
        )

    if cfg["PLT_FIG_4"] is True:
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
            title = r"gaNdalF vs. Balrog: $\texttt{Metacalibration}$ Property Distribution Comparison",
            show_plot=cfg["SHOW_PLOT"],
            save_plot=cfg["SAVE_PLOT"],
            save_name=f"{path_save_plots}/{cfg['RUN_DATE']}_mcal_hist_plot.pdf"
        )


def load_zmean_and_files(cfg, path_zmean_folder, path_data_folder, path_gandalf_mean):
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
    gandalf_means = list(df_sompz_gandalf[['Mean Bin 1', 'Mean Bin 2', 'Mean Bin 3', 'Mean Bin 4']].median())
    gandalf_stds = list(df_sompz_gandalf[['Mean Bin 1', 'Mean Bin 2', 'Mean Bin 3', 'Mean Bin 4']].std())
    if cfg['PLT_MEAN_Z_COMPARE']:
        plot_compare_mean_z_bootstrap(
            data_frame=df_sompz_gandalf,
            n_bins=25,
            title="Compare Boostrap <z>",
            show_plot=cfg['SHOW_PLOT'],
            save_plot=cfg['SAVE_PLOT'],
            save_name=f"{cfg['PATH_SAVE_PLOTS']}/{cfg['RUN_DATE']}_compare_bootstrap_mean_z.pdf"
        )
        exit()

    # Balrog reference lines
    balrog_means = [0.3255, 0.5086, 0.7470, 0.9320]

    return zmean, gandalf_files, balrog_file, gandalf_means, gandalf_stds, balrog_means

def plot_redshift(cfg, path_data_folder, path_zmean_folder, path_gandalf_mean):
    """"""
    zmean, gandalf_files, balrog_file, gandalf_means, gandalf_stds, balrog_means = load_zmean_and_files(
        cfg, path_zmean_folder, path_data_folder, path_gandalf_mean)

    if cfg["PLT_FIG_5"] is True:
        plot_tomo_bin_redshift_bootstrap(
            zmean=zmean,
            gandalf_files=gandalf_files,
            balrog_file=balrog_file,
            gandalf_means=gandalf_means,
            gandalf_stds=gandalf_stds,
            balrog_means=balrog_means,
            title="gaNdalF vs. Balrog: Tomographic Redshift Distribution Comparison",
            show_plot=cfg["SHOW_PLOT"],
            save_plot=cfg["SAVE_PLOT"],
            save_name=f"{cfg['PATH_SAVE_PLOTS']}/{cfg['RUN_DATE']}_redshift_bin_1_bootstrap.pdf",
            all_bins=False
        )
    if cfg["PLT_FIG_6"] is True:
        plot_tomo_bin_redshift_bootstrap_all_in_one(
            zmean=zmean,
            gandalf_files=gandalf_files,
            balrog_file=balrog_file,
            gandalf_means=gandalf_means,
            gandalf_stds=gandalf_stds,
            balrog_means=balrog_means,
            title="gaNdalF vs. Balrog: Tomographic Redshift Distribution Comparison",
            show_plot=cfg["SHOW_PLOT"],
            save_plot=cfg["SAVE_PLOT"],
            save_name=f"{cfg['PATH_SAVE_PLOTS']}/{cfg['RUN_DATE']}_redshift_per_bin_bootstrap.pdf",
        )


def main(cfg, path_data, path_master_cat, path_gandalf_odet, filename_flw_balrog,
         filename_flw_gandalf, path_redshift_hist_folder, path_zmean_folder, path_gandalf_redshift_mean, path_save_plots, calc_metric,
         plt_classf, plt_flow, plt_redshift, plt_trans_norm, flow_columns):
    """"""

    if calc_metric is True:
        df_clf_balrog_detected = None
        df_clf_gandalf_detected = None
        df_clf_balrog_not_detected = None
        df_clf_gandalf_not_detected = None
        df_flw_balrog = None
        df_flw_gandalf = None

        if cfg["CALC_METRIC_DET"] is True or cfg["CALC_METRIC_NOT_DET"] is True:
            df_clf_balrog = pd.read_pickle(f"{cfg['PATH_DATA']}/{cfg['FILENAME_CLF_BALROG']}")
            df_clf_gandalf = pd.read_pickle(f"{cfg['PATH_DATA']}/{cfg['FILENAME_CLF_GANDALF']}")

            df_balrog_clf_deep_cut = apply_deep_cuts(
                path_master_cat=path_master_cat,
                data_frame=df_clf_balrog
            )
            df_gandalf_clf_deep_cut = apply_deep_cuts(
                path_master_cat=path_master_cat,
                data_frame=df_clf_gandalf
            )

            if cfg["CALC_METRIC_DET"] is True:
                df_clf_balrog_detected = df_balrog_clf_deep_cut[df_balrog_clf_deep_cut["detected"] == 1]
                df_clf_gandalf_detected = df_gandalf_clf_deep_cut[df_gandalf_clf_deep_cut["detected"] == 1]

            if cfg["CALC_METRIC_NOT_DET"] is True:
                df_clf_balrog_not_detected = df_balrog_clf_deep_cut[df_balrog_clf_deep_cut["detected"] == 0]
                df_clf_gandalf_not_detected = df_gandalf_clf_deep_cut[df_gandalf_clf_deep_cut["detected"] == 0]

            del df_clf_balrog, df_clf_gandalf

        if cfg["CALC_METRIC_MCAL"] is True:
            df_flw_balrog = pd.read_pickle(f"{cfg['PATH_DATA']}/{cfg['FILENAME_FLW_BALROG']}")
            df_flw_gandalf = pd.read_pickle(f"{cfg['PATH_DATA']}/{cfg['FILENAME_FLW_GANDALF']}")

            df_flw_gandalf = replace_nan(
                data_frame=df_flw_gandalf,
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
                    df_flw_balrog["unsheared/mag_r"].max(),
                    df_flw_balrog["unsheared/mag_i"].max(),
                    df_flw_balrog["unsheared/mag_z"].max(),
                    df_flw_balrog["unsheared/snr"].max(),
                    df_flw_balrog["unsheared/size_ratio"].max(),
                    df_flw_balrog["unsheared/weight"].max(),
                    df_flw_balrog["unsheared/T"].max(),
                ]
            )

            df_flw_gandalf["Color unsheared MAG r-i"] = df_flw_gandalf["unsheared/mag_r"] - df_flw_gandalf[
                "unsheared/mag_i"]
            df_flw_gandalf["Color unsheared MAG i-z"] = df_flw_gandalf["unsheared/mag_i"] - df_flw_gandalf[
                "unsheared/mag_z"]

            df_flw_balrog = check_idf_flux(df_flw_balrog)
            df_flw_gandalf = check_idf_flux(df_flw_gandalf)

            df_flw_balrog = apply_cuts(df_flw_balrog, path_master_cat)
            df_flw_gandalf = apply_cuts(df_flw_gandalf, path_master_cat)

        # Detected +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if cfg["CALC_METRIC_DET"] is True:
            calc_kullback_leibler(
                df_balrog=df_clf_balrog_detected,
                df_gandalf=df_clf_gandalf_detected,
                columns=cfg["CLF_COLUMNS"],
                print_text="KL Divergence for Detected Objects:"
            )

            calc_wasserstein_distance(
                df_balrog=df_clf_balrog_detected,
                df_gandalf=df_clf_gandalf_detected,
                columns=cfg["CLF_COLUMNS"],
                print_text="Wasserstein Distance (Earth Mover’s Distance, EMD) for Detected Objects:"
            )

            calc_mean_std_diff(
                df_balrog=df_clf_balrog_detected,
                df_gandalf=df_clf_gandalf_detected,
                columns=cfg["CLF_COLUMNS"],
                print_text="Mean and Standard Deviation Differences for Detected Objects:"
            )

            calc_total_variation_distance(
                df_balrog=df_clf_balrog_detected,
                df_gandalf=df_clf_gandalf_detected,
                columns=cfg["CLF_COLUMNS"],
                print_text="Total Variation Distance (TVD) for Detected Objects:"
            )

            calc_skewness_kurtosis_diff(
                df_balrog=df_clf_balrog_detected,
                df_gandalf=df_clf_gandalf_detected,
                columns=cfg["CLF_COLUMNS"],
                print_text="Skewness and Kurtosis Differences for Detected Objects:"
            )

            calc_correlation(
                df_balrog=df_clf_balrog_detected,
                df_gandalf=df_clf_gandalf_detected,
                columns = cfg["CLF_COLUMNS"],
                print_text="Pearson and Spearman Correlation for Detected Objects:"
            )

        # Not Detected +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if cfg["CALC_METRIC_NOT_DET"] is True:
            calc_kullback_leibler(
                df_balrog=df_clf_balrog_not_detected,
                df_gandalf=df_clf_gandalf_not_detected,
                columns=cfg["CLF_COLUMNS"],
                print_text="KL Divergence for not Detected Objects:",

            )

            calc_wasserstein_distance(
                df_balrog=df_clf_balrog_not_detected,
                df_gandalf=df_clf_gandalf_not_detected,
                columns=cfg["CLF_COLUMNS"],
                print_text="Wasserstein Distance (Earth Mover’s Distance, EMD) for not Detected Objects:"
            )

            calc_mean_std_diff(
                df_balrog=df_clf_balrog_not_detected,
                df_gandalf=df_clf_gandalf_not_detected,
                columns=cfg["CLF_COLUMNS"],
                print_text="Mean and Standard Deviation Differences for not Detected Objects:"
            )

            calc_total_variation_distance(
                df_balrog=df_clf_balrog_not_detected,
                df_gandalf=df_clf_gandalf_not_detected,
                columns=cfg["CLF_COLUMNS"],
                print_text="Total Variation Distance (TVD) for not Detected Objects:"
            )

            calc_skewness_kurtosis_diff(
                df_balrog=df_clf_balrog_not_detected,
                df_gandalf=df_clf_gandalf_not_detected,
                columns=cfg["CLF_COLUMNS"],
                print_text="Skewness and Kurtosis Differences for not Detected Objects:"
            )

            calc_correlation(
                df_balrog=df_clf_balrog_not_detected,
                df_gandalf=df_clf_gandalf_not_detected,
                columns = cfg["CLF_COLUMNS"],
                print_text="Pearson and Spearman Correlation for not Detected Objects:"
            )

        # Mcal +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if cfg["CALC_METRIC_MCAL"] is True:
            calc_kullback_leibler(
                df_balrog=df_flw_balrog,
                df_gandalf=df_flw_gandalf,
                columns=cfg["FLW_COLUMNS"],
                print_text="KL Divergence for mcal Objects:"
            )

            calc_wasserstein_distance(
                df_balrog=df_flw_balrog,
                df_gandalf=df_flw_gandalf,
                columns=cfg["FLW_COLUMNS"],
                print_text="Wasserstein Distance (Earth Mover’s Distance, EMD) for mcal Objects:"
            )

            calc_mean_std_diff(
                df_balrog=df_flw_balrog,
                df_gandalf=df_flw_gandalf,
                columns=cfg["FLW_COLUMNS"],
                print_text="Mean and Standard Deviation Differences for mcal Objects:"
            )

            calc_total_variation_distance(
                df_balrog=df_flw_balrog,
                df_gandalf=df_flw_gandalf,
                columns=cfg["FLW_COLUMNS"],
                print_text="Total Variation Distance (TVD) for mcal Objects:"
            )

            calc_skewness_kurtosis_diff(
                df_balrog=df_flw_balrog,
                df_gandalf=df_flw_gandalf,
                columns=cfg["FLW_COLUMNS"],
                print_text="Skewness and Kurtosis Differences for mcal Objects:"
            )

            calc_correlation(
                df_balrog=df_flw_balrog,
                df_gandalf=df_flw_gandalf,
                columns=cfg["FLW_COLUMNS"],
                print_text="Pearson and Spearman Correlation for mcal Objects:",
                sample_size = 2_000_000
            )

    if plt_classf is True:
        plot_classifier(
            cfg=cfg,
            path_master_cat=path_master_cat,
            path_save_plots=path_save_plots,
        )

    if plt_flow is True:
        plot_flow(
            cfg=cfg,
            path_data=path_data,
            filename_flw_balrog=filename_flw_balrog,
            filename_flw_gandalf=filename_flw_gandalf,
            path_master_cat=path_master_cat,
            path_save_plots=path_save_plots,
            columns=flow_columns
        )

    if plt_redshift is True:
        plot_redshift(
            cfg=cfg,
            path_data_folder=path_redshift_hist_folder,
            path_zmean_folder=path_zmean_folder,
            path_gandalf_mean=path_gandalf_redshift_mean
        )

    if plt_trans_norm is True:
        df = pd.read_pickle(path_gandalf_odet)
        df = df[['MAGLIM_R', 'MAGLIM_I']]
        df_yj = df.copy()
        df_yj, dict_yj_transformer = get_yj_transformer(
            data_frame=df_yj,
            columns=['MAGLIM_R', 'MAGLIM_I']
        )

        scaler = get_scaler(
            data_frame=df_yj[["MAGLIM_R", 'MAGLIM_I']]
        )
        yj_scaled = scaler.transform(df_yj[["MAGLIM_R", "MAGLIM_I"]])
        df_yj_scaled = pd.DataFrame(yj_scaled, columns=["MAGLIM_R", 'MAGLIM_I'])

        plot_trans_norm_compare(
            data_frame=df,
            data_frame_yj=df_yj,
            data_frame_yj_scaled=df_yj_scaled,
            column="MAGLIM_R",
            ranges=[(23, 24.5), (-3.5, 3), (-0.6, 0.6)],
            bins=30,
            title="Effect of Yeo-Johnson Transformation and Normalization",
            show_plot=cfg["SHOW_PLOT"],
            save_plot=cfg["SAVE_PLOT"],
            save_name=f"{cfg['PATH_SAVE_PLOTS']}/{cfg['RUN_DATE']}_YJ_TRANS_MAXABS_MAGLIM_R.pdf",
        )

def fault_detection(cfg, path_master_cat):
    path_data = '/Volumes/elmichelangelo_external_ssd_1/Data/fault'
    df_run3 = pd.read_hdf(f'{path_data}/2025-04-09_19-01_run3_gandalf_Test_samples_2337398.h5', key='df')
    df_run5 = pd.read_hdf(f'{path_data}/2025-04-09_19-11_run5_gandalf_Test_samples_2334009.h5', key='df')
    df_run15 = pd.read_hdf(f'{path_data}/2025-04-09_20-01_run15_gandalf_Test_samples_2336170.h5', key='df')
    df_run35 = pd.read_hdf(f'{path_data}/2025-04-09_21-41_run35_gandalf_Test_samples_2337990.h5', key='df')

    # print(df_run3.isna().sum().sum())
    # print(df_run5.isna().sum().sum())
    # print(df_run15.isna().sum().sum())
    # print(df_run35.isna().sum().sum())
    # Sample & tag
    df_sampled_run3 = df_run3.copy()#.sample(n=10000, random_state=42)
    df_sampled_run3["source"] = "run3 z_mean_1=0.2931"
    df_sampled_run15 = df_run15.copy()#.sample(n=10000, random_state=42)
    df_sampled_run15["source"] = "run15 z_mean_1=0.3572"
    df_sampled_run5 = df_run5.copy()#.sample(n=10000, random_state=42)
    df_sampled_run5["source"] = "run5 z_mean_1=0.4292"
    df_sampled_run35 = df_run35.copy()#.sample(n=10000, random_state=42)
    df_sampled_run35["source"] = "run35 z_mean_1=0.3372"

    # Combine all
    df_all = pd.concat([df_sampled_run3, df_sampled_run5, df_sampled_run15, df_sampled_run35], ignore_index=True)

    # # Set up plot
    # fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    # axes = axes.flatten()
    #
    # # Define axis combinations and titles
    # panel_defs = [
    #     ("Color BDF MAG U-G", "Color BDF MAG I-Z", "Color U-G vs I-Z", [(-1, 2.5), (-0.25, 1.25)]),
    #     ("MAGLIM_R", "MAGLIM_I", "MAGLIM R vs I", [(23.25, 24.5), (22.75, 24)]),
    #     ("unsheared/weight", "unsheared/T", "weight vs T", [(0, 80), (0, 2)]),
    #     ("unsheared/size_ratio", "unsheared/snr", "size_ratio vs snr", [(-2, 5.5), (-50, 350)]),
    #     ("BDF_MAG_DERED_CALIB_R", "BDF_MAG_DERED_CALIB_I", "BDF_MAG_R vs I", [(18, 25), (18, 24.5)]),
    #     ("BDF_MAG_DERED_CALIB_I", "BDF_MAG_DERED_CALIB_Z", "BDF_MAG_I vs Z", [(18, 24.5), (17.5, 24.5)]),
    #     ("unsheared/mag_r", "unsheared/mag_i", "mag_r vs mag_i", [(18.5, 24.5), (18, 24)]),
    #     ("unsheared/mag_i", "unsheared/mag_z", "mag_i vs mag_z", [(18, 24), (18, 24.5)]),
    # ]

    # for ax, (x, y, title, limits) in zip(axes, panel_defs):
    #     sns.kdeplot(
    #         data=df_all,
    #         x=x,
    #         y=y,
    #         hue="source",
    #         fill=False,
    #         alpha=0.8,
    #         levels=3,
    #         linewidths=2,
    #         ax=ax
    #     )
    #     ax.set_title(title)
    #     if limits is not None:
    #         x_lim = limits[0]
    #         y_lim = limits[1]
    #         ax.set_xlim(x_lim[0], x_lim[1])
    #         ax.set_ylim(y_lim[0], y_lim[1])
    #     ax.grid(True, linestyle='--', alpha=0.5)
    #
    # plt.tight_layout()
    # save_name = f"{cfg['PATH_SAVE_PLOTS']}/{cfg['RUN_DATE']}_compare_outlier_photometry.pdf"
    # # plt.savefig(save_name, dpi=300)
    # plt.show()

    features = [
        "Color BDF MAG U-G", "Color BDF MAG I-Z",
        "MAGLIM_R", "MAGLIM_I",
        "unsheared/weight", "unsheared/T",
        "unsheared/size_ratio", "unsheared/snr",
        "BDF_MAG_DERED_CALIB_R", "BDF_MAG_DERED_CALIB_I",
        "BDF_MAG_DERED_CALIB_Z",
        "unsheared/mag_r", "unsheared/mag_i", "unsheared/mag_z"
    ]

    # Plot-Grid definieren
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()

    # Alle Histogramme erzeugen
    for i, feature in enumerate(features):
        ax = axes[i]
        sns.histplot(
            data=df_all,
            x=feature,
            hue="source",
            element="bars",
            stat="count",
            common_norm=True,
            bins=50,
            ax=ax
        )
        ax.set_title(feature)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_yscale("log")

    # Leere Achsen ausblenden, falls weniger als 16
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

    exit()

    # df_gandalf_flw = replace_nan(
    #     data_frame=df_gandalf_flw,
    #     cols=[
    #         "unsheared/mag_r",
    #         "unsheared/mag_i",
    #         "unsheared/mag_z",
    #         "unsheared/snr",
    #         "unsheared/size_ratio",
    #         "unsheared/weight",
    #         "unsheared/T",
    #     ],
    #     default_values=[
    #         df_balrog_flw["unsheared/mag_r"].max(),
    #         df_balrog_flw["unsheared/mag_i"].max(),
    #         df_balrog_flw["unsheared/mag_z"].max(),
    #         df_balrog_flw["unsheared/snr"].max(),
    #         df_balrog_flw["unsheared/size_ratio"].max(),
    #         df_balrog_flw["unsheared/weight"].max(),
    #         df_balrog_flw["unsheared/T"].max(),
    #     ]
    # )
    #
    # df_gandalf_flw["Color unsheared MAG r-i"] = df_gandalf_flw["unsheared/mag_r"] - df_gandalf_flw["unsheared/mag_i"]
    # df_gandalf_flw["Color unsheared MAG i-z"] = df_gandalf_flw["unsheared/mag_i"] - df_gandalf_flw["unsheared/mag_z"]
    #
    # df_balrog_flw = check_idf_flux(df_balrog_flw)
    # df_gandalf_flw = check_idf_flux(df_gandalf_flw)
    #
    # df_balrog_flw_cut = apply_cuts(df_balrog_flw, path_master_cat)
    # df_gandalf_flw_cut = apply_cuts(df_gandalf_flw, path_master_cat)
    #
    # print(f"Length of Balrog objects: {len(df_balrog_flw)}")
    # print(f"Length of gaNdalF objects: {len(df_gandalf_flw)}")
    # print(f"Length of Balrog objects after mag cut: {len(df_balrog_flw_cut)}")
    # print(f"Length of gaNdalF objects after mag cut: {len(df_gandalf_flw_cut)}")

if __name__ == '__main__':
    import pandas as pd
    from Handler import *
    from datetime import datetime
    import os
    import sys
    import argparse
    import yaml
    import scipy

    path = os.path.abspath(sys.path[0])
    parser = argparse.ArgumentParser(description='Plot gaNdalF')

    if get_os() == "Mac":
        print("Load MAC config file")
        config_file_name = "MAC_paper_plots.cfg"
    elif get_os() == "Linux":
        print("Load LMU config file")
        config_file_name = "LMU_paper_plots.cfg"
    else:
        raise "Error, no config file defined"

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

    now = datetime.now()
    cfg['RUN_DATE'] = now.strftime('%Y-%m-%d_%H-%M')
    if cfg["PLT_FAULT_DETECTION"] is True:
        fault_detection(cfg, path_master_cat=cfg["PATH_MASTER_CAT"],)
    main(
        cfg=cfg,
        path_data=cfg["PATH_DATA"],
        path_master_cat=cfg["PATH_MASTER_CAT"],
        filename_flw_balrog=cfg["FILENAME_FLW_BALROG"],
        filename_flw_gandalf=cfg["FILENAME_FLW_GANDALF"],
        path_gandalf_odet=cfg["PATH_GANDALF_ODET"],
        path_redshift_hist_folder=cfg["PATH_REDSHIFT_HIST_FOLDER"],
        path_zmean_folder=cfg["PATH_ZMEAN_FOLDER"],
        path_gandalf_redshift_mean=cfg["PATH_GANDALF_REDSHIFT_MEAN"],
        path_save_plots=cfg["PATH_SAVE_PLOTS"],
        calc_metric=cfg["CALC_METRIC"],
        plt_classf=cfg["PLT_CLF"],
        plt_flow=cfg["PLT_FLW"],
        plt_redshift=cfg["PLT_REDSHIFT"],
        plt_trans_norm=cfg["PLT_TRANS_NORM"],
        flow_columns=cfg["FLW_COLUMNS"]
    )