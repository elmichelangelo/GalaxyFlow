def calc_kullback_leibler(path_data, filename_clf_balrog, filename_clf_gandalf, filename_flw_balrog,
                          filename_flw_gandalf, path_master_cat, cf_columns, flow_columns):
    import scipy

    df_balrog_clf = pd.read_pickle(f"{path_data}/{filename_clf_balrog}")
    df_gandalf_clf = pd.read_pickle(f"{path_data}/{filename_clf_gandalf}")

    df_balrog_clf_detected = df_balrog_clf[df_balrog_clf["detected"] == 1][cf_columns].dropna()
    df_gandalf_clf_detected = df_gandalf_clf[df_gandalf_clf["detected"] == 1][cf_columns].dropna()

    # Initialize dictionaries to store KL divergence values
    kl_detected = {}

    # Compute KL divergence for each feature
    for column in cf_columns:
        data1 = df_balrog_clf_detected[column]
        data2 = df_gandalf_clf_detected[column]

        # Determine the range and bins
        min_value = min(data1.min(), data2.min())
        max_value = max(data1.max(), data2.max())
        bins = np.linspace(min_value, max_value, 50)  # Adjust the number of bins as needed

        # Compute histograms (probability distributions)
        hist1, _ = np.histogram(data1, bins=bins, density=True)
        hist2, _ = np.histogram(data2, bins=bins, density=True)

        # Add a small constant to avoid zeros (which cause issues in KL divergence)
        hist1 += 1e-10
        hist2 += 1e-10

        # Normalize the histograms to make them probability distributions
        hist1 /= hist1.sum()
        hist2 /= hist2.sum()

        # Compute KL divergence
        kl_value = scipy.stats.entropy(hist1, hist2)
        kl_detected[column] = kl_value

    # Print the KL divergence values
    print("KL Divergence for Detected Objects:")
    for column, kl_value in kl_detected.items():
        print(f"{column}: {kl_value}")

    df_balrog_clf_not_detected = df_balrog_clf[df_balrog_clf["detected"] == 0][cf_columns].dropna()
    df_gandalf_clf_not_detected = df_gandalf_clf[df_gandalf_clf["detected"] == 0][cf_columns].dropna()

    # Initialize dictionaries to store KL divergence values
    kl_not_detected = {}

    # Compute KL divergence for each feature
    for column in cf_columns:
        data1 = df_balrog_clf_not_detected[column]
        data2 = df_gandalf_clf_not_detected[column]

        # Determine the range and bins
        min_value = min(data1.min(), data2.min())
        max_value = max(data1.max(), data2.max())
        bins = np.linspace(min_value, max_value, 50)  # Adjust the number of bins as needed

        # Compute histograms (probability distributions)
        hist1, _ = np.histogram(data1, bins=bins, density=True)
        hist2, _ = np.histogram(data2, bins=bins, density=True)

        # Add a small constant to avoid zeros (which cause issues in KL divergence)
        hist1 += 1e-10
        hist2 += 1e-10

        # Normalize the histograms to make them probability distributions
        hist1 /= hist1.sum()
        hist2 /= hist2.sum()

        # Compute KL divergence
        kl_value = scipy.stats.entropy(hist1, hist2)
        kl_not_detected[column] = kl_value

    # Print the KL divergence values
    print("KL Divergence for Not Detected Objects:")
    for column, kl_value in kl_not_detected.items():
        print(f"{column}: {kl_value}")

    df_balrog_flw = pd.read_pickle(f"{path_data}/{filename_flw_balrog}")
    df_gandalf_flw = pd.read_pickle(f"{path_data}/{filename_flw_gandalf}")

    df_balrog_flw_mcal = apply_cuts(df_balrog_flw, path_master_cat)
    df_gandalf_flw_mcal = apply_cuts(df_gandalf_flw, path_master_cat)

    df_balrog_flw = df_balrog_flw.dropna()
    df_gandalf_flw = df_gandalf_flw.dropna()


    # Initialize dictionaries to store KL divergence values
    kl_flow = {}

    # Compute KL divergence for each feature
    for column in flow_columns:
        data1 = df_balrog_flw[column]
        data2 = df_gandalf_flw[column]

        # Determine the range and bins
        min_value = min(data1.min(), data2.min())
        max_value = max(data1.max(), data2.max())
        bins = np.linspace(min_value, max_value, 50)  # Adjust the number of bins as needed

        # Compute histograms (probability distributions)
        hist1, _ = np.histogram(data1, bins=bins, density=True)
        hist2, _ = np.histogram(data2, bins=bins, density=True)

        # Add a small constant to avoid zeros (which cause issues in KL divergence)
        hist1 += 1e-10
        hist2 += 1e-10

        # Normalize the histograms to make them probability distributions
        hist1 /= hist1.sum()
        hist2 /= hist2.sum()

        # Compute KL divergence
        kl_value = scipy.stats.entropy(hist1, hist2)
        kl_flow[column] = kl_value

    # Print the KL divergence values
    print("KL Divergence for Flow Objects:")
    for column, kl_value in kl_flow.items():
        print(f"{column}: {kl_value}")

    df_balrog_flw_mcal = df_balrog_flw_mcal.dropna()
    df_gandalf_flw_mcal = df_gandalf_flw_mcal.dropna()

    # Initialize dictionaries to store KL divergence values
    kl_flow_mcal = {}

    # Compute KL divergence for each feature
    for column in flow_columns:
        data1 = df_balrog_flw_mcal[column]
        data2 = df_gandalf_flw_mcal[column]

        # Determine the range and bins
        min_value = min(data1.min(), data2.min())
        max_value = max(data1.max(), data2.max())
        bins = np.linspace(min_value, max_value, 50)  # Adjust the number of bins as needed

        # Compute histograms (probability distributions)
        hist1, _ = np.histogram(data1, bins=bins, density=True)
        hist2, _ = np.histogram(data2, bins=bins, density=True)

        # Add a small constant to avoid zeros (which cause issues in KL divergence)
        hist1 += 1e-10
        hist2 += 1e-10

        # Normalize the histograms to make them probability distributions
        hist1 /= hist1.sum()
        hist2 /= hist2.sum()

        # Compute KL divergence
        kl_value = scipy.stats.entropy(hist1, hist2)
        kl_flow_mcal[column] = kl_value

    # Print the KL divergence values
    print("KL Divergence for MCAL Flow Objects:")
    for column, kl_value in kl_flow_mcal.items():
        print(f"{column}: {kl_value}")


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

def plot_classifier(path_data, filename_clf_balrog, filename_clf_gandalf, path_master_cat, path_save_plots):
    """"""
    df_balrog_clf = pd.read_pickle(f"{path_data}/{filename_clf_balrog}")
    df_gandalf_clf = pd.read_pickle(f"{path_data}/{filename_clf_gandalf}")

    df_balrog_clf_deep_cut = apply_deep_cuts(
        path_master_cat=path_master_cat,
        data_frame=df_balrog_clf
    )
    df_gandalf_clf_deep_cut = apply_deep_cuts(
        path_master_cat=path_master_cat,
        data_frame=df_gandalf_clf
    )

    print(f"Length of Balrog detected objects: {len(df_balrog_clf[df_balrog_clf['detected'] == 1])}")
    print(f"Length of Balrog not detected objects: {len(df_balrog_clf[df_balrog_clf['detected'] == 0])}")
    print(f"Length of gaNdalF detected objects: {len(df_gandalf_clf[df_gandalf_clf['detected'] == 1])}")
    print(f"Length of gaNdalF not detected objects: {len(df_gandalf_clf[df_gandalf_clf['detected'] == 0])}")
    print(f"Length of Balrog detected deep cut objects: {len(df_balrog_clf_deep_cut[df_balrog_clf_deep_cut['detected'] == 1])}")
    print(f"Length of Balrog not detected deep cut objects: {len(df_balrog_clf_deep_cut[df_balrog_clf_deep_cut['detected'] == 0])}")
    print(f"Length of gaNdalF detected deep cut objects: {len(df_gandalf_clf_deep_cut[df_gandalf_clf_deep_cut['detected'] == 1])}")
    print(f"Length of gaNdalF not detected deep cut objects: {len(df_gandalf_clf_deep_cut[df_gandalf_clf_deep_cut['detected'] == 0])}")

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
        show_plot=False,
        save_plot=True,
        save_name=f"{path_save_plots}/number_density_fluctuation.png",
        title=f"Number Density Fluctuation Analysis of gaNdalF vs. Balrog Detections")

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
        show_plot=False,
        save_plot=True,
        save_name=f"{path_save_plots}/number_density_fluctuation_deep_cut.png",
        title=f"Number Density Fluctuation Analysis of GANDALF vs. Balrog Detections with Deep Cut Applied"
    )

    plot_multivariate_classifier(
        df_balrog=df_balrog_clf,
        df_gandalf=df_gandalf_clf,
        grid_size=[4, 4],
        x_range=(17.8, 26.3),
        columns={
            "BDF_MAG_DERED_CALIB_R": {
                "label": "BDF Mag R",
                "range": [18.5, 28],
                "position": [0, 0]
            },
            "BDF_MAG_DERED_CALIB_Z": {
                "label": "BDF Mag Z",
                "range": [18, 26],
                "position": [0, 1]
            },
            "BDF_T": {
                "label": "BDF T",
                "range": [-0.25, 1.8],
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
        show_plot=False,
        save_plot=True,
        save_name=f"{path_save_plots}/classifier_multiv.png",
        title=f"Multivariate Comparison of Detection Distributions in gaNdalF and Balrog"
    )

    plot_multivariate_classifier(
        df_balrog=df_balrog_clf_deep_cut,
        df_gandalf=df_gandalf_clf_deep_cut,
        grid_size=[4, 4],
        x_range=(17.8, 26.3),
        columns={
            "BDF_MAG_DERED_CALIB_R": {
                "label": "BDF Mag R",
                "range": [18.5, 28],
                "position": [0, 0]
            },
            "BDF_MAG_DERED_CALIB_Z": {
                "label": "BDF Mag Z",
                "range": [18, 26],
                "position": [0, 1]
            },
            "BDF_T": {
                "label": "BDF T",
                "range": [-0.25, 1.8],
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
        show_plot=False,
        save_plot=True,
        save_name=f"{path_save_plots}/classifier_multiv_deep_cut.png",
        title=f"Multivariate Comparison of Detection Distributions in gaNdalF and Balrog with Deep Cut Applied"
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

    df_balrog_flw = df_balrog_flw[columns]
    df_gandalf_flw = df_gandalf_flw[columns]

    df_balrog_flw_cut = df_balrog_flw_cut[columns]
    df_gandalf_flw_cut = df_gandalf_flw_cut[columns]

    print(f"Length of Balrog objects: {len(df_balrog_flw)}")
    print(f"Length of gaNdalF objects: {len(df_gandalf_flw)}")
    print(f"Length of Balrog objects after mag cut: {len(df_balrog_flw_cut)}")
    print(f"Length of gaNdalF objects after mag cut: {len(df_gandalf_flw_cut)}")

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

def main(path_data, path_master_cat, filename_clf_balrog, filename_clf_gandalf, filename_flw_balrog,
         filename_flw_gandalf, path_save_plots, plt_classf, plt_flow, flow_columns):
    """"""

    calc_kullback_leibler(
        path_data=path_data,
        filename_clf_balrog=filename_clf_balrog,
        filename_clf_gandalf=filename_clf_gandalf,
        filename_flw_balrog=filename_flw_balrog,
        filename_flw_gandalf=filename_flw_gandalf,
        path_master_cat=path_master_cat,
        cf_columns=[
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
        flow_columns=flow_columns
    )

    exit()


    if plt_classf is True:
        plot_classifier(
            path_data=path_data,
            filename_clf_balrog=filename_clf_balrog,
            filename_clf_gandalf=filename_clf_gandalf,
            path_master_cat=path_master_cat,
            path_save_plots=path_save_plots
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

if __name__ == '__main__':
    import pandas as pd
    from Handler import *
    import scipy as sp
    import os
    main(
        path_data="/project/ls-gruen/users/patrick.gebhardt/data/gaNdalF_paper_catalogs",
        path_master_cat="/project/ls-gruen/users/patrick.gebhardt/data/gaNdalF/Y3_mastercat_02_05_21.h5",
        filename_clf_balrog="2024-10-28_08-14_balrog_clf_Test_sample.pkl",
        filename_clf_gandalf="2024-10-28_08-14_gandalf_clf_Test_sample.pkl",
        filename_flw_balrog = "2024-10-28_08-14_balrog_flw_Test_sample.pkl",
        filename_flw_gandalf = "2024-10-28_08-14_gandalf_flw_Test_sample.pkl",
        path_save_plots = "/home/p/P.Gebhardt/Output/gaNdalF_paper",
        plt_classf=False,
        plt_flow=True,
        flow_columns=[
            "Color unsheared MAG r-i",
            "Color unsheared MAG i-z",
            "unsheared/mag_r",
            "unsheared/mag_i",
            "unsheared/mag_z",
            "unsheared/snr",
            "unsheared/size_ratio",
            "unsheared/weight",
            "unsheared/T"
        ]
    )