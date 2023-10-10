from xgboost import XGBClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from Handler.data_loader import load_data
import numpy as np
import matplotlib.pyplot as plt


TRANSFORM_COLS = [
                'AIRMASS_WMEAN_R',
                 'AIRMASS_WMEAN_I',
                 'AIRMASS_WMEAN_Z',
                 'FWHM_WMEAN_R',
                 'FWHM_WMEAN_I',
                 'FWHM_WMEAN_Z',
                 'MAGLIM_R',
                 'MAGLIM_I',
                 'MAGLIM_Z',
                 'EBV_SFD98'
                 ]
target_cols = [
            "detected"
]
training_cols = [
                 'BDF_MAG_DERED_CALIB_R',
                 'BDF_MAG_DERED_CALIB_I',
                 'BDF_MAG_DERED_CALIB_Z',
                 'BDF_MAG_ERR_DERED_CALIB_R',
                 'BDF_MAG_ERR_DERED_CALIB_I',
                 'BDF_MAG_ERR_DERED_CALIB_Z',
                 'Color BDF MAG U-G',
                 'Color BDF MAG G-R',
                 'Color BDF MAG R-I',
                 'Color BDF MAG I-Z',
                 'Color BDF MAG Z-J',
                 'Color BDF MAG J-H',
                 'Color BDF MAG H-K',
                 'BDF_T',
                 'BDF_G',
                 'FWHM_WMEAN_R',
                 'FWHM_WMEAN_I',
                 'FWHM_WMEAN_Z',
                 'AIRMASS_WMEAN_R',
                 'AIRMASS_WMEAN_I',
                 'AIRMASS_WMEAN_Z',
                 'MAGLIM_R',
                 'MAGLIM_I',
                 'MAGLIM_Z',
                'EBV_SFD98'
    ]

training_data, validation_data, test_data = load_data(
            path_training_data="/Users/P.Gebhardt/Development/PhD/data/gandalf_training_data_wundet_ncuts_rdef_rnan_3000000.pkl",
            path_output="/Users/P.Gebhardt/Development/PhD/output/gaNdalF/",
            luminosity_type='MAG',
            selected_scaler=None,
            size_training_dataset=.6,
            size_validation_dataset=.2,
            size_test_dataset=.2,
            reproducible=True,
            run=1,
            lst_yj_transform_cols=TRANSFORM_COLS,
            apply_object_cut=False,
            apply_flag_cut=False,
            apply_airmass_cut=False,
            apply_unsheared_mag_cut=False,
            apply_unsheared_shear_cut=False,
            plot_data=False,
            writer=None
        )

train_data = training_data["data frame training data"]
t_data = test_data["data frame test data"]

pipe = Pipeline(
    [
        ("scaler", MaxAbsScaler()),
        ("clf", XGBClassifier()),
    ]
)
clf = CalibratedClassifierCV(pipe, cv=2, method="isotonic")
clf.fit(train_data[training_cols], train_data[target_cols])
y_prob = clf.predict_proba(t_data[training_cols])[:, 1]
y_pred = y_prob > np.random.rand(len(t_data[target_cols]))

importances = (
                clf.calibrated_classifiers_[0].estimator["clf"].feature_importances_
            )

plot_training_cols = [
                'BDF_R',
                 'BDF_I',
                 'BDF_Z',
                 'BDF_ERR_R',
                 'BDF_ERR_I',
                 'BDF_ERR__Z',
                 'BDF U-G',
                 'BDF G-R',
                 'BDF R-I',
                 'BDF I-Z',
                 'BDF Z-J',
                 'BDF J-H',
                 'BDF H-K',
                 'BDF_T',
                 'BDF_G',
                 'FWHM_R',
                 'FWHM_I',
                 'FWHM_Z',
                 'AIRMASS_R',
                 'AIRMASS_I',
                 'AIRMASS_Z',
                 'MAGLIM_R',
                 'MAGLIM_I',
                 'MAGLIM_Z',
                'EBV_SFD98'
]
plt.bar(plot_training_cols, importances)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
