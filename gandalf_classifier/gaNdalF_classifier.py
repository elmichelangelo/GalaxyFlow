import torch
import os
import seaborn as sns
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from Handler import plot_classification_results, plot_confusion_matrix, plot_roc_curve, plot_recall_curve, plot_probability_hist
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from gandalf_galaxie_dataset import DESGalaxies


class gaNdalFClassifier(object):
    def __init__(self,
                 cfg,
                 lr,
                 bs
                 ):
        super().__init__()
        self.cfg = cfg
        self.bs = bs
        self.best_loss = float('inf')
        self.best_acc = 0.0
        self.best_epoch = 0
        self.lr = lr
        self.lst_loss = []

        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.init_dataset()

        cfg['PATH_OUTPUT_SUBFOLDER_CLASSF'] = f"{cfg['PATH_OUTPUT_CLASSF']}/lr_{self.lr}_bs_{self.bs}"
        cfg['PATH_WRITER_CLASSF'] = f"{cfg['PATH_OUTPUT_SUBFOLDER_CLASSF']}/{cfg['FOLDER_NAME_WRITER_CLASSF']}"
        cfg['PATH_PLOTS_CLASSF'] = f"{cfg['PATH_OUTPUT_SUBFOLDER_CLASSF']}/{cfg['FOLDER_NAME_PLOTS_CLASSF']}"
        cfg['PATH_SAVE_NN_CLASSF'] = f"{cfg['PATH_OUTPUT_SUBFOLDER_CLASSF']}/{cfg['FOLDER_NAME_SAVE_NN_CLASSF']}"

        for plot in cfg['PLOTS_CLASSF']:
            cfg[f'PATH_PLOTS_FOLDER_CLASSF'][plot.upper()] = f"{cfg['PATH_PLOTS_CLASSF']}/{plot}"

        self.make_dirs()

        xgb_clf = XGBClassifier(eval_metric=["error", "logloss"], learning_rate=lr)
        pipe = Pipeline([("clf", xgb_clf)])

        self.model = CalibratedClassifierCV(pipe, cv=2, method="isotonic")
        self.best_validation_loss = float('inf')
        self.best_validation_acc = 0.0
        self.best_validation_epoch = 0
        self.best_model = self.model

    def make_dirs(self):
        """"""
        if not os.path.exists(self.cfg['PATH_OUTPUT_SUBFOLDER_CLASSF']):
            os.mkdir(self.cfg['PATH_OUTPUT_SUBFOLDER_CLASSF'])
        if self.cfg['PLOT_CLASSF'] is True:
            if not os.path.exists(self.cfg['PATH_PLOTS_CLASSF']):
                os.mkdir(self.cfg['PATH_PLOTS_CLASSF'])
            for path_plot in self.cfg['PATH_PLOTS_FOLDER_CLASSF'].values():
                if not os.path.exists(path_plot):
                    os.mkdir(path_plot)

        if self.cfg['SAVE_NN_CLASSF'] is True:
            if not os.path.exists(self.cfg["PATH_SAVE_NN_CLASSF"]):
                os.mkdir(self.cfg["PATH_SAVE_NN_CLASSF"])

    def init_dataset(self):
        galaxies = DESGalaxies(
            cfg=self.cfg,
            kind="classifier_training",
            lst_split=[
                self.cfg['SIZE_TRAINING_DATA_CLASSF'],
                self.cfg['SIZE_VALIDATION_DATA_CLASSF'],
                self.cfg['SIZE_TEST_DATA_CLASSF']
            ]
        )

        # Convert the datasets to numpy arrays for XGBoost compatibility
        X_train, y_train = self.dataset_to_numpy(galaxies.train_dataset)
        X_val, y_val = self.dataset_to_numpy(galaxies.val_dataset)
        X_test, y_test = self.dataset_to_numpy(galaxies.test_dataset)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def run_training(self):
        """"""
        self.model.fit(
            self.X_train,
            self.y_train.ravel()
        )
        df_test_data = pd.DataFrame(self.X_test, columns=self.cfg['INPUT_COLS_MAG_CLASSF'])
        detected_true = self.y_test
        df_test_data['detected_true'] = detected_true
        # Validate the model
        y_pred = self.model.predict(self.X_test)
        y_pred_prob = self.model.predict_proba(self.X_test)[:, 1]
        predictions = y_pred_prob > np.random.rand(len(self.X_test))
        validation_accuracy = accuracy_score(self.y_test, y_pred)
        validation_accuracy_stochastic = accuracy_score(self.y_test, predictions)
        print(f"Accuracy for lr={self.lr}: {validation_accuracy * 100.0:.2f}%")
        print(f"Accuracy stochastic for lr={self.lr}: {validation_accuracy_stochastic * 100.0:.2f}%")

        df_test_data['true_detected'] = predictions
        df_test_data['detected_calibrated'] = predictions
        df_test_data['probability'] = y_pred_prob
        df_test_data['probability_calibrated'] = y_pred_prob

        if self.cfg['PLOT_CLASSF'] is True:
            self.create_plots(df_test_data, self.cfg['EPOCHS_CLASSF'])

        if self.cfg['SAVE_NN_CLASSF'] is True:
            with open(f"{self.cfg['PATH_SAVE_NN_CLASSF']}/{self.cfg[f'SAVE_NAME_NN']}_{self.cfg['RUN_DATE_CLASSF']}.pkl", 'wb') as file:
                pickle.dump(self.model, file)

    def create_plots(self, df_test_data, epoch=''):

        lst_cols = [
            ['BDF_MAG_DERED_CALIB_R', 'BDF_MAG_DERED_CALIB_I'],
            ['BDF_MAG_DERED_CALIB_I', 'BDF_MAG_DERED_CALIB_Z'],
            ['Color BDF MAG U-G', 'Color BDF MAG G-R'],
            ['Color BDF MAG R-I', 'Color BDF MAG I-Z'],
            ['Color BDF MAG Z-J', 'Color BDF MAG J-H'],
            ['BDF_T', 'BDF_G'],
            ['FWHM_WMEAN_R', 'FWHM_WMEAN_I'],
            ['FWHM_WMEAN_I', 'FWHM_WMEAN_Z'],
            ['AIRMASS_WMEAN_R', 'AIRMASS_WMEAN_I'],
            ['AIRMASS_WMEAN_I', 'AIRMASS_WMEAN_Z'],
            ['MAGLIM_R', 'MAGLIM_I'],
            ['MAGLIM_I', 'MAGLIM_Z'],
            ['EBV_SFD98', 'Color BDF MAG H-K']
        ]

        lst_save_names = [
            f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF'][f'MISS-CLASSIFICATION']}/classf_BDF_RI_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF'][f'MISS-CLASSIFICATION']}/classf_BDF_IZ_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF'][f'MISS-CLASSIFICATION']}/classf_Color_UG_GR_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF'][f'MISS-CLASSIFICATION']}/classf_Color_RI_IZ_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF'][f'MISS-CLASSIFICATION']}/classf_Color_ZJ_JH_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF'][f'MISS-CLASSIFICATION']}/classf_BDF_TG_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF'][f'MISS-CLASSIFICATION']}/classf_FWHM_RI_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF'][f'MISS-CLASSIFICATION']}/classf_FWHM_IZ_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF'][f'MISS-CLASSIFICATION']}/classf_AIRMASS_RI_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF'][f'MISS-CLASSIFICATION']}/classf_AIRMASS_IZ_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF'][f'MISS-CLASSIFICATION']}/classf_MAGLIM_RI_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF'][f'MISS-CLASSIFICATION']}/classf_MAGLIM_IZ_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF'][f'MISS-CLASSIFICATION']}/classf_EBV_Color_{epoch}.png",
        ]
        if self.cfg['PLOT_MISS_CLASSF'] is True:
            for idx_cols, cols in enumerate(lst_cols):
                plot_classification_results(
                    data_frame=df_test_data,
                    cols=cols,
                    show_plot=self.cfg['SHOW_PLOT_CLASSF'],
                    save_plot=self.cfg['SAVE_PLOT_CLASSF'],
                    save_name=lst_save_names[idx_cols],
                    title=f"Classification Results, lr={self.lr}, bs={self.bs}, epoch={epoch}"
                )

        if self.cfg['PLOT_MATRIX'] is True:
            plot_confusion_matrix(
                data_frame=df_test_data,
                show_plot=self.cfg['SHOW_PLOT_CLASSF'],
                save_plot=self.cfg['SAVE_PLOT_CLASSF'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF'][f'CONFUSION_MATRIX']}/confusion_matrix_epoch_{epoch}.png",
                title=f"Confusion Matrix, lr={self.lr}, bs={self.bs}, epoch={epoch}"
            )

        # ROC und AUC
        if self.cfg['PLOT_ROC_CURVE'] is True:
            plot_roc_curve(
                data_frame=df_test_data,
                show_plot=self.cfg['SHOW_PLOT_CLASSF'],
                save_plot=self.cfg['SAVE_PLOT_CLASSF'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF'][f'ROC_CURVE']}/roc_curve_epoch_{epoch}.png",
                title=f"Receiver Operating Characteristic (ROC) Curve, lr={self.lr}, bs={self.bs}, epoch={epoch}"
            )

        # Precision-Recall-Kurve
        if self.cfg['PLOT_PRECISION_RECALL_CURVE'] is True:
            plot_recall_curve(
                data_frame=df_test_data,
                show_plot=self.cfg['SHOW_PLOT_CLASSF'],
                save_plot=self.cfg['SAVE_PLOT_CLASSF'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF'][f'PRECISION_RECALL_CURVE']}/precision_recall_curve_epoch_{epoch}.png",
                title=f"recision-Recall Curve, lr={self.lr}, bs={self.bs}, epoch={epoch}"
            )

        # Histogramm der vorhergesagten Wahrscheinlichkeiten
        if self.cfg['PLOT_PROBABILITY_HIST'] is True:
            plot_probability_hist(
                data_frame=df_test_data,
                show_plot=self.cfg['SHOW_PLOT_CLASSF'],
                save_plot=self.cfg['SAVE_PLOT_CLASSF'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF'][f'PROB_HIST']}/probability_histogram{epoch}.png",
                title=f"probability histogram, lr={self.lr}, bs={self.bs}, epoch={epoch}"
            )

        if self.cfg['PLOT_LOSS_CLASSF'] is True:
            self.lst_loss.append(1)
            sns.scatterplot(self.lst_loss)
            if self.cfg['SHOW_PLOT_CLASSF'] is True:
                plt.show()
            if self.cfg['SAVE_PLOT_CLASSF'] is True:
                plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF'][f'LOSS']}/loss_{epoch}.png")
            plt.clf()

    @staticmethod
    def dataset_to_numpy(dataset):
        # Sammeln Sie alle Datenpunkte in einer Liste
        data_list = [dataset[i] for i in range(len(dataset))]
        # Trennen Sie die Daten von den Labels (angenommen, Ihr Dataset gibt ein Tupel von (Daten, Label) zurück)
        data, labels = zip(*data_list)
        # Konvertieren Sie die Listen von Tensoren in Tensoren
        data_tensor = torch.stack(data)
        labels_tensor = torch.stack(labels)
        # Konvertieren Sie die Tensoren in Numpy-Arrays
        data_array = data_tensor.numpy()
        labels_array = labels_tensor.numpy()
        return data_array, labels_array


if __name__ == '__main__':
    pass
