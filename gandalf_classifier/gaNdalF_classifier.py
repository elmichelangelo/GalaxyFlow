import torch
import os
import seaborn as sns
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
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

        # Validate the model
        y_pred = self.model.predict(self.X_test)
        y_pred_prob = self.model.predict_proba(self.X_test)[:, 1]
        predictions = y_pred_prob > np.random.rand(len(self.X_test))
        validation_accuracy = accuracy_score(self.y_test, y_pred)
        validation_accuracy_stochastic = accuracy_score(self.y_test, predictions)
        print(f"Accuracy for lr={self.lr}: {validation_accuracy * 100.0:.2f}%")
        print(f"Accuracy stochastic for lr={self.lr}: {validation_accuracy_stochastic * 100.0:.2f}%")

        if self.cfg['PLOT_CLASSF'] is True:
            self.create_plots(y_pred, y_pred_prob, self.y_test)

        if self.cfg['SAVE_NN_CLASSF'] is True:
            with open(f"{self.cfg['PATH_SAVE_NN_CLASSF']}/{self.cfg[f'SAVE_NAME_NN']}_{self.cfg['RUN_DATE_CLASSF']}.pkl", 'wb') as file:
                pickle.dump(self.model, file)

    def create_plots(self, y_pred, y_pred_prob, y_true):
        # Konfusionsmatrix
        cm = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"])
        fig_matrix = plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, fmt="g")
        plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF']['CONFUSION_MATRIX']}/confusion_matrix.png")
        plt.clf()
        plt.close(fig_matrix)

        # ROC und AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc_curve = plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF']['ROC_CURVE']}/roc_curve.png")
        plt.clf()
        plt.close(fig_roc_curve)

        # Precision-Recall-Kurve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
        fig_recal_curve = plt.figure()
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF']['PRECISION_RECALL_CURVE']}/precision_recall_curve.png")
        plt.clf()
        plt.close(fig_recal_curve)

        # Histogramm der vorhergesagten Wahrscheinlichkeiten
        fig_prob_his = plt.figure()
        plt.hist(y_pred_prob, bins=30, color='skyblue', edgecolor='black')
        plt.title('Histogram of Predicted Probabilities')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF']['PROB_HIST']}/probability_histogram.png")
        plt.clf()
        plt.close(fig_prob_his)

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
