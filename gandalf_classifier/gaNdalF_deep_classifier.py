import torch
import os
import seaborn as sns
from torch.utils.data import DataLoader
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from gandalf_galaxie_dataset import DESGalaxies
from torch import nn
from sklearn.linear_model import LogisticRegression
import joblib


class gaNdalFClassifier(nn.Module):
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
        self.device = torch.device(cfg["DEVICE_CLASSF"])

        self.train_loader, self.valid_loader, self.test_loader, self.galaxies = self.init_dataset()

        cfg['PATH_OUTPUT_SUBFOLDER_CLASSF'] = f"{cfg['PATH_OUTPUT_CLASSF']}/lr_{self.lr}_bs_{self.bs}"
        cfg['PATH_WRITER_CLASSF'] = f"{cfg['PATH_OUTPUT_SUBFOLDER_CLASSF']}/{cfg['FOLDER_NAME_WRITER_CLASSF']}"
        cfg['PATH_PLOTS_CLASSF'] = f"{cfg['PATH_OUTPUT_SUBFOLDER_CLASSF']}/{cfg['FOLDER_NAME_PLOTS_CLASSF']}"
        cfg['PATH_SAVE_NN_CLASSF'] = f"{cfg['PATH_OUTPUT_SUBFOLDER_CLASSF']}/{cfg['FOLDER_NAME_SAVE_NN_CLASSF']}"

        for plot in cfg['PLOTS_CLASSF']:
            cfg[f'PATH_PLOTS_FOLDER_CLASSF'][plot.upper()] = f"{cfg['PATH_PLOTS_CLASSF']}/{plot}"

        self.make_dirs()

        self.model = nn.Sequential(
            nn.Linear(in_features=len(cfg['INPUT_COLS_MAG_CLASSF']), out_features=64),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.5),

            nn.Linear(in_features=64, out_features=128),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.5),

            nn.Linear(in_features=128, out_features=128),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.5),

            # TODO this Layer was additionally added to the model
            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(0.2),

            nn.Linear(in_features=64, out_features=32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid()
        )

        # Loss function to calculate the error of the neural net (binary cross entropy)
        self.loss_function = nn.BCELoss()

        self.loss = 0

        # Optimizer to calculate the weight changes
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.best_validation_loss = float('inf')
        self.best_validation_acc = 0.0
        self.best_validation_epoch = 0
        self.best_model = self.model
        self.calibration_model = None

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
        train_loader = DataLoader(galaxies.train_dataset, batch_size=self.bs, shuffle=True, num_workers=0)
        valid_loader = DataLoader(galaxies.val_dataset, batch_size=self.bs, shuffle=False, num_workers=0)
        test_loader = DataLoader(galaxies.test_dataset, batch_size=self.bs, shuffle=False, num_workers=0)
        return train_loader, valid_loader, test_loader, galaxies

    def train(self, mode=True):
        """"""
        super().train(mode=mode)
        for epoch in range(self.cfg['EPOCHS_CLASSF']):
            self.model.train()  # Set the model to training mode
            train_loss = 0.0
            pbar = tqdm(total=len(self.train_loader.dataset))
            for batch_idx, data in enumerate(self.train_loader):
                input_data = data[0].double()
                output_data = data[1].double()
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)
                self.optimizer.zero_grad()  # Clear the gradients
                outputs = self.model(input_data)  # Forward pass
                loss = self.loss_function(outputs.squeeze(), output_data.squeeze())  # Calculate loss
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update weights
                train_loss += loss.item() * output_data.size(0)
                pbar.update(output_data.size(0))
                pbar.set_description(f"Training,\t"
                                     f"Epoch: {epoch + 1},\t"
                                     f"learning rate: {self.lr},\t"
                                     f"batch size: {self.bs},\t"
                                     f"loss: {train_loss / pbar.n}")

            # Calculate average losses
            train_loss = train_loss / len(self.train_loader.dataset)

            # Validation pass
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                valid_loss = 0.0
                pbar = tqdm(total=len(self.valid_loader.dataset))
                for batch_idx, data in enumerate(self.valid_loader):
                    input_data = data[0].double()
                    output_data = data[1].double()
                    input_data = input_data.to(self.device)
                    output_data = output_data.to(self.device)
                    outputs = self.model(input_data)
                    loss = self.loss_function(outputs.squeeze(), output_data.squeeze())
                    valid_loss += loss.item() * input_data.size(0)
                    pbar.update(output_data.size(0))
                    pbar.set_description(f"Validation,\t"
                                         f"Epoch: {epoch + 1},\t"
                                         f"learning rate: {self.lr},\t"
                                         f"batch size: {self.bs},\t"
                                         f"loss: {valid_loss / pbar.n}")

            valid_loss = valid_loss / len(self.valid_loader.dataset)

            # Print training and validation loss
            print(f'Epoch {epoch + 1} \t Training Loss: {train_loss:.4f} \t Validation Loss: {valid_loss:.4f}')

            # Save the model if the validation loss decreased
            if valid_loss <= self.best_loss:
                print(f'Validation loss decreased ({self.best_loss:.6f} --> {valid_loss:.6f})')
                self.best_loss = valid_loss
        self.calibration_model = self.calibrate(self.valid_loader)  # Kalibrierungsmodell wird hier trainiert

    def forward(self, inputs):
        return self.model(inputs)

    def calibrate(self, valid_loader):
        self.model.eval()  # Set model to evaluation mode
        predictions = []
        tsr_input, tsr_output = self.dataset_to_tensor(self.valid_loader.dataset)
        with torch.no_grad():
            outputs = self.model(tsr_input.double().to(self.device))
            predictions.extend(outputs.squeeze().cpu().numpy())

        # Fit calibration model
        calibration_model = LogisticRegression()
        calibration_model.fit(np.array(predictions).reshape(-1, 1), tsr_output.numpy())
        joblib.dump(self.calibration_model, 'calibration_model.pkl')  # Speichern des Modells
        return calibration_model

    def predict_calibrated(self, y_pred):
        calibrated_outputs = self.calibration_model.predict_proba(y_pred.reshape(-1, 1))[:, 1]
        return calibrated_outputs

    def run_training(self):
        """"""
        self.train()
        self.model.eval()
        tsr_input, tsr_output = self.dataset_to_tensor(self.test_loader.dataset)
        detected_true = tsr_output.numpy()
        # Validate the model
        with torch.no_grad():
            probability = self.model(tsr_input.double().to(self.device)).squeeze().cpu().numpy()
        probability_calibrated = self.predict_calibrated(probability)
        detected = probability > np.random.rand(len(detected_true))
        detected_calibrated = probability_calibrated > np.random.rand(len(detected_true))
        validation_accuracy = accuracy_score(detected_true, detected)
        validation_accuracy_calibrated = accuracy_score(detected_true, detected_calibrated)
        print(f"Accuracy for lr={self.lr}, bs={self.bs}: {validation_accuracy * 100.0:.2f}%")
        print(f"Accuracy calibrated for lr={self.lr}, bs={self.bs}: {validation_accuracy_calibrated * 100.0:.2f}%")

        if self.cfg['PLOT_CLASSF'] is True:
            self.create_plots(detected, detected_calibrated, detected_true, probability, probability_calibrated)

        if self.cfg['SAVE_NN_CLASSF'] is True:
            with open(f"{self.cfg['PATH_SAVE_NN_CLASSF']}/{self.cfg[f'SAVE_NAME_NN']}_{self.cfg['RUN_DATE_CLASSF']}.pkl", 'wb') as file:
                pickle.dump(self.model, file)

    def create_plots(self, detected, detected_calibrated, detected_true, probability, probability_calibrated):
        # Konfusionsmatrix
        cm = confusion_matrix(detected_true, detected)
        df_cm = pd.DataFrame(cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"])
        fig_matrix = plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, fmt="g")
        plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF']['CONFUSION_MATRIX']}/confusion_matrix.png")
        plt.clf()
        plt.close(fig_matrix)

        cm = confusion_matrix(detected_true, detected_calibrated)
        df_cm = pd.DataFrame(cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"])
        fig_matrix = plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, fmt="g")
        plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF']['CONFUSION_MATRIX']}/confusion_matrix_calibrated.png")
        plt.clf()
        plt.close(fig_matrix)

        # ROC und AUC
        fpr, tpr, thresholds = roc_curve(detected_true, detected)
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

        fpr, tpr, thresholds = roc_curve(detected_true, detected_calibrated)
        roc_auc = auc(fpr, tpr)
        fig_roc_curve = plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF']['ROC_CURVE']}/roc_curve_calibrated.png")
        plt.clf()
        plt.close(fig_roc_curve)

        # Precision-Recall-Kurve
        precision, recall, thresholds = precision_recall_curve(detected_true, detected)
        fig_recal_curve = plt.figure()
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF']['PRECISION_RECALL_CURVE']}/precision_recall_curve.png")
        plt.clf()
        plt.close(fig_recal_curve)

        precision, recall, thresholds = precision_recall_curve(detected_true, detected_calibrated)
        fig_recal_curve = plt.figure()
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF']['PRECISION_RECALL_CURVE']}/precision_recall_curve_calibrated.png")
        plt.clf()
        plt.close(fig_recal_curve)

        # Histogramm der vorhergesagten Wahrscheinlichkeiten
        fig_prob_his = plt.figure()
        plt.hist(probability, bins=30, color='skyblue', edgecolor='black')
        plt.title('Histogram of Predicted Probabilities')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF']['PROB_HIST']}/probability_histogram.png")
        plt.clf()
        plt.close(fig_prob_his)

        fig_prob_his = plt.figure()
        plt.hist(probability_calibrated, bins=30, color='skyblue', edgecolor='black')
        plt.title('Histogram of Predicted Probabilities')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF']['PROB_HIST']}/probability_histogram_calibrated.png")
        plt.clf()
        plt.close(fig_prob_his)

    @staticmethod
    def dataset_to_tensor(dataset):
        data_list = [dataset[i] for i in range(len(dataset))]
        input_data, output_data = zip(*data_list)
        tsr_input = torch.stack(input_data)
        tsr_output = torch.stack(output_data)
        return tsr_input, tsr_output

    @staticmethod
    def dataloader_to_numpy(dataloader):
        data_list = []
        label_list = []

        dataloader.dataset.model.eval()

        with torch.no_grad():
            for batch_data in dataloader:
                inputs, labels = batch_data
                inputs = inputs.double().to('cpu')
                labels = labels.double().to('cpu')
                data_list.append(inputs.numpy())
                label_list.append(labels.numpy())

        data_array = np.concatenate(data_list, axis=0)
        label_array = np.concatenate(label_list, axis=0)

        return data_array, label_array


if __name__ == '__main__':
    pass
