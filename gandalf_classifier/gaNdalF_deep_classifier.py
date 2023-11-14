import torch
import os
import seaborn as sns
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from Handler import plot_classification_results, plot_confusion_matrix, plot_roc_curve, plot_recall_curve, plot_probability_hist
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
        self.lst_loss = []

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

            nn.Linear(in_features=128, out_features=64),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.5),
            #
            # # TODO this Layer was additionally added to the model
            # nn.Linear(in_features=128, out_features=256),
            # nn.LeakyReLU(0.2),
            #
            # nn.Linear(in_features=256, out_features=128),
            # nn.LeakyReLU(0.2),
            #
            # nn.Linear(in_features=128, out_features=64),
            # nn.LeakyReLU(0.2),
            # # TODO this Layer was additionally added to the model

            nn.Linear(in_features=64, out_features=32),
            nn.LeakyReLU(0.2),

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

        train_loader = DataLoader(galaxies.train_dataset, batch_size=self.bs, shuffle=True, num_workers=0)
        valid_loader = DataLoader(galaxies.val_dataset, batch_size=self.bs, shuffle=False, num_workers=0)
        test_loader = DataLoader(galaxies.test_dataset, batch_size=self.bs, shuffle=False, num_workers=0)
        return train_loader, valid_loader, test_loader, galaxies

    def train(self, mode=True):
        """
        Trains the neural network model.

        :param mode: A boolean value that determines the training mode of the model.
        """

        super().train(mode=mode)
        # Training cycle that runs for a certain number of epoch
        for epoch in range(self.cfg['EPOCHS_CLASSF']):
            self.model.train()  # Set the model to training mode
            train_loss = 0.0
            pbar_train = tqdm(total=len(self.train_loader.dataset), unit="batch", ncols=200)
            epoch_loss = 0.0
            # Iterates over the training data in batches
            for batch_idx, data in enumerate(self.train_loader):
                input_data = data[0].double()
                output_data = data[1].double()
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)
                self.optimizer.zero_grad()  # Clear the gradients
                outputs = self.model(input_data)  # Forward pass
                loss = self.loss_function(outputs.squeeze(), output_data.squeeze())  # Calculate loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Updates the weights
                train_loss += loss.item() * output_data.size(0)
                pbar_train.update(output_data.size(0))
                pbar_train.set_description(f"Training,\t"
                                           f"Epoch: {epoch + 1},\t"
                                           f"learning rate: {self.lr},\t"
                                           f"batch size: {self.bs},\t"
                                           f"loss: {train_loss / pbar_train.n}")
                epoch_loss += loss.item()
            pbar_train.close()
            self.lst_loss.append(epoch_loss / len(self.train_loader))

            # Calculates the average training loss
            train_loss = train_loss / len(self.train_loader.dataset)

            # Sets the model to evaluation mode
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                valid_loss = 0.0
                pbar_val = tqdm(total=len(self.valid_loader.dataset), unit="batch", ncols=200)

                # Iterates over the validation data in batches
                for batch_idx, data in enumerate(self.valid_loader):
                    input_data = data[0].double()
                    output_data = data[1].double()
                    input_data = input_data.to(self.device)
                    output_data = output_data.to(self.device)
                    outputs = self.model(input_data)
                    loss = self.loss_function(outputs.squeeze(), output_data.squeeze())
                    valid_loss += loss.item() * input_data.size(0)
                    pbar_val.update(output_data.size(0))
                    pbar_val.set_description(f"Validation,\t"
                                             f"Epoch: {epoch + 1},\t"
                                             f"learning rate: {self.lr},\t"
                                             f"batch size: {self.bs},\t"
                                             f"loss: {valid_loss / pbar_val.n}")
                pbar_val.close()

            # Calculates the average validation loss
            valid_loss = valid_loss / len(self.valid_loader.dataset)

            # Prints the training and validation loss
            print(f'Epoch {epoch + 1} \t Training Loss: {train_loss:.4f} \t Validation Loss: {valid_loss:.4f}')

            # Saves the model if the validation loss has decreased
            if valid_loss <= self.best_loss:
                print(f'Validation loss decreased ({self.best_loss:.6f} --> {valid_loss:.6f})')
                self.best_loss = valid_loss

        # Trains the calibration model
        self.calibration_model = self.calibrate()

    def forward(self, inputs):
        return self.model(inputs)

    def save_model(self):
        """"""
        # save config model
        joblib.dump(
            self.calibration_model,
            f"{self.cfg['PATH_SAVE_NN_CLASSF']}/{self.cfg['SAVE_NAME_NN']}_e_{self.cfg['EPOCHS']}_lr_{self.lr}_bs_{self.bs}_scr_{self.cfg['SCALER_CLASSF']}_yjt_{self.cfg['APPLY_YJ_TRANSFORM_CLASSF']}_run_{self.cfg['RUN_DATE_CLASSF']}.pkl"
        )
        # save  model
        torch.save(
            self.model,
            f"{self.cfg['PATH_SAVE_NN_CLASSF']}/{self.cfg['SAVE_NAME_NN']}_e_{self.cfg['EPOCHS']}_lr_{self.lr}_bs_{self.bs}_scr_{self.cfg['SCALER_CLASSF']}_yjt_{self.cfg['APPLY_YJ_TRANSFORM_CLASSF']}_run_{self.cfg['RUN_DATE_CLASSF']}.pt")

    def calibrate(self):
        self.model.eval()  # Set model to evaluation mode
        predictions = []
        tsr_input, tsr_output = self.dataset_to_tensor(self.valid_loader.dataset)
        with torch.no_grad():
            outputs = self.model(tsr_input.double().to(self.device))
            predictions.extend(outputs.squeeze().cpu().numpy())

        # Fit calibration model
        calibration_model = LogisticRegression()
        calibration_model.fit(np.array(predictions).reshape(-1, 1), tsr_output.numpy())
        return calibration_model

    def predict_calibrated(self, y_pred):
        calibrated_outputs = self.calibration_model.predict_proba(y_pred.reshape(-1, 1))[:, 1]
        return calibrated_outputs

    def run_training(self):
        """"""
        self.train()
        self.model.eval()
        tsr_input, tsr_output = self.dataset_to_tensor(self.test_loader.dataset)
        input_data = tsr_input.numpy()
        df_test_data = pd.DataFrame(input_data, columns=self.cfg['INPUT_COLS_MAG_CLASSF'])
        if self.cfg['APPLY_SCALER_CLASSF'] is True:
            df_test_data = pd.DataFrame(self.galaxies.scaler.inverse_transform(df_test_data), columns=df_test_data.keys())

        if self.cfg['APPLY_YJ_TRANSFORM_CLASSF'] is True:
            if self.cfg['TRANSFORM_COLS'] is None:
                trans_col = df_test_data.keys()
            else:
                trans_col = self.cfg['TRANSFORM_COLS_CLASSF']
            df_test_data = self.galaxies.yj_inverse_transform_data(
                data_frame=df_test_data,
                columns=trans_col
            )
        detected_true = tsr_output.numpy()
        df_test_data['detected_true'] = detected_true
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

        df_test_data['detected'] = detected
        df_test_data['detected_calibrated'] = detected_calibrated
        df_test_data['probability'] = probability
        df_test_data['probability_calibrated'] = probability_calibrated

        if self.cfg['PLOT_CLASSF'] is True:
            self.create_plots(df_test_data, self.cfg['EPOCHS_CLASSF'])

        if self.cfg['SAVE_NN_CLASSF'] is True:
            self.save_model()

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
            sns.scatterplot(self.lst_loss)
            if self.cfg['SHOW_PLOT_CLASSF'] is True:
                plt.show()
            if self.cfg['SAVE_PLOT_CLASSF'] is True:
                plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF'][f'LOSS']}/loss_{epoch}.png")
            plt.clf()



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
