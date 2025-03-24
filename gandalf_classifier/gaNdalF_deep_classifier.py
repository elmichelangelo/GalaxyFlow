import torch
import os
import seaborn as sns
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from Handler import plot_classification_results, plot_confusion_matrix, plot_roc_curve, plot_recall_curve, plot_probability_hist, plot_multivariate_clf
from sklearn.metrics import accuracy_score
from gandalf_galaxie_dataset import DESGalaxies
from torch import nn
from sklearn.linear_model import LogisticRegression
import joblib
import random
import logging
import csv


class gaNdalFClassifier(nn.Module):
    def __init__(self,
                 cfg,
                 galaxies,
                 iteration,
                 performance_logger
                 ):
        super().__init__()
        self.cfg = cfg
        self.best_loss = float('inf')
        self.best_acc = 0.0
        self.best_epoch = 0
        self.iteration = iteration
        self.performance_logger = performance_logger
        self.batch_size = 16
        self.galaxies = galaxies
        self.learning_rate = 1
        self.activation = nn.ReLU
        self.number_hidden = []
        self.number_layer = 0
        self.use_batchnorm = False
        self.dropout_prob = 0.0

        self.device = torch.device(cfg["DEVICE_CLASSF"])
        self.lst_loss = []

        self.model = self.build_random_model(
            input_dim=len(self.cfg['INPUT_COLS_MAG_CLASSF']),
            output_dim=1
        )

        self.model = self.model.float()

        # self.train_loader, self.valid_loader, self.test_loader, self.galaxies = self.init_dataset()
        self.train_loader = DataLoader(self.galaxies.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.valid_loader = DataLoader(self.galaxies.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(self.galaxies.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.cfg['PATH_PLOTS_FOLDER'] = {}
        self.cfg['PATH_OUTPUT_SUBFOLDER'] = f"{self.cfg['PATH_OUTPUT']}/iteration_{self.iteration}"
        self.cfg['PATH_WRITER'] = f"{self.cfg['PATH_OUTPUT_SUBFOLDER']}/{self.cfg['FOLDER_WRITER']}"
        self.cfg['PATH_PLOTS'] = f"{self.cfg['PATH_OUTPUT_SUBFOLDER']}/{self.cfg['FOLDER_PLOTS']}"
        self.cfg['PATH_SAVE_NN'] = f"{self.cfg['PATH_OUTPUT_SUBFOLDER']}/{self.cfg['FOLDER_SAVE_NN']}"

        for plot in self.cfg['PLOTS_CLASSF']:
            self.cfg[f'PATH_PLOTS_FOLDER'][plot.upper()] = f"{self.cfg['PATH_PLOTS']}/{plot}"

        self.make_dirs()

        print(f"Number of Layers: {self.number_layer}")
        print(f"Hidden Sizes: {self.number_hidden}")
        print(f"Activation Functions: {self.activation}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Yeo-Johnson Transformation: {self.cfg['APPLY_YJ_TRANSFORM_CLASSF']}")
        print(f"MaxAbs Scaler: {self.cfg['APPLY_SCALER_CLASSF']}")

        self.training_logger = logging.getLogger("TrainingLogger")
        self.training_logger.setLevel(logging.INFO)
        self.training_handler = logging.FileHandler(f"{self.cfg['PATH_OUTPUT']}/model_info_{self.cfg['RUN_DATE']}.log", mode='a')
        self.training_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(message)s'))
        self.training_logger.addHandler(self.training_handler)

        self.training_logger.info(f"#########################################################################")
        self.training_logger.info(f"######################## Iteration {self.iteration} #####################")
        self.training_logger.info(f"Number of Layers: {self.number_layer}")
        self.training_logger.info(f"Hidden Sizes: {self.number_hidden}")
        self.training_logger.info(f"Activation Functions: {self.activation}")
        self.training_logger.info(f"Learning Rate: {self.learning_rate}")
        self.training_logger.info(f"Batch Size: {self.batch_size}")
        self.training_logger.info(f"Yeo-Johnson Transformation: {self.cfg['APPLY_YJ_TRANSFORM_CLASSF']}")
        self.training_logger.info(f"MaxAbs Scaler: {self.cfg['APPLY_SCALER_CLASSF']}")

        self.performance_logger.info(f"######################## Iteration {self.iteration} #####################")
        self.performance_logger.info(f"Number of Layers: {self.number_layer}")
        self.performance_logger.info(f"Hidden Sizes: {self.number_hidden}")
        self.performance_logger.info(f"Activation Functions: {self.activation}")
        self.performance_logger.info(f"Learning Rate: {self.learning_rate}")
        self.performance_logger.info(f"Batch Size: {self.batch_size}")
        self.performance_logger.info(f"Yeo-Johnson Transformation: {self.cfg['APPLY_YJ_TRANSFORM_CLASSF']}")
        self.performance_logger.info(f"MaxAbs Scaler: {self.cfg['APPLY_SCALER_CLASSF']}")

        self.model.to(self.device)

        # Loss function to calculate the error of the neural net (binary cross entropy)
        self.loss_function = nn.BCELoss()

        self.loss = 0

        # Optimizer to calculate the weight changes
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.best_validation_loss = float('inf')
        self.best_validation_acc = 0.0
        self.best_validation_epoch = 0
        self.best_model = self.model
        self.calibration_model = None

    def build_random_model(self, input_dim, output_dim):
        num_layers = random.choice(self.cfg["POSSIBLE_NUM_LAYERS"])
        self.number_layer = num_layers
        chosen_act_fn = random.choice(self.cfg["ACTIVATIONS"])()
        self.activation = chosen_act_fn.__class__.__name__
        self.learning_rate = random.choice(self.cfg["LEARNING_RATE_CLASSF"])
        self.batch_size = random.choice(self.cfg["BATCH_SIZE_CLASSF"])
        self.cfg["APPLY_YJ_TRANSFORM_CLASSF"] = random.choice(self.cfg["YJ_TRANSFORMATION"])
        self.cfg["APPLY_SCALER_CLASSF"] = random.choice(self.cfg["MAXABS_SCALER"])

        self.use_batchnorm = random.choice(self.cfg["USE_BATCHNORM_CLASSF"])
        self.dropout_prob = random.choice(self.cfg["DROPOUT_PROB_CLASSF"])

        layers = []
        in_features = input_dim
        for _ in range(num_layers):
            out_features = random.choice(self.cfg["POSSIBLE_HIDDEN_SIZES"])
            self.number_hidden.append(out_features)
            layers.append(nn.Linear(in_features, out_features))

            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(out_features))

            layers.append(chosen_act_fn)

            if self.dropout_prob > 0.0:
                layers.append(nn.Dropout(self.dropout_prob))

            in_features = out_features

        # Output-Layer
        layers.append(nn.Linear(in_features, output_dim))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def make_dirs(self):
        """"""
        if not os.path.exists(self.cfg['PATH_OUTPUT_SUBFOLDER']):
            os.mkdir(self.cfg['PATH_OUTPUT_SUBFOLDER'])
        if self.cfg['PLOT_CLASSF'] is True:
            if not os.path.exists(self.cfg['PATH_PLOTS']):
                os.mkdir(self.cfg['PATH_PLOTS'])
            for path_plot in self.cfg['PATH_PLOTS_FOLDER'].values():
                if not os.path.exists(path_plot):
                    os.mkdir(path_plot)

        if self.cfg['SAVE_NN_CLASSF'] is True:
            if not os.path.exists(self.cfg["PATH_SAVE_NN"]):
                os.mkdir(self.cfg["PATH_SAVE_NN"])

    def init_dataset(self):
        galaxies = DESGalaxies(
            cfg=self.cfg,
            kind="classifier_training"
        )

        train_loader = DataLoader(galaxies.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(galaxies.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(galaxies.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
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
            # pbar_train = tqdm(total=len(self.train_loader.dataset), unit="batch", ncols=200)
            epoch_loss = 0.0
            # Iterates over the training data in batches
            for batch_idx, data in enumerate(self.train_loader):
                input_data = data[0].float()
                output_data = data[1].float()
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)
                self.optimizer.zero_grad()  # Clear the gradients
                outputs = self.model(input_data)  # Forward pass
                loss = self.loss_function(outputs.squeeze(), output_data.squeeze())  # Calculate loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Updates the weights
                train_loss += loss.item() * output_data.size(0)
                # pbar_train.update(output_data.size(0))
                # pbar_train.set_description(f"Training,\t"
                #                            f"Epoch: {epoch + 1},\t"
                #                            f"learning rate: {self.learning_rate},\t"
                #                            f"batch size: {self.batch_size},\t"
                #                            f"number of layer: {self.number_layer},\t"
                #                            f"hidden size: {self.number_hidden},\t"
                #                            f"activation: {self.activation},\t"
                #                            f"loss: {train_loss / pbar_train.n}")
                epoch_loss += loss.item()
            # pbar_train.close()
            self.lst_loss.append(epoch_loss / len(self.train_loader))

            # Calculates the average training loss
            train_loss = train_loss / len(self.train_loader.dataset)

            # Sets the model to evaluation mode
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                valid_loss = 0.0
                # pbar_val = tqdm(total=len(self.valid_loader.dataset), unit="batch", ncols=200)

                # Iterates over the validation data in batches
                for batch_idx, data in enumerate(self.valid_loader):
                    input_data = data[0].float()
                    output_data = data[1].float()
                    input_data = input_data.to(self.device)
                    output_data = output_data.to(self.device)
                    outputs = self.model(input_data)
                    loss = self.loss_function(outputs.squeeze(), output_data.squeeze())
                    valid_loss += loss.item() * input_data.size(0)
                #     pbar_val.update(output_data.size(0))
                #     pbar_val.set_description(f"Validation,\t"
                #                              f"Epoch: {epoch + 1},\t"
                #                              f"learning rate: {self.learning_rate},\t"
                #                              f"batch size: {self.batch_size},\t"
                #                              f"number of layer: {self.number_layer},\t"
                #                              f"hidden size: {self.number_hidden},\t"
                #                              f"activation: {self.activation},\t"
                #                              f"loss: {valid_loss / pbar_val.n}")
                # pbar_val.close()

            # Calculates the average validation loss
            valid_loss = valid_loss / len(self.valid_loader.dataset)

            # Prints the training and validation loss
            print(f'Epoch {epoch + 1} \t Training Loss: {train_loss:.4f} \t Validation Loss: {valid_loss:.4f}')
            self.training_logger.info(f'Epoch {epoch + 1} \t Training Loss: {train_loss:.4f} \t Validation Loss: {valid_loss:.4f}')

            # Saves the model if the validation loss has decreased
            if valid_loss <= self.best_loss:
                print(f'Validation loss decreased ({self.best_loss:.6f} --> {valid_loss:.6f})')
                self.best_loss = valid_loss

            if self.cfg['PLOT_MULTIVARIATE_CLF_TRAINING'] is True:
                self.model.eval()

                # Trains the calibration model
                self.calibration_model = self.calibrate()

                tsr_input, tsr_output = self.dataset_to_tensor(self.test_loader.dataset)

                input_data = tsr_input.numpy()
                arr_flow_output = np.ones((len(input_data), len(self.cfg["OUTPUT_COLS_MAG_FLOW"])))
                df_test_data = pd.DataFrame(
                    np.concatenate((input_data, arr_flow_output), axis=1),
                    columns=self.cfg['INPUT_COLS_MAG_CLASSF'] + self.cfg['OUTPUT_COLS_MAG_FLOW']
                )
                if self.cfg['APPLY_SCALER_CLASSF'] is True:
                    df_test_data = pd.DataFrame(
                        self.galaxies.scaler.inverse_transform(df_test_data),
                        columns=df_test_data.keys()
                    )

                if self.cfg['APPLY_YJ_TRANSFORM_CLASSF'] is True:
                    if self.cfg['TRANSFORM_COLS_CLASSF'] is None:
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
                    probability = self.model(tsr_input.float().to(self.device)).squeeze().cpu().numpy()
                probability_calibrated = self.predict_calibrated(probability)
                detected = probability > np.random.rand(len(detected_true))
                detected_calibrated = probability_calibrated > np.random.rand(len(detected_true))
                # validation_accuracy = accuracy_score(detected_true, detected)
                # validation_accuracy_calibrated = accuracy_score(detected_true, detected_calibrated)
                # print(f"Accuracy for learning_rate={self.learning_rate}, batch_size={self.batch_size}: {validation_accuracy * 100.0:.2f}%")
                # logging.info(f'Accuracy (normal) for learning_rate={self.learning_rate}, batch_size={self.batch_size}: {validation_accuracy * 100.0:.2f}%')
                # print(
                #     f"Accuracy calibrated for learning_rate={self.learning_rate}, batch_size={self.batch_size}: {validation_accuracy_calibrated * 100.0:.2f}%")
                # logging.info(
                #     f'Accuracy (calibrated) for learning_rate={self.learning_rate}, batch_size={self.batch_size}: {validation_accuracy_calibrated * 100.0:.2f}%')

                df_test_data['true_detected'] = detected
                df_test_data['detected_calibrated'] = detected_calibrated
                df_test_data['probability'] = probability
                df_test_data['probability_calibrated'] = probability_calibrated

                plot_multivariate_clf(
                    df_balrog_detected=df_test_data[df_test_data['true_detected'] == 1],
                    df_gandalf_detected=df_test_data[df_test_data['detected_calibrated'] == 1],
                    df_balrog_not_detected=df_test_data[df_test_data['true_detected'] == 0],
                    df_gandalf_not_detected=df_test_data[df_test_data['detected_calibrated'] == 0],
                    train_plot=True,
                    columns={
                        # "BDF_MAG_DERED_CALIB_R": {
                        #     "label": "BDF Mag R",
                        #     "range": [17.5, 26.5],
                        #     "position": [0, 0]
                        # },
                        # "BDF_MAG_DERED_CALIB_Z": {
                        #     "label": "BDF Mag Z",
                        #     "range": [17.5, 26.5],
                        #     "position": [0, 1]
                        # },
                        "BDF_T": {
                            "label": "BDF T",
                            "range": [-2, 3],
                            "position": [0, 2]
                        },
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
                        #     "label": "FWHM Z",
                        #     "range": [0.6, 1.16],
                        #     "position": [2, 0]
                        # },
                        # "AIRMASS_WMEAN_R": {
                        #     "label": "AIRMASS R",
                        #     "range": [0.95, 1.45],
                        #     "position": [2, 1]
                        # },
                        # "AIRMASS_WMEAN_I": {
                        #     "label": "AIRMASS I",
                        #     "range": [1, 1.45],
                        #     "position": [2, 2]
                        # },
                        # "AIRMASS_WMEAN_Z": {
                        #     "label": "AIRMASS Z",
                        #     "range": [1, 1.4],
                        #     "position": [2, 3]
                        # },
                        # "MAGLIM_R": {
                        #     "label": "MAGLIM R",
                        #     "range": [23, 24.8],
                        #     "position": [3, 0]
                        # },
                        # "MAGLIM_I": {
                        #     "label": "MAGLIM I",
                        #     "range": [22.4, 24.0],
                        #     "position": [3, 1]
                        # },
                        # "MAGLIM_Z": {
                        #     "label": "MAGLIM Z",
                        #     "range": [21.8, 23.2],
                        #     "position": [3, 2]
                        # },
                        # "EBV_SFD98": {
                        #     "label": "EBV SFD98",
                        #     "range": [-0.01, 0.10],
                        #     "position": [3, 3]
                        # }
                    },
                    show_plot=self.cfg["SHOW_PLOT_CLASSF"],
                    save_plot=self.cfg["SAVE_PLOT_CLASSF"],
                    save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'MULTIVARIATE_CLF']}/{epoch}_classifier_multiv.pdf",
                    sample_size=100000,  # None,
                    x_range=(17.5, 26.5),
                    title=f"gaNdalF vs. Balrog: Photometric Property Distribution Comparison"
                )

        # Trains the calibration model
        self.calibration_model = self.calibrate()

    def forward(self, inputs):
        return self.model(inputs)

    def save_model(self):
        """"""
        # save config model
        joblib.dump(
            self.calibration_model,
            f"{self.cfg['PATH_SAVE_NN']}/gaNdalF_classifier_e_{self.cfg['EPOCHS_CLASSF']}_lr_{self.learning_rate}_bs_{self.batch_size}_scr_{self.cfg['APPLY_SCALER_CLASSF']}_yjt_{self.cfg['APPLY_YJ_TRANSFORM_CLASSF']}_run_{self.cfg['RUN_DATE']}.pkl"
        )
        # save  model
        torch.save(
            self.model,
            f"{self.cfg['PATH_SAVE_NN']}/gaNdalF_classifier_e_{self.cfg['EPOCHS_CLASSF']}_lr_{self.learning_rate}_bs_{self.batch_size}_scr_{self.cfg['APPLY_SCALER_CLASSF']}_yjt_{self.cfg['APPLY_YJ_TRANSFORM_CLASSF']}_run_{self.cfg['RUN_DATE']}.pt")

    def calibrate(self):
        self.model.eval()  # Set model to evaluation mode
        predictions = []
        tsr_input, tsr_output = self.dataset_to_tensor(self.valid_loader.dataset)
        with torch.no_grad():
            outputs = self.model(tsr_input.float().to(self.device))
            predictions.extend(outputs.squeeze().cpu().numpy())

        # Fit calibration model
        calibration_model = LogisticRegression()
        calibration_model.fit(np.array(predictions).reshape(-1, 1), tsr_output.numpy().ravel())
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
        arr_flow_output = np.ones((len(input_data), len(self.cfg["OUTPUT_COLS_MAG_FLOW"])))
        df_test_data = pd.DataFrame(
            np.concatenate((input_data, arr_flow_output), axis=1),
            columns=self.cfg['INPUT_COLS_MAG_CLASSF'] + self.cfg['OUTPUT_COLS_MAG_FLOW']
        )
        if self.cfg['APPLY_SCALER_CLASSF'] is True:
            df_test_data = pd.DataFrame(
                self.galaxies.scaler.inverse_transform(df_test_data),
                columns=df_test_data.keys()
            )

        if self.cfg['APPLY_YJ_TRANSFORM_CLASSF'] is True:
            if self.cfg['TRANSFORM_COLS_CLASSF'] is None:
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
            probability = self.model(tsr_input.float().to(self.device)).squeeze().cpu().numpy()
        probability_calibrated = self.predict_calibrated(probability)
        detected = probability > np.random.rand(len(detected_true))
        detected_calibrated = probability_calibrated > np.random.rand(len(detected_true))
        validation_accuracy = accuracy_score(detected_true, detected)
        validation_accuracy_calibrated = accuracy_score(detected_true, detected_calibrated)
        print(f"Accuracy for lr={self.learning_rate}, bs={self.batch_size}: {validation_accuracy * 100.0:.2f}%")
        self.training_logger.info(f'Accuracy (normal) for lr={self.learning_rate}, bs={self.batch_size}: {validation_accuracy * 100.0:.2f}%')
        print(f"Accuracy calibrated for lr={self.learning_rate}, bs={self.batch_size}: {validation_accuracy_calibrated * 100.0:.2f}%")
        self.training_logger.info(f'Accuracy (calibrated) for lr={self.learning_rate}, bs={self.batch_size}: {validation_accuracy_calibrated * 100.0:.2f}%')

        df_test_data['true_detected'] = detected
        df_test_data['detected_calibrated'] = detected_calibrated
        df_test_data['probability'] = probability
        df_test_data['probability_calibrated'] = probability_calibrated

        csv_path = os.path.join(self.cfg['PATH_OUTPUT'], "training_results.csv")
        pkl_path = os.path.join(self.cfg['PATH_OUTPUT'], "training_results.pkl")

        # Prepare data as dictionary
        data = {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "hidden_sizes": ",".join(str(h) for h in self.number_hidden),
            "number_of_layers": self.number_layer,
            "activation_functions": ",".join(str(a) for a in self.activation),
            "batch_norm_layer": self.use_batchnorm,
            "dropout": self.dropout_prob,
            "yj_transformation": self.cfg["APPLY_YJ_TRANSFORM_CLASSF"],
            "maxabs_scaler": self.cfg["APPLY_SCALER_CLASSF"],
            "accuracy": validation_accuracy,
            "calibrated_accuracy": validation_accuracy_calibrated
        }

        # Check if csv already exists
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, delimiter=';')
        else:
            df = pd.DataFrame(columns=data.keys())

        # Append new data
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

        # Save CSV and Pickle
        df.to_csv(csv_path, index=False, sep=';')
        df.to_pickle(pkl_path)

        # with open(csv_path, "a", newline='') as f:
        #     writer = csv.writer(f, delimiter=";")  # Oder Komma, je nach Bedarf
        #     if write_header:
        #         writer.writerow([
        #             "learning_rate",
        #             "batch_size",
        #             "hidden_sizes",
        #             "number_of_layers",
        #             "activation_functions",
        #             "batch_norm_layer",
        #             "dropout",
        #             "yj_transformation",
        #             "maxabs_scaler",
        #             "accuracy",
        #             "calibrated_accuracy"
        #         ])
        #
        #     writer.writerow([
        #         self.learning_rate,
        #         self.batch_size,
        #         ",".join(str(h) for h in self.number_hidden),
        #         self.number_layer,
        #         ",".join(str(a) for a in self.activation),
        #         self.use_batchnorm,
        #         self.dropout_prob,
        #         self.cfg["APPLY_YJ_TRANSFORM_CLASSF"],
        #         self.cfg["APPLY_SCALER_CLASSF"],
        #         f"{validation_accuracy:.4f}",
        #         f"{validation_accuracy_calibrated:.4f}"
        #     ])

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
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_BDF_RI_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_BDF_IZ_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_Color_UG_GR_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_Color_RI_IZ_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_Color_ZJ_JH_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_BDF_TG_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_FWHM_RI_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_FWHM_IZ_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_AIRMASS_RI_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_AIRMASS_IZ_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_MAGLIM_RI_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_MAGLIM_IZ_epoch_{epoch}.png",
            f"{self.cfg['PATH_PLOTS_FOLDER'][f'MISS-CLASSIFICATION']}/classf_EBV_Color_{epoch}.png",
        ]
        if self.cfg['PLOT_MULTIVARIATE_CLF'] is True:
            plot_multivariate_clf(
                df_balrog_detected=df_test_data[df_test_data['true_detected'] == 1],
                df_gandalf_detected=df_test_data[df_test_data['detected_calibrated'] == 1],
                df_balrog_not_detected=df_test_data[df_test_data['true_detected'] == 0],
                df_gandalf_not_detected=df_test_data[df_test_data['detected_calibrated'] == 0],
                train_plot=False,
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
                show_plot=self.cfg["SHOW_PLOT_CLASSF"],
                save_plot=self.cfg["SAVE_PLOT_CLASSF"],
                save_name=f"{self.cfg['PATH_OUTPUT']}/{self.iteration}_classifier_multiv.pdf",
                sample_size=None,
                x_range=(17.5, 26.5),
                title=f"nl: {self.number_layer}; nh: {self.number_hidden}; af: {self.activation}; lr: {self.learning_rate}; bs: {self.batch_size}; YJ: {self.cfg['APPLY_YJ_TRANSFORM_CLASSF']}; scaler: {self.cfg['APPLY_SCALER_CLASSF']}"
            )
        if self.cfg['PLOT_MISS_CLASSF'] is True:
            for idx_cols, cols in enumerate(lst_cols):
                plot_classification_results(
                    data_frame=df_test_data,
                    cols=cols,
                    show_plot=self.cfg['SHOW_PLOT_CLASSF'],
                    save_plot=self.cfg['SAVE_PLOT_CLASSF'],
                    save_name=lst_save_names[idx_cols],
                    title=f"Classification Results, lr={self.learning_rate}, bs={self.batch_size}, epoch={epoch}"
                )

        if self.cfg['PLOT_MATRIX'] is True:
            plot_confusion_matrix(
                data_frame=df_test_data,
                show_plot=self.cfg['SHOW_PLOT_CLASSF'],
                save_plot=self.cfg['SAVE_PLOT_CLASSF'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'CONFUSION_MATRIX']}/confusion_matrix_epoch_{epoch}.png",
                title=f"Confusion Matrix, lr={self.learning_rate}, bs={self.batch_size}, epoch={epoch}"
            )

        # ROC und AUC
        if self.cfg['PLOT_ROC_CURVE'] is True:
            plot_roc_curve(
                data_frame=df_test_data,
                show_plot=self.cfg['SHOW_PLOT_CLASSF'],
                save_plot=self.cfg['SAVE_PLOT_CLASSF'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'ROC_CURVE']}/roc_curve_epoch_{epoch}.png",
                title=f"Receiver Operating Characteristic (ROC) Curve, lr={self.learning_rate}, bs={self.batch_size}, epoch={epoch}"
            )

        # Precision-Recall-Kurve
        if self.cfg['PLOT_PRECISION_RECALL_CURVE'] is True:
            plot_recall_curve(
                data_frame=df_test_data,
                show_plot=self.cfg['SHOW_PLOT_CLASSF'],
                save_plot=self.cfg['SAVE_PLOT_CLASSF'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'PRECISION_RECALL_CURVE']}/precision_recall_curve_epoch_{epoch}.png",
                title=f"recision-Recall Curve, lr={self.learning_rate}, bs={self.batch_size}, epoch={epoch}"
            )

        # Histogramm der vorhergesagten Wahrscheinlichkeiten
        if self.cfg['PLOT_PROBABILITY_HIST'] is True:
            plot_probability_hist(
                data_frame=df_test_data,
                show_plot=self.cfg['SHOW_PLOT_CLASSF'],
                save_plot=self.cfg['SAVE_PLOT_CLASSF'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER'][f'PROB_HIST']}/probability_histogram{epoch}.png",
                title=f"probability histogram, lr={self.learning_rate}, bs={self.batch_size}, epoch={epoch}"
            )

        if self.cfg['PLOT_LOSS_CLASSF'] is True:
            sns.scatterplot(self.lst_loss)
            if self.cfg['SHOW_PLOT_CLASSF'] is True:
                plt.show()
            if self.cfg['SAVE_PLOT_CLASSF'] is True:
                plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER'][f'LOSS']}/loss_{epoch}.png")
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
                inputs = inputs.float().to('cpu')
                labels = labels.float().to('cpu')
                data_list.append(inputs.numpy())
                label_list.append(labels.numpy())

        data_array = np.concatenate(data_list, axis=0)
        label_array = np.concatenate(label_list, axis=0)

        return data_array, label_array


if __name__ == '__main__':
    pass
