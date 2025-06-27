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
                 classifier_logger
                 ):
        super().__init__()
        self.cfg = cfg
        self.best_loss = float('inf')
        self.best_acc = 0.0
        self.best_epoch = 0
        self.iteration = iteration
        self.classifier_logger = classifier_logger
        self.batch_size = 16
        self.galaxies = galaxies
        self.learning_rate = 1
        self.activation = nn.ReLU
        self.number_hidden = []
        self.number_layer = 0
        self.use_batchnorm = False
        self.dropout_prob = 0.0

        self.scalers = joblib.load(f"{self.cfg['PATH_TRANSFORMERS']}/{self.cfg['FILENAME_SCALER']}")

        self.device = torch.device(cfg["DEVICE_CLASSF"])
        self.lst_loss = []

        self.model = self.build_random_model(
            input_dim=len(self.cfg['INPUT_COLS']),
            output_dim=1
        )

        self.model = self.model.float()

        # self.train_loader, self.valid_loader, self.test_loader, self.galaxies = self.init_dataset()
        # self.train_loader = DataLoader(self.galaxies.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        # self.valid_loader = DataLoader(self.galaxies.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        # self.test_loader = DataLoader(self.galaxies.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.cfg['PATH_PLOTS_FOLDER'] = {}
        self.cfg['PATH_OUTPUT_SUBFOLDER'] = f"{self.cfg['PATH_OUTPUT']}/iteration_{self.iteration}"
        self.cfg['PATH_WRITER'] = f"{self.cfg['PATH_OUTPUT_SUBFOLDER']}/{self.cfg['FOLDER_WRITER']}"
        self.cfg['PATH_PLOTS'] = f"{self.cfg['PATH_OUTPUT_SUBFOLDER']}/{self.cfg['FOLDER_PLOTS']}"
        self.cfg['PATH_SAVE_NN'] = f"{self.cfg['PATH_OUTPUT_SUBFOLDER']}/{self.cfg['FOLDER_SAVE_NN']}"

        for plot in self.cfg['PLOTS_CLASSF']:
            self.cfg[f'PATH_PLOTS_FOLDER'][plot.upper()] = f"{self.cfg['PATH_PLOTS']}/{plot}"

        self.make_dirs()

        self.classifier_logger.log_info_stream(f"#########################################################################")
        self.classifier_logger.log_info_stream(f"######################## Iteration {self.iteration} #####################")
        self.classifier_logger.log_info_stream(f"Number of Layers: {self.number_layer}")
        self.classifier_logger.log_info_stream(f"Hidden Sizes: {self.number_hidden}")
        self.classifier_logger.log_info_stream(f"Activation Functions: {self.activation}")
        self.classifier_logger.log_info_stream(f"Learning Rate: {self.learning_rate}")
        self.classifier_logger.log_info_stream(f"Batch Size: {self.batch_size}")

        self.model.to(self.device)

        self.loss_function = nn.BCELoss()

        self.loss = 0

        # Optimizer to calculate the weight changes
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.best_validation_loss = float('inf')
        self.best_validation_acc = 0.0
        self.best_validation_epoch = 0
        self.best_model = self.model

    def build_random_model(self, input_dim, output_dim):
        new_catch = False
        pkl_path = os.path.join(self.cfg['PATH_OUTPUT'], "training_results.pkl")

        while not new_catch:
            self.number_hidden = []

            num_layers = random.choice(self.cfg["POSSIBLE_NUM_LAYERS"])
            self.number_layer = num_layers
            chosen_act_fn = random.choice(self.cfg["ACTIVATIONS"])()
            self.activation = chosen_act_fn.__class__.__name__
            self.learning_rate = random.choice(self.cfg["LEARNING_RATE"])
            self.batch_size = random.choice(self.cfg["BATCH_SIZE"])

            self.use_batchnorm = random.choice(self.cfg["USE_BATCHNORM"])
            self.dropout_prob = random.choice(self.cfg["DROPOUT_PROB"])

            layers = []
            in_features = input_dim
            for idx, _ in enumerate(range(num_layers)):
                if self.cfg["RANDOM"] is True:
                    out_features = random.choice(self.cfg["POSSIBLE_HIDDEN_SIZES"])
                else:
                    out_features = self.cfg["HIDDEN_SIZES"][idx]

                self.number_hidden.append(out_features)
                layers.append(nn.Linear(in_features, out_features))

                if self.use_batchnorm:
                    layers.append(nn.BatchNorm1d(out_features))

                layers.append(chosen_act_fn)

                if self.dropout_prob > 0.0:
                    layers.append(nn.Dropout(self.dropout_prob))

                in_features = out_features



            data = {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "hidden_sizes": ",".join(str(h) for h in self.number_hidden),
                "number_of_layers": self.number_layer,
                "activation_functions": ",".join(str(a) for a in self.activation),
                "batch_norm_layer": self.use_batchnorm,
                "dropout": self.dropout_prob
            }

            # Check if csv already exists
            if os.path.exists(pkl_path):
                df = pd.read_pickle(pkl_path)
                if not ((df[list(data)] == pd.Series(data)).all(axis=1)).any():
                    new_catch = True
            else:
                new_catch = True

        # Output-Layer
        layers.append(nn.Linear(in_features, output_dim))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def make_dirs(self):
        """"""
        os.makedirs(self.cfg['PATH_OUTPUT_SUBFOLDER'], exist_ok=True)
        
        if self.cfg['PLOT_CLASSF'] is True:
            os.makedirs(self.cfg['PATH_PLOTS'], exist_ok=True)
            for path_plot in self.cfg['PATH_PLOTS_FOLDER'].values():
                os.makedirs(path_plot, exist_ok=True)
                
        if self.cfg['SAVE_NN_CLASSF'] is True:
            os.makedirs(self.cfg['PATH_SAVE_NN'], exist_ok=True)

    def train(self, mode=True):
        """
        Trains the neural network model.

        :param mode: A boolean value that determines the training mode of the model.
        """

        super().train(mode=mode)

        df_train = self.galaxies.train_dataset
        df_valid = self.galaxies.valid_dataset
        num_samples = len(df_train)
        batch_size = self.batch_size
        n_batches = (num_samples + batch_size - 1) // batch_size  # ceil division

        num_valid_samples = len(df_valid)
        n_valid_batches = (num_valid_samples + batch_size - 1) // batch_size

        # Training cycle that runs for a certain number of epoch
        for epoch in range(self.cfg['EPOCHS']):
            self.model.train()  # Set the model to training mode
            train_loss = 0.0
            pbar_train = tqdm(total=len(df_train), unit="batch", ncols=200)
            df_train_shuffled = df_train.sample(frac=1).reset_index(drop=True)

            # Iterates over the training data in batches
            for i in range(n_batches):
                batch_df = df_train_shuffled.iloc[i * batch_size:(i + 1) * batch_size]

                input_data = torch.tensor(batch_df[self.cfg["INPUT_COLS"]].values, dtype=torch.float32)
                output_data = torch.tensor(batch_df[self.cfg["OUTPUT_COLS"]].values, dtype=torch.float32)

                for j, col in enumerate(self.cfg["INPUT_COLS"]):
                    if col in self.cfg.get("COLUMNS_LOG1P", []):
                        input_data[:, j] = torch.log1p(input_data[:, j])
                    scale = self.scalers[col].scale_[0]
                    min_ = self.scalers[col].min_[0]
                    input_data[:, j] = input_data[:, j] * scale + min_

                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(input_data)
                loss = self.loss_function(outputs.squeeze(), output_data.squeeze()).mean()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * output_data.size(0)
                pbar_train.update(output_data.size(0))
                pbar_train.set_description(f"Training,\t"
                                           f"Epoch: {epoch + 1},\t"
                                           f"learning rate: {self.learning_rate},\t"
                                           f"batch size: {self.batch_size},\t"
                                           f"number of layer: {self.number_layer},\t"
                                           f"hidden size: {self.number_hidden},\t"
                                           f"activation: {self.activation},\t"
                                           f"loss: {train_loss / pbar_train.n}")
            avg_train_loss = train_loss / num_samples
            pbar_train.close()
            self.lst_loss.append(avg_train_loss)

            # Sets the model to evaluation mode
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                valid_loss = 0.0
                pbar_val = tqdm(total=num_valid_samples, unit="batch", ncols=200)

                for i in range(n_valid_batches):
                    batch_df = df_valid.iloc[i * batch_size:(i + 1) * batch_size]

                    input_data = torch.tensor(batch_df[self.cfg["INPUT_COLS"]].values, dtype=torch.float32)
                    output_data = torch.tensor(batch_df[self.cfg["OUTPUT_COLS"]].values, dtype=torch.float32)

                    for j, col in enumerate(self.cfg["INPUT_COLS"]):
                        if col in self.cfg.get("COLUMNS_LOG1P", []):
                            input_data[:, j] = torch.log1p(input_data[:, j])
                        scale = self.scalers[col].scale_[0]
                        min_ = self.scalers[col].min_[0]
                        input_data[:, j] = input_data[:, j] * scale + min_

                    input_data = input_data.to(self.device)
                    output_data = output_data.to(self.device)

                    outputs = self.model(input_data)
                    loss = self.loss_function(outputs.squeeze(), output_data.squeeze()).mean()

                    valid_loss += loss.item() * output_data.size(0)
                    pbar_val.update(output_data.size(0))
                    pbar_val.set_description(f"Validation,\t"
                                             f"Epoch: {epoch + 1},\t"
                                             f"learning rate: {self.learning_rate},\t"
                                             f"batch size: {self.batch_size},\t"
                                             f"number of layer: {self.number_layer},\t"
                                             f"hidden size: {self.number_hidden},\t"
                                             f"activation: {self.activation},\t"
                                             f"loss: {valid_loss / pbar_val.n}")
                pbar_val.close()

            # Calculates the average validation loss
            valid_loss = valid_loss / num_valid_samples

            # Prints the training and validation loss
            self.classifier_logger.log_info_stream(f'Epoch {epoch + 1} \t Training Loss: {train_loss:.4f} \t Validation Loss: {valid_loss:.4f}')

            # Saves the model if the validation loss has decreased
            if valid_loss <= self.best_loss:
                self.classifier_logger.log_info_stream(f'Validation loss decreased ({self.best_loss:.6f} --> {valid_loss:.6f})')
                self.best_loss = valid_loss

    def forward(self, inputs):
        return self.model(inputs)

    def save_model(self):
        """"""
        # save config model
        # joblib.dump(
        #     self.calibration_model,
        #     f"{self.cfg['PATH_SAVE_NN']}/gaNdalF_classifier_e_{self.cfg['EPOCHS_CLASSF']}_lr_{self.learning_rate}_bs_{self.batch_size}_scr_{self.cfg['APPLY_SCALER_CLASSF']}_yjt_{self.cfg['APPLY_YJ_TRANSFORM_CLASSF']}_run_{self.cfg['RUN_DATE']}.pkl"
        # )
        # save  model
        torch.save(
            self.model,
            f"{self.cfg['PATH_SAVE_NN']}/gaNdalF_classifier_e_{self.cfg['EPOCHS_CLASSF']}_lr_{self.learning_rate}_bs_{self.batch_size}_scr_{self.cfg['APPLY_SCALER_CLASSF']}_yjt_{self.cfg['APPLY_YJ_TRANSFORM_CLASSF']}_run_{self.cfg['RUN_DATE']}.pt")

    # def calibrate(self):
    #     self.model.eval()
    #     predictions = []
    #     tsr_input, tsr_output = self.dataset_to_tensor(self.valid_loader.dataset)
    #     with torch.no_grad():
    #         outputs = self.model(tsr_input.float().to(self.device))
    #         predictions = outputs.squeeze().cpu().numpy()
    #
    #     # Drop samples with NaN predictions
    #     mask = ~np.isnan(predictions)
    #     predictions = np.array(predictions)[mask].reshape(-1, 1)
    #     labels = tsr_output.numpy().ravel()[mask]
    #
    #     calibration_model = LogisticRegression()
    #     calibration_model.fit(predictions, labels)
    #     return calibration_model
    #
    # def predict_calibrated(self, y_pred):
    #     calibrated_outputs = self.calibration_model.predict_proba(y_pred.reshape(-1, 1))[:, 1]
    #     return calibrated_outputs

    def run_training(self):
        """"""
        self.train()
        self.model.eval()
        df_test = self.galaxies.test_dataset
        df_test_input = df_test[self.cfg["INPUT_COLS"]].copy()
        # tsr_output = torch.tensor(df_test[self.cfg["OUTPUT_COLS"]].values, dtype=torch.float32)
        detected_true = df_test['detected'].values
        for j, col in enumerate(self.cfg["INPUT_COLS"]):
            if col in self.cfg.get("COLUMNS_LOG1P", []):
                df_test_input[col] = np.log1p(df_test_input[col])
            df_test_input[col] = self.scalers[col].transform(df_test_input[[col]])
        tsr_input = torch.tensor(df_test_input.values, dtype=torch.float32)

        # tsr_input, tsr_output = self.dataset_to_tensor(self.test_loader.dataset)
        # input_data = tsr_input.numpy()
        # arr_flow_output = np.ones((len(input_data), len(self.cfg["OUTPUT_COLS_MAG_FLOW"])))
        # df_test_data = pd.DataFrame(
        #     np.concatenate((input_data, arr_flow_output), axis=1),
        #     columns=self.cfg['INPUT_COLS_MAG_CLASSF'] + self.cfg['OUTPUT_COLS_MAG_FLOW']
        # )
        # detected_true = tsr_output.numpy().reshape(-1).astype(int)
        # df_test_data['detected_true'] = detected_true

        # Validate the model
        with torch.no_grad():
            probability = self.model(tsr_input.to(self.device)).squeeze().cpu().numpy()
        detected = (probability > np.random.rand(len(probability))).astype(int)
        detected = detected.astype(int)


        df_test['gandalf_probability'] = probability
        df_test['gandalf_detected'] = detected

        validation_accuracy = accuracy_score(detected_true, detected)
        self.classifier_logger.log_info_stream(f"Accuracy for lr={self.learning_rate}, bs={self.batch_size}: {validation_accuracy * 100.0:.2f}%")
        self.classifier_logger.log_info_stream(f'Accuracy (normal) for lr={self.learning_rate}, bs={self.batch_size}: {validation_accuracy * 100.0:.2f}%')

        self.classifier_logger.log_info_stream(f"detected shape: {detected.shape}")
        self.classifier_logger.log_info_stream(f"detected_true shape: {detected_true.shape}")

        false_positives = np.sum((detected == 1) & (detected_true == 0))
        false_negatives = np.sum((detected == 0) & (detected_true == 1))

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

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
            "accuracy": validation_accuracy,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "length": len(detected_true),
            "trainable_params": trainable_params
        }

        self.classifier_logger.log_info_stream(data)

        df_cvs = pd.DataFrame([data])
        write_header = not os.path.exists(csv_path)
        df_cvs.to_csv(csv_path, index=False, sep=';', mode="a", header=write_header)

        if os.path.exists(pkl_path):
            df_pkl = pd.read_pickle(pkl_path)
        else:
            df_pkl = pd.DataFrame(columns=data.keys())
        df_pkl = pd.concat([df_pkl, pd.DataFrame([data])], ignore_index=True)
        df_pkl.to_pickle(pkl_path)
        #
        # if self.cfg['PLOT_CLASSF'] is True:
        #     self.create_plots(df_test_data, self.cfg['EPOCHS_CLASSF'])
        #
        # if self.cfg['SAVE_NN_CLASSF'] is True:
        #     self.save_model()

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
                df_balrog_detected=df_test_data[df_test_data['detected_true'] == 1],
                df_gandalf_detected=df_test_data[df_test_data['detected_calibrated'] == 1],
                df_balrog_not_detected=df_test_data[df_test_data['detected_true'] == 0],
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
                save_name=f"{self.cfg['PATH_OUTPUT']}/{self.iteration}_balrog_classifier_multiv.pdf",
                sample_size=None,
                x_range=(17.5, 26.5),
                title=f"nl: {self.number_layer}; nh: {self.number_hidden}; af: {self.activation}; lr: {self.learning_rate}; bs: {self.batch_size}; YJ: {self.cfg['APPLY_YJ_TRANSFORM_CLASSF']}; scaler: {self.cfg['APPLY_SCALER_CLASSF']}"
            )
            plot_multivariate_clf(
                df_balrog_detected=df_test_data[df_test_data['detected_non_calibrated'] == 1],
                df_gandalf_detected=df_test_data[df_test_data['detected_calibrated'] == 1],
                df_balrog_not_detected=df_test_data[df_test_data['detected_non_calibrated'] == 0],
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
                save_name=f"{self.cfg['PATH_OUTPUT']}/{self.iteration}_classifier_calibrated_non_calibrated_multiv.pdf",
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
