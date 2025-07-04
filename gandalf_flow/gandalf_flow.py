import copy
import torch
import torch.optim as optim
import torch.utils.data

import torch.nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Handler import count_parameters, fnn, calc_color, make_gif, loss_plot, plot_compare_corner, residual_plot, plot_mean_or_std, plot_features
from gandalf_galaxie_dataset import DESGalaxies
import pandas as pd
import numpy as np
import seaborn as sns
import os
import joblib
from torch.utils.data import DataLoader, TensorDataset
from Handler.helper_functions import inverse_transform_df
import matplotlib.pyplot as plt
import math
from datetime import datetime

torch.set_default_dtype(torch.float64)


class gaNdalFFlow(object):

    def __init__(self,
                 cfg,
                 learning_rate,
                 weight_decay,
                 number_hidden,
                 number_blocks,
                 batch_size,
                 train_flow_logger
                 ):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg["DEVICE"])
        self.act = cfg["ACTIVATION_FUNCTION"]
        self.train_flow_logger = train_flow_logger

        self.lst_epochs = []
        self.lst_mean_mag_r = []
        self.lst_mean_mag_i = []
        self.lst_mean_mag_z = []
        self.lst_mean_snr = []
        self.lst_mean_size_ratio = []
        self.lst_mean_t = []
        self.lst_std_mag_r = []
        self.lst_std_mag_i = []
        self.lst_std_mag_z = []
        self.lst_std_snr = []
        self.lst_std_size_ratio = []
        self.lst_std_t = []
        self.lst_mean_mag_r_cut = []
        self.lst_mean_mag_i_cut = []
        self.lst_mean_mag_z_cut = []
        self.lst_mean_snr_cut = []
        self.lst_mean_size_ratio_cut = []
        self.lst_mean_t_cut = []
        self.lst_std_mag_r_cut = []
        self.lst_std_mag_i_cut = []
        self.lst_std_mag_z_cut = []
        self.lst_std_snr_cut = []
        self.lst_std_size_ratio_cut = []
        self.lst_std_t_cut = []
        self.lst_train_loss_per_batch = []
        self.lst_train_loss_per_epoch = []
        self.lst_valid_loss_per_batch = []
        self.lst_valid_loss_per_epoch = []

        self.dict_delta_color_color = {}
        self.dict_delta_color_color_mcal = {}
        self.dict_delta_unsheared = {}
        self.dict_delta_unsheared_mcal = {}
        self.dict_delta_color_diff = {}
        self.dict_delta_color_diff_mcal = {}

        if cfg["SCALER_TYPE"].lower() == "minmax":
            self.scalers = joblib.load(f"{self.cfg['PATH_TRANSFORMERS']}/{self.cfg['FILENAME_MINMAX_SCALER']}")
        else:
            self.scalers = joblib.load(f"{self.cfg['PATH_TRANSFORMERS']}/{self.cfg['FILENAME_STANDARD_SCALER']}")

        self.yj_transformer = joblib.load(f"{self.cfg['PATH_TRANSFORMERS']}/{self.cfg['FILENAME_YJ_TRANSFORMER']}")

        self.bs = batch_size
        self.lr = learning_rate
        self.wd = weight_decay
        self.nh = number_hidden
        self.nb = number_blocks

        self.cfg['PATH_PLOTS_FOLDER'] = {}
        self.cfg['PATH_OUTPUT_SUBFOLDER'] = f"{self.cfg['PATH_OUTPUT']}/lr_{self.lr}_wd_{self.wd}_nh_{self.nh}_nb_{self.nb}_bs_{self.bs}"
        self.cfg['PATH_OUTPUT_SUBFOLDER_CATALOGS'] = f"{self.cfg['PATH_OUTPUT_CATALOGS']}/lr_{self.lr}_wd_{self.wd}_nh_{self.nh}_nb_{self.nb}_bs_{self.bs}"
        self.cfg['PATH_WRITER'] = (f"{self.cfg['PATH_OUTPUT_SUBFOLDER']}/{self.cfg['FOLDER_WRITER']}/"
                                   f"lr_{self.lr}_nh_{self.nh}_nb_{self.nb}_bs_{self.bs}")
        self.cfg['PATH_PLOTS'] = f"{self.cfg['PATH_OUTPUT_SUBFOLDER']}/{self.cfg['FOLDER_PLOTS']}"
        self.cfg['PATH_SAVE_NN'] = f"{self.cfg['PATH_OUTPUT_SUBFOLDER']}/{self.cfg['FOLDER_SAVE_NN']}"

        for plot in self.cfg['PLOT_FOLDERS_FLOW']:
            self.cfg[f'PATH_PLOTS_FOLDER'][plot.upper()] = f"{self.cfg['PATH_PLOTS']}/{plot}"

        self.make_dirs()
        self.writer = SummaryWriter(
            log_dir=cfg['PATH_WRITER'],
            comment=f"learning rate: {self.lr} "
                    f"number hidden: {self.nh}_"
                    f"number blocks: {self.nb}_"
                    f"batch size: {self.bs}"
        )
        self.galaxies = self.init_dataset()  # self.train_loader, self.valid_loader, self.test_sampled_data, '

        self.model, self.optimizer = self.init_network(
            num_outputs=len(cfg[f"OUTPUT_COLS"]),
            num_input=len(cfg[f"INPUT_COLS"])
        )

        with torch.no_grad():
            for i in range((len(self.galaxies.train_dataset) + self.bs - 1) // self.bs):
                batch_df = self.galaxies.train_dataset.iloc[i * self.bs:(i + 1) * self.bs]
                input_data = torch.tensor(batch_df[self.cfg["INPUT_COLS"]].values, dtype=torch.float64)
                output_data = torch.tensor(batch_df[self.cfg["OUTPUT_COLS"]].values, dtype=torch.float64)
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)
                self.writer.add_graph(self.model, (output_data, input_data))
                self.writer.close()
                break

            total_parameters = count_parameters(self.model)
            self.train_flow_logger.log_info_stream(f"Total trainable parameters: {total_parameters}")

        self.global_step = 0
        self.best_validation_loss = np.float64('inf')
        self.best_validation_epoch = 1
        self.best_train_loss = np.float64('inf')
        self.best_train_epoch = 1
        self.best_model = self.model

    def make_dirs(self):
        """"""
        if not os.path.exists(self.cfg['PATH_OUTPUT_SUBFOLDER']):
            os.mkdir(self.cfg['PATH_OUTPUT_SUBFOLDER'])
        if not os.path.exists(self.cfg['PATH_OUTPUT_SUBFOLDER_CATALOGS']):
            os.mkdir(self.cfg['PATH_OUTPUT_SUBFOLDER_CATALOGS'])
        if self.cfg['PLOT_TEST_FLOW'] is True:
            if not os.path.exists(self.cfg['PATH_PLOTS']):
                os.mkdir(self.cfg['PATH_PLOTS'])
            for path_plot in self.cfg['PATH_PLOTS_FOLDER'].values():
                if not os.path.exists(path_plot):
                    os.mkdir(path_plot)

        if self.cfg['SAVE_NN_FLOW'] is True:
            if not os.path.exists(self.cfg["PATH_SAVE_NN"]):
                os.mkdir(self.cfg["PATH_SAVE_NN"])

    def init_dataset(self):
        """"""
        galaxies = DESGalaxies(
            dataset_logger=self.train_flow_logger,
            cfg=self.cfg
        )

        # Create DataLoaders for training, validation, and testing
        # train_loader = DataLoader(galaxies.train_dataset, batch_size=self.bs, shuffle=False, num_workers=0)
        # valid_loader = DataLoader(galaxies.valid_dataset, batch_size=self.bs, shuffle=False, num_workers=0)
        # test_sampled_data = galaxies.test_dataset.sample(self.cfg["NUMBER_TEST_SAMPLES"], replace=True)
        return galaxies  # train_loader, valid_loader, test_sampled_data, galaxies

    def init_network(self, num_outputs, num_input):
        modules = []
        for _ in range(self.nb):
            modules += [
                fnn.MADE(num_inputs=num_outputs, num_hidden=self.nh, num_cond_inputs=num_input, act=self.act),
                fnn.BatchNormFlow(num_outputs),
                fnn.Reverse(num_outputs)
            ]
        model = fnn.FlowSequential(*modules)
        model = model.to(dtype=torch.float64)
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)

        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)
        return model, optimizer

    def run_training(self):
        today = datetime.now().strftime("%Y%m%d_%H%M%S")
        for epoch in range(self.cfg["EPOCHS_FLOW"]):
            self.train_flow_logger.log_info_stream(f"Epoch: {epoch+1}/{self.cfg['EPOCHS_FLOW']}")
            self.train_flow_logger.log_info_stream(f"Train")
            train_loss_epoch = self.train(
                epoch=epoch
            )

            self.train_flow_logger.log_info_stream(f"Validation")
            validation_loss = self.validate(
                epoch=epoch
            )

            self.lst_epochs.append(epoch)
            self.lst_train_loss_per_epoch.append(train_loss_epoch)
            self.lst_valid_loss_per_epoch.append(validation_loss)

            if validation_loss < self.best_validation_loss:
                self.best_validation_epoch = epoch
                self.best_validation_loss = validation_loss
                # self.best_model = copy.deepcopy(self.model)
                self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
                # self.best_model.eval()

            if train_loss_epoch < self.best_train_loss:
                self.best_train_epoch = epoch
                self.best_train_loss = train_loss_epoch

            if self.cfg['PLOT_TRAINING_FLOW'] is True:
                self.plot_data(epoch=epoch, today=today)

            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy().astype(np.float64), epoch+1)
        self.train_flow_logger.log_info_stream(f"End Training")
        self.train_flow_logger.log_info_stream(f"Best validation epoch: {self.best_validation_epoch + 1}\t"
                                               f"best validation loss: {-self.best_validation_loss}\t"
                                               f"learning rate: {self.lr}\t"
                                               f"num_hidden: {self.nh}\t"
                                               f"num_blocks: {self.nb}\t"
                                               f"batch_size: {self.bs}")
        self.writer.add_hparams(
            hparam_dict={
                "learning rate": self.lr,
                "batch size": self.bs,
                "number hidden": self.nh,
                "number blocks": self.nb},
            metric_dict={
                "hparam/last training loss": train_loss_epoch,
                "hparam/last validation loss": validation_loss,
                "hparam/best validation loss": self.best_validation_loss,
                "hparam/best train loss": self.best_train_loss,
                "hparam/best validation epoch": self.best_validation_epoch,
                "hparam/best train epoch": self.best_train_epoch,
            },
            run_name=f"final result"
        )

        if self.cfg['PLOT_TEST_FLOW'] is True:
            self.plot_data(epoch=self.cfg["EPOCHS_FLOW"] - 1, today=today)

        if self.cfg['SAVE_NN_FLOW'] is True:
            torch.save(
                self.best_model,
                f"{self.cfg['PATH_SAVE_NN']}/best_model_e_{self.best_validation_epoch+1}_lr_{self.lr}_bs_{self.bs}_run_{self.cfg['RUN_DATE']}.pt")
            torch.save(
                self.model,
                f"{self.cfg['PATH_SAVE_NN']}/last_model_e_{self.cfg['EPOCHS_FLOW']}_lr_{self.lr}_bs_{self.bs}_run_{self.cfg['RUN_DATE']}.pt")

        self.writer.flush()
        self.writer.close()

    def plot_feature_histograms(self, df, columns, title_prefix="Feature", n_cols=3):
        n_features = len(columns)
        n_rows = math.ceil(n_features / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(columns):
            sns.histplot(df[col], bins=30, ax=axes[i])
            axes[i].set_title(f"{title_prefix}: {col}")
            axes[i].set_yscale("log")

        # Leere Subplots ausschalten
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def plot_output_comparison_histograms(self, true_df, generated_df, columns, n_cols=3, epoch=1, counter=1):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import math

        n_features = len(columns)
        n_rows = math.ceil(n_features / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(columns):
            ax = axes[i]
            sns.histplot(true_df[col], bins=30, label='true', ax=ax, stat="density", color='blue', alpha=0.5)
            sns.histplot(generated_df[col], bins=30, label='generated', ax=ax, stat="density", color='orange',
                         alpha=0.5)
            ax.set_title(f"Output: {col}, Epoch: {epoch}, Counter: {counter}")
            ax.set_yscale("log")
            ax.legend()

        # Entferne ungenutzte Subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(f"/Users/P.Gebhardt/Development/PhD/output/gaNdalF_flow_training/{epoch}_{counter}-training_output.png", bbox_inches='tight', dpi=600)

    def train(self, epoch):
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 0.1

        train_loss = 0.0
        total_samples = 0
        df_train = self.galaxies.train_dataset
        # COLS = df_train.columns
        # df_train = self.yj_transformer.transform(df_train)
        # df_train = pd.DataFrame(df_train, columns=COLS)
        # df_train = self.scalers.transform(df_train)
        # df_train = pd.DataFrame(df_train, columns=COLS)

        input_data = torch.tensor(df_train[self.cfg["INPUT_COLS"]].values, dtype=torch.float64)
        output_data = torch.tensor(df_train[self.cfg["OUTPUT_COLS"]].values, dtype=torch.float64)

        for j, col in enumerate(self.cfg["INPUT_COLS"]):
            if col in self.cfg.get("COLUMNS_LOG1P", []):
                input_data[:, j] = torch.log1p(input_data[:, j])
            scaler = self.scalers[col]
            if hasattr(scaler, "min_"):
                scale = scaler.scale_[0]
                min_ = scaler.min_[0]
                input_data[:, j] = input_data[:, j] * scale + min_
            else:
                mean = scaler.mean_[0]
                scale = scaler.scale_[0]
                input_data[:, j] = (input_data[:, j] - mean) / scale

        for j, col in enumerate(self.cfg["OUTPUT_COLS"]):
            if col in self.cfg.get("COLUMNS_LOG1P", []):
                output_data[:, j] = torch.log1p(output_data[:, j])
            scaler = self.scalers[col]
            if hasattr(scaler, "min_"):
                scale = scaler.scale_[0]
                min_ = scaler.min_[0]
                input_data[:, j] = input_data[:, j] * scale + min_
            else:
                mean = scaler.mean_[0]
                scale = scaler.scale_[0]
                input_data[:, j] = (input_data[:, j] - mean) / scale

        train_dataset = TensorDataset(input_data, output_data)
        train_loader = DataLoader(train_dataset, batch_size=self.bs, shuffle=True)

        pbar = tqdm(
            train_loader,
            desc=f"Training, Epoch {epoch + 1}",
            ncols=200
        )

        for input_data, output_data in pbar:
            input_data = input_data.to(self.device)
            output_data = output_data.to(self.device)
            self.optimizer.zero_grad()
            loss = -self.model.log_probs(output_data, input_data).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, error_if_nonfinite=False)
            self.optimizer.step()
            total_samples += output_data.size(0)
            train_loss += loss.item() * output_data.size(0)
            pbar.set_postfix(loss=train_loss / total_samples)
            self.lst_train_loss_per_batch.append(loss.item())
            self.global_step += 1
        pbar.close()

        train_loss = train_loss / len(self.galaxies.train_dataset)
        self.train_flow_logger.log_info_stream(f"Training,\t"
                                               f"Epoch: {epoch + 1},\t"
                                               f"learning rate: {self.lr},\t"
                                               f"number hidden: {self.nh},\t"
                                               f"number blocks: {self.nb},\t"
                                               f"batch size: {self.bs},\t"
                                               f"training loss: {train_loss}")
        self.writer.add_scalar('training loss', train_loss, epoch+1)

        self.model.train()
        for module in self.model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 0
        with torch.no_grad():
            self.model(
                torch.tensor(self.galaxies.train_dataset[self.cfg["OUTPUT_COLS"]].values, dtype=torch.float64),
                cond_inputs=torch.tensor(self.galaxies.train_dataset[self.cfg["INPUT_COLS"]].values, dtype=torch.float64)
            )
        for module in self.model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 0.1
        self.model.eval()

        return train_loss

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        total_samples = 0

        df_valid = self.galaxies.valid_dataset
        # COLS = df_valid.columns
        # df_valid = self.yj_transformer.transform(df_valid)
        # df_valid = pd.DataFrame(df_valid, columns=COLS)
        # df_valid = self.scalers.transform(df_valid)
        # df_valid = pd.DataFrame(df_valid, columns=COLS)

        input_data = torch.tensor(df_valid[self.cfg["INPUT_COLS"]].values, dtype=torch.float64)
        output_data = torch.tensor(df_valid[self.cfg["OUTPUT_COLS"]].values, dtype=torch.float64)

        for j, col in enumerate(self.cfg["INPUT_COLS"]):
            if col in self.cfg.get("COLUMNS_LOG1P", []):
                input_data[:, j] = torch.log1p(input_data[:, j])
            scaler = self.scalers[col]
            if hasattr(scaler, "min_"):
                scale = scaler.scale_[0]
                min_ = scaler.min_[0]
                input_data[:, j] = input_data[:, j] * scale + min_
            else:
                mean = scaler.mean_[0]
                scale = scaler.scale_[0]
                input_data[:, j] = (input_data[:, j] - mean) / scale

        for j, col in enumerate(self.cfg["OUTPUT_COLS"]):
            if col in self.cfg.get("COLUMNS_LOG1P", []):
                output_data[:, j] = torch.log1p(output_data[:, j])
            scaler = self.scalers[col]
            if hasattr(scaler, "min_"):
                scale = scaler.scale_[0]
                min_ = scaler.min_[0]
                input_data[:, j] = input_data[:, j] * scale + min_
            else:
                mean = scaler.mean_[0]
                scale = scaler.scale_[0]
                input_data[:, j] = (input_data[:, j] - mean) / scale

        valid_dataset = TensorDataset(input_data, output_data)
        valid_loader = DataLoader(valid_dataset, batch_size=self.bs, shuffle=True)

        pbar = tqdm(
            valid_loader,
            desc=f"Validation, Epoch {epoch + 1}",
            ncols=200
        )

        for input_data, output_data in pbar:
            input_data = input_data.to(self.device)
            output_data = output_data.to(self.device)

            with torch.no_grad():
                loss = -self.model.log_probs(output_data, input_data).mean()
                self.lst_valid_loss_per_batch.append(loss.item())
                val_loss += loss.data.item() * output_data.size(0)
            total_samples += output_data.size(0)
            pbar.set_postfix(loss=val_loss / total_samples)
        pbar.close()
        val_loss = val_loss / len(self.galaxies.valid_dataset)
        self.train_flow_logger.log_info_stream(f"Validation,\t"
                                               f"Epoch: {epoch + 1},\t"
                                               f"learning rate: {self.lr},\t"
                                               f"number hidden: {self.nh},\t"
                                               f"number blocks: {self.nb},\t"
                                               f"batch size: {self.bs},\t"
                                               f"validation loss: {val_loss}")
        self.writer.add_scalar('validation loss', val_loss, epoch+1)
        return val_loss

    def plot_data(self, epoch, today):
        """"""
        self.train_flow_logger.log_info_stream("Plot test")
        sns.set_theme()
        self.model.eval()

        df_balrog = self.galaxies.test_dataset

        # COLS = df_balrog.columns
        # df_balrog = self.yj_transformer.transform(df_balrog)
        # df_balrog = pd.DataFrame(df_balrog, columns=COLS)
        # df_balrog = self.scalers.transform(df_balrog)
        # df_balrog = pd.DataFrame(df_balrog, columns=COLS)

        df_gandalf = df_balrog.copy()

        input_data = torch.tensor(df_balrog[self.cfg["INPUT_COLS"]].values, dtype=torch.float64)

        for j, col in enumerate(self.cfg["INPUT_COLS"]):
            if col in self.cfg.get("COLUMNS_LOG1P", []):
                input_data[:, j] = torch.log1p(input_data[:, j])
            scaler = self.scalers[col]
            if hasattr(scaler, "min_"):
                scale = scaler.scale_[0]
                min_ = scaler.min_[0]
                input_data[:, j] = input_data[:, j] * scale + min_
            else:
                mean = scaler.mean_[0]
                scale = scaler.scale_[0]
                input_data[:, j] = (input_data[:, j] - mean) / scale

        input_data = input_data.to(self.device)

        with torch.no_grad():
            arr_gandalf_output = self.model.sample(len(input_data), cond_inputs=input_data).detach()

        for j, col in enumerate(self.cfg["OUTPUT_COLS"]):
            scaler = self.scalers[col]
            # Zuerst den Skaler rückgängig machen
            if hasattr(scaler, "min_"):
                # MinMaxScaler: x_original = (x_scaled * (max - min)) + min
                scale = scaler.scale_[0]
                min_ = scaler.min_[0]
                arr_gandalf_output[:, j] = (arr_gandalf_output[:, j] - min_) / scale
            else:
                # StandardScaler: x_original = (x_scaled * std) + mean
                mean = scaler.mean_[0]
                scale = scaler.scale_[0]
                arr_gandalf_output[:, j] = arr_gandalf_output[:, j] * scale + mean
            # Dann ggf. log1p rückgängig machen (exp - 1)
            if col in self.cfg.get("COLUMNS_LOG1P", []):
                arr_gandalf_output[:, j] = torch.expm1(arr_gandalf_output[:, j])

        df_gandalf[self.cfg["OUTPUT_COLS"]] = arr_gandalf_output.cpu().numpy()

        plot_features(cfg=self.cfg, plot_log=self.train_flow_logger, df_gandalf=df_gandalf, df_balrog=df_balrog, columns=self.cfg["OUTPUT_COLS"], title_prefix="compare", epoch=epoch, today=today)
        # df_balrog = inverse_transform_df(self.scalers, df_balrog)
        # df_gandalf = inverse_transform_df(self.scalers, df_gandalf)
        # df_balrog = inverse_transform_df(self.yj_transformer, df_balrog)
        # df_gandalf = inverse_transform_df(self.yj_transformer, df_gandalf)

        # df_balrog = self.scalers.inverse_transform(df_balrog)
        # df_balrog = pd.DataFrame(df_balrog, columns=COLS)
        # df_balrog = self.yj_transformer.inverse_transform(df_balrog)
        # df_balrog = pd.DataFrame(df_balrog, columns=COLS)
        #
        # df_gandalf = self.scalers.inverse_transform(df_gandalf)
        # df_gandalf = pd.DataFrame(df_gandalf, columns=COLS)
        # df_gandalf = self.yj_transformer.inverse_transform(df_gandalf)
        # df_gandalf = pd.DataFrame(df_gandalf, columns=COLS)

        self.train_flow_logger.log_info_stream(f"gaNdalF NaNs: {df_gandalf.isna().sum()}")

        return

        df_gandalf[self.cfg['CUT_COLS_FLOW']] = self.galaxies.df_test_cut_cols.to_numpy()
        df_balrog[self.cfg['CUT_COLS_FLOW']] = self.galaxies.df_test_cut_cols.to_numpy()

        for col in self.cfg['FILL_NA_FLOW'].keys():
            df_gandalf[col] = df_gandalf[col].fillna(self.cfg['FILL_NA_FLOW'][col])
        if self.cfg['DROP_NA_FLOW'] is True:
            df_gandalf = df_gandalf.dropna()

        for b in self.cfg['BANDS_FLOW']:
            df_gandalf[f"meas {b} - true {b}"] = df_gandalf[f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}_{b}"] - \
                                                 df_gandalf[
                                                     f"BDF_{self.cfg['LUM_TYPE_FLOW'].upper()}_DERED_CALIB_{b.upper()}"]
            df_balrog[f"meas {b} - true {b}"] = df_balrog[f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}_{b}"] - \
                                                df_balrog[
                                                    f"BDF_{self.cfg['LUM_TYPE_FLOW'].upper()}_DERED_CALIB_{b.upper()}"]

        df_balrog_cut = df_balrog.copy()

        if self.cfg['APPLY_OBJECT_CUT_FLOW'] is not True:
            df_balrog_cut = self.galaxies.unsheared_object_cuts(data_frame=df_balrog_cut)
        if self.cfg['APPLY_FLAG_CUT_FLOW'] is not True:
            df_balrog_cut = self.galaxies.flag_cuts(data_frame=df_balrog_cut)
        if self.cfg['APPLY_UNSHEARED_MAG_CUT_FLOW'] is not True:
            df_balrog_cut = self.galaxies.unsheared_mag_cut(data_frame=df_balrog_cut)
        if self.cfg['APPLY_UNSHEARED_SHEAR_CUT_FLOW'] is not True:
            df_balrog_cut = self.galaxies.unsheared_shear_cuts(data_frame=df_balrog_cut)
        if self.cfg['APPLY_AIRMASS_CUT_FLOW'] is not True:
            df_balrog_cut = self.galaxies.airmass_cut(data_frame=df_balrog_cut)

        df_gandalf_cut = df_gandalf.copy()

        if self.cfg['APPLY_OBJECT_CUT_FLOW'] is not True:
            df_gandalf_cut = self.galaxies.unsheared_object_cuts(data_frame=df_gandalf_cut)
        if self.cfg['APPLY_FLAG_CUT_FLOW'] is not True:
            df_gandalf_cut = self.galaxies.flag_cuts(data_frame=df_gandalf_cut)
        if self.cfg['APPLY_UNSHEARED_MAG_CUT_FLOW'] is not True:
            df_gandalf_cut = self.galaxies.unsheared_mag_cut(data_frame=df_gandalf_cut)
        if self.cfg['APPLY_UNSHEARED_SHEAR_CUT_FLOW'] is not True:
            df_gandalf_cut = self.galaxies.unsheared_shear_cuts(data_frame=df_gandalf_cut)
        if self.cfg['APPLY_AIRMASS_CUT_FLOW'] is not True:
            df_gandalf_cut = self.galaxies.airmass_cut(data_frame=df_gandalf_cut)

        df_gandalf = calc_color(
            data_frame=df_gandalf,
            colors=self.cfg['COLORS_FLOW'],
            column_name=f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}"
        )
        df_balrog = calc_color(
            data_frame=df_balrog,
            colors=self.cfg['COLORS_FLOW'],
            column_name=f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}"
        )
        df_gandalf_cut = calc_color(
            data_frame=df_gandalf_cut,
            colors=self.cfg['COLORS_FLOW'],
            column_name=f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}"
        )
        df_balrog_cut = calc_color(
            data_frame=df_balrog_cut,
            colors=self.cfg['COLORS_FLOW'],
            column_name=f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}"
        )

        if self.cfg['PLOT_LOSS_FLOW'] is True:
            img_grid = loss_plot(
                epoch=epoch,
                lst_train_loss_per_batch=self.lst_train_loss_per_batch,
                lst_train_loss_per_epoch=self.lst_train_loss_per_epoch,
                lst_valid_loss_per_batch=self.lst_valid_loss_per_batch,
                lst_valid_loss_per_epoch=self.lst_valid_loss_per_epoch,
                show_plot=self.cfg['SHOW_PLOT_FLOW'],
                save_plot=self.cfg['SAVE_PLOT_FLOW'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER']['LOSS_PLOT']}/loss_{epoch + 1}.png"
            )
            self.writer.add_image("loss plot", img_grid, epoch + 1)

        if self.cfg['PLOT_COLOR_COLOR_FLOW'] is True:
            try:
                img_grid, self.dict_delta_color_color = plot_compare_corner(
                    data_frame_generated=df_gandalf,
                    data_frame_true=df_balrog,
                    dict_delta=self.dict_delta_color_color,
                    epoch=epoch + 1,
                    title=f"color-color plot",
                    columns=["r-i", "i-z"],
                    labels=["r-i", "i-z"],
                    show_plot=self.cfg['SHOW_PLOT_FLOW'],
                    save_plot=self.cfg['SAVE_PLOT_FLOW'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['COLOR_COLOR_PLOT']}/color_color_{epoch + 1}.png",
                    ranges=[(-4, 4), (-4, 4)]
                )
                self.writer.add_image("color color plot", img_grid, epoch + 1)
            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['COLOR_COLOR_PLOT']}/color_color_{epoch + 1}.png")
            try:
                img_grid, self.dict_delta_color_color_mcal = plot_compare_corner(
                    data_frame_generated=df_gandalf_cut,
                    data_frame_true=df_balrog_cut,
                    dict_delta=self.dict_delta_color_color_mcal,
                    epoch=epoch + 1,
                    title=f"mcal color-color plot",
                    columns=["r-i", "i-z"],
                    labels=["r-i", "i-z"],
                    show_plot=self.cfg['SHOW_PLOT_FLOW'],
                    save_plot=self.cfg['SAVE_PLOT_FLOW'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_COLOR_COLOR_PLOT']}/mcal_color_color_{epoch + 1}.png",
                    ranges=[(-1.2, 1.8), (-1.5, 1.5)]
                )
                self.writer.add_image("color color plot mcal", img_grid, epoch + 1)
            except Exception as e:
                print(
                    f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_COLOR_COLOR_PLOT']}/mcal_color_color_{epoch + 1}.png")
        if self.cfg['PLOT_RESIDUAL_FLOW']:
            try:
                img_grid = residual_plot(
                    data_frame_generated=df_gandalf,
                    data_frame_true=df_balrog,
                    luminosity_type=self.cfg['LUM_TYPE_FLOW'],
                    plot_title=f"residual, epoch {epoch + 1}",
                    bands=self.cfg['BANDS_FLOW'],
                    show_plot=self.cfg['SHOW_PLOT_FLOW'],
                    save_plot=self.cfg['SAVE_PLOT_FLOW'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['RESIDUAL_PLOT']}/residual_plot_{epoch + 1}.png"
                )
                self.writer.add_image("residual plot", img_grid, epoch + 1)
            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['RESIDUAL_PLOT']}/residual_plot_{epoch + 1}.png")

            try:
                img_grid = residual_plot(
                    data_frame_generated=df_gandalf_cut,
                    data_frame_true=df_balrog_cut,
                    luminosity_type=self.cfg['LUM_TYPE_FLOW'],
                    plot_title=f"mcal residual, epoch {epoch + 1}",
                    bands=self.cfg['BANDS_FLOW'],
                    show_plot=self.cfg['SHOW_PLOT_FLOW'],
                    save_plot=self.cfg['SAVE_PLOT_FLOW'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_RESIDUAL_PLOT']}/mcal_residual_plot_{epoch + 1}.png"
                )
                self.writer.add_image("residual plot mcal", img_grid, epoch + 1)
            except Exception as e:
                print(
                    f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_RESIDUAL_PLOT']}/mcal_residual_plot_{epoch + 1}.png")

        if self.cfg['PLOT_CHAIN_FLOW'] is True:
            try:
                img_grid, self.dict_delta_unsheared = plot_compare_corner(
                    data_frame_generated=df_gandalf,
                    data_frame_true=df_balrog,
                    dict_delta=self.dict_delta_unsheared,
                    epoch=epoch + 1,
                    title=f"chain plot",
                    show_plot=self.cfg['SHOW_PLOT_FLOW'],
                    save_plot=self.cfg['SAVE_PLOT_FLOW'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['CHAIN_PLOT']}/chainplot_{epoch + 1}.png",
                    columns=[
                        f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}_r",
                        f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}_i",
                        f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T"
                    ],
                    labels=[
                        f"{self.cfg['LUM_TYPE_FLOW'].lower()}_r",
                        f"{self.cfg['LUM_TYPE_FLOW'].lower()}_i",
                        f"{self.cfg['LUM_TYPE_FLOW'].lower()}_z",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    ranges=[(15, 30), (15, 30), (15, 30), (-15, 75), (-1.5, 4), (-1.5, 2)]
                )
                self.writer.add_image("chain plot", img_grid, epoch + 1)
            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['CHAIN_PLOT']}/chainplot_{epoch + 1}.png")

            try:
                img_grid, self.dict_delta_unsheared_mcal = plot_compare_corner(
                    data_frame_generated=df_gandalf_cut,
                    data_frame_true=df_balrog_cut,
                    dict_delta=self.dict_delta_unsheared_mcal,
                    epoch=epoch + 1,
                    title=f"mcal chain plot",
                    show_plot=self.cfg['SHOW_PLOT_FLOW'],
                    save_plot=self.cfg['SAVE_PLOT_FLOW'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_CHAIN_PLOT']}/mcal_chainplot_{epoch + 1}.png",
                    columns=[
                        f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}_r",
                        f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}_i",
                        f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T",
                    ],
                    labels=[
                        f"{self.cfg['LUM_TYPE_FLOW'].lower()}_r",
                        f"{self.cfg['LUM_TYPE_FLOW'].lower()}_i",
                        f"{self.cfg['LUM_TYPE_FLOW'].lower()}_z",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    ranges=[(15, 30), (15, 30), (15, 30), (-75, 200), (-1.5, 6), (-1, 4)]
                )
                self.writer.add_image("chain plot mcal", img_grid, epoch + 1)
            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_CHAIN_PLOT']}/mcal_chainplot_{epoch + 1}.png")

            try:
                img_grid, self.dict_delta_color_diff = plot_compare_corner(
                    data_frame_generated=df_gandalf,
                    data_frame_true=df_balrog,
                    dict_delta=self.dict_delta_color_diff,
                    epoch=epoch + 1,
                    title=f"color diff plot",
                    show_plot=self.cfg['SHOW_PLOT_FLOW'],
                    save_plot=self.cfg['SAVE_PLOT_FLOW'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['COLOR_DIFF_PLOT']}/color_diff_{epoch + 1}.png",
                    columns=[
                        "meas r - true r",
                        "meas i - true i",
                        "meas z - true z"
                    ],
                    labels=[
                        "meas r - true r",
                        "meas i - true i",
                        "meas z - true z"
                    ],
                    ranges=[(-4, 4), (-4, 4), (-4, 4)]
                )
                self.writer.add_image("color diff plot", img_grid, epoch + 1)
            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['COLOR_DIFF_PLOT']}/color_diff_{epoch + 1}.png")

            try:
                img_grid, self.dict_delta_color_diff_mcal = plot_compare_corner(
                    data_frame_generated=df_gandalf_cut,
                    data_frame_true=df_balrog_cut,
                    dict_delta=self.dict_delta_color_diff_mcal,
                    epoch=epoch + 1,
                    title=f"mcal color diff plot",
                    show_plot=self.cfg['SHOW_PLOT_FLOW'],
                    save_plot=self.cfg['SAVE_PLOT_FLOW'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_COLOR_DIFF_PLOT']}/mcal_color_diff_{epoch + 1}.png",
                    columns=[
                        "meas r - true r",
                        "meas i - true i",
                        "meas z - true z"
                    ],
                    labels=[
                        "meas r - true r",
                        "meas i - true i",
                        "meas z - true z"
                    ],
                    ranges=[(-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5)]
                )
                self.writer.add_image("color diff plot mcal", img_grid, epoch + 1)
            except Exception as e:
                print(
                    f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_COLOR_DIFF_PLOT']}/mcal_color_diff_{epoch + 1}.png")

        if self.cfg['PLOT_MEAN_FLOW'] is True:
            try:
                lists_mean_to_plot = [
                    self.lst_mean_mag_r,
                    self.lst_mean_mag_i,
                    self.lst_mean_mag_z,
                    self.lst_mean_snr,
                    self.lst_mean_size_ratio,
                    self.lst_mean_t,
                ]

                lists_mean_to_plot_updated, img_grid = plot_mean_or_std(
                    data_frame_generated=df_gandalf,
                    data_frame_true=df_balrog,
                    lists_to_plot=lists_mean_to_plot,
                    list_epochs=self.lst_epochs,

                    columns=[
                        f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}_r",
                        f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}_i",
                        f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T"
                    ],
                    lst_labels=[
                        f"{self.cfg['LUM_TYPE_FLOW'].lower()}_r",
                        f"{self.cfg['LUM_TYPE_FLOW'].lower()}_i",
                        f"{self.cfg['LUM_TYPE_FLOW'].lower()}_z",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    lst_marker=["o", "^", "X", "d", "s", "*"],
                    lst_color=["blue", "red", "green", "orange", "purple", "black"],
                    plot_title="mean ratio",
                    show_plot=self.cfg['SHOW_PLOT_FLOW'],
                    save_plot=self.cfg['SAVE_PLOT_FLOW'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['MEAN_PLOT']}/mean_{epoch + 1}.png",
                    statistic_type="mean"
                )
                self.writer.add_image("mean plot", img_grid, epoch + 1)
                for idx_plot, lst_plot in enumerate(lists_mean_to_plot):
                    lst_plot = lists_mean_to_plot_updated[idx_plot]

            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['MEAN_PLOT']}/mean_{epoch + 1}.png")
                print(
                    f"Mean shapes: \t epoch \t {self.cfg['LUM_TYP_FLOWE'].lower()} r \t {self.cfg['LUM_TYPE_FLOW'].lower()} i \t {self.cfg['LUM_TYPE_FLOW'].lower()} z \t snr \t size_ratio \t T")
                print(
                    f"\t           \t {len(self.lst_epochs)} \t {len(self.lst_mean_mag_r)} \t {len(self.lst_mean_mag_i)} \t {len(self.lst_mean_mag_z)} \t {len(self.lst_mean_snr)} \t {len(self.lst_mean_size_ratio)} \t {len(self.lst_mean_t)}")

            try:
                lists_mean_to_plot_cut = [
                    self.lst_mean_mag_r_cut,
                    self.lst_mean_mag_i_cut,
                    self.lst_mean_mag_z_cut,
                    self.lst_mean_snr_cut,
                    self.lst_mean_size_ratio_cut,
                    self.lst_mean_t_cut,
                ]

                lists_mean_to_plot_cut_updated, img_grid = plot_mean_or_std(
                    data_frame_generated=df_gandalf_cut,
                    data_frame_true=df_balrog_cut,
                    lists_to_plot=lists_mean_to_plot_cut,
                    list_epochs=self.lst_epochs,
                    columns=[
                        f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}_r",
                        f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}_i",
                        f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T"
                    ],
                    lst_labels=[
                        f"{self.cfg['LUM_TYPE_FLOW'].lower()}_r",
                        f"{self.cfg['LUM_TYPE_FLOW'].lower()}_i",
                        f"{self.cfg['LUM_TYPE_FLOW'].lower()}_z",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    lst_marker=["o", "^", "X", "d", "s", "*"],
                    lst_color=["blue", "red", "green", "orange", "purple", "black"],
                    plot_title="mcal mean ratio",
                    show_plot=self.cfg['SHOW_PLOT_FLOW'],
                    save_plot=self.cfg['SAVE_PLOT_FLOW'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_MEAN_PLOT']}/mcal_mean_{epoch + 1}.png",
                    statistic_type="mean"
                )
                self.writer.add_image("mean plot mcal", img_grid, epoch + 1)
                for idx_plot_cut, lst_plot_cut in enumerate(lists_mean_to_plot_cut):
                    lst_plot_cut = lists_mean_to_plot_cut_updated[idx_plot_cut]

            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_MEAN_PLOT']}/mcal_mean_{epoch + 1}.png")
                print(
                    f"Mean mcal shapes: \t epoch \t {self.cfg['LUM_TYPE_FLOW'].lower()} r \t {self.cfg['LUM_TYPE_FLOW'].lower()} i \t {self.cfg['LUM_TYPE_FLOW'].lower()} z \t snr \t size_ratio \t T")
                print(
                    f"\t           \t {len(self.lst_epochs)} \t {len(self.lst_mean_mag_r_cut)} \t {len(self.lst_mean_mag_i_cut)} \t {len(self.lst_mean_mag_z_cut)} \t {len(self.lst_mean_snr_cut)} \t {len(self.lst_mean_size_ratio_cut)} \t {len(self.lst_mean_t_cut)}")

        if self.cfg['PLOT_STD_FLOW'] is True:
            try:
                lists_std_to_plot = [
                    self.lst_std_mag_r,
                    self.lst_std_mag_i,
                    self.lst_std_mag_z,
                    self.lst_std_snr,
                    self.lst_std_size_ratio,
                    self.lst_std_t,
                ]

                lists_std_to_plot_updated, img_grid = plot_mean_or_std(
                    data_frame_generated=df_gandalf,
                    data_frame_true=df_balrog,
                    lists_to_plot=lists_std_to_plot,
                    list_epochs=self.lst_epochs,
                    columns=[
                        f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}_r",
                        f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}_i",
                        f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T"
                    ],
                    lst_labels=[
                        f"{self.cfg['LUM_TYPE_FLOW'].lower()}_r",
                        f"{self.cfg['LUM_TYPE_FLOW'].lower()}_i",
                        f"{self.cfg['LUM_TYPE_FLOW'].lower()}_z",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    lst_marker=["o", "^", "X", "d", "s", "*"],
                    lst_color=["blue", "red", "green", "orange", "purple", "black"],
                    plot_title="std ratio",
                    show_plot=self.cfg['SHOW_PLOT_FLOW'],
                    save_plot=self.cfg['SAVE_PLOT_FLOW'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['STD_PLOT']}/std_{epoch + 1}.png",
                    statistic_type="std"
                )
                self.writer.add_image("std plot", img_grid, epoch + 1)
                for idx_plot, lst_plot in enumerate(lists_std_to_plot):
                    lst_plot = lists_std_to_plot_updated[idx_plot]

            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['STD_PLOT']}/std_{epoch + 1}.png")
                print(
                    f"Std shapes: \t epoch \t {self.cfg['LUM_TYPE_FLOW'].lower()} r \t {self.cfg['LUM_TYPE_FLOW'].lower()} i \t {self.cfg['LUM_TYPE_FLOW'].lower()} z \t snr \t size_ratio \t T")
                print(
                    f"\t           \t {len(self.lst_epochs)} \t {len(self.lst_std_mag_r)} \t {len(self.lst_std_mag_i)} \t {len(self.lst_std_mag_z)} \t {len(self.lst_std_snr)} \t {len(self.lst_std_size_ratio)} \t {len(self.lst_std_t)}")

            try:
                lists_std_to_plot_cut = [
                    self.lst_std_mag_r_cut,
                    self.lst_std_mag_i_cut,
                    self.lst_std_mag_z_cut,
                    self.lst_std_snr_cut,
                    self.lst_std_size_ratio_cut,
                    self.lst_std_t_cut,
                ]

                lists_std_to_plot_cut_updated, img_grid = plot_mean_or_std(
                    data_frame_generated=df_gandalf_cut,
                    data_frame_true=df_balrog_cut,
                    lists_to_plot=lists_std_to_plot_cut,
                    list_epochs=self.lst_epochs,
                    columns=[
                        f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}_r",
                        f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}_i",
                        f"unsheared/{self.cfg['LUM_TYPE_FLOW'].lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T"
                    ],
                    lst_labels=[
                        f"{self.cfg['LUM_TYPE_FLOW'].lower()}_r",
                        f"{self.cfg['LUM_TYPE_FLOW'].lower()}_i",
                        f"{self.cfg['LUM_TYPE_FLOW'].lower()}_z",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    lst_marker=["o", "^", "X", "d", "s", "*"],
                    lst_color=["blue", "red", "green", "orange", "purple", "black"],
                    plot_title="mcal std ratio",
                    show_plot=self.cfg['SHOW_PLOT_FLOW'],
                    save_plot=self.cfg['SAVE_PLO_FLOWT'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_STD_PLOT']}/mcal_std_{epoch + 1}.png",
                    statistic_type="std"
                )
                self.writer.add_image("std plot mcal", img_grid, epoch + 1)
                for idx_plot_cut, lst_plot_cut in enumerate(lists_std_to_plot_cut):
                    lst_plot_cut = lists_std_to_plot_cut_updated[idx_plot_cut]

            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_STD_PLOT']}/mcal_std_{epoch + 1}.png")
                print(
                    f"Std mcal shapes: \t epoch \t {self.cfg['LUM_TYPE_FLOW'].lower()} r \t {self.cfg['LUM_TYPE_FLOW'].lower()} i \t {self.cfg['LUM_TYPE_FLOW'].lower()} z \t snr \t size_ratio \t T")
                print(
                    f"\t           \t {len(self.lst_epochs)} \t {len(self.lst_std_mag_r_cut)} \t {len(self.lst_std_mag_i_cut)} \t {len(self.lst_std_mag_z_cut)} \t {len(self.lst_std_snr_cut)} \t {len(self.lst_std_size_ratio_cut)} \t {len(self.lst_std_t_cut)}")