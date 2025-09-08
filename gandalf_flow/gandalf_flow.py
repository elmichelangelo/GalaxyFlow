import copy
import torch
import torch.optim as optim
import torch.utils.data

import torch.nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from Handler import count_parameters, fnn, calc_color, loss_plot, plot_compare_corner, residual_plot, plot_mean_or_std, plot_features, plot_single_feature_dist
from gandalf_galaxie_dataset import DESGalaxies
import pandas as pd
import numpy as np
import seaborn as sns
import os
import joblib
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import math
from datetime import datetime
from ray import tune
import random
import signal

torch.set_default_dtype(torch.float64)


class gaNdalFFlow(object):

    def __init__(self,
                 cfg,
                 learning_rate,
                 number_hidden,
                 number_blocks,
                 batch_size,
                 number_layers,
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

        self.start_epoch = 0
        self.current_epoch = 0

        self.scalers = joblib.load(f"{self.cfg['PATH_TRANSFORMERS']}/{self.cfg['FILENAME_STANDARD_SCALER']}")

        self.bs = batch_size
        self.lr = learning_rate
        self.nh = number_hidden
        self.nb = number_blocks
        self.nl = number_layers

        self.cfg['PATH_PLOTS_FOLDER'] = {}
        self.cfg['PATH_OUTPUT_SUBFOLDER'] = f"{self.cfg['PATH_OUTPUT']}/lr_{self.lr}_nh_{self.nh}_nb_{self.nb}_nl_{self.nl}_bs_{self.bs}"
        self.cfg['PATH_OUTPUT_SUBFOLDER_CATALOGS'] = f"{self.cfg['PATH_OUTPUT_CATALOGS']}/lr_{self.lr}_nh_{self.nh}_nb_{self.nb}_nl_{self.nl}_bs_{self.bs}"
        self.cfg['PATH_WRITER'] = (f"{self.cfg['PATH_OUTPUT_SUBFOLDER']}/{self.cfg['FOLDER_WRITER']}/"
                                   f"lr_{self.lr}_nh_{self.nh}_nb_{self.nb}_nl_{self.nl}_bs_{self.bs}")
        self.cfg['PATH_PLOTS'] = f"{self.cfg['PATH_OUTPUT_SUBFOLDER']}/{self.cfg['FOLDER_PLOTS']}"
        self.cfg['PATH_SAVE_NN'] = f"{self.cfg['PATH_OUTPUT_SUBFOLDER']}/{self.cfg['FOLDER_SAVE_NN']}"

        for plot in self.cfg['PLOT_FOLDERS_FLOW']:
            self.cfg[f'PATH_PLOTS_FOLDER'][plot.upper()] = f"{self.cfg['PATH_PLOTS']}/{plot}"

        self.make_dirs()
        # self.writer = SummaryWriter(
        #     log_dir=cfg['PATH_WRITER'],
        #     comment=f"learning rate: {self.lr} "
        #             f"number hidden: {self.nh}_"
        #             f"number blocks: {self.nb}_"
        #             f"number layers: {self.nl}_"
        #             f"batch size: {self.bs}"
        # )
        self.galaxies = self.init_dataset()  # self.train_loader, self.valid_loader, self.test_sampled_data, '

        self.model, self.optimizer = self.init_network(
            num_outputs=len(cfg[f"OUTPUT_COLS"]),
            num_input=len(cfg[f"INPUT_COLS"])
        )

        # with torch.no_grad():
        #     for i in range((len(self.galaxies.train_dataset) + self.bs - 1) // self.bs):
        #         batch_df = self.galaxies.train_dataset.iloc[i * self.bs:(i + 1) * self.bs]
        #         input_data = torch.tensor(batch_df[self.cfg["INPUT_COLS"]].values, dtype=torch.float64)
        #         output_data = torch.tensor(batch_df[self.cfg["OUTPUT_COLS"]].values, dtype=torch.float64)
        #         input_data = input_data.to(self.device)
        #         output_data = output_data.to(self.device)
        #         self.writer.add_graph(self.model, (output_data, input_data))
        #         break
        #
        #     total_parameters = count_parameters(self.model)
        #     self.train_flow_logger.log_info_stream(f"Total trainable parameters: {total_parameters}")

        self.global_step = 0
        self.best_validation_loss = np.float64('inf')
        self.best_validation_epoch = 1
        self.best_train_loss = np.float64('inf')
        self.best_train_epoch = 1
        self.best_model = self.model

        self.load_checkpoint_if_any()
        self._install_sigterm_handler()

    def make_dirs(self):
        """"""
        if not os.path.exists(self.cfg['PATH_OUTPUT_SUBFOLDER']):
            os.mkdir(self.cfg['PATH_OUTPUT_SUBFOLDER'])
        if not os.path.exists(self.cfg['PATH_OUTPUT_SUBFOLDER_CATALOGS']):
            os.mkdir(self.cfg['PATH_OUTPUT_SUBFOLDER_CATALOGS'])
        if self.cfg['PLOT_TRAINING'] is True:
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
        return galaxies

    def init_network(self, num_outputs, num_input):
        modules = []
        for _ in range(self.nb):
            modules += [
                fnn.MADE(num_inputs=num_outputs, num_hidden=self.nh, num_cond_inputs=num_input, act=self.act, num_layers=self.nl),
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
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, )
        return model, optimizer

    def _checkpoint_path(self):
        return os.path.join(self.cfg['PATH_OUTPUT_SUBFOLDER'], "last.ckpt.pt")

    def _state_dict(self, epoch):
        return {
            "epoch": int(epoch),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_validation_loss": float(self.best_validation_loss),
            "best_validation_epoch": int(self.best_validation_epoch),
            "best_train_loss": float(self.best_train_loss),
            "best_train_epoch": int(self.best_train_epoch),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "hist": {
                "lst_epochs": self.lst_epochs,
                "lst_train_loss_per_epoch": self.lst_train_loss_per_epoch,
                "lst_valid_loss_per_epoch": self.lst_valid_loss_per_epoch,
            },
            "cfg_snapshot": {
                "lr": self.lr, "bs": self.bs, "nh": self.nh, "nb": self.nb, "nl": self.nl
            }
        }

    def save_checkpoint(self, epoch):
        path = self._checkpoint_path()
        tmp = path + ".tmp"
        torch.save(self._state_dict(epoch), tmp)
        os.replace(tmp, path)  # atomic

    def load_checkpoint_if_any(self):
        path = self._checkpoint_path()
        if not os.path.exists(path):
            self.start_epoch = 0
            return False
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.best_validation_loss = ckpt.get("best_validation_loss", self.best_validation_loss)
        self.best_validation_epoch = ckpt.get("best_validation_epoch", self.best_validation_epoch)
        self.best_train_loss = ckpt.get("best_train_loss", self.best_train_loss)
        self.best_train_epoch = ckpt.get("best_train_epoch", self.best_train_epoch)
        random.setstate(ckpt["rng"]["python"])
        np.random.set_state(ckpt["rng"]["numpy"])
        torch.random.set_rng_state(ckpt["rng"]["torch"])
        if torch.cuda.is_available() and ckpt["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
        self.start_epoch = int(ckpt["epoch"]) + 1
        self.train_flow_logger.log_info_stream(f"Resumed from checkpoint at epoch {self.start_epoch}")
        return True

    def _install_sigterm_handler(self):
        import threading, os, signal
        if threading.current_thread() is not threading.main_thread():
            self.train_flow_logger.log_info_stream("Skip SIGTERM handler: not in main thread.")
            return
        if os.environ.get("TUNE_ORIG_WORKING_DIR") or os.environ.get("RAY_ADDRESS"):
            self.train_flow_logger.log_info_stream("Skip SIGTERM handler: running under Ray/Tune.")
            return

        def _handler(signum, frame):
            try:
                ep = getattr(self, "current_epoch", 0)
                self.train_flow_logger.log_info_stream("SIGTERM – saving checkpoint...")
                self.save_checkpoint(ep)
            finally:
                os._exit(0)

        try:
            signal.signal(signal.SIGTERM, _handler)
            self.train_flow_logger.log_info_stream("Installed SIGTERM handler.")
        except ValueError as e:
            self.train_flow_logger.log_info_stream(f"Skip SIGTERM handler ({e}).")

    def run_training(self):
        today = datetime.now().strftime("%Y%m%d_%H%M%S")
        epochs_no_improve = 0
        min_delta = 1e-4

        for epoch in range(self.start_epoch, self.cfg["EPOCHS_FLOW"]):
            self.current_epoch = epoch

            self.train_flow_logger.log_info_stream(f"Epoch: {epoch+1}/{self.cfg['EPOCHS_FLOW']}")
            self.train_flow_logger.log_info_stream(f"Train")
            train_loss_epoch = self.train(
                epoch=epoch,
                date=today
            )

            self.train_flow_logger.log_info_stream(f"Validation")
            validation_loss = self.validate(
                epoch=epoch,
                date=today
            )

            self.lst_epochs.append(epoch)
            self.lst_train_loss_per_epoch.append(train_loss_epoch)
            self.lst_valid_loss_per_epoch.append(validation_loss)

            if validation_loss < self.best_validation_loss - min_delta:
                self.best_validation_epoch = epoch
                self.best_validation_loss = validation_loss
                self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if train_loss_epoch < self.best_train_loss - min_delta:
                self.best_train_epoch = epoch
                self.best_train_loss = train_loss_epoch

            if self.cfg['PLOT_TRAINING'] is True:
                self.plot_data(epoch=epoch, today=today)

            # for name, param in self.model.named_parameters():
            #     self.writer.add_histogram(name, param.clone().cpu().data.numpy().astype(np.float64), epoch+1)
            yield epoch, validation_loss, train_loss_epoch

            self.save_checkpoint(epoch)

        self.train_flow_logger.log_info_stream(f"End Training")
        self.train_flow_logger.log_info_stream(f"Best validation epoch: {self.best_validation_epoch + 1}\t"
                                               f"best validation loss: {self.best_validation_loss}\t"
                                               f"learning rate: {self.lr}\t"
                                               f"num_hidden: {self.nh}\t"
                                               f"num_blocks: {self.nb}\t"
                                               f"num_layers: {self.nl}\t"
                                               f"batch_size: {self.bs}"
                                               )
        # self.writer.add_scalar('validation loss (last)', validation_loss, self.current_epoch + 1)
        # self.writer.add_hparams(
        #     hparam_dict={
        #         "learning rate": self.lr,
        #         "batch size": self.bs,
        #         "number hidden": self.nh,
        #         "number blocks": self.nb,
        #         "number layers": self.nl
        #     },
        #     metric_dict={
        #         "hparam/last training loss": train_loss_epoch,
        #         "hparam/last validation loss": validation_loss,
        #         "hparam/best validation loss": self.best_validation_loss,
        #         "hparam/best train loss": self.best_train_loss,
        #         "hparam/best validation epoch": self.best_validation_epoch,
        #         "hparam/best train epoch": self.best_train_epoch,
        #     },
        # )

        if self.cfg['SAVE_NN_FLOW'] is True:
            torch.save(
                self.best_model,
                f"{self.cfg['PATH_SAVE_NN']}/best_model_e_{self.best_validation_epoch+1}_lr_{self.lr}_bs_{self.bs}_run_{self.cfg['RUN_DATE']}.pt")
            torch.save(
                self.model,
                f"{self.cfg['PATH_SAVE_NN']}/last_model_e_{self.cfg['EPOCHS_FLOW']}_lr_{self.lr}_bs_{self.bs}_run_{self.cfg['RUN_DATE']}.pt")

        # self.writer.flush()
        # self.writer.close()
        return self.best_validation_loss

    def train(self, epoch, date):
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 0.1

        train_loss = 0.0
        total_samples = 0
        df_train = self.galaxies.train_dataset

        if epoch == 0:
            for col in self.cfg["NF_COLUMNS_OF_INTEREST"]:
                scaler = self.scalers[col]
                mean = scaler.mean_[0]
                scale = scaler.scale_[0]
                df_train[col] = (df_train[col] - mean) / scale

        if self.cfg["PLOT_TRAINING_INPUT_OUTPUT"] is True:
            os.makedirs(f"{self.cfg['PATH_OUTPUT_PLOTS']}/{date}_output_plots/", exist_ok=True)
            plot_single_feature_dist(
                df=df_train,
                columns=self.cfg["INPUT_COLS"],
                title_prefix=f"Train_Scaled_Input",
                save_name=f"{self.cfg['PATH_OUTPUT_PLOTS']}/{date}_output_plots/{date}_{epoch + 1}_Train_Scaled_Input.pdf",
                epoch=epoch + 1
            )
            plot_single_feature_dist(
                df=df_train,
                columns=self.cfg["OUTPUT_COLS"],
                title_prefix=f"Train_Scaled_Output",
                save_name=f"{self.cfg['PATH_OUTPUT_PLOTS']}/{date}_output_plots/{date}_{epoch + 1}_Train_Scaled_Output.pdf",
                epoch=epoch + 1
            )

        input_data = torch.tensor(df_train[self.cfg["INPUT_COLS"]].values, dtype=torch.float64)
        output_data = torch.tensor(df_train[self.cfg["OUTPUT_COLS"]].values, dtype=torch.float64)

        train_dataset = TensorDataset(input_data, output_data)
        train_loader = DataLoader(train_dataset, batch_size=self.bs, shuffle=True)

        for input_data, output_data in train_loader:
            input_data = input_data.to(self.device)
            output_data = output_data.to(self.device)

            self.optimizer.zero_grad()
            loss = -self.model.log_probs(output_data, input_data).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, error_if_nonfinite=False)
            self.optimizer.step()

            bs_curr = output_data.size(0)
            total_samples += bs_curr
            train_loss += loss.item() * bs_curr
            self.lst_train_loss_per_batch.append(train_loss / total_samples)
            self.global_step += 1

        train_loss = train_loss / len(self.galaxies.train_dataset)
        self.train_flow_logger.log_info_stream(f"Training,\t"
                                               f"Epoch: {epoch + 1},\t"
                                               f"learning rate: {self.lr},\t"
                                               f"number hidden: {self.nh},\t"
                                               f"number blocks: {self.nb},\t"
                                               f"number layers: {self.nl},\t"
                                               f"batch size: {self.bs},\t"
                                               f"training loss: {train_loss}")
        # self.writer.add_scalar('training loss', train_loss, epoch+1)

        self.model.train()
        for module in self.model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 0
        with torch.no_grad():
            y = torch.tensor(
                self.galaxies.train_dataset[self.cfg["OUTPUT_COLS"]].values,
                dtype=torch.float64, device=self.device
            )
            x = torch.tensor(
                df_train[self.cfg["INPUT_COLS"]].values,
                dtype=torch.float64, device=self.device
            )
            self.model(y, cond_inputs=x)
        for module in self.model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 0.1
        self.model.eval()

        return train_loss

    def validate(self, epoch, date):
        self.model.eval()
        val_loss = 0
        total_samples = 0

        df_valid = self.galaxies.valid_dataset

        if epoch == 0:
            for col in self.cfg["NF_COLUMNS_OF_INTEREST"]:
                scaler = self.scalers[col]
                mean = scaler.mean_[0]
                scale = scaler.scale_[0]
                df_valid[col] = (df_valid[col] - mean) / scale

        if self.cfg["PLOT_TRAINING_INPUT_OUTPUT"] is True:
            plot_single_feature_dist(
                df=df_valid,
                columns=self.cfg["INPUT_COLS"],
                title_prefix=f"Valid_Scaled_Input",
                save_name=f"{self.cfg['PATH_OUTPUT_PLOTS']}/{date}_output_plots/{date}_{epoch + 1}_Valid_Scaled_Input.pdf",
                epoch=epoch + 1
            )
            plot_single_feature_dist(
                df=df_valid,
                columns=self.cfg["OUTPUT_COLS"],
                title_prefix=f"Valid_Scaled_Output",
                save_name=f"{self.cfg['PATH_OUTPUT_PLOTS']}/{date}_output_plots/{date}_{epoch + 1}_Valid_Scaled_Output.pdf",
                epoch=epoch + 1
            )

        input_data = torch.tensor(df_valid[self.cfg["INPUT_COLS"]].values, dtype=torch.float64)
        output_data = torch.tensor(df_valid[self.cfg["OUTPUT_COLS"]].values, dtype=torch.float64)

        valid_dataset = TensorDataset(input_data, output_data)
        valid_loader = DataLoader(valid_dataset, batch_size=self.bs, shuffle=False)

        for input_data, output_data in valid_loader:
            input_data = input_data.to(self.device)
            output_data = output_data.to(self.device)

            with torch.no_grad():
                loss = -self.model.log_probs(output_data, input_data).mean()

                bs_curr = output_data.size(0)
                val_loss += loss.item() * bs_curr
            total_samples += bs_curr
            self.lst_valid_loss_per_batch.append(val_loss / max(1, total_samples))

        val_loss = val_loss / len(self.galaxies.valid_dataset)
        self.train_flow_logger.log_info_stream(f"Validation,\t"
                                               f"Epoch: {epoch + 1},\t"
                                               f"learning rate: {self.lr},\t"
                                               f"number hidden: {self.nh},\t"
                                               f"number blocks: {self.nb},\t"
                                               f"number layers: {self.nl},\t"
                                               f"batch size: {self.bs},\t"
                                               f"validation loss: {val_loss}")
        # self.writer.add_scalar('validation loss', val_loss, epoch+1)
        return val_loss

    def plot_data(self, epoch, today):
        """"""
        self.train_flow_logger.log_info_stream("Plot test")
        self.model.eval()
        os.makedirs(f"{self.cfg['PATH_OUTPUT_PLOTS']}/{today}_loss_plots/", exist_ok=True)
        os.makedirs(f"{self.cfg['PATH_OUTPUT_PLOTS']}/{today}_output_plots/", exist_ok=True)
        if self.cfg["PLOT_TRAINING_LOSS"] is True:
            loss_plot(
                epoch=epoch,
                lst_train_loss_per_batch=self.lst_train_loss_per_batch,
                lst_train_loss_per_epoch=self.lst_train_loss_per_epoch,
                lst_valid_loss_per_batch=self.lst_valid_loss_per_batch,
                lst_valid_loss_per_epoch=self.lst_valid_loss_per_epoch,
                show_plot=False,
                save_plot=True,
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER']['LOSS_PLOT']}/{today}_loss_{epoch + 1}.pdf",
                title=f"bs {self.bs}; lr {self.lr}; nh {self.nh}; nb {self.nb}; nl {self.nl}"
            )
            img_grid_loss = loss_plot(
                epoch=epoch,
                lst_train_loss_per_batch=self.lst_train_loss_per_batch,
                lst_train_loss_per_epoch=self.lst_train_loss_per_epoch,
                lst_valid_loss_per_batch=self.lst_valid_loss_per_batch,
                lst_valid_loss_per_epoch=self.lst_valid_loss_per_epoch,
                show_plot=False,
                save_plot=True,
                save_name=f"{self.cfg['PATH_PLOTS']}/{today}_loss.pdf",
                title=f"bs {self.bs}; lr {self.lr}; nh {self.nh}; nb {self.nb}; nl {self.nl}"
            )
            # self.writer.add_image("loss plot", img_grid_loss, epoch + 1)
        if self.cfg["PLOT_TRAINING_LOG_LOSS"] is True:
            loss_plot(
                epoch=epoch,
                lst_train_loss_per_batch=self.lst_train_loss_per_batch,
                lst_train_loss_per_epoch=self.lst_train_loss_per_epoch,
                lst_valid_loss_per_batch=self.lst_valid_loss_per_batch,
                lst_valid_loss_per_epoch=self.lst_valid_loss_per_epoch,
                show_plot=False,
                save_plot=True,
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER']['LOSS_PLOT']}/{today}_loss_{epoch + 1}_logscale.pdf",
                title=f"bs {self.bs}; lr {self.lr}; nh {self.nh}; nb {self.nb}; nl {self.nl}",
                log_scale=True
            )
            img_grid_log_loss = loss_plot(
                epoch=epoch,
                lst_train_loss_per_batch=self.lst_train_loss_per_batch,
                lst_train_loss_per_epoch=self.lst_train_loss_per_epoch,
                lst_valid_loss_per_batch=self.lst_valid_loss_per_batch,
                lst_valid_loss_per_epoch=self.lst_valid_loss_per_epoch,
                show_plot=False,
                save_plot=True,
                save_name=f"{self.cfg['PATH_PLOTS']}/{today}_loss_logscale.pdf",
                title=f"bs {self.bs}; lr {self.lr}; nh {self.nh}; nb {self.nb}; nl {self.nl}",
                log_scale=True
            )
            # self.writer.add_image("log loss plot", img_grid_log_loss, epoch + 1)

        df_balrog = self.galaxies.test_dataset

        if epoch == 0:
            for col in self.cfg["NF_COLUMNS_OF_INTEREST"]:
                scaler = self.scalers[col]
                mean = scaler.mean_[0]
                scale = scaler.scale_[0]
                df_balrog[col] = (df_balrog[col] - mean) / scale

        if self.cfg["PLOT_TRAINING_INPUT_OUTPUT"] is True:
            img_grid_input = plot_single_feature_dist(
                df=df_balrog,
                columns=self.cfg["INPUT_COLS"],
                title_prefix=f"Test_Scaled_Input",
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER']['INPUT_OUTPUT_PLOT']}/{today}_{epoch + 1}_Test_Scaled_Input.pdf",
                epoch=epoch + 1
            )
            # self.writer.add_image("scaled input plot", img_grid_input, epoch + 1)
            img_grid_output = plot_single_feature_dist(
                df=df_balrog,
                columns=self.cfg["OUTPUT_COLS"],
                title_prefix=f"Test_Scaled_Output",
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER']['INPUT_OUTPUT_PLOT']}/{today}_{epoch + 1}_Test_Scaled_Output.pdf",
                epoch=epoch + 1
            )
            # self.writer.add_image("scaled output plot", img_grid_output, epoch + 1)

        input_data = torch.tensor(df_balrog[self.cfg["INPUT_COLS"]].values, dtype=torch.float64)
        output_data = torch.tensor(df_balrog[self.cfg["OUTPUT_COLS"]].values, dtype=torch.float64)

        df_gandalf = df_balrog.copy()

        input_data = input_data.to(self.device)

        with torch.no_grad():
            arr_gandalf_output = self.model.sample(len(input_data), cond_inputs=input_data).detach()

        output_data_np = arr_gandalf_output.cpu().numpy()

        input_data_np_true = input_data.cpu().numpy()
        output_data_np_true = output_data.cpu().numpy()
        arr_all_true = np.concatenate([input_data_np_true, output_data_np_true], axis=1)
        arr_all = np.concatenate([input_data_np_true, output_data_np], axis=1)


        df_output_true = pd.DataFrame(arr_all_true, columns=list(self.cfg["INPUT_COLS"]) + list(self.cfg["OUTPUT_COLS"]))
        df_output_true = df_output_true[self.cfg["NF_COLUMNS_OF_INTEREST"]]

        df_output_gandalf = pd.DataFrame(arr_all, columns=list(self.cfg["INPUT_COLS"]) + list(self.cfg["OUTPUT_COLS"]))
        df_output_gandalf = df_output_gandalf[self.cfg["NF_COLUMNS_OF_INTEREST"]]

        if self.cfg['PLOT_TRAINING_FEATURES'] is True:
            img_grid_feature_hist = plot_features(
                cfg=self.cfg,
                plot_log=self.train_flow_logger,
                df_gandalf=df_output_gandalf,
                df_balrog=df_output_true,
                columns=self.cfg["OUTPUT_COLS"],
                title_prefix=f"bs {self.bs}; lr {self.lr}; nh {self.nh}; nb {self.nb} ; nl {self.nl} - ",
                epoch=epoch,
                today=today,
                savename=f"{self.cfg['PATH_PLOTS_FOLDER']['FEATURE_HIST_PLOT']}/{today}_{epoch+1}_compare_output.pdf"
            )
            # self.writer.add_image("Feature Histogram", img_grid_feature_hist, epoch + 1)

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
            # self.writer.add_image("loss plot", img_grid, epoch + 1)

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
                # self.writer.add_image("color color plot", img_grid, epoch + 1)
            except Exception as e:
                self.train_flow_logger.log_info_stream(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['COLOR_COLOR_PLOT']}/color_color_{epoch + 1}.png")
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
                # self.writer.add_image("color color plot mcal", img_grid, epoch + 1)
            except Exception as e:
                self.train_flow_logger.log_info_stream(
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
                # self.writer.add_image("residual plot", img_grid, epoch + 1)
            except Exception as e:
                self.train_flow_logger.log_info_stream(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['RESIDUAL_PLOT']}/residual_plot_{epoch + 1}.png")

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
                # self.writer.add_image("residual plot mcal", img_grid, epoch + 1)
            except Exception as e:
                self.train_flow_logger.log_info_stream(
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
                # self.writer.add_image("chain plot", img_grid, epoch + 1)
            except Exception as e:
                self.train_flow_logger.log_info_stream(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['CHAIN_PLOT']}/chainplot_{epoch + 1}.png")

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
                # self.writer.add_image("chain plot mcal", img_grid, epoch + 1)
            except Exception as e:
                self.train_flow_logger.log_info_stream(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_CHAIN_PLOT']}/mcal_chainplot_{epoch + 1}.png")

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
                # self.writer.add_image("color diff plot", img_grid, epoch + 1)
            except Exception as e:
                self.train_flow_logger.log_info_stream(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['COLOR_DIFF_PLOT']}/color_diff_{epoch + 1}.png")

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
                # self.writer.add_image("color diff plot mcal", img_grid, epoch + 1)
            except Exception as e:
                self.train_flow_logger.log_info_stream(
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
                # self.writer.add_image("mean plot", img_grid, epoch + 1)
                for idx_plot, lst_plot in enumerate(lists_mean_to_plot):
                    lst_plot = lists_mean_to_plot_updated[idx_plot]

            except Exception as e:
                self.train_flow_logger.log_info_stream(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['MEAN_PLOT']}/mean_{epoch + 1}.png")
                self.train_flow_logger.log_info_stream(
                    f"Mean shapes: \t epoch \t {self.cfg['LUM_TYP_FLOWE'].lower()} r \t {self.cfg['LUM_TYPE_FLOW'].lower()} i \t {self.cfg['LUM_TYPE_FLOW'].lower()} z \t snr \t size_ratio \t T")
                self.train_flow_logger.log_info_stream(
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
                # self.writer.add_image("mean plot mcal", img_grid, epoch + 1)
                for idx_plot_cut, lst_plot_cut in enumerate(lists_mean_to_plot_cut):
                    lst_plot_cut = lists_mean_to_plot_cut_updated[idx_plot_cut]

            except Exception as e:
                self.train_flow_logger.log_info_stream(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_MEAN_PLOT']}/mcal_mean_{epoch + 1}.png")
                self.train_flow_logger.log_info_stream(
                    f"Mean mcal shapes: \t epoch \t {self.cfg['LUM_TYPE_FLOW'].lower()} r \t {self.cfg['LUM_TYPE_FLOW'].lower()} i \t {self.cfg['LUM_TYPE_FLOW'].lower()} z \t snr \t size_ratio \t T")
                self.train_flow_logger.log_info_stream(
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
                # self.writer.add_image("std plot", img_grid, epoch + 1)
                for idx_plot, lst_plot in enumerate(lists_std_to_plot):
                    lst_plot = lists_std_to_plot_updated[idx_plot]

            except Exception as e:
                self.train_flow_logger.log_info_stream(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['STD_PLOT']}/std_{epoch + 1}.png")
                self.train_flow_logger.log_info_stream(
                    f"Std shapes: \t epoch \t {self.cfg['LUM_TYPE_FLOW'].lower()} r \t {self.cfg['LUM_TYPE_FLOW'].lower()} i \t {self.cfg['LUM_TYPE_FLOW'].lower()} z \t snr \t size_ratio \t T")
                self.train_flow_logger.log_info_stream(
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
                # self.writer.add_image("std plot mcal", img_grid, epoch + 1)
                for idx_plot_cut, lst_plot_cut in enumerate(lists_std_to_plot_cut):
                    lst_plot_cut = lists_std_to_plot_cut_updated[idx_plot_cut]

            except Exception as e:
                self.train_flow_logger.log_info_stream(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_STD_PLOT']}/mcal_std_{epoch + 1}.png")
                self.train_flow_logger.log_info_stream(
                    f"Std mcal shapes: \t epoch \t {self.cfg['LUM_TYPE_FLOW'].lower()} r \t {self.cfg['LUM_TYPE_FLOW'].lower()} i \t {self.cfg['LUM_TYPE_FLOW'].lower()} z \t snr \t size_ratio \t T")
                self.train_flow_logger.log_info_stream(
                    f"\t           \t {len(self.lst_epochs)} \t {len(self.lst_std_mag_r_cut)} \t {len(self.lst_std_mag_i_cut)} \t {len(self.lst_std_mag_z_cut)} \t {len(self.lst_std_snr_cut)} \t {len(self.lst_std_size_ratio_cut)} \t {len(self.lst_std_t_cut)}")