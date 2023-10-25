import copy
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, random_split
import torch.nn
import torchvision.utils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from tensorboard.plugins.hparams import api as hp
import galaxyflow.flow as fnn
from galaxyflow import GalaxyDataset
import pandas as pd
from Handler.data_loader import load_data
from chainconsumer import ChainConsumer
from Handler.helper_functions import *
from Handler.cut_functions import *
from Handler.plot_functions import *
from scipy.stats import binned_statistic, median_abs_deviation
import seaborn as sns


class TrainFlow(object):

    def __init__(self,
                 cfg,
                 learning_rate,
                 weight_decay,
                 number_hidden,
                 number_blocks,
                 batch_size
                 ):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg["DEVICE"])
        self.act = cfg["ACTIVATION_FUNCTION"]

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

        self.bs = batch_size
        self.lr = learning_rate
        self.wd = weight_decay
        self.nh = number_hidden
        self.nb = number_blocks

        cfg['PATH_OUTPUT'] = f"{cfg['PATH_OUTPUT']}/lr_{self.lr}_wd_{self.wd}_nh_{self.nh}_nb_{self.nb}_bs_{self.bs}"
        cfg['PATH_OUTPUT_CATALOGS'] = f"{cfg['PATH_OUTPUT_CATALOGS']}/lr_{self.lr}_wd_{self.wd}_nh_{self.nh}_nb_{self.nb}_bs_{self.bs}"
        cfg['PATH_WRITER'] = (f"{cfg['PATH_OUTPUT']}/{cfg['PATH_WRITER']}/"
                              f"lr_{self.lr}_nh_{self.nh}_nb_{self.nb}_bs_{self.bs}")
        cfg['PATH_PLOTS'] = f"{cfg['PATH_OUTPUT']}/{cfg['PATH_PLOTS']}"
        cfg['PATH_SAVE_NN'] = f"{cfg['PATH_OUTPUT']}/{cfg['PATH_SAVE_NN']}"

        for plot in cfg['PLOTS']:
            cfg[f'PATH_PLOTS_FOLDER'][plot.upper()] = f"{cfg['PATH_PLOTS']}/{plot}"

        self.make_dirs()
        self.writer = SummaryWriter(
            log_dir=cfg['PATH_WRITER'],
            comment=f"learning rate: {self.lr} "
                    f"number hidden: {self.nh}_"
                    f"number blocks: {self.nb}_"
                    f"batch size: {self.bs}"
        )
        self.train_loader, self.valid_loader, self.df_test, self.galaxies = self.init_dataset()

        self.model, self.optimizer = self.init_network(
            num_outputs=len(cfg[f"OUTPUT_COLS_{cfg['LUM_TYPE']}"]),
            num_input=len(cfg[f"INPUT_COLS_{cfg['LUM_TYPE']}"])
        )
        with torch.no_grad():
            for batch_idx, data in enumerate(self.train_loader):
                input_data = data[0].float()
                output_data = data[1].float()
                input_data = input_data.to(self.device)
                output_data = output_data.to(self.device)
                self.writer.add_graph(self.model, (output_data, input_data))
                self.writer.close()
                break

            total_parameters = count_parameters(self.model)
            print(f"Total trainable parameters: {total_parameters}")

        self.global_step = 0
        self.best_validation_loss = float('inf')
        self.best_validation_epoch = 1
        self.best_train_loss = float('inf')
        self.best_train_epoch = 1
        self.best_model = self.model

    def make_dirs(self):
        """"""
        if not os.path.exists(self.cfg['PATH_OUTPUT']):
            os.mkdir(self.cfg['PATH_OUTPUT'])
        if not os.path.exists(self.cfg['PATH_OUTPUT_CATALOGS']):
            os.mkdir(self.cfg['PATH_OUTPUT_CATALOGS'])
        if self.cfg['PLOT_TEST'] is True:
            if not os.path.exists(self.cfg['PATH_PLOTS']):
                os.mkdir(self.cfg['PATH_PLOTS'])
            for path_plot in self.cfg['PATH_PLOTS_FOLDER'].values():
                if not os.path.exists(path_plot):
                    os.mkdir(path_plot)

        if self.cfg['SAVE_NN'] is True:
            if not os.path.exists(self.cfg["PATH_SAVE_NN"]):
                os.mkdir(self.cfg["PATH_SAVE_NN"])

    def init_dataset(self):
        """"""
        galaxies = GalaxyDataset(
            cfg=self.cfg,
            lst_split=[self.cfg['SIZE_TRAINING_DATA'], self.cfg['SIZE_VALIDATION_DATA'], self.cfg['SIZE_TEST_DATA']]
        )

        # Create DataLoaders for training, validation, and testing
        train_loader = DataLoader(galaxies.train_dataset, batch_size=self.bs, shuffle=True, num_workers=0)
        valid_loader = DataLoader(galaxies.val_dataset, batch_size=self.bs, shuffle=False, num_workers=0)
        test_loader = DataLoader(galaxies.test_dataset, batch_size=self.bs, shuffle=False, num_workers=0)

        # training_data, validation_data, test_data = load_data(
        #     cfg=self.cfg,
        #     writer=self.writer
        # )
        #
        # train_output = torch.from_numpy(
        #     training_data[f"data frame training data"][self.cfg[f"OUTPUT_COLS_{self.cfg['LUM_TYPE']}"]].to_numpy())
        # train_input = torch.from_numpy(
        #     training_data[f"data frame training data"][self.cfg[f"INPUT_COLS_{self.cfg['LUM_TYPE']}"]].to_numpy())
        # train_dataset = torch.utils.data.TensorDataset(train_output, train_input)
        #
        # valid_output = torch.from_numpy(
        #     validation_data[f"data frame validation data"][self.cfg[f"OUTPUT_COLS_{self.cfg['LUM_TYPE']}"]].to_numpy())
        # valid_input = torch.from_numpy(
        #     validation_data[f"data frame validation data"][self.cfg[f"INPUT_COLS_{self.cfg['LUM_TYPE']}"]].to_numpy())
        # valid_dataset = torch.utils.data.TensorDataset(valid_output, valid_input)
        #
        # if self.cfg['VALIDATION_BATCH_SIZE'] == -1:
        #     self.cfg['VALIDATION_BATCH_SIZE'] = len(validation_data[f"data frame validation data"])
        #
        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset,
        #     batch_size=self.bs,
        #     shuffle=False,
        #     # **kwargs
        # )
        #
        # valid_loader = torch.utils.data.DataLoader(
        #     valid_dataset,
        #     batch_size=self.cfg['VALIDATION_BATCH_SIZE'],
        #     shuffle=False,
        #     drop_last=False,
        #     # **kwargs
        # )

        return train_loader, valid_loader, test_loader, galaxies  # test_data, test_data[f"scaler"]

    def init_network(self, num_outputs, num_input):
        modules = []
        for _ in range(self.nb):
            modules += [
                fnn.MADE(num_outputs, self.nh, num_input, act=self.act),
                fnn.BatchNormFlow(num_outputs),
                fnn.Reverse(num_outputs)
            ]
        model = fnn.FlowSequential(*modules)

        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)

        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)
        return model, optimizer

    def run_training(self):

        for epoch in range(self.cfg["EPOCHS"]):
            train_loss_epoch = self.train(
                epoch=epoch
            )
            validation_loss = self.validate(
                epoch=epoch
            )

            self.lst_epochs.append(epoch)
            self.lst_train_loss_per_epoch.append(train_loss_epoch)
            self.lst_valid_loss_per_epoch.append(validation_loss)

            if validation_loss < self.best_validation_loss:
                self.best_validation_epoch = epoch
                self.best_validation_loss = validation_loss
                self.best_model = copy.deepcopy(self.model)
                self.best_model.eval()

            if train_loss_epoch < self.best_train_loss:
                self.best_train_epoch = epoch
                self.best_train_loss = train_loss_epoch

            # if epoch - self.best_validation_epoch >= 30:
            #     break

            print(f"Best validation epoch: {self.best_validation_epoch + 1}\t"
                  f"best validation loss: {-self.best_validation_loss}\t"
                  f"learning rate: {self.lr}\t"
                  f"num_hidden: {self.nh}\t"
                  f"num_blocks: {self.nb}\t"
                  f"batch_size: {self.bs}")

            if self.cfg['PLOT_TRAINING'] is True:
                self.plot_data(epoch=epoch)

            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy().astype(np.float64), epoch+1)

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

        if self.cfg['PLOT_TRAINING'] is False:
            self.plot_data(epoch=self.cfg["EPOCHS"] - 1)

        if self.cfg['PLOT_TEST'] is True:
            if self.cfg['PLOT_TRAINING'] is True:
                lst_gif = []
                for plot in self.cfg['PLOTS']:
                    if plot != 'Gif':
                        lst_gif.append((
                            self.cfg[f'PATH_PLOTS_FOLDER'][plot.upper()],
                            f"{self.cfg[f'PATH_PLOTS_FOLDER']['GIF']}/{plot.lower()}.gif"
                        ))

                for gif in lst_gif:
                    try:
                        make_gif(gif[0], gif[1])
                    except Exception as e:
                        print(f"Could not make gif for {gif[0]}. Error message: {e}")

        if self.cfg['SAVE_NN'] is True:
            torch.save(
                self.best_model,
                f"{self.cfg['PATH_SAVE_NN']}/best_model_epoch_{self.best_validation_epoch+1}_run_{self.cfg['RUN_DATE']}.pt")
            torch.save(
                self.model,
                f"{self.cfg['PATH_SAVE_NN']}/last_model_epoch_{self.cfg['EPOCHS']}_run_{self.cfg['RUN_DATE']}.pt")

        self.writer.flush()
        self.writer.close()

    def train(self, epoch):
        self.model.train()
        train_loss = 0.0
        pbar = tqdm(total=len(self.train_loader.dataset))
        for batch_idx, data in enumerate(self.train_loader):
            input_data = data[0].float()
            output_data = data[1].float()
            input_data = input_data.to(self.device)
            output_data = output_data.to(self.device)
            self.optimizer.zero_grad()
            loss = -self.model.log_probs(output_data, input_data).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, error_if_nonfinite=False)
            self.optimizer.step()
            train_loss += loss.data.item() * output_data.size(0)
            pbar.update(output_data.size(0))
            pbar.set_description(f"Training,\t"
                                 f"Epoch: {epoch+1},\t"
                                 f"learning rate: {self.lr},\t"
                                 f"number hidden: {self.nh},\t"
                                 f"number blocks: {self.nb},\t"
                                 f"batch size: {self.bs},\t"
                                 f"loss: {train_loss / pbar.n}")
            self.lst_train_loss_per_batch.append(loss.item())
            self.global_step += 1
        pbar.close()

        train_loss = train_loss / len(self.train_loader.dataset)

        self.writer.add_scalar('training loss', train_loss, epoch+1)

        for module in self.model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 0

        with torch.no_grad():
            self.model(
                inputs=self.train_loader.dataset[:][1].to(output_data.device).float(),
                cond_inputs= self.train_loader.dataset[:][0].to(output_data.device).float()
            )

        for module in self.model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 1
        return train_loss

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        pbar = tqdm(total=len(self.valid_loader.dataset))
        for batch_idx, data in enumerate(self.valid_loader):
            input_data = data[0].float()
            output_data = data[1].float()
            input_data = input_data.to(self.device)
            output_data = output_data.to(self.device)
            with torch.no_grad():
                loss = -self.model.log_probs(output_data, input_data).mean()
                self.lst_valid_loss_per_batch.append(loss.item())
                val_loss += loss.data.item() * output_data.size(0)
            pbar.update(output_data.size(0))
            pbar.set_description(f"Validation,\t"
                                 f"Epoch: {epoch + 1},\t"
                                 f"learning rate: {self.lr},\t"
                                 f"number hidden: {self.nh},\t"
                                 f"number blocks: {self.nb},\t"
                                 f"batch size: {self.bs},\t"
                                 f"loss: {val_loss / pbar.n}")
        pbar.close()
        val_loss = val_loss / len(self.valid_loader.dataset)
        self.writer.add_scalar('validation loss', val_loss, epoch+1)
        return val_loss

    def plot_data(self, epoch):
        """"""
        sns.set_theme()
        self.model.eval()

        # cond_data = torch.Tensor(self.df_test[f"data frame test data"][self.cfg[f"INPUT_COLS_{self.cfg['LUM_TYPE']}"]].to_numpy())
        input_data = self.galaxies.test_dataset[:][0].float()
        output_data_true = self.galaxies.test_dataset[:][1].float()
        input_data = input_data.to(self.device)
        output_data_true = output_data_true.to(self.device)
        with torch.no_grad():
            output_data_gandalf = self.model.sample(len(input_data), cond_inputs=input_data).detach()  # .cpu()

        df_gandalf = pd.DataFrame()
        df_gandalf[self.cfg[f"INPUT_COLS_{self.cfg['LUM_TYPE']}"]] = input_data.cpu().numpy()
        df_gandalf[self.cfg[f"OUTPUT_COLS_{self.cfg['LUM_TYPE']}"]] = output_data_gandalf.cpu().numpy()
        df_gandalf = df_gandalf.astype('float64')

        df_balrog = pd.DataFrame()
        df_balrog[self.cfg[f"INPUT_COLS_{self.cfg['LUM_TYPE']}"]] = input_data.cpu().numpy()
        df_balrog[self.cfg[f"OUTPUT_COLS_{self.cfg['LUM_TYPE']}"]] = output_data_true.cpu().numpy()
        df_balrog = df_balrog.astype('float64')

        if self.cfg['APPLY_SCALER'] is True:
            df_gandalf = pd.DataFrame(self.galaxies.scaler.inverse_transform(df_gandalf), columns=df_gandalf.keys())
            df_balrog = pd.DataFrame(self.galaxies.scaler.inverse_transform(df_balrog), columns=df_balrog.keys())

        if self.cfg['APPLY_YJ_TRANSFORM'] is True:
            if self.cfg['TRANSFORM_COLS'] is None:
                trans_col = df_balrog.keys()
            else:
                trans_col = self.cfg['TRANSFORM_COLS']
            df_balrog = self.galaxies.yj_inverse_transform_data(
                data_frame=df_balrog,
                columns=trans_col
            )
            df_gandalf = self.galaxies.yj_inverse_transform_data(
                data_frame=df_gandalf,
                columns=trans_col
            )

        df_gandalf[self.cfg['CUT_COLS']] = self.galaxies.df_cut_cols.iloc[self.galaxies.dict_indices['test']].to_numpy()
        df_balrog[self.cfg['CUT_COLS']] = self.galaxies.df_cut_cols.iloc[self.galaxies.dict_indices['test']].to_numpy()

        # df_generated_label = pd.DataFrame(input_data.numpy(), columns=self.cfg[f"INPUT_COLS_{self.cfg['LUM_TYPE']}"])
        # df_generated_output = pd.DataFrame(output_data_gandalf.cpu().numpy(), columns=self.cfg[f"OUTPUT_COLS_{self.cfg['LUM_TYPE']}"])
        # df_generated_output = pd.concat([df_generated_label, df_generated_output], axis=1)
        # df_generated_scaled[self.cfg[f"INPUT_COLS_{self.cfg['LUM_TYPE']}"]+self.cfg[f"OUTPUT_COLS_{self.cfg['LUM_TYPE']}"]] = df_generated_output
        # generated_rescaled = self.scaler.inverse_transform(df_generated_scaled)
        # df_generated = pd.DataFrame(generated_rescaled, columns=df_generated_scaled.keys())
        #
        # true = self.scaler.inverse_transform(df_true_scaled)
        # df_true = pd.DataFrame(true, columns=df_generated_scaled.keys())

        if self.cfg['APPLY_FILL_NA'] is True:
            for col in self.cfg['FILL_NA'].keys():
                df_gandalf[col] = df_gandalf[col].fillna(self.cfg['FILL_NA'][col])

        for b in self.cfg['BANDS']:
            df_gandalf[f"meas {b} - true {b}"] = df_gandalf[f"unsheared/{self.cfg['LUM_TYPE'].lower()}_{b}"] - df_gandalf[f"BDF_{self.cfg['LUM_TYPE'].upper()}_DERED_CALIB_{b.upper()}"]
            df_balrog[f"meas {b} - true {b}"] = df_balrog[f"unsheared/{self.cfg['LUM_TYPE'].lower()}_{b}"] - df_balrog[f"BDF_{self.cfg['LUM_TYPE'].upper()}_DERED_CALIB_{b.upper()}"]

        df_balrog_cut = df_balrog.copy()

        if self.cfg['APPLY_OBJECT_CUT'] is not True:
            df_balrog_cut = self.galaxies.unsheared_object_cuts(data_frame=df_balrog_cut)
        if self.cfg['APPLY_FLAG_CUT'] is not True:
            df_balrog_cut = self.galaxies.flag_cuts(data_frame=df_balrog_cut)
        if self.cfg['APPLY_UNSHEARED_MAG_CUT'] is not True:
            df_balrog_cut = self.galaxies.unsheared_mag_cut(data_frame=df_balrog_cut)
        if self.cfg['APPLY_UNSHEARED_SHEAR_CUT'] is not True:
            df_balrog_cut = self.galaxies.unsheared_shear_cuts(data_frame=df_balrog_cut)
        if self.cfg['APPLY_AIRMASS_CUT'] is not True:
            df_balrog_cut = self.galaxies.airmass_cut(data_frame=df_balrog_cut)

        df_gandalf_cut = df_gandalf.copy()

        if self.cfg['APPLY_OBJECT_CUT'] is not True:
            df_gandalf_cut = self.galaxies.unsheared_object_cuts(data_frame=df_gandalf_cut)
        if self.cfg['APPLY_FLAG_CUT'] is not True:
            df_gandalf_cut = self.galaxies.flag_cuts(data_frame=df_gandalf_cut)
        if self.cfg['APPLY_UNSHEARED_MAG_CUT'] is not True:
            df_gandalf_cut = self.galaxies.unsheared_mag_cut(data_frame=df_gandalf_cut)
        if self.cfg['APPLY_UNSHEARED_SHEAR_CUT'] is not True:
            df_gandalf_cut = self.galaxies.unsheared_shear_cuts(data_frame=df_gandalf_cut)
        if self.cfg['APPLY_AIRMASS_CUT'] is not True:
            df_gandalf_cut = self.galaxies.airmass_cut(data_frame=df_gandalf_cut)

        df_gandalf = calc_color(
            data_frame=df_gandalf,
            colors=self.cfg['COLORS'],
            column_name=f"unsheared/{self.cfg['LUM_TYPE'].lower()}"
        )
        df_balrog = calc_color(
            data_frame=df_balrog,
            colors=self.cfg['COLORS'],
            column_name=f"unsheared/{self.cfg['LUM_TYPE'].lower()}"
        )
        df_gandalf_cut = calc_color(
            data_frame=df_gandalf_cut,
            colors=self.cfg['COLORS'],
            column_name=f"unsheared/{self.cfg['LUM_TYPE'].lower()}"
        )
        df_balrog_cut = calc_color(
            data_frame=df_balrog_cut,
            colors=self.cfg['COLORS'],
            column_name=f"unsheared/{self.cfg['LUM_TYPE'].lower()}"
        )

        if self.cfg['PLOT_LOSS'] is True:
            img_grid = loss_plot(
                epoch=epoch,
                lst_train_loss_per_batch=self.lst_train_loss_per_batch,
                lst_train_loss_per_epoch=self.lst_train_loss_per_epoch,
                lst_valid_loss_per_batch=self.lst_valid_loss_per_batch,
                lst_valid_loss_per_epoch=self.lst_valid_loss_per_epoch,
                show_plot=self.cfg['SHOW_PLOT'],
                save_plot=self.cfg['SAVE_PLOT'],
                save_name=f"{self.cfg['PATH_PLOTS_FOLDER']['LOSS_PLOT']}/loss_{epoch + 1}.png"
            )
            self.writer.add_image("loss plot", img_grid, epoch + 1)

        if self.cfg['PLOT_COLOR_COLOR'] is True:
            try:
                img_grid, self.dict_delta_color_color = plot_compare_corner(
                    data_frame_generated=df_gandalf,
                    data_frame_true=df_balrog,
                    dict_delta=self.dict_delta_color_color,
                    epoch=epoch+1,
                    title=f"color-color plot",
                    columns=["r-i", "i-z"],
                    labels=["r-i", "i-z"],
                    show_plot=self.cfg['SHOW_PLOT'],
                    save_plot=self.cfg['SAVE_PLOT'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['COLOR_COLOR_PLOT']}/color_color_{epoch + 1}.png",
                    ranges=[(-4, 4), (-4, 4)]
                )
                self.writer.add_image("color color plot", img_grid, epoch+1)
            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['COLOR_COLOR_PLOT']}/color_color_{epoch + 1}.png")
            try:
                img_grid, self.dict_delta_color_color_mcal = plot_compare_corner(
                    data_frame_generated=df_gandalf_cut,
                    data_frame_true=df_balrog_cut,
                    dict_delta=self.dict_delta_color_color_mcal,
                    epoch=epoch+1,
                    title=f"mcal color-color plot",
                    columns=["r-i", "i-z"],
                    labels=["r-i", "i-z"],
                    show_plot=self.cfg['SHOW_PLOT'],
                    save_plot=self.cfg['SAVE_PLOT'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_COLOR_COLOR_PLOT']}/mcal_color_color_{epoch + 1}.png",
                    ranges=[(-1.2, 1.8), (-1.5, 1.5)]
                )
                self.writer.add_image("color color plot mcal", img_grid, epoch+1)
            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_COLOR_COLOR_PLOT']}/mcal_color_color_{epoch + 1}.png")
        if self.cfg['PLOT_RESIDUAL']:
            try:
                img_grid = residual_plot(
                    data_frame_generated=df_gandalf,
                    data_frame_true=df_balrog,
                    luminosity_type=self.cfg['LUM_TYPE'],
                    plot_title=f"residual, epoch {epoch+1}",
                    bands=self.cfg['BANDS'],
                    show_plot=self.cfg['SHOW_PLOT'],
                    save_plot=self.cfg['SAVE_PLOT'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['RESIDUAL_PLOT']}/residual_plot_{epoch + 1}.png"
                )
                self.writer.add_image("residual plot", img_grid, epoch + 1)
            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['RESIDUAL_PLOT']}/residual_plot_{epoch + 1}.png")

            try:
                img_grid = residual_plot(
                    data_frame_generated=df_gandalf_cut,
                    data_frame_true=df_balrog_cut,
                    luminosity_type=self.cfg['LUM_TYPE'],
                    plot_title=f"mcal residual, epoch {epoch+1}",
                    bands=self.cfg['BANDS'],
                    show_plot=self.cfg['SHOW_PLOT'],
                    save_plot=self.cfg['SAVE_PLOT'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_RESIDUAL_PLOT']}/mcal_residual_plot_{epoch + 1}.png"
                )
                self.writer.add_image("residual plot mcal", img_grid, epoch + 1)
            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_RESIDUAL_PLOT']}/mcal_residual_plot_{epoch + 1}.png")

        if self.cfg['PLOT_CHAIN'] is True:
            try:
                img_grid, self.dict_delta_unsheared = plot_compare_corner(
                    data_frame_generated=df_gandalf,
                    data_frame_true=df_balrog,
                    dict_delta=self.dict_delta_unsheared,
                    epoch=epoch+1,
                    title=f"chain plot",
                    show_plot=self.cfg['SHOW_PLOT'],
                    save_plot=self.cfg['SAVE_PLOT'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['CHAIN_PLOT']}/chainplot_{epoch + 1}.png",
                    columns=[
                        f"unsheared/{self.cfg['LUM_TYPE'].lower()}_r",
                        f"unsheared/{self.cfg['LUM_TYPE'].lower()}_i",
                        f"unsheared/{self.cfg['LUM_TYPE'].lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T"
                    ],
                    labels=[
                        f"{self.cfg['LUM_TYPE'].lower()}_r",
                        f"{self.cfg['LUM_TYPE'].lower()}_i",
                        f"{self.cfg['LUM_TYPE'].lower()}_z",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    ranges=[(15, 30), (15, 30), (15, 30), (-2, 4), (-3.5, 4), (-1.5, 2)]
                )
                self.writer.add_image("chain plot", img_grid, epoch + 1)
            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['CHAIN_PLOT']}/chainplot_{epoch + 1}.png")

            try:
                img_grid, self.dict_delta_unsheared_mcal = plot_compare_corner(
                    data_frame_generated=df_gandalf_cut,
                    data_frame_true=df_balrog_cut,
                    dict_delta=self.dict_delta_unsheared_mcal,
                    epoch=epoch+1,
                    title=f"mcal chain plot",
                    show_plot=self.cfg['SHOW_PLOT'],
                    save_plot=self.cfg['SAVE_PLOT'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_CHAIN_PLOT']}/mcal_chainplot_{epoch + 1}.png",
                    columns=[
                        f"unsheared/{self.cfg['LUM_TYPE'].lower()}_r",
                        f"unsheared/{self.cfg['LUM_TYPE'].lower()}_i",
                        f"unsheared/{self.cfg['LUM_TYPE'].lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T",
                    ],
                    labels=[
                        f"{self.cfg['LUM_TYPE'].lower()}_r",
                        f"{self.cfg['LUM_TYPE'].lower()}_i",
                        f"{self.cfg['LUM_TYPE'].lower()}_z",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    ranges=[(15, 30), (15, 30), (15, 30), (-75, 425), (-1.5, 6), (-1, 4)]
                )
                self.writer.add_image("chain plot mcal", img_grid, epoch + 1)
            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_CHAIN_PLOT']}/mcal_chainplot_{epoch + 1}.png")

            try:
                img_grid, self.dict_delta_color_diff = plot_compare_corner(
                    data_frame_generated=df_gandalf,
                    data_frame_true=df_balrog,
                    dict_delta=self.dict_delta_color_diff,
                    epoch=epoch+1,
                    title=f"color diff plot",
                    show_plot=self.cfg['SHOW_PLOT'],
                    save_plot=self.cfg['SAVE_PLOT'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['COLOR_DIFF_PLOT']}/color_diff_{epoch + 1}.png",
                    columns=[
                        # f"BDF_{cfg['LUM_TYPE'].upper()}_DERED_CALIB_R",
                        # f"BDF_{cfg['LUM_TYPE'].upper()}_DERED_CALIB_I",
                        # f"BDF_{cfg['LUM_TYPE'].upper()}_DERED_CALIB_Z",
                        "meas r - true r",
                        "meas i - true i",
                        "meas z - true z"
                    ],
                    labels=[
                        # "true r",
                        # "true i",
                        # "true z",
                        "meas r - true r",
                        "meas i - true i",
                        "meas z - true z"
                    ],
                    ranges=[(-4, 4), (-4, 4), (-4, 4)]  # (18, 30), (18, 30), (18, 30),
                )
                self.writer.add_image("color diff plot", img_grid, epoch + 1)
            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['COLOR_DIFF_PLOT']}/color_diff_{epoch + 1}.png")

            try:
                img_grid, self.dict_delta_color_diff_mcal = plot_compare_corner(
                    data_frame_generated=df_gandalf_cut,
                    data_frame_true=df_balrog_cut,
                    dict_delta=self.dict_delta_color_diff_mcal,
                    epoch=epoch+1,
                    title=f"mcal color diff plot",
                    show_plot=self.cfg['SHOW_PLOT'],
                    save_plot=self.cfg['SAVE_PLOT'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_COLOR_DIFF_PLOT']}/mcal_color_diff_{epoch + 1}.png",
                    columns=[
                        # f"BDF_{cfg['LUM_TYPE'].upper()}_DERED_CALIB_R",
                        # f"BDF_{cfg['LUM_TYPE'].upper()}_DERED_CALIB_I",
                        # f"BDF_{cfg['LUM_TYPE'].upper()}_DERED_CALIB_Z",
                        "meas r - true r",
                        "meas i - true i",
                        "meas z - true z"
                    ],
                    labels=[
                        # "true r",
                        # "true i",
                        # "true z",
                        "meas r - true r",
                        "meas i - true i",
                        "meas z - true z"
                    ],
                    ranges=[(-1.5, 1.5), (-1.5, 1.5), (-1.5, 1.5)]  # (18, 30), (18, 30), (18, 30),
                )
                self.writer.add_image("color diff plot mcal", img_grid, epoch + 1)
            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_COLOR_DIFF_PLOT']}/mcal_color_diff_{epoch + 1}.png")

        if self.cfg['PLOT_MEAN'] is True:
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
                        f"unsheared/{self.cfg['LUM_TYPE'].lower()}_r",
                        f"unsheared/{self.cfg['LUM_TYPE'].lower()}_i",
                        f"unsheared/{self.cfg['LUM_TYPE'].lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T"
                    ],
                    lst_labels=[
                        f"{self.cfg['LUM_TYPE'].lower()}_r",
                        f"{self.cfg['LUM_TYPE'].lower()}_i",
                        f"{self.cfg['LUM_TYPE'].lower()}_z",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    lst_marker=["o", "^", "X", "d", "s", "*"],
                    lst_color=["blue", "red", "green", "orange", "purple", "black"],
                    plot_title="mean ratio",
                    show_plot=self.cfg['SHOW_PLOT'],
                    save_plot=self.cfg['SAVE_PLOT'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['MEAN_PLOT']}/mean_{epoch+1}.png",
                    statistic_type="mean"
                )
                self.writer.add_image("mean plot", img_grid, epoch + 1)
                for idx_plot, lst_plot in enumerate(lists_mean_to_plot):
                    lst_plot = lists_mean_to_plot_updated[idx_plot]

            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['MEAN_PLOT']}/mean_{epoch+1}.png")
                print(f"Mean shapes: \t epoch \t {self.cfg['LUM_TYPE'].lower()} r \t {self.cfg['LUM_TYPE'].lower()} i \t {self.cfg['LUM_TYPE'].lower()} z \t snr \t size_ratio \t T")
                print(f"\t           \t {len(self.lst_epochs)} \t {len(self.lst_mean_mag_r)} \t {len(self.lst_mean_mag_i)} \t {len(self.lst_mean_mag_z)} \t {len(self.lst_mean_snr)} \t {len(self.lst_mean_size_ratio)} \t {len(self.lst_mean_t)}")

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
                        f"unsheared/{self.cfg['LUM_TYPE'].lower()}_r",
                        f"unsheared/{self.cfg['LUM_TYPE'].lower()}_i",
                        f"unsheared/{self.cfg['LUM_TYPE'].lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T"
                    ],
                    lst_labels=[
                        f"{self.cfg['LUM_TYPE'].lower()}_r",
                        f"{self.cfg['LUM_TYPE'].lower()}_i",
                        f"{self.cfg['LUM_TYPE'].lower()}_z",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    lst_marker=["o", "^", "X", "d", "s", "*"],
                    lst_color=["blue", "red", "green", "orange", "purple", "black"],
                    plot_title="mcal mean ratio",
                    show_plot=self.cfg['SHOW_PLOT'],
                    save_plot=self.cfg['SAVE_PLOT'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_MEAN_PLOT']}/mcal_mean_{epoch + 1}.png",
                    statistic_type="mean"
                )
                self.writer.add_image("mean plot mcal", img_grid, epoch + 1)
                for idx_plot_cut, lst_plot_cut in enumerate(lists_mean_to_plot_cut):
                    lst_plot_cut = lists_mean_to_plot_cut_updated[idx_plot_cut]

            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_MEAN_PLOT']}/mcal_mean_{epoch + 1}.png")
                print(f"Mean mcal shapes: \t epoch \t {self.cfg['LUM_TYPE'].lower()} r \t {self.cfg['LUM_TYPE'].lower()} i \t {self.cfg['LUM_TYPE'].lower()} z \t snr \t size_ratio \t T")
                print(f"\t           \t {len(self.lst_epochs)} \t {len(self.lst_mean_mag_r_cut)} \t {len(self.lst_mean_mag_i_cut)} \t {len(self.lst_mean_mag_z_cut)} \t {len(self.lst_mean_snr_cut)} \t {len(self.lst_mean_size_ratio_cut)} \t {len(self.lst_mean_t_cut)}")

        if self.cfg['PLOT_STD'] is True:
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
                        f"unsheared/{self.cfg['LUM_TYPE'].lower()}_r",
                        f"unsheared/{self.cfg['LUM_TYPE'].lower()}_i",
                        f"unsheared/{self.cfg['LUM_TYPE'].lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T"
                    ],
                    lst_labels=[
                        f"{self.cfg['LUM_TYPE'].lower()}_r",
                        f"{self.cfg['LUM_TYPE'].lower()}_i",
                        f"{self.cfg['LUM_TYPE'].lower()}_z",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    lst_marker=["o", "^", "X", "d", "s", "*"],
                    lst_color=["blue", "red", "green", "orange", "purple", "black"],
                    plot_title="std ratio",
                    show_plot=self.cfg['SHOW_PLOT'],
                    save_plot=self.cfg['SAVE_PLOT'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['STD_PLOT']}/std_{epoch + 1}.png",
                    statistic_type="std"
                )
                self.writer.add_image("std plot", img_grid, epoch + 1)
                for idx_plot, lst_plot in enumerate(lists_std_to_plot):
                    lst_plot = lists_std_to_plot_updated[idx_plot]

            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['STD_PLOT']}/std_{epoch + 1}.png")
                print(f"Std shapes: \t epoch \t {self.cfg['LUM_TYPE'].lower()} r \t {self.cfg['LUM_TYPE'].lower()} i \t {self.cfg['LUM_TYPE'].lower()} z \t snr \t size_ratio \t T")
                print(f"\t           \t {len(self.lst_epochs)} \t {len(self.lst_std_mag_r)} \t {len(self.lst_std_mag_i)} \t {len(self.lst_std_mag_z)} \t {len(self.lst_std_snr)} \t {len(self.lst_std_size_ratio)} \t {len(self.lst_std_t)}")

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
                        f"unsheared/{self.cfg['LUM_TYPE'].lower()}_r",
                        f"unsheared/{self.cfg['LUM_TYPE'].lower()}_i",
                        f"unsheared/{self.cfg['LUM_TYPE'].lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T"
                    ],
                    lst_labels=[
                        f"{self.cfg['LUM_TYPE'].lower()}_r",
                        f"{self.cfg['LUM_TYPE'].lower()}_i",
                        f"{self.cfg['LUM_TYPE'].lower()}_z",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    lst_marker=["o", "^", "X", "d", "s", "*"],
                    lst_color=["blue", "red", "green", "orange", "purple", "black"],
                    plot_title="mcal std ratio",
                    show_plot=self.cfg['SHOW_PLOT'],
                    save_plot=self.cfg['SAVE_PLOT'],
                    save_name=f"{self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_STD_PLOT']}/mcal_std_{epoch + 1}.png",
                    statistic_type="std"
                )
                self.writer.add_image("std plot mcal", img_grid, epoch + 1)
                for idx_plot_cut, lst_plot_cut in enumerate(lists_std_to_plot_cut):
                    lst_plot_cut = lists_std_to_plot_cut_updated[idx_plot_cut]

            except Exception as e:
                print(f"Error {e}: {self.cfg[f'PATH_PLOTS_FOLDER']['MCAL_STD_PLOT']}/mcal_std_{epoch + 1}.png")
                print(f"Std mcal shapes: \t epoch \t {self.cfg['LUM_TYPE'].lower()} r \t {self.cfg['LUM_TYPE'].lower()} i \t {self.cfg['LUM_TYPE'].lower()} z \t snr \t size_ratio \t T")
                print(f"\t           \t {len(self.lst_epochs)} \t {len(self.lst_std_mag_r_cut)} \t {len(self.lst_std_mag_i_cut)} \t {len(self.lst_std_mag_z_cut)} \t {len(self.lst_std_snr_cut)} \t {len(self.lst_std_size_ratio_cut)} \t {len(self.lst_std_t_cut)}")

