import copy
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import ray
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn
import torchvision.utils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from tensorboard.plugins.hparams import api as hp
import galaxyflow.flow as fnn
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
                 path_train_data,
                 size_training_dataset,
                 size_validation_dataset,
                 size_test_dataset,
                 luminosity_type,
                 path_output,
                 col_label_flow,
                 col_output_flow,
                 plot_test,
                 show_plot,
                 save_plot,
                 save_nn,
                 learning_rate,
                 weight_decay,
                 number_hidden,
                 number_blocks,
                 epochs,
                 device,
                 activation_function,
                 batch_size,
                 valid_batch_size,
                 selected_scaler,
                 do_loss_plot,
                 do_color_color_plot,
                 do_residual_plot,
                 do_chain_plot,
                 do_mean_plot,
                 do_std_plot,
                 do_flags_plot,
                 do_detected_plot,
                 run_hyperparameter_tuning,
                 lst_replace_transform_cols,
                 lst_replace_values,
                 lst_fill_na,
                 run,
                 reproducible,
                 apply_fill_na,
                 apply_object_cut,
                 apply_flag_cut,
                 apply_airmass_cut,
                 apply_unsheared_mag_cut,
                 apply_unsheared_shear_cut,
                 plot_load_data
                 ):
        super().__init__()
        self.size_training_dataset = size_training_dataset
        self.size_validation_dataset = size_validation_dataset
        self.size_test_dataset = size_test_dataset
        self.do_loss_plot = do_loss_plot
        self.run = run
        self.do_color_color_plot = do_color_color_plot
        self.do_residual_plot = do_residual_plot
        self.do_chain_plot = do_chain_plot
        self.do_mean_plot = do_mean_plot
        self.do_std_plot = do_std_plot
        self.do_flags_plot = do_flags_plot
        self.do_detected_plot = do_detected_plot
        self.epochs = epochs
        self.plot_test = plot_test
        self.show_plot = show_plot
        self.save_plot = save_plot
        self.save_nn = save_nn
        self.reproducible = reproducible
        self.device = torch.device(device)
        self.act = activation_function
        self.actal_selected_values = None
        self.plot_load_data = plot_load_data
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = batch_size
        self.lst_replace_transform_cols = lst_replace_transform_cols
        self.lst_replace_values = lst_replace_values
        self.lst_fill_na = lst_fill_na
        self.lst_epochs = []
        self.lst_epochs_cut = []
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

        self.path_train_data = path_train_data
        if run is not None:
            path_output = f"{path_output}/run_{run}"
            if not os.path.exists(path_output):
                os.mkdir(path_output)
        self.path_output = path_output
        self.selected_scaler = selected_scaler
        self.col_label_flow = col_label_flow
        self.col_output_flow = col_output_flow
        self.apply_fill_na = apply_fill_na
        self.apply_object_cut = apply_object_cut
        self.apply_flag_cut = apply_flag_cut
        self.apply_airmass_cut = apply_airmass_cut
        self.apply_unsheared_mag_cut = apply_unsheared_mag_cut
        self.apply_unsheared_shear_cut = apply_unsheared_shear_cut
        self.luminosity_type = luminosity_type
        self.run_hyperparameter_tuning = run_hyperparameter_tuning

        if self.run_hyperparameter_tuning is not True:
            self.lr = learning_rate
            self.wd = weight_decay
            self.num_hidden = number_hidden
            self.num_blocks = number_blocks
            self.path_output_flow = f"{self.path_output}/Flow_" \
                                    f"lr_{self.lr}_" \
                                    f"wd_{self.wd}_" \
                                    f"num_hidden_{self.num_hidden}_" \
                                    f"num_blocks_{self.num_blocks}_" \
                                    f"batch_size_{self.batch_size}"
            self.path_writer = f"{self.path_output}/writer/" \
                               f"lr_{self.lr}_" \
                               f"num_hidden_{self.num_hidden}_" \
                               f"num_blocks_{self.num_blocks}_" \
                               f"batch_size_{self.batch_size}"
            self.path_plots = f"{self.path_output_flow}/plots"
            self.path_chain_plot = f"{self.path_plots}/chain_plot"
            self.path_loss_plot = f"{self.path_plots}/loss_plot"
            self.path_mean_plot = f"{self.path_plots}/mean_plot"
            self.path_std_plot = f"{self.path_plots}/std_plot"
            self.path_residual_plot = f"{self.path_plots}/residual_plot"
            self.path_color_diff_plot = f"{self.path_plots}/color_diff_plot"
            self.path_color_color_plot = f"{self.path_plots}/color_color_plot"
            self.path_chain_plot_mcal = f"{self.path_plots}/chain_plot_mcal"
            self.path_mean_plot_mcal = f"{self.path_plots}/mean_plot_mcal"
            self.path_std_plot_mcal = f"{self.path_plots}/std_plot_mcal"
            self.path_residual_plot_mcal = f"{self.path_plots}/residual_plot_mcal"
            self.path_color_diff_plot_mcal = f"{self.path_plots}/color_diff_plot_mcal"
            self.path_color_color_plot_mcal = f"{self.path_plots}/color_color_plot_mcal"
            self.path_gifs = f"{self.path_plots}/gifs"
            self.path_save_nn = f"{self.path_output_flow}/nn"
            self.make_dirs()
            self.writer = SummaryWriter(
                log_dir=self.path_writer,
                comment=f"learning rate: {self.lr} "
                        f"number hidden: {self.num_hidden}_"
                        f"number blocks: {self.num_blocks}_"
                        f"batch size: {self.batch_size}"
            )
            self.train_loader, self.valid_loader, self.df_test, self.scaler = self.init_dataset(
                path_train_data=path_train_data,
                selected_scaler=selected_scaler
            )

            self.model, self.optimizer = self.init_network(
                num_inputs=len(col_output_flow),
                num_cond_inputs=len(col_label_flow)
            )
            with torch.no_grad():
                for batch_idx, data in enumerate(self.train_loader):
                    cond_data = data[1].float()
                    cond_data = cond_data.to(self.device)
                    data = data[0]
                    data = data.to(self.device)
                    self.writer.add_graph(self.model, (data, cond_data))
                    self.writer.close()
                    break

            self.global_step = 0
            self.best_validation_loss = float('inf')
            self.best_validation_epoch = 1
            self.best_train_loss = float('inf')
            self.best_train_epoch = 1
            self.best_model = self.model

    def make_dirs(self):
        """"""
        if self.plot_test is True:
            if not os.path.exists(self.path_output_flow):
                os.mkdir(self.path_output_flow)
            if not os.path.exists(self.path_plots):
                os.mkdir(self.path_plots)
            if not os.path.exists(self.path_chain_plot):
                os.mkdir(self.path_chain_plot)
            if not os.path.exists(self.path_loss_plot):
                os.mkdir(self.path_loss_plot)
            if not os.path.exists(self.path_mean_plot):
                os.mkdir(self.path_mean_plot)
            if not os.path.exists(self.path_std_plot):
                os.mkdir(self.path_std_plot)
            if not os.path.exists(self.path_color_diff_plot):
                os.mkdir(self.path_color_diff_plot)
            if not os.path.exists(self.path_color_color_plot):
                os.mkdir(self.path_color_color_plot)
            if not os.path.exists(self.path_residual_plot):
                os.mkdir(self.path_residual_plot)
            if not os.path.exists(self.path_chain_plot_mcal):
                os.mkdir(self.path_chain_plot_mcal)
            if not os.path.exists(self.path_mean_plot_mcal):
                os.mkdir(self.path_mean_plot_mcal)
            if not os.path.exists(self.path_std_plot_mcal):
                os.mkdir(self.path_std_plot_mcal)
            if not os.path.exists(self.path_color_diff_plot_mcal):
                os.mkdir(self.path_color_diff_plot_mcal)
            if not os.path.exists(self.path_color_color_plot_mcal):
                os.mkdir(self.path_color_color_plot_mcal)
            if not os.path.exists(self.path_residual_plot_mcal):
                os.mkdir(self.path_residual_plot_mcal)
            if not os.path.exists(self.path_gifs):
                os.mkdir(self.path_gifs)

        if self.save_nn is True:
            if not os.path.exists(self.path_save_nn):
                os.mkdir(self.path_save_nn)

    def init_dataset(self, path_train_data, selected_scaler):
        """"""
        training_data, validation_data, test_data = load_data(
            path_training_data=path_train_data,
            path_output=self.path_output,
            luminosity_type=self.luminosity_type,
            selected_scaler=selected_scaler,
            size_training_dataset=self.size_training_dataset,
            size_validation_dataset=self.size_validation_dataset,
            size_test_dataset=self.size_test_dataset,
            reproducible=self.reproducible,
            run=self.run,
            lst_replace_transform_cols=self.lst_replace_transform_cols,
            lst_replace_values=self.lst_replace_values,
            apply_object_cut=self.apply_object_cut,
            apply_flag_cut=self.apply_flag_cut,
            apply_airmass_cut=self.apply_airmass_cut,
            apply_unsheared_mag_cut=self.apply_unsheared_mag_cut,
            apply_unsheared_shear_cut=self.apply_unsheared_shear_cut,
            plot_data=self.plot_load_data,
            writer=self.writer
        )

        train_tensor = torch.from_numpy(training_data[f"data frame training data"][self.col_output_flow].to_numpy())
        train_labels = torch.from_numpy(training_data[f"data frame training data"][self.col_label_flow].to_numpy())
        train_dataset = torch.utils.data.TensorDataset(train_tensor, train_labels)

        valid_tensors = torch.from_numpy(validation_data[f"data frame validation data"][self.col_output_flow].to_numpy())
        valid_labels = torch.from_numpy(validation_data[f"data frame validation data"][self.col_label_flow].to_numpy())
        valid_dataset = torch.utils.data.TensorDataset(valid_tensors, valid_labels)
        if self.valid_batch_size == -1:
            self.valid_batch_size = len(validation_data[f"data frame validation data"])

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            # **kwargs
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
            drop_last=False,
            # **kwargs
        )

        return train_loader, valid_loader, test_data, test_data[f"scaler"]

    def init_network(self, num_inputs, num_cond_inputs):
        modules = []
        for _ in range(self.num_blocks):
            modules += [
                fnn.MADE(num_inputs, self.num_hidden, num_cond_inputs, act=self.act),
                fnn.BatchNormFlow(num_inputs),
                fnn.Reverse(num_inputs)
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

    def hyperparameter_tuning(self, config):
        self.lr = config["learning_rate"]
        self.num_hidden = config["number_hidden"]
        self.num_blocks = config["number_blocks"]
        self.batch_size = config["batch_size"]
        self.wd = config["weight_decay"]
        self.train_loader, self.valid_loader, self.df_test, self.scaler = self.init_dataset(
            path_train_data=self.path_train_data,
            selected_scaler=self.selected_scaler
        )

        self.model, self.optimizer = self.init_network(
            num_inputs=len(self.col_output_flow),
            num_cond_inputs=len(self.col_label_flow)
        )
        self.lst_epochs = []
        self.lst_epochs_cut = []
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
        self.global_step = 0
        self.best_validation_loss = float('inf')
        self.best_validation_epoch = 0
        self.best_model = self.model
        self.path_output_flow = f"{self.path_output}/Flow_hypertuning_" \
                                f"lr_{self.lr}_" \
                                f"wd_{self.wd}_" \
                                f"num_hidden_{self.num_hidden}_" \
                                f"num_blocks_{self.num_blocks}_" \
                                f"batch_size_{self.batch_size}"
        self.path_plots = f"{self.path_output_flow}/plots"
        self.path_chain_plot = f"{self.path_plots}/chain_plot"
        self.path_loss_plot = f"{self.path_plots}/loss_plot"
        self.path_mean_plot = f"{self.path_plots}/mean_plot"
        self.path_std_plot = f"{self.path_plots}/std_plot"
        self.path_residual_plot = f"{self.path_plots}/residual_plot"
        self.path_color_diff_plot = f"{self.path_plots}/color_diff_plot"
        self.path_color_color_plot = f"{self.path_plots}/color_color_plot"
        self.path_chain_plot_mcal = f"{self.path_plots}/chain_plot_mcal"
        self.path_mean_plot_mcal = f"{self.path_plots}/mean_plot_mcal"
        self.path_std_plot_mcal = f"{self.path_plots}/std_plot_mcal"
        self.path_residual_plot_mcal = f"{self.path_plots}/residual_plot_mcal"
        self.path_color_diff_plot_mcal = f"{self.path_plots}/color_diff_plot_mcal"
        self.path_color_color_plot_mcal = f"{self.path_plots}/color_color_plot_mcal"
        self.path_gifs = f"{self.path_plots}/gifs"
        self.path_save_nn = f"{self.path_output_flow}/nn"
        self.writer = SummaryWriter(log_dir=f"{self.path_output_flow}/writer", comment=f"_lr_{self.lr}")
        self.make_dirs()
        self.run_training()

    def run_training(self):

        for epoch in range(self.epochs):
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

            if epoch - self.best_validation_epoch >= 30:
                break

            print(f"Best validation epoch: {self.best_validation_epoch + 1}\t"
                  f"best validation loss: {-self.best_validation_loss}\t"
                  f"learning rate: {self.lr}\t"
                  f"num_hidden: {self.num_hidden}\t"
                  f"num_blocks: {self.num_blocks}\t"
                  f"batch_size: {self.batch_size}")

            if self.plot_test is True:
                self.plot_data(epoch=epoch)

            if self.run_hyperparameter_tuning is True:
                ray.tune.report(loss=train_loss_epoch)

            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy().astype(np.float64), epoch+1)

        self.writer.add_hparams(
            hparam_dict={
                "learning rate": self.lr,
                "batch size": self.batch_size,
                "number hidden": self.num_hidden,
                "number blocks": self.num_blocks},
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

        if self.plot_test is True:
            lst_gif = [
                (self.path_chain_plot, f"{self.path_gifs}/chain_plot.gif"),
                (self.path_loss_plot, f"{self.path_gifs}/loss_plot.gif"),
                (self.path_mean_plot, f"{self.path_gifs}/mean_plot.gif"),
                (self.path_std_plot, f"{self.path_gifs}/std_plot.gif"),
                (self.path_color_color_plot, f"{self.path_gifs}/color_color_plot.gif"),
                (self.path_residual_plot, f"{self.path_gifs}/residual_plot.gif"),
                (self.path_chain_plot_mcal, f"{self.path_gifs}/chain_plot_mcal.gif"),
                (self.path_mean_plot_mcal, f"{self.path_gifs}/mean_plot_mcal.gif"),
                (self.path_std_plot_mcal, f"{self.path_gifs}/std_plot_mcal.gif"),
                (self.path_color_color_plot_mcal, f"{self.path_gifs}/color_color_plot_mcal.gif"),
                (self.path_residual_plot_mcal, f"{self.path_gifs}/residual_plot_mcal.gif")
            ]

            for gif in lst_gif:
                try:
                    make_gif(gif[0], gif[1])
                except:
                    pass

        if self.save_nn is True:
            torch.save(self.best_model, f"{self.path_save_nn}/best_model_des_epoch_{self.best_validation_epoch+1}_run_{self.run}.pt")
            torch.save(self.model, f"{self.path_save_nn}/last_model_des_epoch_{self.epochs}_run_{self.run}.pt")

        self.writer.flush()
        self.writer.close()

    def train(self, epoch):
        self.model.train()
        train_loss = 0.0
        pbar = tqdm(total=len(self.train_loader.dataset))
        for batch_idx, data in enumerate(self.train_loader):
            cond_data = data[1].float()
            cond_data = cond_data.to(self.device)
            data = data[0]
            data = data.to(self.device)
            self.optimizer.zero_grad()
            loss = -self.model.log_probs(data, cond_data).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1, error_if_nonfinite=False)
            self.optimizer.step()
            train_loss += loss.data.item() * data.size(0)
            pbar.update(data.size(0))
            pbar.set_description(f"Training,\t"
                                 f"Epoch: {epoch+1},\t"
                                 f"learning rate: {self.lr},\t"
                                 f"number hidden: {self.num_hidden},\t"
                                 f"number blocks: {self.num_blocks},\t"
                                 f"batch size: {self.batch_size},\t"
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
            self.model(self.train_loader.dataset.tensors[0].to(data.device),
                       self.train_loader.dataset.tensors[1].to(data.device).float())

        for module in self.model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 1
        return train_loss

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        pbar = tqdm(total=len(self.valid_loader.dataset))
        for batch_idx, data in enumerate(self.valid_loader):
            cond_data = data[1].float()
            cond_data = cond_data.to(self.device)
            data = data[0]
            data = data.to(self.device)
            with torch.no_grad():
                loss = -self.model.log_probs(data, cond_data).mean()
                self.lst_valid_loss_per_batch.append(loss.item())
                val_loss += loss.data.item() * data.size(0)
            pbar.update(data.size(0))
            pbar.set_description(f"Validation,\t"
                                 f"Epoch: {epoch + 1},\t"
                                 f"learning rate: {self.lr},\t"
                                 f"number hidden: {self.num_hidden},\t"
                                 f"number blocks: {self.num_blocks},\t"
                                 f"batch size: {self.batch_size},\t"
                                 f"loss: {val_loss / pbar.n}")
        pbar.close()
        val_loss = val_loss / len(self.valid_loader.dataset)
        self.writer.add_scalar('validation loss', val_loss, epoch+1)
        return val_loss

    def plot_data(self, epoch):
        """"""
        sns.set_theme()
        self.model.eval()

        colors = [
            ("r", "i"),
            ("i", "z")
        ]

        bands = [
            "r",
            "i",
            "z"
        ]

        cond_data = torch.Tensor(self.df_test[f"data frame test data"][self.col_label_flow].to_numpy())
        with torch.no_grad():
            tensor_output = self.model.sample(len(cond_data), cond_inputs=cond_data).detach().cpu()

        df_generated_scaled = self.df_test[f"data frame test data"].copy()
        df_true_scaled = self.df_test[f"data frame test data"].copy()

        df_generated_label = pd.DataFrame(cond_data.numpy(), columns=self.col_label_flow)
        df_generated_output = pd.DataFrame(tensor_output.cpu().numpy(), columns=self.col_output_flow)
        df_generated_output = pd.concat([df_generated_label, df_generated_output], axis=1)
        df_generated_scaled[self.col_label_flow+self.col_output_flow] = df_generated_output
        generated_rescaled = self.scaler.inverse_transform(df_generated_scaled)
        df_generated = pd.DataFrame(generated_rescaled, columns=df_generated_scaled.keys())

        true = self.scaler.inverse_transform(df_true_scaled)
        df_true = pd.DataFrame(true, columns=df_generated_scaled.keys())

        if self.apply_fill_na is True:
            for na in self.lst_fill_na:
                na_tuple = eval(na)
                df_generated[na_tuple[0]] = df_generated[na_tuple[0]].fillna(na_tuple[1])

        for b in bands:
            df_generated[f"meas {b} - true {b}"] = (df_generated[f'unsheared/{self.luminosity_type.lower()}_{b}'] - df_generated[f'BDF_{self.luminosity_type.upper()}_DERED_CALIB_{b.upper()}'])
            df_true[f"meas {b} - true {b}"] = df_true[f'unsheared/{self.luminosity_type.lower()}_{b}'] - df_true[f'BDF_{self.luminosity_type.upper()}_DERED_CALIB_{b.upper()}']

        df_true_cut = df_true.copy()
        df_true_cut = unreplace_and_untransform_data(
            data_frame=df_true_cut,
            dict_pt=self.df_test["power transformer"],
            columns=self.lst_replace_transform_cols,
            replace_value=self.lst_replace_values
        )

        if self.apply_object_cut is not True:
            df_true_cut = unsheared_object_cuts(data_frame=df_true_cut)
        if self.apply_flag_cut is not True:
            df_true_cut = flag_cuts(data_frame=df_true_cut)
        if self.apply_unsheared_mag_cut is not True:
            df_true_cut = unsheared_mag_cut(data_frame=df_true_cut)
        if self.apply_unsheared_shear_cut is not True:
            df_true_cut = unsheared_shear_cuts(data_frame=df_true_cut)
        if self.apply_airmass_cut is not True:
            df_true_cut = airmass_cut(data_frame=df_true_cut)

        df_generated_cut = df_generated.copy()
        df_generated_cut = unreplace_and_untransform_data(
            data_frame=df_generated_cut,
            dict_pt=self.df_test["power transformer"],
            columns=self.lst_replace_transform_cols,
            replace_value=self.lst_replace_values
        )

        if self.apply_object_cut is not True:
            df_generated_cut = unsheared_object_cuts(data_frame=df_generated_cut)
        if self.apply_flag_cut is not True:
            df_generated_cut = flag_cuts(data_frame=df_generated_cut)
        if self.apply_unsheared_mag_cut is not True:
            df_generated_cut = unsheared_mag_cut(data_frame=df_generated_cut)
        if self.apply_unsheared_shear_cut is not True:
            df_generated_cut = unsheared_shear_cuts(data_frame=df_generated_cut)
        if self.apply_airmass_cut is not True:
            df_generated_cut = airmass_cut(data_frame=df_generated_cut)

        if self.do_loss_plot is True:
            img_grid = loss_plot(
                epoch=epoch,
                lst_train_loss_per_batch=self.lst_train_loss_per_batch,
                lst_train_loss_per_epoch=self.lst_train_loss_per_epoch,
                lst_valid_loss_per_batch=self.lst_valid_loss_per_batch,
                lst_valid_loss_per_epoch=self.lst_valid_loss_per_epoch,
                show_plot=self.show_plot,
                save_plot=self.save_plot,
                save_name=f"{self.path_loss_plot}/loss_{epoch + 1}.png"
            )
            self.writer.add_image("loss plot", img_grid, epoch + 1)

        if self.do_color_color_plot is True:
            try:
                img_grid = color_color_plot(
                    data_frame_generated=df_generated,
                    data_frame_true=df_true,
                    luminosity_type=self.luminosity_type,
                    colors=colors,
                    show_plot=self.show_plot,
                    save_name=f'{self.path_color_color_plot}/color_color_{epoch + 1}.png',
                    extents={
                        f"unsheared/{self.luminosity_type.lower()} r-i": (-4, 4),
                        f"unsheared/{self.luminosity_type.lower()} i-z": (-4, 4)
                    }
                )
                self.writer.add_image("color color plot", img_grid, epoch+1)
            except Exception as e:
                print(f"Error {e}: {self.path_color_color_plot}/color_color_{epoch + 1}.png")

            try:
                img_grid = color_color_plot(
                    data_frame_generated=df_generated_cut,
                    data_frame_true=df_true_cut,
                    luminosity_type=self.luminosity_type,
                    colors=colors,
                    show_plot=self.show_plot,
                    save_name=f'{self.path_color_color_plot_mcal}/mcal_color_color_{epoch + 1}.png',
                    extents={
                        f"unsheared/{self.luminosity_type.lower()} r-i": (-1.2, 1.8),
                        f"unsheared/{self.luminosity_type.lower()} i-z": (-1.5, 1.5)
                    }
                )
                self.writer.add_image("color color plot mcal", img_grid, epoch+1)
            except Exception as e:
                print(f"Error {e}: {self.path_color_color_plot_mcal}/mcal_color_color_{epoch + 1}.png")

        if self.do_residual_plot:
            try:
                img_grid = residual_plot(
                    data_frame_generated=df_generated,
                    data_frame_true=df_true,
                    luminosity_type=self.luminosity_type,
                    plot_title=f"residual, epoch {epoch+1}",
                    bands=bands,
                    show_plot=self.show_plot,
                    save_plot=self.save_plot,
                    save_name=f"{self.path_residual_plot}/residual_plot_{epoch + 1}.png"
                )
                self.writer.add_image("residual plot", img_grid, epoch + 1)
            except Exception as e:
                print(f"Error {e}: {self.path_residual_plot}/residual_plot_{epoch + 1}.png")

            try:
                img_grid = residual_plot(
                    data_frame_generated=df_generated_cut,
                    data_frame_true=df_true_cut,
                    luminosity_type=self.luminosity_type,
                    plot_title=f"mcal residual, epoch {epoch+1}",
                    bands=bands,
                    show_plot=self.show_plot,
                    save_plot=self.save_plot,
                    save_name=f"{self.path_residual_plot_mcal}/mcal_residual_plot_{epoch + 1}.png"
                )
                self.writer.add_image("residual plot mcal", img_grid, epoch + 1)
            except Exception as e:
                print(f"Error {e}: {self.path_residual_plot_mcal}/mcal_residual_plot_{epoch + 1}.png")

        if self.do_chain_plot is True:
            try:
                img_grid = plot_chain_compare(
                    data_frame_generated=df_generated,
                    data_frame_true=df_true,
                    epoch=epoch,
                    show_plot=self.show_plot,
                    save_name=f'{self.path_chain_plot}/chainplot_{epoch + 1}.png',
                    columns=[
                        f"unsheared/{self.luminosity_type.lower()}_r",
                        f"unsheared/{self.luminosity_type.lower()}_i",
                        f"unsheared/{self.luminosity_type.lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T"
                    ],
                    parameter=[
                        f"{self.luminosity_type.lower()}_r",
                        f"{self.luminosity_type.lower()}_i",
                        f"{self.luminosity_type.lower()}_z",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    extends={
                        f"{self.luminosity_type.lower()}_r": (15, 30),
                        f"{self.luminosity_type.lower()}_i": (15, 30),
                        f"{self.luminosity_type.lower()}_z": (15, 30),
                        "snr": (-2, 4),
                        "size_ratio": (-3.5, 4),
                        "T": (-1.5, 2)
                    },
                    max_ticks=5,
                    shade_alpha=0.8,
                    tick_font_size=12,
                    label_font_size=12
                )
                self.writer.add_image("chain plot", img_grid, epoch + 1)
            except Exception as e:
                print(f"Error {e}: {self.path_chain_plot}/chainplot_{epoch + 1}.png")

            try:
                img_grid = plot_chain_compare(
                    data_frame_generated=df_generated_cut,
                    data_frame_true=df_true_cut,
                    epoch=epoch,
                    show_plot=self.show_plot,
                    save_name=f'{self.path_chain_plot_mcal}/mcal_chainplot_{epoch + 1}.png',
                    columns=[
                        f"unsheared/{self.luminosity_type.lower()}_r",
                        f"unsheared/{self.luminosity_type.lower()}_i",
                        f"unsheared/{self.luminosity_type.lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T",
                    ],
                    parameter=[
                        f"{self.luminosity_type.lower()}_r",
                        f"{self.luminosity_type.lower()}_i",
                        f"{self.luminosity_type.lower()}_z",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    extends={
                        f"{self.luminosity_type.lower()}_r": (15, 30),
                        f"{self.luminosity_type.lower()}_i": (15, 30),
                        f"{self.luminosity_type.lower()}_z": (15, 30),
                        "snr": (-75, 425),
                        "size_ratio": (-1.5, 6),
                        "T": (-1, 4)
                    },
                    max_ticks=5,
                    shade_alpha=0.8,
                    tick_font_size=12,
                    label_font_size=12
                )
                self.writer.add_image("chain plot mcal", img_grid, epoch + 1)
            except Exception as e:
                print(f"Error {e}: {self.path_chain_plot_mcal}/mcal_chainplot_{epoch + 1}.png")

            try:
                img_grid = plot_chain_compare(
                    data_frame_generated=df_generated,
                    data_frame_true=df_true,
                    epoch=epoch,
                    show_plot=self.show_plot,
                    save_name=f"{self.path_color_diff_plot}/color_diff_{epoch + 1}.png",
                    columns=[
                        f"BDF_{self.luminosity_type.upper()}_DERED_CALIB_R",
                        f"BDF_{self.luminosity_type.upper()}_DERED_CALIB_I",
                        f"BDF_{self.luminosity_type.upper()}_DERED_CALIB_Z",
                        "meas r - true r",
                        "meas i - true i",
                        "meas z - true z"
                    ],
                    parameter=[
                        "true r",
                        "true i",
                        "true z",
                        "meas r - true r",
                        "meas i - true i",
                        "meas z - true z"
                    ],
                    extends={
                        "true r": (18, 30),
                        "true i": (18, 30),
                        "true z": (18, 30),
                        "meas r - true r": (-4, 4),
                        "meas i - true i": (-4, 4),
                        "meas z - true z": (-4, 4),
                    },
                    max_ticks=5,
                    shade_alpha=0.8,
                    tick_font_size=12,
                    label_font_size=12
                )
                self.writer.add_image("color diff plot", img_grid, epoch + 1)
            except Exception as e:
                print(f"Error {e}: {self.path_color_diff_plot}/color_diff_{epoch + 1}.png")

            try:
                img_grid = plot_chain_compare(
                    data_frame_generated=df_generated_cut,
                    data_frame_true=df_true_cut,
                    epoch=epoch,
                    show_plot=self.show_plot,
                    save_name=f"{self.path_color_diff_plot_mcal}/mcal_color_diff_{epoch + 1}.png",
                    columns=[
                        f"BDF_{self.luminosity_type.upper()}_DERED_CALIB_R",
                        f"BDF_{self.luminosity_type.upper()}_DERED_CALIB_I",
                        f"BDF_{self.luminosity_type.upper()}_DERED_CALIB_Z",
                        "meas r - true r",
                        "meas i - true i",
                        "meas z - true z"
                    ],
                    parameter=[
                        "true r",
                        "true i",
                        "true z",
                        "meas r - true r",
                        "meas i - true i",
                        "meas z - true z"
                    ],
                    extends={
                        "true r": (18, 30),
                        "true i": (18, 30),
                        "true z": (18, 30),
                        "meas r - true r": (-1.5, 1.5),
                        "meas i - true i": (-1.5, 1.5),
                        "meas z - true z": (-1.5, 1.5),
                    },
                    max_ticks=5,
                    shade_alpha=0.8,
                    tick_font_size=12,
                    label_font_size=12
                )
                self.writer.add_image("color diff plot mcal", img_grid, epoch + 1)
            except Exception as e:
                print(f"Error {e}: {self.path_color_diff_plot_mcal}/mcal_color_diff_{epoch + 1}.png")

        if self.do_mean_plot is True:
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
                    data_frame_generated=df_generated,
                    data_frame_true=df_true,
                    lists_to_plot=lists_mean_to_plot,
                    list_epochs=self.lst_epochs,

                    columns=[
                        f"unsheared/{self.luminosity_type.lower()}_r",
                        f"unsheared/{self.luminosity_type.lower()}_i",
                        f"unsheared/{self.luminosity_type.lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T"
                    ],
                    lst_labels=[
                        f"{self.luminosity_type.lower()}_r",
                        f"{self.luminosity_type.lower()}_i",
                        f"{self.luminosity_type.lower()}_z",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    lst_marker=["o", "^", "X", "d", "s", "*"],
                    lst_color=["blue", "red", "green", "orange", "purple", "black"],
                    plot_title="mean ratio",
                    show_plot=self.show_plot,
                    save_plot=self.save_plot,
                    save_name=f"{self.path_mean_plot}/mean_{epoch+1}.png",
                    statistic_type="mean"
                )
                self.writer.add_image("mean plot", img_grid, epoch + 1)
                for idx_plot, lst_plot in enumerate(lists_mean_to_plot):
                    lst_plot = lists_mean_to_plot_updated[idx_plot]

            except Exception as e:
                print(f"Error {e}: {self.path_mean_plot}/mean_{epoch+1}.png")
                print(f"Mean shapes: \t epoch \t {self.luminosity_type.lower()} r \t {self.luminosity_type.lower()} i \t {self.luminosity_type.lower()} z \t snr \t size_ratio \t T")
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
                    data_frame_generated=df_generated_cut,
                    data_frame_true=df_true_cut,
                    lists_to_plot=lists_mean_to_plot_cut,
                    list_epochs=self.lst_epochs,
                    columns=[
                        f"unsheared/{self.luminosity_type.lower()}_r",
                        f"unsheared/{self.luminosity_type.lower()}_i",
                        f"unsheared/{self.luminosity_type.lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T"
                    ],
                    lst_labels=[
                        f"{self.luminosity_type.lower()}_r",
                        f"{self.luminosity_type.lower()}_i",
                        f"{self.luminosity_type.lower()}_z",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    lst_marker=["o", "^", "X", "d", "s", "*"],
                    lst_color=["blue", "red", "green", "orange", "purple", "black"],
                    plot_title="mcal mean ratio",
                    show_plot=self.show_plot,
                    save_plot=self.save_plot,
                    save_name=f"{self.path_mean_plot_mcal}/mcal_mean_{epoch + 1}.png",
                    statistic_type="mean"
                )
                self.writer.add_image("mean plot mcal", img_grid, epoch + 1)
                for idx_plot_cut, lst_plot_cut in enumerate(lists_mean_to_plot_cut):
                    lst_plot_cut = lists_mean_to_plot_cut_updated[idx_plot_cut]

            except Exception as e:
                print(f"Error {e}: {self.path_mean_plot_mcal}/mcal_mean_{epoch + 1}.png")
                print(f"Mean mcal shapes: \t epoch \t {self.luminosity_type.lower()} r \t {self.luminosity_type.lower()} i \t {self.luminosity_type.lower()} z \t snr \t size_ratio \t T")
                print(f"\t           \t {len(self.lst_epochs)} \t {len(self.lst_mean_mag_r_cut)} \t {len(self.lst_mean_mag_i_cut)} \t {len(self.lst_mean_mag_z_cut)} \t {len(self.lst_mean_snr_cut)} \t {len(self.lst_mean_size_ratio_cut)} \t {len(self.lst_mean_t_cut)}")

        if self.do_std_plot is True:
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
                    data_frame_generated=df_generated,
                    data_frame_true=df_true,
                    lists_to_plot=lists_std_to_plot,
                    list_epochs=self.lst_epochs,
                    columns=[
                        f"unsheared/{self.luminosity_type.lower()}_r",
                        f"unsheared/{self.luminosity_type.lower()}_i",
                        f"unsheared/{self.luminosity_type.lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T"
                    ],
                    lst_labels=[
                        f"{self.luminosity_type.lower()}_r",
                        f"{self.luminosity_type.lower()}_i",
                        f"{self.luminosity_type.lower()}_z",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    lst_marker=["o", "^", "X", "d", "s", "*"],
                    lst_color=["blue", "red", "green", "orange", "purple", "black"],
                    plot_title="std ratio",
                    show_plot=self.show_plot,
                    save_plot=self.save_plot,
                    save_name=f"{self.path_std_plot}/std_{epoch + 1}.png",
                    statistic_type="std"
                )
                self.writer.add_image("std plot", img_grid, epoch + 1)
                for idx_plot, lst_plot in enumerate(lists_std_to_plot):
                    lst_plot = lists_std_to_plot_updated[idx_plot]

            except Exception as e:
                print(f"Error {e}: {self.path_std_plot}/std_{epoch + 1}.png")
                print(f"Std shapes: \t epoch \t {self.luminosity_type.lower()} r \t {self.luminosity_type.lower()} i \t {self.luminosity_type.lower()} z \t snr \t size_ratio \t T")
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
                    data_frame_generated=df_generated_cut,
                    data_frame_true=df_true_cut,
                    lists_to_plot=lists_std_to_plot_cut,
                    list_epochs=self.lst_epochs,
                    columns=[
                        f"unsheared/{self.luminosity_type.lower()}_r",
                        f"unsheared/{self.luminosity_type.lower()}_i",
                        f"unsheared/{self.luminosity_type.lower()}_z",
                        "unsheared/snr",
                        "unsheared/size_ratio",
                        "unsheared/T"
                    ],
                    lst_labels=[
                        f"{self.luminosity_type.lower()}_r",
                        f"{self.luminosity_type.lower()}_i",
                        f"{self.luminosity_type.lower()}_z",
                        "snr",
                        "size_ratio",
                        "T",
                    ],
                    lst_marker=["o", "^", "X", "d", "s", "*"],
                    lst_color=["blue", "red", "green", "orange", "purple", "black"],
                    plot_title="mcal std ratio",
                    show_plot=self.show_plot,
                    save_plot=self.save_plot,
                    save_name=f"{self.path_std_plot_mcal}/mcal_std_{epoch + 1}.png",
                    statistic_type="std"
                )
                self.writer.add_image("std plot mcal", img_grid, epoch + 1)
                for idx_plot_cut, lst_plot_cut in enumerate(lists_std_to_plot_cut):
                    lst_plot_cut = lists_std_to_plot_cut_updated[idx_plot_cut]

            except Exception as e:
                print(f"Error {e}: {self.path_std_plot_mcal}/mcal_std_{epoch + 1}.png")
                print(f"Std mcal shapes: \t epoch \t {self.luminosity_type.lower()} r \t {self.luminosity_type.lower()} i \t {self.luminosity_type.lower()} z \t snr \t size_ratio \t T")
                print(f"\t           \t {len(self.lst_epochs)} \t {len(self.lst_std_mag_r_cut)} \t {len(self.lst_std_mag_i_cut)} \t {len(self.lst_std_mag_z_cut)} \t {len(self.lst_std_snr_cut)} \t {len(self.lst_std_size_ratio_cut)} \t {len(self.lst_std_t_cut)}")

