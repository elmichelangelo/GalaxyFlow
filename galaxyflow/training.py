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
from tqdm import tqdm
from tensorboardX import SummaryWriter
import galaxyflow.flow as fnn
import pandas as pd
from Handler.data_loader import load_data
from chainconsumer import ChainConsumer
from Handler.helper_functions import unreplace_and_untransform_data
from Handler.cut_functions import unsheared_mag_cut, unsheared_object_cuts, flag_cuts, airmass_cut, unsheared_shear_cuts
from Handler.plot_functions import make_gif, loss_plot, color_color_plot, residual_plot, plot_chain_compare,\
    plot_mean_or_std
from scipy.stats import binned_statistic, median_abs_deviation
import seaborn as sns


class TrainFlow(object):

    def __init__(self,
                 path_train_data,
                 size_training_dataset,
                 size_validation_dataset,
                 size_test_dataset,
                 path_output,
                 col_label_flow,
                 col_output_flow,
                 plot_test,
                 show_plot,
                 save_plot,
                 save_nn,
                 learning_rate,
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
                 run,
                 reproducible):
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
        self.device = torch.device(device)
        self.act = activation_function
        self.actal_selected_values = None
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = batch_size
        self.lst_replace_transform_cols = lst_replace_transform_cols
        self.lst_replace_values = lst_replace_values
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

        # if run_hyperparameter_tuning is not True:
        self.lr = learning_rate
        self.num_hidden = number_hidden
        self.num_blocks = number_blocks
        self.path_output_flow = f"{path_output}/Flow_" \
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
        self.writer = SummaryWriter(log_dir=f"{self.path_output_flow}/writer", comment=f"_lr_{self.lr}")
        self.train_loader, self.valid_loader, self.df_test, self.scaler = self.init_dataset(
            path_train_data=path_train_data,
            selected_scaler=selected_scaler,
            reproducible=reproducible
        )

        self.model, self.optimizer = self.init_network(
            num_inputs=len(col_output_flow),
            num_cond_inputs=len(col_label_flow)
        )

        self.global_step = 0
        self.best_validation_loss = float('inf')
        self.best_validation_epoch = 0
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

    def init_dataset(self, path_train_data, selected_scaler, reproducible):
        """"""
        training_data, validation_data, test_data = load_data(
            path_training_data=path_train_data,
            path_output=self.path_output,
            selected_scaler=selected_scaler,
            size_training_dataset=self.size_training_dataset,
            size_validation_dataset=self.size_validation_dataset,
            size_test_dataset=self.size_test_dataset,
            reproducible=reproducible,
            run=self.run,
            lst_replace_transform_cols=self.lst_replace_transform_cols,
            lst_replace_values=self.lst_replace_values,
            apply_cuts=True
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
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-6)

        return model, optimizer

    def hyperparameter_tuning(self, learning_rate, number_hidden, number_blocks, batch_size, reproducible):
        # , number_hidden, number_blocks
        self.lr = learning_rate,
        self.num_hidden = number_hidden
        self.num_blocks = number_blocks
        self.batch_size = batch_size
        self.train_loader, self.valid_loader, self.test_loader, self.scaler = self.init_dataset(
            path_train_data=self.path_train_data,
            selected_scaler=self.selected_scaler,
            reproducible=reproducible
        )

        self.model, self.optimizer = self.init_network(
            num_inputs=len(self.col_output_flow),
            num_cond_inputs=len(self.col_label_flow)
        )

        self.global_step = 0
        self.best_validation_loss = float('inf')
        self.best_validation_epoch = 0
        self.best_model = self.model
        self.path_output_flow = f"{self.path_output}/Flow_hyper_" \
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
        self.writer = SummaryWriter(log_dir=f"{self.path_output_flow}/writer", comment=f"_lr_{self.lr}")
        self.make_dirs()
        self.run_training()

    def run_training(self):
        for epoch in range(self.epochs):
            print('\nEpoch: {}'.format(epoch + 1))

            train_loss, train_loss_epoch = self.train()
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

            if epoch - self.best_validation_epoch >= 30:
                break

            print(f"Best validation at epoch {self.best_validation_epoch + 1}"
                  f"\t Average Log Likelihood {-self.best_validation_loss}")

            if self.plot_test is True:
                # try:
                self.plot_data(epoch=epoch)
                # except:
                #     print(f"Error epoch {epoch+1}")

        if self.plot_test is True:
            make_gif(self.path_chain_plot, f"{self.path_gifs}/chain_plot.gif")
            make_gif(self.path_loss_plot, f"{self.path_gifs}/loss_plot.gif")
            make_gif(self.path_mean_plot, f"{self.path_gifs}/mean_plot.gif")
            make_gif(self.path_std_plot, f"{self.path_gifs}/std_plot.gif")
            make_gif(self.path_color_color_plot, f"{self.path_gifs}/color_color_plot.gif")
            make_gif(self.path_residual_plot, f"{self.path_gifs}/residual_plot.gif")
            make_gif(self.path_chain_plot_mcal, f"{self.path_gifs}/chain_plot_mcal.gif")
            make_gif(self.path_mean_plot_mcal, f"{self.path_gifs}/mean_plot_mcal.gif")
            make_gif(self.path_std_plot_mcal, f"{self.path_gifs}/std_plot_mcal.gif")
            make_gif(self.path_color_color_plot_mcal, f"{self.path_gifs}/color_color_plot_mcal.gif")
            make_gif(self.path_residual_plot_mcal, f"{self.path_gifs}/residual_plot_mcal.gif")
        if self.save_nn is True:
            torch.save(self.best_model, f"{self.path_save_nn}/best_model_des_epoch_{self.best_validation_epoch+1}_run_{self.run}.pt")
            torch.save(self.model, f"{self.path_save_nn}/last_model_des_epoch_{self.epochs}_run_{self.run}.pt")
        self.writer.close()

    def train(self):
        self.model.train()
        train_loss = 0
        pbar = tqdm(total=len(self.train_loader.dataset))
        for batch_idx, data in enumerate(self.train_loader):
            cond_data = data[1].float()
            cond_data = cond_data.to(self.device)
            data = data[0]
            data = data.to(self.device)
            self.optimizer.zero_grad()
            loss = -self.model.log_probs(data, cond_data).mean()
            if loss.isnan():
                print(loss)
            self.lst_train_loss_per_batch.append(loss.item())
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            pbar.update(data.size(0))
            pbar.set_description(f'Train, Log likelihood: {train_loss / (batch_idx + 1)}')
            self.writer.add_scalar('training/loss', loss.item(), self.global_step)
            self.global_step += 1
        pbar.close()

        for module in self.model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 0

        with torch.no_grad():
            self.model(self.train_loader.dataset.tensors[0].to(data.device),
                  self.train_loader.dataset.tensors[1].to(data.device).float())

        for module in self.model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                module.momentum = 1
        return train_loss, train_loss / (batch_idx + 1)

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0

        pbar = tqdm(total=len(self.valid_loader.dataset))
        pbar.set_description('Eval')
        for batch_idx, data in enumerate(self.valid_loader):
            cond_data = data[1].float()
            cond_data = cond_data.to(self.device)
            data = data[0]
            data = data.to(self.device)
            with torch.no_grad():
                val_loss += -self.model.log_probs(data, cond_data).sum().item()
            pbar.update(data.size(0))
            pbar.set_description(f'Val, Log likelihood in nats: {val_loss / pbar.n}')
            self.lst_valid_loss_per_batch.append(val_loss / pbar.n)

        self.writer.add_scalar('validation/LL', val_loss / len(self.valid_loader.dataset), epoch)

        pbar.close()
        return val_loss / len(self.valid_loader.dataset)

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

        # lst_replace_transform_cols = [
        #     "unsheared/T",
        #     "unsheared/snr",
        #     "unsheared/size_ratio",
        #     "AIRMASS_WMEAN_R",
        #     "AIRMASS_WMEAN_I",
        #     "AIRMASS_WMEAN_Z",
        #     "FWHM_WMEAN_R",
        #     "FWHM_WMEAN_I",
        #     "FWHM_WMEAN_Z",
        #     "MAGLIM_R",
        #     "MAGLIM_I",
        #     "MAGLIM_Z",
        #     "EBV_SFD98"
        # ]
        # lst_replace_values = [
        #     (-9999, -1),
        #     (-7070.360705084288, 0),
        #     None,
        #     (-9999, 0),
        #     (-9999, 0),
        #     (-9999, 0),
        #     (-9999, 0),
        #     (-9999, 0),
        #     (-9999, 0),
        #     (-9999, 0),
        #     (-9999, 0),
        #     (-9999, 0),
        #     None
        # ]

        lst_fill_na = [
            ("unsheared/snr", -7070.360705084288),
            ("unsheared/T", -9999),
            ("unsheared/size_ratio", -1),
            ("unsheared/flags", -1),
            ("unsheared/flux_r", -99999),
            ("unsheared/flux_i", -99999),
            ("unsheared/flux_z", -99999),
            ("unsheared/flux_err_r", -99999),
            ("unsheared/flux_err_i", -99999),
            ("unsheared/flux_err_z", -99999),
            ("AIRMASS_WMEAN_R", -9999),
            ("AIRMASS_WMEAN_I", -9999),
            ("AIRMASS_WMEAN_Z", -9999),
            ("FWHM_WMEAN_R", -9999),
            ("FWHM_WMEAN_I", -9999),
            ("FWHM_WMEAN_Z", -9999),
            ("MAGLIM_R", -9999),
            ("MAGLIM_I", -9999),
            ("MAGLIM_Z", -9999),
            ("EBV_SFD98", -9999),
        ]

        # for batch_idx, data in enumerate(self.df_test):
        cond_data = torch.Tensor(self.df_test[f"data frame test data"][self.col_label_flow].to_numpy())
        cond_data = cond_data.to(self.device)
        with torch.no_grad():
            tensor_output = self.model.sample(len(self.df_test[f"data frame test data"]),
                                              cond_inputs=cond_data).detach().cpu()

        df_generated_scaled = self.df_test[f"data frame test data"].copy()
        df_true_scaled = self.df_test[f"data frame test data"].copy()

        df_generated_label = pd.DataFrame(cond_data.numpy(), columns=self.col_label_flow)
        df_generated_output = pd.DataFrame(tensor_output.numpy(), columns=self.col_output_flow)
        df_generated_concat = pd.concat([df_generated_label, df_generated_output], axis=1)
        df_generated_scaled.update(df_generated_concat)

        test_generated_rescaled = self.scaler.inverse_transform(df_generated_scaled)
        df_generated = pd.DataFrame(test_generated_rescaled, columns=df_generated_scaled.keys())

        true = self.scaler.inverse_transform(df_true_scaled)
        df_true = pd.DataFrame(true, columns=df_generated_scaled.keys())

        for b in bands:
            df_generated[f"meas {b} - true {b}"] = df_generated[f'unsheared/lupt_{b}'] - df_generated[f'BDF_LUPT_DERED_CALIB_{b.upper()}']
            df_true[f"meas {b} - true {b}"] = df_true[f'unsheared/lupt_{b}'] - df_true[f'BDF_LUPT_DERED_CALIB_{b.upper()}']

        print("true na", df_generated.isna().sum().sum())

        df_true = unreplace_and_untransform_data(
            data_frame=df_true,
            dict_pt=self.df_test["power transformer"],
            columns=self.lst_replace_transform_cols,
            replace_value=self.lst_replace_values
        )

        print("true na", df_generated.isna().sum().sum())

        # df_true_cut = unsheared_object_cuts(data_frame=df_true)
        # df_true_cut = flag_cuts(data_frame=df_true_cut)
        df_true_cut = airmass_cut(data_frame=df_true)
        df_true_cut = unsheared_mag_cut(data_frame=df_true_cut)
        df_true_cut = unsheared_shear_cuts(data_frame=df_true_cut)

        print("generated na", df_generated.isna().sum().sum())

        df_generated = unreplace_and_untransform_data(
            data_frame=df_generated,
            dict_pt=self.df_test["power transformer"],
            columns=self.lst_replace_transform_cols,
            replace_value=self.lst_replace_values
        )

        print("generated na", df_generated.isna().sum().sum())

        # df_generated_cut = unsheared_object_cuts(data_frame=df_generated)
        # df_generated_cut = flag_cuts(data_frame=df_generated_cut)
        df_generated_cut = airmass_cut(data_frame=df_generated)
        df_generated_cut = unsheared_mag_cut(data_frame=df_generated_cut)
        df_generated_cut = unsheared_shear_cuts(data_frame=df_generated_cut)

        for na in lst_fill_na:
            df_generated[na[0]] = df_generated[na[0]].fillna(na[1])

        if self.do_loss_plot is True:
            loss_plot(
                epoch=epoch,
                lst_train_loss_per_batch=self.lst_train_loss_per_batch,
                lst_train_loss_per_epoch=self.lst_train_loss_per_epoch,
                lst_valid_loss_per_batch=self.lst_valid_loss_per_batch,
                lst_valid_loss_per_epoch=self.lst_valid_loss_per_epoch,
                show_plot=self.show_plot,
                save_plot=self.save_plot,
                save_name=f"{self.path_loss_plot}/loss_{epoch + 1}.png"
            )

        if self.do_color_color_plot is True:
            color_color_plot(
                data_frame_generated=df_generated,
                data_frame_true=df_true,
                colors=colors,
                show_plot=self.show_plot,
                save_name=f'{self.path_color_color_plot}/color_color_{epoch + 1}.png',
                extents={
                    "unsheared/lupt r-i": (-6, 6),
                    "unsheared/lupt i-z": (-25, 25)
                }
            )
            color_color_plot(
                data_frame_generated=df_generated_cut,
                data_frame_true=df_true_cut,
                colors=colors,
                show_plot=self.show_plot,
                save_name=f'{self.path_color_color_plot_mcal}/mcal_color_color_{epoch + 1}.png',
                extents={
                    "unsheared/lupt r-i": (-6, 6),
                    "unsheared/lupt i-z": (-25, 25)
                }
            )

        if self.do_residual_plot:
            residual_plot(
                data_frame_generated=df_generated,
                data_frame_true=df_true,
                plot_title=f"residual, epoch {epoch+1}",
                bands=bands,
                show_plot=self.show_plot,
                save_plot=self.save_plot,
                save_name=f"{self.path_residual_plot}/residual_plot_{epoch + 1}.png"
            )

            residual_plot(
                data_frame_generated=df_generated_cut,
                data_frame_true=df_true_cut,
                plot_title=f"mcal residual, epoch {epoch+1}",
                bands=bands,
                show_plot=self.show_plot,
                save_plot=self.save_plot,
                save_name=f"{self.path_residual_plot_mcal}/mcal_residual_plot_{epoch + 1}.png"
            )

        if self.do_chain_plot is True:
            plot_chain_compare(
                data_frame_generated=df_generated,
                data_frame_true=df_true,
                epoch=epoch,
                show_plot=self.show_plot,
                save_name=f'{self.path_chain_plot}/chainplot_{epoch + 1}.png',
                columns=[
                    "unsheared/lupt_r",
                    "unsheared/lupt_i",
                    "unsheared/lupt_z",
                    "unsheared/snr",
                    "unsheared/size_ratio",
                    "unsheared/T"
                ],
                parameter=[
                    "lupt_r",
                    "lupt_i",
                    "lupt_z",
                    "snr",
                    "size_ratio",
                    "T",
                ],
                extends=None,
                max_ticks=5,
                shade_alpha=0.8,
                tick_font_size=12,
                label_font_size=12
            )

            plot_chain_compare(
                data_frame_generated=df_generated_cut,
                data_frame_true=df_true_cut,
                epoch=epoch,
                show_plot=self.show_plot,
                save_name=f'{self.path_chain_plot_mcal}/mcal_chainplot_{epoch + 1}.png',
                columns=[
                    "unsheared/lupt_r",
                    "unsheared/lupt_i",
                    "unsheared/lupt_z",
                    "unsheared/snr",
                    "unsheared/size_ratio",
                    "unsheared/T",
                ],
                parameter=[
                    "lupt_r",
                    "lupt_i",
                    "lupt_z",
                    "snr",
                    "size_ratio",
                    "T",
                ],
                max_ticks=5,
                shade_alpha=0.8,
                tick_font_size=12,
                label_font_size=12
            )

            plot_chain_compare(
                data_frame_generated=df_generated,
                data_frame_true=df_true,
                epoch=epoch,
                show_plot=self.show_plot,
                save_name=f"{self.path_color_diff_plot}/color_diff_{epoch + 1}.png",
                columns=[
                    "BDF_LUPT_DERED_CALIB_R",
                    "BDF_LUPT_DERED_CALIB_I",
                    "BDF_LUPT_DERED_CALIB_Z",
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
                max_ticks=5,
                shade_alpha=0.8,
                tick_font_size=12,
                label_font_size=12
            )

            plot_chain_compare(
                data_frame_generated=df_generated_cut,
                data_frame_true=df_true_cut,
                epoch=epoch,
                show_plot=self.show_plot,
                save_name=f"{self.path_color_diff_plot_mcal}/mcal_color_diff_{epoch + 1}.png",
                columns=[
                    "BDF_LUPT_DERED_CALIB_R",
                    "BDF_LUPT_DERED_CALIB_I",
                    "BDF_LUPT_DERED_CALIB_Z",
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
                max_ticks=5,
                shade_alpha=0.8,
                tick_font_size=12,
                label_font_size=12
            )

        if self.do_mean_plot is True:
            lists_mean_to_plot = [
                self.lst_mean_mag_r,
                self.lst_mean_mag_i,
                self.lst_mean_mag_z,
                self.lst_mean_snr,
                self.lst_mean_size_ratio,
                self.lst_mean_t,
            ]

            lists_mean_to_plot_updated = plot_mean_or_std(
                data_frame_generated=df_generated,
                data_frame_true=df_true,
                lists_to_plot=lists_mean_to_plot,
                list_epochs=self.lst_epochs,
                columns=[
                    "unsheared/lupt_r",
                    "unsheared/lupt_i",
                    "unsheared/lupt_z",
                    "unsheared/snr",
                    "unsheared/size_ratio",
                    "unsheared/T"
                ],
                lst_labels=[
                    "lupt_r",
                    "lupt_i",
                    "lupt_z",
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

            for idx_plot, lst_plot in enumerate(lists_mean_to_plot):
                lst_plot = lists_mean_to_plot_updated[idx_plot]

            lists_mean_to_plot_cut = [
                self.lst_mean_mag_r_cut,
                self.lst_mean_mag_i_cut,
                self.lst_mean_mag_z_cut,
                self.lst_mean_snr_cut,
                self.lst_mean_size_ratio_cut,
                self.lst_mean_t_cut,
            ]

            lists_mean_to_plot_cut_updated = plot_mean_or_std(
                data_frame_generated=df_generated_cut,
                data_frame_true=df_true_cut,
                lists_to_plot=lists_mean_to_plot_cut,
                list_epochs=self.lst_epochs,
                columns=[
                    "unsheared/lupt_r",
                    "unsheared/lupt_i",
                    "unsheared/lupt_z",
                    "unsheared/snr",
                    "unsheared/size_ratio",
                    "unsheared/T"
                ],
                lst_labels=[
                    "lupt_r",
                    "lupt_i",
                    "lupt_z",
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

            for idx_plot_cut, lst_plot_cut in enumerate(lists_mean_to_plot_cut):
                lst_plot_cut = lists_mean_to_plot_cut_updated[idx_plot_cut]

        if self.do_std_plot is True:
            lists_std_to_plot = [
                self.lst_std_mag_r,
                self.lst_std_mag_i,
                self.lst_std_mag_z,
                self.lst_std_snr,
                self.lst_std_size_ratio,
                self.lst_std_t,
            ]

            lists_std_to_plot_updated = plot_mean_or_std(
                data_frame_generated=df_generated,
                data_frame_true=df_true,
                lists_to_plot=lists_std_to_plot,
                list_epochs=self.lst_epochs,
                columns=[
                    "unsheared/lupt_r",
                    "unsheared/lupt_i",
                    "unsheared/lupt_z",
                    "unsheared/snr",
                    "unsheared/size_ratio",
                    "unsheared/T"
                ],
                lst_labels=[
                    "lupt_r",
                    "lupt_i",
                    "lupt_z",
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

            for idx_plot, lst_plot in enumerate(lists_std_to_plot):
                lst_plot = lists_std_to_plot_updated[idx_plot]

            lists_std_to_plot_cut = [
                self.lst_std_mag_r_cut,
                self.lst_std_mag_i_cut,
                self.lst_std_mag_z_cut,
                self.lst_std_snr_cut,
                self.lst_std_size_ratio_cut,
                self.lst_std_t_cut,
            ]

            lists_std_to_plot_cut_updated = plot_mean_or_std(
                data_frame_generated=df_generated_cut,
                data_frame_true=df_true_cut,
                lists_to_plot=lists_std_to_plot_cut,
                list_epochs=self.lst_epochs,
                columns=[
                    "unsheared/lupt_r",
                    "unsheared/lupt_i",
                    "unsheared/lupt_z",
                    "unsheared/snr",
                    "unsheared/size_ratio",
                    "unsheared/T"
                ],
                lst_labels=[
                    "lupt_r",
                    "lupt_i",
                    "lupt_z",
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

            for idx_plot_cut, lst_plot_cut in enumerate(lists_std_to_plot_cut):
                lst_plot_cut = lists_std_to_plot_cut_updated[idx_plot_cut]

