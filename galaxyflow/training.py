import copy
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
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
from Handler.helper_functions import make_gif
from scipy.stats import binned_statistic, median_abs_deviation
import seaborn as sns


class TrainFlow(object):

    def __init__(self,
                 path_train_data,
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
                 plot_loss,
                 plot_color_color,
                 plot_residual,
                 plot_chain,
                 plot_mean,
                 plot_std,
                 plot_flags,
                 plot_detected):
        super().__init__()
        self.plot_loss = plot_loss
        self.plot_color_color = plot_color_color
        self.plot_residual = plot_residual
        self.plot_chain = plot_chain
        self.plot_mean = plot_mean
        self.plot_std = plot_std
        self.plot_flags = plot_flags
        self.plot_detected = plot_detected
        self.lr = learning_rate
        self.num_hidden = number_hidden
        self.num_blocks = number_blocks
        self.epochs = epochs
        self.plot_test = plot_test
        self.show_plot = show_plot
        self.save_plot = save_plot
        self.save_nn = save_nn
        self.device = torch.device(device)
        self.act = activation_function
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = batch_size
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
        self.lst_train_loss_per_batch = []
        self.lst_train_loss_per_epoch = []
        self.lst_valid_loss_per_batch = []
        self.lst_valid_loss_per_epoch = []
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
        self.path_flag_plot = f"{self.path_plots}/flag_plot"
        self.path_color_diff_plot = f"{self.path_plots}/color_diff_plot"
        self.path_color_color_plot = f"{self.path_plots}/color_color_plot"
        self.path_detection_plot = f"{self.path_plots}/detection_plot"
        self.path_gifs = f"{self.path_plots}/gifs"
        self.path_save_nn = f"{self.path_output_flow}/nn"
        self.make_dirs()

        self.writer = SummaryWriter(log_dir=f"{self.path_output_flow}/writer", comment=f"_lr_{self.lr}")
        self.col_label_flow = col_label_flow
        self.col_output_flow = col_output_flow

        self.train_loader, self.valid_loader, self.test_loader, self.scaler = self.init_dataset(
            path_train_data=path_train_data,
            selected_scaler=selected_scaler
        )

        self.model, self.optimizer  = self.init_network(
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
            if not os.path.exists(self.path_flag_plot):
                os.mkdir(self.path_flag_plot)
            if not os.path.exists(self.path_detection_plot):
                os.mkdir(self.path_detection_plot)
            if not os.path.exists(self.path_gifs):
                os.mkdir(self.path_gifs)
            if not os.path.exists(self.path_color_diff_plot):
                os.mkdir(self.path_color_diff_plot)
            if not os.path.exists(self.path_color_color_plot):
                os.mkdir(self.path_color_color_plot)
            if not os.path.exists(self.path_residual_plot):
                os.mkdir(self.path_residual_plot)

        if self.save_nn is True:
            if not os.path.exists(self.path_save_nn):
                os.mkdir(self.path_save_nn)

    def init_dataset(self, path_train_data, selected_scaler):
        """"""
        training_data, validation_data, test_data = load_data(
            path_training_data=path_train_data,
            input_flow=self.col_label_flow,
            output_flow=self.col_output_flow,
            selected_scaler=selected_scaler
        )

        train_tensor = torch.from_numpy(training_data[f"output flow in order {self.col_output_flow}"])
        train_labels = torch.from_numpy(training_data[f"label flow in order {self.col_label_flow}"])
        train_dataset = torch.utils.data.TensorDataset(train_tensor, train_labels)

        valid_tensors = torch.from_numpy(validation_data[f"output flow in order {self.col_output_flow}"])
        valid_labels = torch.from_numpy(validation_data[f"label flow in order {self.col_label_flow}"])
        valid_dataset = torch.utils.data.TensorDataset(valid_tensors, valid_labels)
        if self.valid_batch_size == -1:
            self.valid_batch_size = len(validation_data[f"output flow in order {self.col_output_flow}"])

        test_tensor = torch.from_numpy(test_data[f"output flow in order {self.col_output_flow}"])
        test_labels = torch.from_numpy(test_data[f"label flow in order {self.col_label_flow}"])
        test_dataset = torch.utils.data.TensorDataset(test_tensor, test_labels)
        self.test_batch_size = len(test_data[f"output flow in order {self.col_output_flow}"])

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

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
            # **kwargs
        )

        return train_loader, valid_loader, test_loader, test_data[f"scaler"]

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

    def run_training(self):
        for epoch in range(self.epochs):
            print('\nEpoch: {}'.format(epoch + 1))

            train_loss, train_loss_epoch = self.train()
            validation_loss = self.validate(
                epoch=epoch,
                loader=self.valid_loader
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
                try:
                    self.plot_data(epoch=epoch)
                except:
                    print(f"Error epoch {epoch}")

        if self.plot_test is True:
            make_gif(self.path_chain_plot, f"{self.path_gifs}/chain_plot.gif")
            make_gif(self.path_loss_plot, f"{self.path_gifs}/loss_plot.gif")
            make_gif(self.path_mean_plot, f"{self.path_gifs}/mean_plot.gif")
            make_gif(self.path_std_plot, f"{self.path_gifs}/std_plot.gif")
            make_gif(self.path_flag_plot, f"{self.path_gifs}/flag_plot.gif")
            make_gif(self.path_detection_plot, f"{self.path_gifs}/detection_plot.gif")
            make_gif(self.path_color_color_plot, f"{self.path_gifs}/color_color_plot.gif")
            make_gif(self.path_residual_plot, f"{self.path_gifs}/residual_plot.gif")
        if self.save_nn is True:
            torch.save(self.best_model, f"{self.path_save_nn}/best_model_des_epoch_{self.best_validation_epoch+1}.pt")
            torch.save(self.model, f"{self.path_save_nn}/last_model_des_epoch_{self.epochs}.pt")
        self.validate(
            epoch=self.best_validation_epoch,
            loader=self.test_loader
        )
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

    def validate(self, epoch, loader):
        self.model.eval()
        val_loss = 0

        pbar = tqdm(total=len(loader.dataset))
        pbar.set_description('Eval')
        for batch_idx, data in enumerate(loader):
            cond_data = data[1].float()
            cond_data = cond_data.to(self.device)
            data = data[0]
            data = data.to(self.device)
            with torch.no_grad():
                val_loss += -self.model.log_probs(data, cond_data).sum().item()
            pbar.update(data.size(0))
            pbar.set_description(f'Val, Log likelihood in nats: {val_loss / pbar.n}')
            self.lst_valid_loss_per_batch.append(val_loss / pbar.n)

        self.writer.add_scalar('validation/LL', val_loss / len(loader.dataset), epoch)

        pbar.close()
        return val_loss / len(loader.dataset)

    def plot_data(self, epoch):
        """"""
        sns.set_theme()
        self.model.eval()

        if self.plot_loss is True:
            statistical_figure, ((stat_ax1, stat_ax2), (stat_ax3, stat_ax4)) = plt.subplots(nrows=2, ncols=2)
            statistical_figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)
            statistical_figure.suptitle(f"Epoch: {epoch}", fontsize=16)

            # Create dataframe of progress list
            df_test_loss_per_batch = pd.DataFrame({
                "training loss": self.lst_train_loss_per_batch
            })
            df_test_loss_per_epoch = pd.DataFrame({
                "training loss": self.lst_train_loss_per_epoch
            })
            df_valid_loss_per_batch = pd.DataFrame({
                "validation loss": self.lst_valid_loss_per_batch
            })
            df_valid_loss_per_epoch = pd.DataFrame({
                "validation loss": self.lst_valid_loss_per_epoch
            })


            # Create plot
            df_test_loss_per_batch.plot(
                figsize=(16, 9),
                alpha=0.5,
                marker=".",
                grid=True,
                # yticks=(0, 0.25, 0.5, 0.69, 1.0, 5.0),
                ax=stat_ax1)

            stat_ax1.set_xlabel("batch", fontsize=10, loc='right')
            stat_ax1.set_ylabel("loss", fontsize=12, loc='top')
            stat_ax1.set_title(f"Loss per batch")

            df_test_loss_per_epoch.plot(
                figsize=(16, 9),
                alpha=0.5,
                marker=".",
                grid=True,
                # yticks=(0, 0.25, 0.5, 0.69, 1.0, 5.0),
                ax=stat_ax2)

            stat_ax2.set_xlabel("epoch", fontsize=10, loc='right')
            stat_ax2.set_ylabel("loss", fontsize=12, loc='top')
            stat_ax2.set_title(f"Loss per epoch")

            # Create plot
            df_valid_loss_per_batch.plot(
                figsize=(16, 9),
                alpha=0.5,
                marker=".",
                grid=True,
                # yticks=(0, 0.25, 0.5, 0.69, 1.0, 5.0),
                ax=stat_ax3)

            stat_ax3.set_xlabel("batch", fontsize=10, loc='right')
            stat_ax3.set_ylabel("loss", fontsize=12, loc='top')
            stat_ax3.set_title(f"Loss per batch")

            df_valid_loss_per_epoch.plot(
                figsize=(16, 9),
                alpha=0.5,
                marker=".",
                grid=True,
                # yticks=(0, 0.25, 0.5, 0.69, 1.0, 5.0),
                ax=stat_ax4)

            stat_ax4.set_xlabel("epoch", fontsize=10, loc='right')
            stat_ax4.set_ylabel("loss", fontsize=12, loc='top')
            stat_ax4.set_title(f"Loss per epoch")

            if self.show_plot is True:
                statistical_figure.show()
            if self.save_plot is True:
                statistical_figure.savefig(f"{self.path_loss_plot}/loss_{epoch+1}.png", dpi=200)

            # Clear and close open figure to avoid memory overload
            statistical_figure.clf()
            plt.close(statistical_figure)


        colors = [
            ("r", "i"),
            ("i", "z")
        ]

        bands = [
            "r",
            "i",
            "z"
        ]

        for batch_idx, data in enumerate(self.test_loader):
            cond_data = data[1].float()
            cond_data = cond_data.to(self.device)
            data = data[0].float()
            with torch.no_grad():
                test_output = self.model.sample(self.test_batch_size, cond_inputs=cond_data).detach().cpu()

            df_generator_label = pd.DataFrame(cond_data.numpy(), columns=self.col_label_flow)
            df_generator_output = pd.DataFrame(test_output.numpy(), columns=self.col_output_flow)
            df_generator_scaled = pd.concat([df_generator_label, df_generator_output], axis=1)

            # r, _ = np.where(df_generator_output.isin([np.nan, np.inf, -np.inf]))
            # r = np.unique(r)
            # df_generator_output = df_generator_output.drop(index=r)
            # df_generator_label = df_generator_label.drop(index=r)
            # df_generator_scaled = pd.concat([df_generator_label, df_generator_output], axis=1)
            # try:
            #     generator_rescaled = self.scaler.inverse_transform(df_generator_scaled)
            # except ValueError:
            #     print("Value Error")
            #     continue

            generator_rescaled = self.scaler.inverse_transform(df_generator_scaled)
            df_generated = pd.DataFrame(generator_rescaled, columns=df_generator_scaled.columns)

            df_true_output = pd.DataFrame(data, columns=self.col_output_flow)
            df_true_scaled = pd.concat([df_generator_label, df_true_output], axis=1)
            true_rescaled = self.scaler.inverse_transform(df_true_scaled)
            df_true = pd.DataFrame(true_rescaled, columns=df_true_scaled.columns)

            if self.plot_color_color is True:
                df_generated_measured = pd.DataFrame({})
                df_true_measured = pd.DataFrame({})
                for color in colors:
                    df_generated_measured[f"{color[0]}-{color[1]}"] = \
                        np.array(df_generated[f"unsheared/mag_{color[0]}"]) - np.array(
                            df_generated[f"unsheared/mag_{color[1]}"])
                    df_true_measured[f"{color[0]}-{color[1]}"] = \
                        np.array(df_true[f"unsheared/mag_{color[0]}"]) - np.array(df_true[f"unsheared/mag_{color[1]}"])

                arr_true = df_true_measured.to_numpy()
                arr_generated = df_generated_measured.to_numpy()
                parameter = [
                    "unsheared/mag r-i",
                    "unsheared/mag i-z"
                ]
                chainchat = ChainConsumer()
                chainchat.add_chain(arr_true, parameters=parameter, name="true observed properties: chat")
                chainchat.add_chain(arr_generated, parameters=parameter, name="generated observed properties: chat*")
                chainchat.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
                chainchat.plotter.plot(
                    filename=f'{self.path_color_color_plot}/color_color_{epoch+1}.png',
                    figsize="page",
                    extents={
                        "unsheared/mag r-i": (-6, 6),
                        "unsheared/mag i-z": (-25, 25)
                    }
                )
                if self.show_plot is True:
                    plt.show()
                plt.clf()
                plt.close()

            if self.plot_residual:
                hist_figure, ((stat_ax1), (stat_ax2), (stat_ax3)) = \
                    plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
                hist_figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
                hist_figure.suptitle(r"residual", fontsize=16)

                lst_axis_res = [
                    stat_ax1,
                    stat_ax2,
                    stat_ax3
                ]

                lst_xlim_res = [
                    (-2.5, 2.5),
                    (-2.5, 2.5),
                    (-2.5, 2.5)
                ]

                df_hist_balrog = pd.DataFrame({
                    "dataset": ["skillz" for _ in range(len(df_true[f"unsheared/mag_r"]))]
                })
                df_hist_generated = pd.DataFrame({
                    "dataset": ["generated" for _ in range(len(df_true[f"unsheared/mag_r"]))]
                })
                for band in bands:
                    df_hist_balrog[f"BDF_MAG_DERED_CALIB - unsheared/mag {band}"] = df_true[
                                                                                        f"BDF_MAG_DERED_CALIB_{band.upper()}"] - \
                                                                                    df_true[f"unsheared/mag_{band}"]
                    df_hist_generated[f"BDF_MAG_DERED_CALIB - unsheared/mag {band}"] = df_true[
                                                                                           f"BDF_MAG_DERED_CALIB_{band.upper()}"] - \
                                                                                       df_generated[f"unsheared/mag_{band}"]

                for idx, band in enumerate(bands):
                    sns.histplot(
                        data=df_hist_balrog,
                        x=f"BDF_MAG_DERED_CALIB - unsheared/mag {band}",
                        ax=lst_axis_res[idx],
                        element="step",
                        stat="density",
                        color="dodgerblue",
                        bins=50,
                        label="balrog"
                    )
                    sns.histplot(
                        data=df_hist_generated,
                        x=f"BDF_MAG_DERED_CALIB - unsheared/mag {band}",
                        ax=lst_axis_res[idx],
                        element="step",
                        stat="density",
                        color="darkorange",
                        fill=False,
                        bins=50,
                        label="generated"
                    )
                    lst_axis_res[idx].axvline(
                        x=df_hist_balrog[f"BDF_MAG_DERED_CALIB - unsheared/mag {band}"].median(),
                        color='dodgerblue',
                        ls='--',
                        lw=1.5,
                        label="Mean balrog"
                    )
                    lst_axis_res[idx].axvline(
                        x=df_hist_generated[f"BDF_MAG_DERED_CALIB - unsheared/mag {band}"].median(),
                        color='darkorange',
                        ls='--',
                        lw=1.5,
                        label="Mean generated"
                    )
                    lst_axis_res[idx].set_xlim(lst_xlim_res[idx][0], lst_xlim_res[idx][1])
                    if idx == 0:
                        lst_axis_res[idx].legend()
                    else:
                        lst_axis_res[idx].legend([], [], frameon=False)
                hist_figure.tight_layout()
                if self.show_plot is True:
                    plt.show()

                if self.save_plot is True:
                    plt.savefig(f"{self.path_residual_plot}/residual_plot_{epoch+1}.png")
                plt.clf()
                plt.close()

            if self.plot_chain is True:
                df_generated_measured = pd.DataFrame({
                    "unsheared/mag_r": np.array(df_generated["unsheared/mag_r"]),
                    "unsheared/mag_i": np.array(df_generated["unsheared/mag_i"]),
                    "unsheared/mag_z": np.array(df_generated["unsheared/mag_z"]),
                    "unsheared/snr": np.array(df_generated["unsheared/snr"]),
                    "unsheared/size_ratio": np.array(df_generated["unsheared/size_ratio"]),
                    "unsheared/T": np.array(df_generated["unsheared/T"])
                })

                df_analytical_output = pd.DataFrame(data, columns=self.col_output_flow)
                df_analytical_scaled = pd.concat([df_generator_label, df_analytical_output], axis=1)
                analytical_rescaled = self.scaler.inverse_transform(df_analytical_scaled)
                df_balrog = pd.DataFrame(analytical_rescaled, columns=df_analytical_scaled.columns)
                df_balrog_measured = pd.DataFrame({
                    "unsheared/mag_r": np.array(df_balrog["unsheared/mag_r"]),
                    "unsheared/mag_i": np.array(df_balrog["unsheared/mag_i"]),
                    "unsheared/mag_z": np.array(df_balrog["unsheared/mag_z"]),
                    "unsheared/snr": np.array(df_balrog["unsheared/snr"]),
                    "unsheared/size_ratio": np.array(df_balrog["unsheared/size_ratio"]),
                    "unsheared/T": np.array(df_balrog["unsheared/T"])
                })

                arr_balrog = df_balrog_measured.to_numpy()
                arr_generated = df_generated_measured.to_numpy()
                parameter = [
                    "mag r",
                    "mag i",
                    "mag z",
                    "snr",
                    "size ratio",
                    "T"
                ]
                chainchat = ChainConsumer()
                chainchat.add_chain(arr_balrog, parameters=parameter, name="balrog observed properties: chat")
                chainchat.add_chain(arr_generated, parameters=parameter, name="generated observed properties: chat*")
                chainchat.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
                try:
                    chainchat.plotter.plot(
                        filename=f'{self.path_chain_plot}/chainplot_{epoch + 1}.png',
                        figsize="page",
                        # extents={
                        #     "mag r": (17.5, 26),
                        #     "mag i": (17.5, 26),
                        #     "mag z": (17.5, 26),
                        #     "snr": (-11, 100),
                        #     "size ratio": (-4, 4),
                        #     "T": (-2, 2)
                        # }
                    )
                except:
                    print("chain error at epoch", epoch + 1)
                plt.clf()

                chaincolor = ChainConsumer()
                df_compare_balrog = pd.DataFrame({
                    'true r': df_balrog['BDF_MAG_DERED_CALIB_R'],
                    'true i': df_balrog['BDF_MAG_DERED_CALIB_I'],
                    'true z': df_balrog['BDF_MAG_DERED_CALIB_Z'],
                    'meas r - true r': df_balrog['unsheared/mag_r'] - df_balrog['BDF_MAG_DERED_CALIB_R'],
                    'meas i - true i': df_balrog['unsheared/mag_i'] - df_balrog['BDF_MAG_DERED_CALIB_I'],
                    'meas z - true z': df_balrog['unsheared/mag_z'] - df_balrog['BDF_MAG_DERED_CALIB_Z']
                })
                df_compare_generated = pd.DataFrame({
                    'true r': df_generated['BDF_MAG_DERED_CALIB_R'],
                    'true i': df_generated['BDF_MAG_DERED_CALIB_I'],
                    'true z': df_generated['BDF_MAG_DERED_CALIB_Z'],
                    'meas r - true r': df_generated['unsheared/mag_r'] - df_generated['BDF_MAG_DERED_CALIB_R'],
                    'meas i - true i': df_generated['unsheared/mag_i'] - df_generated['BDF_MAG_DERED_CALIB_I'],
                    'meas z - true z': df_generated['unsheared/mag_z'] - df_generated['BDF_MAG_DERED_CALIB_Z']
                })
                plot_parameter = [
                    "true r",
                    "true i",
                    "true z",
                    'meas r - true r',
                    'meas i - true i',
                    'meas z - true z'
                ]
                chaincolor.add_chain(df_compare_balrog.to_numpy(), parameters=plot_parameter, name="Balrog color")
                chaincolor.add_chain(df_compare_generated.to_numpy(), parameters=plot_parameter, name="generated color")
                chaincolor.plotter.plot(
                    figsize="page",
                    # extents={
                    #     "true r": (18, 24),
                    #     "true i": (18, 24),
                    #     "true z": (18, 24),
                    #     "meas r - true r": (-1, 1),
                    #     "meas i - true i": (-1, 1),
                    #     "meas z - true z": (-1, 1)
                    # }
                )
                if self.show_plot is True:
                    plt.show()
                if self.save_plot is True:
                    plt.savefig(f"{self.path_color_diff_plot}/color_diff_{epoch + 1}.png", dpi=200)
                plt.clf()
                plt.close()

            if self.plot_mean is True:
                self.lst_mean_mag_r.append(df_generated["unsheared/mag_r"].mean() / df_true["unsheared/mag_r"].mean())
                self.lst_mean_mag_i.append(df_generated["unsheared/mag_i"].mean() / df_true["unsheared/mag_i"].mean())
                self.lst_mean_mag_z.append(df_generated["unsheared/mag_z"].mean() / df_true["unsheared/mag_z"].mean())
                self.lst_mean_snr.append(df_generated["unsheared/snr"].mean() / df_true["unsheared/snr"].mean())
                self.lst_mean_size_ratio.append(
                    df_generated["unsheared/size_ratio"].mean() / df_true["unsheared/size_ratio"].mean())
                self.lst_mean_t.append(df_generated["unsheared/T"].mean() / df_true["unsheared/T"].mean())

                plt.plot(self.lst_epochs, self.lst_mean_mag_r, marker="o", linestyle='-', color="blue", label="mag r")
                plt.plot(self.lst_epochs, self.lst_mean_mag_i, marker="^", linestyle='-', color="red", label="mag i")
                plt.plot(self.lst_epochs, self.lst_mean_mag_z, marker="X", linestyle='-', color="green", label="mag z")
                plt.plot(self.lst_epochs, self.lst_mean_snr, marker="d", linestyle='-', color="orange", label="snr")
                plt.plot(self.lst_epochs, self.lst_mean_size_ratio, marker="s", linestyle='-', color="purple",
                         label="size ratio")
                plt.plot(self.lst_epochs, self.lst_mean_t, marker="*", linestyle='-', color="black", label="T")
                plt.legend()
                plt.title("plot ratio mean")
                plt.xlabel("epoch")
                plt.ylabel("mean(chat*) / mean(chat)")

                if self.show_plot is True:
                    plt.show()
                if self.save_plot is True:
                    plt.savefig(f"{self.path_mean_plot}/mean_{epoch+1}.png", dpi=200)
                plt.clf()
                plt.close()

            if self.plot_std is True:
                self.lst_std_mag_r.append(df_generated["unsheared/mag_r"].std() / df_true["unsheared/mag_r"].std())
                self.lst_std_mag_i.append(df_generated["unsheared/mag_i"].std() / df_true["unsheared/mag_i"].std())
                self.lst_std_mag_z.append(df_generated["unsheared/mag_z"].std() / df_true["unsheared/mag_z"].std())
                self.lst_std_snr.append(df_generated["unsheared/snr"].std() / df_true["unsheared/snr"].std())
                self.lst_std_size_ratio.append(
                    df_generated["unsheared/size_ratio"].std() / df_true["unsheared/size_ratio"].std())
                self.lst_std_t.append(df_generated["unsheared/T"].std() / df_true["unsheared/T"].std())

                plt.plot(self.lst_epochs, self.lst_std_mag_r, marker="o", linestyle='-', color="blue", label="mag r")
                plt.plot(self.lst_epochs, self.lst_std_mag_i, marker="^", linestyle='-', color="red", label="mag i")
                plt.plot(self.lst_epochs, self.lst_std_mag_z, marker="X", linestyle='-', color="green", label="mag z")
                plt.plot(self.lst_epochs, self.lst_std_snr, marker="d", linestyle='-', color="orange", label="snr")
                plt.plot(self.lst_epochs, self.lst_std_size_ratio, marker="s", linestyle='-', color="purple",
                         label="size ratio")
                plt.plot(self.lst_epochs, self.lst_std_t, marker="*", linestyle='-', color="black", label="T")
                plt.legend()
                plt.title("plot ratio standard deviation")
                plt.xlabel("epoch")
                plt.ylabel("std(chat*) / std(chat)")

                if self.show_plot is True:
                    plt.show()
                if self.save_plot is True:
                    plt.savefig(f"{self.path_std_plot}/std_{epoch+1}.png", dpi=200)
                plt.clf()
                plt.close()

            if self.plot_flags is True:
                plt.plot(df_generated["unsheared/flags"], ".b", label="generated flag")
                plt.plot(df_true["unsheared/flags"], ".g", label="true flag")
                plt.title(f"Compare flags, epoch {epoch}")
                plt.xlabel("flags")
                plt.legend()
                plt.ylim(-0.5, 0.5)
                if self.show_plot is True:
                    plt.show()
                if self.save_plot is True:
                    plt.savefig(f"{self.path_flag_plot}/flags_{epoch}.png", dpi=200)
                plt.clf()

            if self.plot_detected is True:
                plt.plot(df_generated["detected"], ".b", label="generated detected")
                plt.plot(df_true["detected"], ".g", label="true detected")
                plt.title(f"Compare detected, epoch {epoch}")
                plt.xlabel("detected")
                plt.legend()
                plt.ylim(0.5, 1.5)
                if self.show_plot is True:
                    plt.show()
                if self.save_plot is True:
                    plt.savefig(f"{self.path_detection_plot}/detection_{epoch}.png", dpi=200)
                plt.clf()

