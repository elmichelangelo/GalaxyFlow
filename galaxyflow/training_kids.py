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
from Handler.data_loader import load_data_kidz
from chainconsumer import ChainConsumer
from Handler.helper_functions import make_gif
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
                 apply_cuts):
        super().__init__()
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

        self.lst_mean_mag_u = []
        self.lst_mean_mag_g = []
        self.lst_mean_mag_r = []
        self.lst_mean_mag_i = []
        self.lst_mean_mag_Z = []
        self.lst_mean_mag_Y = []
        self.lst_mean_mag_J = []
        self.lst_mean_mag_H = []
        self.lst_mean_mag_Ks = []

        self.lst_std_mag_u = []
        self.lst_std_mag_g = []
        self.lst_std_mag_r = []
        self.lst_std_mag_i = []
        self.lst_std_mag_Z = []
        self.lst_std_mag_Y = []
        self.lst_std_mag_J = []
        self.lst_std_mag_H = []
        self.lst_std_mag_Ks = []

        self.lst_train_loss_per_batch = []
        self.lst_train_loss_per_epoch = []
        self.lst_valid_loss_per_batch = []
        self.lst_valid_loss_per_epoch = []
        self.path_output_flow = f"{path_output}/Kids_flow_" \
                                f"lr_{self.lr}_" \
                                f"num_hidden_{self.num_hidden}_" \
                                f"num_blocks_{self.num_blocks}_" \
                                f"batch_size_{self.batch_size}_"
        self.path_plots = f"{self.path_output_flow}/plots"
        self.path_chain_plot = f"{self.path_plots}/chain_plot"
        self.path_loss_plot = f"{self.path_plots}/loss_plot"
        self.path_mean_plot = f"{self.path_plots}/mean_plot"
        self.path_std_plot = f"{self.path_plots}/std_plot"
        self.path_flag_plot = f"{self.path_plots}/flag_plot"
        self.path_color_diff_plot = f"{self.path_plots}/color_diff_plot"
        self.path_detection_plot = f"{self.path_plots}/detection_plot"
        self.path_gifs = f"{self.path_plots}/gifs"
        self.path_save_nn = f"{self.path_output_flow}/nn"
        self.make_dirs()

        self.writer = SummaryWriter(log_dir=f"{self.path_output_flow}/writer", comment=f"_lr_{self.lr}")
        self.col_label_flow = col_label_flow
        self.col_output_flow = col_output_flow

        self.train_loader, self.valid_loader, self.test_loader, self.scaler = self.init_dataset(
            path_train_data=path_train_data,
            selected_scaler=selected_scaler,
            apply_cuts=apply_cuts
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

        if self.save_nn is True:
            if not os.path.exists(self.path_save_nn):
                os.mkdir(self.path_save_nn)

    def init_dataset(self, path_train_data, selected_scaler, apply_cuts):
        """"""
        training_data, validation_data, test_data = load_data_kidz(
            path_training_data=path_train_data,
            input_flow=self.col_label_flow,
            output_flow=self.col_output_flow,
            selected_scaler=selected_scaler,
            apply_cuts=apply_cuts
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

        # for module in model.modules():
        #     if isinstance(module, nn.Linear):
        #         nn.init.orthogonal_(module.weight)
        #         if hasattr(module, 'bias') and module.bias is not None:
        #             module.bias.data.fill_(0)

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

            # if epoch - self.best_validation_epoch >= 30:
            #     break

            if validation_loss < self.best_validation_loss:
                self.best_validation_epoch = epoch
                self.best_validation_loss = validation_loss
                self.best_model = copy.deepcopy(self.model)
                self.best_model.eval()

            print(f"Best validation at epoch {self.best_validation_epoch + 1}"
                  f"\t Average Log Likelihood {-self.best_validation_loss}")

            if self.plot_test is True:
                self.plot_data(epoch=epoch)

        if self.plot_test is True:
            make_gif(self.path_chain_plot, f"{self.path_gifs}/chain_plot.gif")
            make_gif(self.path_loss_plot, f"{self.path_gifs}/loss_plot.gif")
            make_gif(self.path_mean_plot, f"{self.path_gifs}/mean_plot.gif")
            make_gif(self.path_std_plot, f"{self.path_gifs}/std_plot.gif")
            make_gif(self.path_flag_plot, f"{self.path_gifs}/flag_plot.gif")
            make_gif(self.path_detection_plot, f"{self.path_gifs}/detection_plot.gif")
        if self.save_nn is True:
            torch.save(self.best_model, f"{self.path_save_nn}/best_model_epoch_{self.best_validation_epoch + 1}.pt")
            torch.save(self.model, f"{self.path_save_nn}/last_model_epoch_{epoch + 1}.pt")
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

        for batch_idx, data in enumerate(self.test_loader):
            cond_data = data[1].float()
            cond_data = cond_data.to(self.device)
            data = data[0].float()
            with torch.no_grad():
                test_output = self.model.sample(self.test_batch_size, cond_inputs=cond_data).detach().cpu()

            df_generator_label = pd.DataFrame(cond_data.numpy(), columns=self.col_label_flow)
            df_generator_output = pd.DataFrame(test_output.numpy(), columns=self.col_output_flow)

            r, _ = np.where(df_generator_output.isin([np.nan, np.inf, -np.inf]))
            r = np.unique(r)
            df_generator_output = df_generator_output.drop(index=r)
            df_generator_label = df_generator_label.drop(index=r)

            df_generator_scaled = pd.concat([df_generator_label, df_generator_output], axis=1)
            try:
                generator_rescaled = self.scaler.inverse_transform(df_generator_scaled)
            except ValueError:
                print("Value Error")
                continue

            df_generated = pd.DataFrame(generator_rescaled, columns=df_generator_scaled.columns)
            df_generated_measured = pd.DataFrame({
                "luptize_u": np.array(df_generated["luptize_u"]),
                "luptize_r": np.array(df_generated["luptize_r"]),
                "luptize_g": np.array(df_generated["luptize_g"]),
                "luptize_i": np.array(df_generated["luptize_i"]),
                "luptize_Z": np.array(df_generated["luptize_Z"]),
                "luptize_Y": np.array(df_generated["luptize_Y"]),
                "luptize_J": np.array(df_generated["luptize_J"]),
                "luptize_H": np.array(df_generated["luptize_H"]),
                "luptize_Ks": np.array(df_generated["luptize_Ks"])
            })

            df_true_output = pd.DataFrame(data, columns=self.col_output_flow)
            df_true_scaled = pd.concat([df_generator_label, df_true_output], axis=1)
            true_rescaled = self.scaler.inverse_transform(df_true_scaled)
            df_true = pd.DataFrame(true_rescaled, columns=df_true_scaled.columns)
            df_true_measured = pd.DataFrame({
                "luptize_u": np.array(df_true["luptize_u"]),
                "luptize_r": np.array(df_true["luptize_r"]),
                "luptize_g": np.array(df_true["luptize_g"]),
                "luptize_i": np.array(df_true["luptize_i"]),
                "luptize_Z": np.array(df_true["luptize_Z"]),
                "luptize_Y": np.array(df_true["luptize_Y"]),
                "luptize_J": np.array(df_true["luptize_J"]),
                "luptize_H": np.array(df_true["luptize_H"]),
                "luptize_Ks": np.array(df_true["luptize_Ks"])
            })

            arr_true = df_true_measured.to_numpy()
            arr_generated = df_generated_measured.to_numpy()
            parameter = [
                "luptize u",
                "luptize r",
                "luptize g",
                "luptize i",
                "luptize Z",
                "luptize Y",
                "luptize J",
                "luptize H",
                "luptize Ks",
            ]
            chainchat = ChainConsumer()
            chainchat.add_chain(arr_true, parameters=parameter, name="true observed properties: chat")
            chainchat.add_chain(arr_generated, parameters=parameter, name="generated observed properties: chat*")
            chainchat.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
            try:
                chainchat.plotter.plot(
                    filename=f'{self.path_chain_plot}/chainplot_{epoch + 1}.png',
                    figsize="page",
                    # extents={
                    #     "luptize u": (-0.7E5, 0.7E5),
                    #     "luptize r": (-0.5E6, 0.5E6),
                    #     "luptize g": (-0.5E6, 0.5E6),
                    #     "luptize i": (-0.5E6, 0.5E6),
                    #     "luptize Z": (-0.5E6, 0.5E6),
                    #     "luptize Y": (-0.5E6, 0.5E6),
                    #     "luptize J": (-0.5E6, 0.5E6),
                    #     "luptize H": (-0.5E6, 0.5E6),
                    #     "luptize Ks": (-0.5E6, 0.5E6)
                    # }
                )
            except:
                print("chain error at epoch", epoch + 1)
            plt.clf()

            # self.lst_mean_mag_u.append(df_generated["FLUX_GAAP_u"].mean() / df_true["FLUX_GAAP_u"].mean())
            # self.lst_mean_mag_g.append(df_generated["FLUX_GAAP_g"].mean() / df_true["FLUX_GAAP_g"].mean())
            # self.lst_mean_mag_r.append(df_generated["FLUX_GAAP_r"].mean() / df_true["FLUX_GAAP_r"].mean())
            # self.lst_mean_mag_i.append(df_generated["FLUX_GAAP_i"].mean() / df_true["FLUX_GAAP_i"].mean())
            # self.lst_mean_mag_Z.append(df_generated["FLUX_GAAP_Z"].mean() / df_true["FLUX_GAAP_Z"].mean())
            # self.lst_mean_mag_Y.append(df_generated["FLUX_GAAP_Y"].mean() / df_true["FLUX_GAAP_Y"].mean())
            # self.lst_mean_mag_J.append(df_generated["FLUX_GAAP_J"].mean() / df_true["FLUX_GAAP_J"].mean())
            # self.lst_mean_mag_H.append(df_generated["FLUX_GAAP_H"].mean() / df_true["FLUX_GAAP_H"].mean())
            # self.lst_mean_mag_Ks.append(df_generated["FLUX_GAAP_Ks"].mean() / df_true["FLUX_GAAP_Ks"].mean())
            #
            # plt.plot(self.lst_epochs, self.lst_mean_mag_u, marker="o", linestyle='-', color="blue", label="mag u")
            # plt.plot(self.lst_epochs, self.lst_mean_mag_g, marker="^", linestyle='-', color="red", label="mag g")
            # plt.plot(self.lst_epochs, self.lst_mean_mag_r, marker="X", linestyle='-', color="green", label="mag r")
            # plt.plot(self.lst_epochs, self.lst_mean_mag_i, marker="d", linestyle='-', color="orange", label="mag i")
            # plt.plot(self.lst_epochs, self.lst_mean_mag_Z, marker="s", linestyle='-', color="purple", label="mag Z")
            # plt.plot(self.lst_epochs, self.lst_mean_mag_Y, marker="*", linestyle='-', color="black", label="mag Y")
            # plt.plot(self.lst_epochs, self.lst_mean_mag_J, marker="a", linestyle='-', color="grey", label="mag J")
            # plt.plot(self.lst_epochs, self.lst_mean_mag_Ks, marker="w", linestyle='-', color="darkred", label="mag Ks")
            # plt.legend()
            # plt.title("plot ratio mean")
            # plt.xlabel("epoch")
            # plt.ylabel("mean(chat*) / mean(chat)")
            #
            # if self.show_plot is True:
            #     plt.show()
            # if self.save_plot is True:
            #     plt.savefig(f"{self.path_mean_plot}/mean_{epoch+1}.png", dpi=200)
            # plt.clf()
            # plt.close()
            #
            # self.lst_std_mag_r.append(df_generated["unsheared/mag_r"].std() / df_true["unsheared/mag_r"].std())
            # self.lst_std_mag_i.append(df_generated["unsheared/mag_i"].std() / df_true["unsheared/mag_i"].std())
            # self.lst_std_mag_z.append(df_generated["unsheared/mag_z"].std() / df_true["unsheared/mag_z"].std())
            # self.lst_std_snr.append(df_generated["unsheared/snr"].std() / df_true["unsheared/snr"].std())
            # self.lst_std_size_ratio.append(
            #     df_generated["unsheared/size_ratio"].std() / df_true["unsheared/size_ratio"].std())
            # self.lst_std_t.append(df_generated["unsheared/T"].std() / df_true["unsheared/T"].std())
            #
            # plt.plot(self.lst_epochs, self.lst_std_mag_r, marker="o", linestyle='-', color="blue", label="mag r")
            # plt.plot(self.lst_epochs, self.lst_std_mag_i, marker="^", linestyle='-', color="red", label="mag i")
            # plt.plot(self.lst_epochs, self.lst_std_mag_z, marker="X", linestyle='-', color="green", label="mag z")
            # plt.plot(self.lst_epochs, self.lst_std_snr, marker="d", linestyle='-', color="orange", label="snr")
            # plt.plot(self.lst_epochs, self.lst_std_size_ratio, marker="s", linestyle='-', color="purple",
            #          label="size ratio")
            # plt.plot(self.lst_epochs, self.lst_std_t, marker="*", linestyle='-', color="black", label="T")
            # plt.legend()
            # plt.title("plot ratio standard deviation")
            # plt.xlabel("epoch")
            # plt.ylabel("std(chat*) / std(chat)")
            #
            # if self.show_plot is True:
            #     plt.show()
            # if self.save_plot is True:
            #     plt.savefig(f"{self.path_std_plot}/std_{epoch+1}.png", dpi=200)
            # plt.clf()
            # plt.close()
            #
            # plt.plot(df_generated["unsheared/flags"], ".b", label="generated flag")
            # plt.plot(df_true["unsheared/flags"], ".g", label="true flag")
            # plt.title(f"Compare flags, epoch {epoch}")
            # plt.xlabel("flags")
            # plt.legend()
            # if self.show_plot is True:
            #     plt.show()
            # if self.save_plot is True:
            #     plt.savefig(f"{self.path_flag_plot}/flags_{epoch}.png", dpi=200)
            # plt.clf()
            #
            # plt.plot(df_generated["detected"], ".b", label="generated detected")
            # plt.plot(df_true["detected"], ".g", label="true detected")
            # plt.title(f"Compare detected, epoch {epoch}")
            # plt.xlabel("detected")
            # plt.legend()
            # if self.show_plot is True:
            #     plt.show()
            # if self.save_plot is True:
            #     plt.savefig(f"{self.path_detection_plot}/detection_{epoch}.png", dpi=200)
            # plt.clf()

