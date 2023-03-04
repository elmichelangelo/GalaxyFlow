import argparse
import copy
import math
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
from tensorboardX import SummaryWriter
import datasets
import src.GANdalfflow.flow as fnn
import pandas as pd
import utils
from src.GANdalfflow.flow import FlowSequential
from src.Handler.data_loader import load_data
from chainconsumer import ChainConsumer
from src.Handler.gif_maker import make_gif
sys.path.append(os.path.dirname(__file__))


def run(path_train_data, path_output, col_input_discriminator, col_label_generator, col_output_generator, plot_test):
    global global_step, writer
    # for lr in [0.0001, 0.00001, 0.000001, 0.0000001]:
    #     for num_hidden in [64, 128]:
    #         for num_blocks in [2, 3, 4, 5]:
    lr = 1E-6
    num_hidden = 64
    num_blocks = 3
    epochs = 250
    mnist_dataset = 'MNIST'
    flow = 'maf'
    device = torch.device("cpu")
    # kwargs = {}  # {'num_workers': 4, 'pin_memory': True}  # if cuda else {}

    training_data, test_data = load_data(
        path_training_data=path_train_data,
        apply_cuts=False,
        input_discriminator=col_input_discriminator,
        input_generator=col_label_generator,
        output_generator=col_output_generator,
        analytical_data=True,
        only_detected=True,
        selected_scaler="MaxAbsScaler"
    )
    length = len(training_data[f"output generator in order {col_output_generator}"])
    train_arr = training_data[f"output generator in order {col_output_generator}"][:int(0.8*length)]
    train_labels_arr = training_data[f"label generator in order {col_label_generator}"][:int(0.8*length)]
    train_tensor = torch.from_numpy(train_arr)
    train_labels = torch.from_numpy(train_labels_arr)
    train_dataset = torch.utils.data.TensorDataset(train_tensor, train_labels)

    valid_arr = training_data[f"output generator in order {col_output_generator}"][int(0.8*length):]
    valid_labels_arr = training_data[f"label generator in order {col_label_generator}"][int(0.8 * length):]
    valid_tensors = torch.from_numpy(valid_arr)
    valid_labels = torch.from_numpy(valid_labels_arr)
    valid_dataset = torch.utils.data.TensorDataset(valid_tensors, valid_labels)

    test_arr = test_data[f"output generator in order {col_output_generator}"]
    test_labels_arr = test_data[f"label generator in order {col_label_generator}"]
    test_tensor = torch.from_numpy(test_arr)
    test_labels = torch.from_numpy(test_labels_arr)
    test_dataset = torch.utils.data.TensorDataset(test_tensor, test_labels)

    num_cond_inputs = len(col_label_generator)
    num_inputs = len(col_output_generator)

    batch_size = 64
    valid_batch_size = len(valid_arr)
    test_batch_size = len(test_arr)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # **kwargs
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_batch_size,
        shuffle=False,
        drop_last=False,
        # **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        # **kwargs
    )

    act = 'tanh'
    modules = []

    for _ in range(num_blocks):
        modules += [
            fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]

    model = fnn.FlowSequential(*modules)

    # for module in model.modules():
    #     if isinstance(module, nn.Linear):
    #         nn.init.orthogonal_(module.weight)
    #         if hasattr(module, 'bias') and module.bias is not None:
    #             module.bias.data.fill_(0)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    writer = SummaryWriter(comment=flow + "_" + mnist_dataset)
    global_step = 0
    best_validation_loss = float('inf')
    best_validation_epoch = 0
    best_model = model

    if plot_test is True:
        # Set output paths
        path_test_output = f"{path_output}/Flow_test_" \
                           f"lr_{lr}_" \
                           f"num_hidden_{num_hidden}_" \
                           f"num_blocks_{num_blocks}"
        path_test_plots = f"{path_test_output}/plots"
        path_chain_plot = f"{path_test_plots}/chain_plot"
        path_gifs = f"{path_test_plots}/gifs"

    for epoch in range(epochs):
        print('\nEpoch: {}'.format(epoch+1))

        train(
            model=model,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer
        )
        validation_loss = validate(epoch, model, valid_loader, device)

        if epoch - best_validation_epoch >= 30:
            break

        if validation_loss < best_validation_loss:
            best_validation_epoch = epoch
            best_validation_loss = validation_loss
            best_model = copy.deepcopy(model)

        print(
            '\n Best validation at epoch {}: Average Log Likelihood in nats: {:.4f}'.
            format(best_validation_epoch+1, -best_validation_loss))

        best_model.eval()
        if plot_test is True:
            if not os.path.exists(path_test_output):
                os.mkdir(path_test_output)
            if not os.path.exists(path_test_plots):
                os.mkdir(path_test_plots)
            if not os.path.exists(path_chain_plot):
                os.mkdir(path_chain_plot)
            if not os.path.exists(path_gifs):
                os.mkdir(path_gifs)
            for batch_idx, data in enumerate(test_loader):
                cond_data = data[1].float()
                cond_data = cond_data.to(device)
                data = data[0].float()
                with torch.no_grad():
                    test_output = best_model.sample(test_batch_size, cond_inputs=cond_data).detach().cpu()

                df_generator_label = pd.DataFrame(cond_data.numpy(), columns=test_data[f"columns label generator"])
                df_generator_output = pd.DataFrame(test_output.numpy(), columns=test_data[f"columns output generator"])

                r, _ = np.where(df_generator_output.isin([np.nan, np.inf, -np.inf]))
                r = np.unique(r)
                df_generator_output = df_generator_output.drop(index=r)
                df_generator_label = df_generator_label.drop(index=r)

                df_generator_scaled = pd.concat([df_generator_label, df_generator_output], axis=1)
                try:
                    generator_rescaled = test_data[f"scaler"].inverse_transform(df_generator_scaled)
                except ValueError:
                    print("Value Error")
                    pass
                df_generator = pd.DataFrame(generator_rescaled, columns=df_generator_scaled.columns)
                df_generated = pd.DataFrame({
                    "unsheared/mag_r": np.array(df_generator["unsheared/mag_r"]),
                    "unsheared/mag_i": np.array(df_generator["unsheared/mag_i"]),
                    "unsheared/mag_z": np.array(df_generator["unsheared/mag_z"]),
                    "unsheared/snr": np.array(df_generator["unsheared/snr"]),
                    "unsheared/size_ratio": np.array(df_generator["unsheared/size_ratio"]),
                    "unsheared/T": np.array(df_generator["unsheared/T"])
                })

                df_analytical_output = pd.DataFrame(data, columns=test_data[f"columns output generator"])
                df_analytical_scaled = pd.concat([df_generator_label, df_analytical_output], axis=1)
                analytical_rescaled = test_data[f"scaler"].inverse_transform(df_analytical_scaled)
                df_balrog = pd.DataFrame(analytical_rescaled, columns=df_analytical_scaled.columns)
                df_balrog = pd.DataFrame({
                    "unsheared/mag_r": np.array(df_balrog["unsheared/mag_r"]),
                    "unsheared/mag_i": np.array(df_balrog["unsheared/mag_i"]),
                    "unsheared/mag_z": np.array(df_balrog["unsheared/mag_z"]),
                    "unsheared/snr": np.array(df_balrog["unsheared/snr"]),
                    "unsheared/size_ratio": np.array(df_balrog["unsheared/size_ratio"]),
                    "unsheared/T": np.array(df_balrog["unsheared/T"])
                })

                arr_balrog = df_balrog.to_numpy()
                arr_generated = df_generated.to_numpy()
                parameter = [
                    "mag r",
                    "mag i",
                    "mag z",
                    "snr",
                    "size ratio",
                    "T"
                ]
                chaincon = ChainConsumer()
                chaincon.add_chain(arr_balrog, parameters=parameter, name="balrog observed properties: chat")
                chaincon.add_chain(arr_generated, parameters=parameter, name="generated observed properties: chat*")
                chaincon.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
                try:
                    chaincon.plotter.plot(
                        filename=f'{path_chain_plot}/chainplot_{epoch+1}.png',
                        figsize="page",
                        extents={
                            "mag r": (17.5, 26),
                            "mag i": (17.5, 26),
                            "mag z": (17.5, 26),
                            "snr": (-11, 55),
                            "size ratio": (-1.5, 4),
                            "T": (-1, 2.5)
                        }
                    )
                except:
                    print("chain error at epoch", epoch+1)
                plt.clf()
                plt.close()
    plt.clf()
    plt.close()
    make_gif(path_chain_plot, f"{path_gifs}/chain_plot.gif")
    validate(
        epoch=best_validation_epoch,
        model=best_model,
        loader=test_loader,
        device=device
    )


def train(model, train_loader, device, optimizer):
    global global_step, writer
    model.train()
    train_loss = 0

    pbar = tqdm(total=len(train_loader.dataset))
    for batch_idx, data in enumerate(train_loader):
        cond_data = data[1].float()
        cond_data = cond_data.to(device)
        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        loss = -model.log_probs(data, cond_data).mean()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pbar.update(data.size(0))
        pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
            -train_loss / (batch_idx + 1)))

        writer.add_scalar('training/loss', loss.item(), global_step)
        global_step += 1

    pbar.close()

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 0

    with torch.no_grad():
        model(train_loader.dataset.tensors[0].to(data.device),
              train_loader.dataset.tensors[1].to(data.device).float())

    for module in model.modules():
        if isinstance(module, fnn.BatchNormFlow):
            module.momentum = 1


def validate(epoch, model, loader, device):
    global global_step, writer

    model.eval()
    val_loss = 0

    pbar = tqdm(total=len(loader.dataset))
    pbar.set_description('Eval')
    for batch_idx, data in enumerate(loader):
        cond_data = data[1].float()
        cond_data = cond_data.to(device)
        data = data[0]
        data = data.to(device)
        with torch.no_grad():
            val_loss += -model.log_probs(data, cond_data).sum().item()  # sum up batch loss
        pbar.update(data.size(0))
        pbar.set_description('Val, Log likelihood in nats: {:.6f}'.format(
            -val_loss / pbar.n))

    writer.add_scalar('validation/LL', val_loss / len(loader.dataset), epoch)

    pbar.close()
    return val_loss / len(loader.dataset)


if __name__ == '__main__':
    path = os.path.abspath(sys.path[1])
    lst_label_generator = [
        "BDF_MAG_DERED_CALIB_R",
        "BDF_MAG_DERED_CALIB_I",
        "BDF_MAG_DERED_CALIB_Z",
        "Color Mag U-G",
        "Color Mag G-R",
        "Color Mag R-I",
        "Color Mag I-Z",
        "Color Mag Z-J",
        "Color Mag J-H",
        "Color Mag H-KS",
        "BDF_T",
        "BDF_G",
        "FWHM_WMEAN_R",
        "FWHM_WMEAN_I",
        "FWHM_WMEAN_Z",
        "AIRMASS_WMEAN_R",
        "AIRMASS_WMEAN_I",
        "AIRMASS_WMEAN_Z",
        "MAGLIM_R",
        "MAGLIM_I",
        "MAGLIM_Z",
        "EBV_SFD98"
    ]

    lst_output_generator = [
        "unsheared/mag_r",
        "unsheared/mag_i",
        "unsheared/mag_z",
        "unsheared/snr",
        "unsheared/size_ratio",
        "unsheared/flags",
        "unsheared/T",
        "detected"
    ]

    lst_input_discriminator = [
        "BDF_MAG_DERED_CALIB_R",
        "BDF_MAG_DERED_CALIB_I",
        "BDF_MAG_DERED_CALIB_Z",
        "Color Mag U-G",
        "Color Mag G-R",
        "Color Mag R-I",
        "Color Mag I-Z",
        "Color Mag Z-J",
        "Color Mag J-H",
        "Color Mag H-KS",
        "BDF_T",
        "BDF_G",
        "FWHM_WMEAN_R",
        "FWHM_WMEAN_I",
        "FWHM_WMEAN_Z",
        "AIRMASS_WMEAN_R",
        "AIRMASS_WMEAN_I",
        "AIRMASS_WMEAN_Z",
        "MAGLIM_R",
        "MAGLIM_I",
        "MAGLIM_Z",
        "EBV_SFD98",
        "unsheared/mag_r",
        "unsheared/mag_i",
        "unsheared/mag_z",
        "unsheared/snr",
        "unsheared/size_ratio",
        "unsheared/flags",
        "unsheared/T",
        "detected"
    ]
    run(
        path_train_data=f"{path}/Data/Balrog_2_data_MAG_250000.pkl",
        path_output=f"{path}/Output",
        col_input_discriminator=lst_input_discriminator,
        col_output_generator=lst_output_generator,
        col_label_generator=lst_label_generator,
        plot_test=True
    )
