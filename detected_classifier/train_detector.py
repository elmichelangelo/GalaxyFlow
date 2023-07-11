import torch
import copy
import os
import sys
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from Handler.data_loader import load_data
from torch import nn
from torch.utils.data import Dataset


class TrainDet(object):
    def __init__(self,
                 path_train_data,
                 path_output,
                 path_loss_txt_output,
                 selected_scaler,
                 reproducible,
                 lr,
                 run,
                 size_training_dataset,
                 size_validation_dataset,
                 size_test_dataset,
                 batch_size,
                 valid_batch_size
                 ):
        super().__init__()
        self.path_train_data = path_train_data
        self.path_output = path_output
        self.path_loss_txt_output = path_loss_txt_output
        self.run = run
        self.size_training_dataset = size_training_dataset
        self.size_validation_dataset = size_validation_dataset
        self.size_test_dataset = size_test_dataset
        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = batch_size
        self.best_loss = float('inf')
        self.best_acc = 0.0
        self.best_epoch = 0
        self.lr = lr
        self.bs = bs

        self.col_label_det = [
            "BDF_MAG_DERED_CALIB_U",
            "BDF_MAG_DERED_CALIB_G",
            "BDF_MAG_DERED_CALIB_R",
            "BDF_MAG_DERED_CALIB_I",
            "BDF_MAG_DERED_CALIB_Z",
            "BDF_MAG_DERED_CALIB_J",
            "BDF_MAG_DERED_CALIB_H",
            "BDF_MAG_DERED_CALIB_K",
            "BDF_MAG_ERR_DERED_CALIB_U",
            "BDF_MAG_ERR_DERED_CALIB_G",
            "BDF_MAG_ERR_DERED_CALIB_R",
            "BDF_MAG_ERR_DERED_CALIB_I",
            "BDF_MAG_ERR_DERED_CALIB_Z",
            "BDF_MAG_ERR_DERED_CALIB_J",
            "BDF_MAG_ERR_DERED_CALIB_H",
            "BDF_MAG_ERR_DERED_CALIB_K",
            "Color Mag U-G",
            "Color Mag G-R",
            "Color Mag R-I",
            "Color Mag I-Z",
            "Color Mag Z-J",
            "Color Mag J-H",
            "Color Mag H-K",
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

        self.col_output_det = ["detected"]

        self.train_loader, self.valid_loader, self.test_loader, self.scaler = self.init_dataset(
            selected_scaler=selected_scaler,
            reproducible=reproducible
        )

        self.model = BinaryClassifier(num_features=len(self.col_label_det), learning_rate=lr)
        self.best_validation_loss = float('inf')
        self.best_validation_acc = 0.0
        self.best_validation_epoch = 0
        self.best_model = self.model

    def init_dataset(self, selected_scaler, reproducible):
        """"""
        training_data, validation_data, test_data, all_data = load_data(
            path_training_data=self.path_train_data,
            path_output=self.path_output,
            input_flow=self.col_label_det,
            output_flow=self.col_output_det,
            selected_scaler=selected_scaler,
            size_training_dataset=self.size_training_dataset,
            size_validation_dataset=self.size_validation_dataset,
            size_test_dataset=self.size_test_dataset,
            reproducible=reproducible,
            run=self.run
        )

        train_tensor = torch.from_numpy(training_data[f"output flow in order {self.col_output_det}"])
        train_labels = torch.from_numpy(training_data[f"label flow in order {self.col_label_det}"])
        train_dataset = torch.utils.data.TensorDataset(train_labels, train_tensor)

        valid_tensors = torch.from_numpy(validation_data[f"output flow in order {self.col_output_det}"])
        valid_labels = torch.from_numpy(validation_data[f"label flow in order {self.col_label_det}"])
        valid_dataset = torch.utils.data.TensorDataset(valid_labels, valid_tensors)
        if self.valid_batch_size == -1:
            self.valid_batch_size = len(validation_data[f"output flow in order {self.col_output_det}"])

        test_tensor = torch.from_numpy(test_data[f"output flow in order {self.col_output_det}"])
        test_labels = torch.from_numpy(test_data[f"label flow in order {self.col_label_det}"])
        test_dataset = torch.utils.data.TensorDataset(test_labels, test_tensor)
        self.test_batch_size = len(test_data[f"output flow in order {self.col_output_det}"])

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

    def run_training(self, epochs):
        """"""
        # Training
        lst_loss = []
        lst_acc = []
        lst_acc_val = []
        for e in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0
            epoch_acc = 0
            for X_batch, y_batch in self.train_loader:
                self.model.optimizer.zero_grad()

                # Convert tensors to float
                y_batch = y_batch.float()
                X_batch = X_batch.float()

                y_pred = self.model(X_batch)

                loss = self.model.loss_function(y_pred, y_batch)
                acc = binary_acc(y_pred, y_batch)

                if loss < self.best_loss:
                    self.best_epoch = e
                    self.best_loss = loss

                if acc > self.best_acc:
                    self.best_acc = acc

                loss.backward()
                self.model.optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

            lst_loss.append(epoch_loss / len(self.train_loader))
            lst_acc.append(epoch_acc / len(self.train_loader))

            lst_acc_val, validation_loss, validation_acc = self.validate_network(
                e=e,
                lst_loss=lst_loss,
                lst_acc=lst_acc,
                lst_acc_val=lst_acc_val
            )

            if validation_acc > self.best_validation_acc:
                self.best_validation_acc = validation_acc

            if validation_loss < self.best_validation_loss:
                self.best_validation_epoch = e
                self.best_validation_loss = validation_loss
                self.best_model = copy.deepcopy(self.model)
                self.best_model.eval()

            print(f'Epoch {e + 0:03}: | Training Loss: {epoch_loss / len(self.train_loader):.5f} | Best Training Loss: {self.best_loss:.5f}')
            print(f'Epoch {e + 0:03}: | Training Acc: {epoch_acc / len(self.train_loader):.3f} | Best Training Acc: {self.best_acc:.3f}')
            print(f'Epoch {e + 0:03}: | Validation Loss: {validation_loss:.5f} | Best validation Loss: {self.best_validation_loss :.5f}')
            print(f'Epoch {e + 0:03}: | Validation Acc: {validation_acc:.3f} | Best validation Acc: {self.best_validation_acc :.3f}')

            if e - self.best_validation_epoch >= 30:
                break

        with open(f"{self.path_loss_txt_output}/loss.txt", "a") as f:
            f.write(f"Learning Rate {self.lr} \t|\t "
                    f"Batch Size {self.bs} \t|\t "
                    f"Best training loss {self.best_loss :.5f} \t|\t "
                    f"Best training acc {self.best_acc :.5f} \t|\t "
                    f"Best training epoch {self.best_epoch + 0:03} \t|\t "
                    f"Best validation loss {self.best_validation_loss :.5f} \t|\t "
                    f"Best validation acc {self.best_validation_acc + 0:03} \t|\t "
                    f"Best validation epoch {self.best_validation_epoch + 0:03}\n")

        torch.save(self.best_model, f"{self.path_loss_txt_output}/best_model_des_epoch_{self.best_validation_epoch + 1}_lr_{self.lr}_bs_{self.bs}.pt")
        torch.save(self.model, f"{self.path_loss_txt_output}/last_model_des_epoch_{epochs}_lr_{self.lr}_bs_{self.bs}.pt")

    def validate_network(self, e, lst_loss, lst_acc, lst_acc_val):
        lst_loss_valid = []
        lst_acc_valid = []
        for X_batch, y_batch in self.valid_loader:
            # Convert tensors to float
            self.model.eval()
            lst_y_pred = []
            lst_y = []
            y_batch = y_batch.float()
            X_batch = X_batch.float()
            y_pred_valid = self.model(X_batch)
            acc_valid = binary_acc(y_pred_valid, y_batch)
            lst_loss_valid.append(self.model.loss_function(y_pred_valid, y_batch).detach())
            lst_acc_valid.append(acc_valid.item() / len(self.valid_loader))
            lst_acc_val.append(acc_valid.item() / len(self.valid_loader))

            y_pred_valid_round = torch.where(y_pred_valid <= 0.05, torch.zeros_like(y_pred_valid), y_pred_valid)
            y_pred_valid_round = torch.where(y_pred_valid_round >= 0.95, torch.ones_like(y_pred_valid_round), y_pred_valid_round)

            lst_y_full = list(y_batch.numpy())
            for idx, y in enumerate(list(y_pred_valid_round.detach().numpy())):  # y_pred_valid
                lst_y_pred.append(float(y))
                lst_y.append(float(lst_y_full[idx]))

            df_y = pd.DataFrame({
                "detected": lst_y + lst_y_pred,
                "type": ["true" for _ in range(len(lst_y))] + ["predicted" for _ in range(len(lst_y_pred))],
                "id": [i for i in range(len(lst_y))] + [j for j in range(len(lst_y_pred))]
            })

            sns.histplot(data=df_y, x="detected", hue="type")
            plt.savefig(f"{self.path_output}/y_pred/y_pred_{e}.png")
            plt.clf()
            break

        df_loss = pd.DataFrame(lst_loss, columns=["loss"])
        df_acc = pd.DataFrame(lst_acc, columns=["acc"])
        df_acc_valid = pd.DataFrame(lst_acc_val, columns=["acc validation"])
        self.plot_data(df_acc_valid["acc validation"], f"/acc_validation/acc_val_{e}")
        self.plot_data(df_loss["loss"], f"/loss/loss_{e}")
        self.plot_data(df_acc["acc"], f"/acc/acc_{e}")
        return lst_acc_val, np.mean(lst_loss_valid), np.mean(lst_acc_valid)

    def plot_data(self, data, save_name):
        """"""
        sns.scatterplot(data)
        plt.savefig(f"{self.path_output}/{save_name}.png")
        plt.clf()


class BinaryClassifier(nn.Module):

    def __init__(self, num_features, learning_rate):
        super(BinaryClassifier, self).__init__()

        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # if self.torch_device == "mps":
        #     self.model.to(self.torch_device)

        # Loss function to calculate the error of the neural net (binary cross entropy)
        self.loss_function = nn.BCELoss()

        self.loss = 0

        # Optimizer to calculate the weight changes
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, inputs):
        return self.model(inputs)


def binary_acc(y_pred, y_test):
    # y_pred_tag = torch.round(y_pred)
    plt.hist(y_pred.detach().numpy())
    exit()
    y_pred_tag = torch.where(y_pred <= 0.2, torch.zeros_like(y_pred), y_pred)
    y_pred_tag = torch.where(y_pred_tag >= 0.8, torch.ones_like(y_pred_tag), y_pred_tag)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


def main(
        path_train_data,
        path_output,
        path_loss_txt_output,
        selected_scaler,
        reproducible,
        run,
        epochs,
        size_training_dataset,
        size_validation_dataset,
        batch_size,
        valid_batch_size,
        lr,
        size_test_dataset):
    """"""

    train_detector = TrainDet(
        path_train_data=path_train_data,
        size_training_dataset=size_training_dataset,
        size_validation_dataset=size_validation_dataset,
        size_test_dataset=size_test_dataset,
        path_output=path_output,
        path_loss_txt_output=path_loss_txt_output,
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        selected_scaler=selected_scaler,
        run=run,
        lr=lr,
        reproducible=reproducible
    )

    train_detector.run_training(epochs)


if __name__ == '__main__':
    i = 0
    path = os.path.abspath(sys.path[i])

    while not os.path.exists(f"{path}/Data/balrog_all_w_undetected_3000000.pkl"):
        i += 1
        path = os.path.abspath(sys.path[i])
    for lr in [0.001, 0.0001, 0.00001, 0.000001]:
        for bs in [8, 16, 32, 64]:
            if not os.path.exists(f"{path}/Output/plots/lr_{lr}_bs_{bs}"):
                os.mkdir(f"{path}/Output/plots/lr_{lr}_bs_{bs}")
                os.mkdir(f"{path}/Output/plots/lr_{lr}_bs_{bs}/loss")
                os.mkdir(f"{path}/Output/plots/lr_{lr}_bs_{bs}/acc")
                os.mkdir(f"{path}/Output/plots/lr_{lr}_bs_{bs}/acc_validation")
                os.mkdir(f"{path}/Output/plots/lr_{lr}_bs_{bs}/y_pred")
            main(
                path_train_data=f"{path}/Data/balrog_all_w_undetected_3000000.pkl",
                size_training_dataset=.6,
                size_validation_dataset=.2,
                size_test_dataset=.2,
                epochs=150,
                path_output=f"{path}/Output/plots/lr_{lr}_bs_{bs}",
                path_loss_txt_output=f"{path}/Output/plots",
                selected_scaler="MaxAbsScaler",
                run=1,
                batch_size=bs,
                valid_batch_size=-1,
                lr=lr,
                reproducible=False
            )