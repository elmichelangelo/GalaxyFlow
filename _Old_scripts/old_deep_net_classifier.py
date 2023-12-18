import torch
import copy
import os
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import pandas as pd
import matplotlib.pyplot as plt
from gandalf_galaxie_dataset import DESGalaxies
from torch.utils.data import DataLoader
from torch import nn


class TrainDet(object):
    def __init__(self,
                 cfg,
                 lr,
                 bs
                 ):
        super().__init__()
        self.cfg = cfg
        self.bs = bs
        self.best_loss = float('inf')
        self.best_acc = 0.0
        self.best_epoch = 0
        self.lr = lr

        self.train_loader, self.valid_loader, self.df_test, self.galaxies = self.init_dataset()

        cfg['PATH_OUTPUT_CLASSF'] = f"{cfg['PATH_OUTPUT_CLASSF']}/lr_{self.lr}_bs_{self.bs}"
        cfg['PATH_WRITER_CLASSF'] = f"{cfg['PATH_OUTPUT_CLASSF']}/{cfg['PATH_WRITER_CLASSF']}"
        cfg['PATH_PLOTS_CLASSF'] = f"{cfg['PATH_OUTPUT_CLASSF']}/{cfg['PATH_PLOTS_CLASSF']}"
        cfg['PATH_SAVE_NN_CLASSF'] = f"{cfg['PATH_OUTPUT_CLASSF']}/{cfg['PATH_SAVE_NN_CLASSF']}"

        for plot in cfg['PLOTS_CLASSF']:
            cfg[f'PATH_PLOTS_FOLDER_CLASSF'][plot.upper()] = f"{cfg['PATH_PLOTS_CLASSF']}/{plot}"

        self.make_dirs()

        self.model = BinaryClassifier(num_features=len(self.cfg['INPUT_COLS_MAG_CLASSF']), learning_rate=lr)
        self.best_validation_loss = float('inf')
        self.best_validation_acc = 0.0
        self.best_validation_epoch = 0
        self.best_model = self.model

    def make_dirs(self):
        """"""
        if not os.path.exists(self.cfg['PATH_OUTPUT_CLASSF']):
            os.mkdir(self.cfg['PATH_OUTPUT_CLASSF'])
        if self.cfg['PLOT_TEST_CLASSF'] is True:
            if not os.path.exists(self.cfg['PATH_PLOTS_CLASSF']):
                os.mkdir(self.cfg['PATH_PLOTS_CLASSF'])
            for path_plot in self.cfg['PATH_PLOTS_FOLDER_CLASSF'].values():
                if not os.path.exists(path_plot):
                    os.mkdir(path_plot)

        if self.cfg['SAVE_NN_CLASSF'] is True:
            if not os.path.exists(self.cfg["PATH_SAVE_NN_CLASSF"]):
                os.mkdir(self.cfg["PATH_SAVE_NN_CLASSF"])

    def init_dataset(self):
        """"""
        galaxies = DESGalaxies(
            cfg=self.cfg,
            kind="classifier_training",
            lst_split=[
                self.cfg['SIZE_TRAINING_DATA_CLASSF'],
                self.cfg['SIZE_VALIDATION_DATA_CLASSF'],
                self.cfg['SIZE_TEST_DATA_CLASSF']
            ]
        )

        # Create DataLoaders for training, validation, and testing
        train_loader = DataLoader(galaxies.train_dataset, batch_size=self.bs, shuffle=True, num_workers=0)
        valid_loader = DataLoader(galaxies.val_dataset, batch_size=self.bs, shuffle=False, num_workers=0)
        test_loader = DataLoader(galaxies.test_dataset, batch_size=self.bs, shuffle=False, num_workers=0)

        return train_loader, valid_loader, test_loader, galaxies

    def run_training(self):
        """"""
        # Training
        lst_loss = []
        lst_acc = []
        lst_acc_val = []
        for e in range(1, self.cfg['EPOCHS_CLASSF'] + 1):
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

        # Plot misclassification error over epoch
        misclassification_errors = [1 - acc / 100 for acc in lst_acc]
        plt.figure()
        plt.plot(range(1, len(misclassification_errors) + 1), misclassification_errors, marker='o', color='red')
        plt.title('Misclassification Error Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Misclassification Error')
        plt.savefig(
            f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF']['MISCLASS_ERROR']}/misclassification_error_over_epochs.png")
        plt.clf()

        with open(f"{self.cfg['PATH_WRITER_CLASSF']}/loss.txt", "a") as f:
            f.write(f"Learning Rate {self.lr} \t|\t "
                    f"Batch Size {self.bs} \t|\t "
                    f"Best training loss {self.best_loss :.5f} \t|\t "
                    f"Best training acc {self.best_acc :.5f} \t|\t "
                    f"Best training epoch {self.best_epoch + 0:03} \t|\t "
                    f"Best validation loss {self.best_validation_loss :.5f} \t|\t "
                    f"Best validation acc {self.best_validation_acc + 0:03} \t|\t "
                    f"Best validation epoch {self.best_validation_epoch + 0:03}\n")

        torch.save(self.best_model, f"{self.cfg['PATH_SAVE_NN_CLASSF']}/best_model_des_epoch_{self.best_validation_epoch + 1}_lr_{self.lr}_bs_{self.bs}.pt")
        torch.save(self.model, f"{self.cfg['PATH_SAVE_NN_CLASSF']}/last_model_des_epoch_{self.cfg['EPOCHS_CLASSF']}_lr_{self.lr}_bs_{self.bs}.pt")

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
                lst_y.append(int(lst_y_full[idx]))

            df_y = pd.DataFrame({
                "true_detected": lst_y + lst_y_pred,
                "type": ["true" for _ in range(len(lst_y))] + ["predicted" for _ in range(len(lst_y_pred))],
                "id": [i for i in range(len(lst_y))] + [j for j in range(len(lst_y_pred))]
            })

            sns.histplot(data=df_y, x="true_detected", hue="type")
            plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF']['Y_PRED']}/y_pred_{e}.png")
            plt.clf()

            # Compute confusion matrix
            y_pred_tag = torch.where(y_pred_valid >= 0.5, torch.ones_like(y_pred_valid), torch.zeros_like(y_pred_valid))
            cm = confusion_matrix(lst_y, y_pred_tag)
            df_cm = pd.DataFrame(cm, columns=["Predicted 0", "Predicted 1"], index=["Actual 0", "Actual 1"])
            plt.figure(figsize=(10, 7))
            sns.heatmap(df_cm, annot=True, fmt="g")
            plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF']['CONFUSION_MATRIX']}/confusion_matrix_{e}.png")
            plt.clf()

            # Compute ROC and AUC
            fpr, tpr, thresholds = roc_curve(lst_y, lst_y_pred)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF']['ROC_CURVE']}/roc_curve_{e}.png")
            plt.clf()

            # Compute precision-recall curve
            precision, recall, thresholds = precision_recall_curve(lst_y, lst_y_pred)
            plt.figure()
            plt.plot(recall, precision, color='blue', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.savefig(
                f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF']['PRECISION_RECALL_CURVE']}/precision_recall_curve_{e}.png")
            plt.clf()

            # Histogram of predicted probabilities
            plt.figure()
            plt.hist(lst_y_pred, bins=30, color='skyblue', edgecolor='black')
            plt.title('Histogram of Predicted Probabilities')
            plt.xlabel('Probability')
            plt.ylabel('Frequency')
            plt.savefig(f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF']['PROB_HIST']}/probability_histogram_{e}.png")
            plt.clf()
            break

        df_loss = pd.DataFrame(lst_loss, columns=["loss"])
        df_acc = pd.DataFrame(lst_acc, columns=["acc"])
        df_acc_valid = pd.DataFrame(lst_acc_val, columns=["acc validation"])
        self.plot_data(df_acc_valid["acc validation"], f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF']['ACC_VALIDATION']}/acc_val_{e}.png")
        self.plot_data(df_loss["loss"], f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF']['LOSS']}/loss_{e}.png")
        self.plot_data(df_acc["acc"], f"{self.cfg['PATH_PLOTS_FOLDER_CLASSF']['ACC']}/acc_{e}.png")
        return lst_acc_val, np.mean(lst_loss_valid), np.mean(lst_acc_valid)

    def plot_data(self, data, save_name):
        """"""
        sns.scatterplot(data)
        plt.savefig(save_name)
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
    y_pred_tag = torch.round(y_pred)
    plt.hist(y_pred.detach().numpy())

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


if __name__ == '__main__':
    pass