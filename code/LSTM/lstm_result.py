#!/bin/bash python
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import random
from lstm import LSTM
from utils import *
import torch.nn as nn
import os


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():
        for i, (data, targets) in enumerate(data_loader):
            data, targets = data.to(device), targets.to(device)
            output = model(data)

            total_loss += (
                criterion(output[:, -1:, :], targets[:, -1:, :]).detach().cpu().numpy()
            )

    return total_loss


def predict_model(model, test_loader, window_size, epoch, plot=True):
    model.eval()
    test_rollout = torch.Tensor(0)
    test_result = torch.Tensor(0)
    predict_loss = 0.0
    truth = torch.Tensor(0)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():
        for i, (data, targets) in enumerate(test_loader):
            # state_h, state_c = model.init_state(1)###change
            if i == 0:
                data_in = data
                test_rollout = targets
            else:
                data_in = test_rollout[:, -window_size:, :]
            # state_h, state_c = state_h.to(device), state_c.to(device)
            data_in, targets = data_in.to(device), targets.to(device)
            output = model(data_in)
            predict_loss += (
                criterion(output[:, -1:, :], targets[:, -1:, :]).detach().cpu().numpy()
            )

            test_rollout = torch.cat(
                [test_rollout, output[:, -1:, :].detach().cpu()], dim=1
            )
            test_result = torch.cat(
                (test_result, output[:, -1, :].view(-1).detach().cpu()), 0
            )
            truth = torch.cat((truth, targets[:, -1, :].view(-1).detach().cpu()), 0)

    if plot == True:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        ax.plot(test_result, label="forecast")
        ax.plot(truth, label="truth")
        ax.plot(test_result - truth, ls="--", label="residual")
        # ax.grid(True, which='both')
        ax.axhline(y=0)
        ax.legend(loc="upper right")
        fig.savefig(root_dir + f"/figs/lstm_epoch{epoch}_pred.png")
        plt.close(fig)


class early_stopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.early_stop = False
        self.best_loss = None
        self.counter = 0
        self.best_model = None

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model
            torch.save(model, "best_lstm.pth")
        else:
            self.counter += 1
            if self.counter == self.patience:
                self.early_stop = True
                print("Early stopping")
            print(
                f"----Current loss {val_loss} higher than best loss {self.best_loss}, early stop counter {self.counter}----"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--requires_training", default=False, action="store_true")
    parser.add_argument("--use_pre_trained", default=False, action="store_true")
    parser.add_argument("--use_nasa_test_range", default=False, action="store_true")
    parser.add_argument("--pre_trained_file_name")
    args = parser.parse_args()

    requires_training = args.requires_training
    use_pre_trained = args.use_pre_trained
    pre_trained_file_name = args.pre_trained_file_name
    use_nasa_test_range = args.use_nasa_test_range

    torch.cuda.manual_seed(1008)
    torch.cuda.manual_seed_all(1008)
    np.random.seed(1008)
    random.seed(1008)
    torch.manual_seed(1008)

    sns.set_style("whitegrid")
    sns.set_palette(["#57068c", "#E31212", "#01AD86"])

    root_dir = "/kaggle/working/datas/"  # specify where results will be saved

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    fig_dir = "/kaggle/working/datas/figs"

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # test
    best_config = {
        "hidden_size": 512,
        "num_layers": 3,
        "dropout": 0.1,
        "lr": 0.001,
        "window_size": 192,
        "batch_size": 128,
        "optim_step": 2,
        "lr_decay": 0.7,
        "bidirectional": False,
    }
    # #future
    # best_config = {
    #     "hidden_size": 216,
    #     "num_layers": 1,
    #     "dropout": 0.1,
    #     "lr": 5e-05,
    #     "window_size": 192,
    #     "batch_size": 128,
    #     "optim_step": 15,
    #     "lr_decay": 0.85,
    #     "bidirectional": False,
    # }

    train_proportion = 0.7
    test_proportion = 0  # test_size is fixed to same as NASA's range and train+val will fill out the rest of the time points if use_nasa_test_range = True.
    val_proportion = 0.3
    input_size = 1  # best_config['input_size']
    hidden_size = best_config["hidden_size"]
    num_layers = best_config["num_layers"]
    optim_step = best_config["optim_step"]
    lr_decay = best_config["lr_decay"]
    dropout = best_config["dropout"]
    lr = best_config["lr"]
    window_size = best_config["window_size"]
    batch_size = best_config["batch_size"]

    train_val_loader, train_loader, val_loader, test_loader, scaler = get_data_loaders(
        train_proportion,
        test_proportion,
        val_proportion,
        window_size=window_size,
        pred_size=1,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=False,
        test_mode=True,
        use_nasa_test_range=use_nasa_test_range,
    )

    if requires_training == True:
        if use_pre_trained == True:
            model = torch.load(pre_trained_file_name)
        else:
            model = LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=False,
            )
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
        print("Using device: ", device)
        model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, optim_step, gamma=lr_decay)
        # writer = tensorboard.SummaryWriter('') #specify where tensorboard results will be saved

        epochs = 800
        train_losses = []
        test_losses = []
        best_test_loss = float("inf")
        Early_Stopping = early_stopping(patience=20)

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for data, targets in train_val_loader:
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, targets)
                total_loss += loss.item()
                loss.backward(retain_graph=True)
                optimizer.step()
            if epoch % 10 == 0:
                print(f"Saving prediction for epoch {epoch}")
                predict_model(model, test_loader, window_size, epoch, plot=False)
            train_losses.append(total_loss * batch_size)
            test_loss = evaluate(model, test_loader, criterion)
            test_losses.append(test_loss / len(test_loader.dataset))
            if epoch == 1:  ###DEBUG
                print(
                    f"Total of {len(train_val_loader.dataset)} samples in training set and {len(test_loader.dataset)} samples in test set"
                )
            print(
                f"Epoch: {epoch}, train_loss: {total_loss*batch_size/len(train_val_loader.dataset)}, test_loss: {test_loss/len(test_loader.dataset)}, lr: {scheduler.get_last_lr()}"
            )
            Early_Stopping(model, test_loss / len(test_loader))
            if Early_Stopping.early_stop:
                break
            # writer.add_scalar('train_loss',total_loss,epoch)
            # writer.add_scalar('val_loss',test_loss,epoch)

            scheduler.step()
        ## Plot losses
        xs = np.arange(len(train_losses))
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        ax.plot(xs, train_losses)
        fig.savefig(root_dir + "figs/lstm_train_loss.png")
        plt.close(fig)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        ax.plot(xs, test_losses)
        fig.savefig(root_dir + "figs/lstm_test_loss.png")
        plt.close(fig)

    ### Predict
    if not requires_training:
        model = torch.load(pre_trained_file_name)
    else:
        model = torch.load("best_lstm.pth")
    model.eval()
    test_rollout = torch.Tensor(0)
    test_result = torch.Tensor(0)
    predict_loss = 0.0
    truth = torch.Tensor(0)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():
        for i, (data, targets) in enumerate(test_loader):
            if i == 0:
                data_in = data
                test_rollout = targets
            else:
                data_in = test_rollout[:, -window_size:, :]
            data_in, targets = data_in.to(device), targets.to(device)
            output = model(data_in)

            test_rollout = torch.cat(
                [test_rollout, output[:, -1:, :].detach().cpu()], dim=1
            )
            test_result = torch.cat(
                (test_result, output[:, -1, :].view(-1).detach().cpu()), 0
            )
            truth = torch.cat((truth, targets[:, -1, :].view(-1).detach().cpu()), 0)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
    ax.plot(test_result, label="forecast")
    ax.plot(truth, label="truth")
    ax.plot(test_result - truth, ls="--", label="residual")
    # ax.grid(True, which='both')
    ax.axhline(y=0)
    ax.legend(loc="upper right")
    fig.savefig(root_dir + "figs/lstm_pred.png")
    plt.close(fig)

    ### Check MSE, MAE
    test_result = test_result.numpy().reshape(-1, 1)
    test_result = scaler.inverse_transform(test_result)
    truth = truth.numpy().reshape(-1, 1)
    truth = scaler.inverse_transform(truth)
    RMSE = mean_squared_error(truth, test_result) ** 0.5
    MAE = mean_absolute_error(truth, test_result)
    print(f"RMSE:{RMSE}, MAE: {MAE}")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
    ax.plot(test_result, label="forecast")
    ax.plot(truth, label="truth")
    ax.plot(test_result - truth, ls="--", label="residual")
    # ax.grid(True, which='both')
    ax.axhline(y=0)
    ax.legend(loc="upper right")
    fig.savefig(root_dir + "figs/lstm_inverse_prediction.png")

    test_result_df = pd.DataFrame(test_result, columns=["predictions"])
    test_result_df["truth"] = truth
    test_result_df.to_csv(root_dir + "lstm_prediction.csv")
