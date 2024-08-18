#!/bin/bash python
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from informer import Informer
from utils import *
import torch.nn as nn
import os


def process_one_batch(batch_x, batch_y):
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)

    dec_inp = torch.zeros([batch_y.shape[0], 1, batch_y.shape[-1]]).float().to(device)
    dec_inp = (
        torch.cat([batch_y[:, : (window_size - 1), :], dec_inp], dim=1)
        .float()
        .to(device)
    )
    outputs = model(batch_x, dec_inp)

    return outputs, batch_y


def evaluate(model, data_loader, criterion, scaler):
    torch.cuda.manual_seed(1008)
    torch.cuda.manual_seed_all(1008)
    np.random.seed(1008)
    random.seed(1008)
    torch.manual_seed(1008)

    model.eval()
    test_rollout = torch.Tensor(0)
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    device = "cpu"
    total_loss = 0.0
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():
        for i, (data, targets) in enumerate(data_loader):
            if i == 0:
                enc_in = data
                dec_in = targets
                test_rollout = targets
            else:
                enc_in = test_rollout[:, -window_size:, :]
                dec_in = torch.zeros([enc_in.shape[0], 1, enc_in.shape[-1]]).float()
                dec_in = torch.cat(
                    [enc_in[:, : (window_size - 1), :], dec_in], dim=1
                ).float()
                # dec_in = enc_in[:,:(window_size-1),:]
            enc_in, dec_in, targets = (
                enc_in.to(device),
                dec_in.to(device),
                targets.to(device),
            )
            output = model(enc_in, dec_in)

            total_loss += (
                criterion(output[:, -1:, :], targets[:, -1:, :]).detach().cpu().numpy()
            )
            test_rollout = torch.cat(
                [test_rollout, output[:, -1:, :].detach().cpu()], dim=1
            )
            test_result = torch.cat(
                (test_result, output[:, -1, :].view(-1).detach().cpu()), 0
            )
            truth = torch.cat((truth, targets[:, -1, :].view(-1).detach().cpu()), 0)
    return total_loss


def predict_model(model, test_loader, window_size, epoch, plot=True):
    model.eval()
    test_rollout = torch.Tensor(0)
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():
        for i, (data, targets) in enumerate(test_loader):
            if i == 0:
                enc_in = data
                dec_in = targets
                test_rollout = targets
            else:
                enc_in = test_rollout[:, -window_size:, :]
                dec_in = torch.zeros([enc_in.shape[0], 1, enc_in.shape[-1]]).float()
                dec_in = torch.cat(
                    [enc_in[:, : (window_size - 1), :], dec_in], dim=1
                ).float()
                # dec_in = enc_in[:,:(window_size-1),:]
            enc_in, dec_in, targets = (
                enc_in.to(device),
                dec_in.to(device),
                targets.to(device),
            )
            output = model(enc_in, dec_in)

            test_rollout = torch.cat(
                [test_rollout, output[:, -1:, :].detach().cpu()], dim=1
            )
            test_result = torch.cat(
                (test_result, output[:, -1, :].view(-1).detach().cpu()), 0
            )
            truth = torch.cat((truth, targets[:, -1, :].view(-1).detach().cpu()), 0)

    if plot is True:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        ax.plot(test_result, label="forecast")
        ax.plot(truth, label="truth")
        ax.plot(test_result - truth, ls="--", label="residual")
        # ax.grid(True, which='both')
        ax.axhline(y=0)
        ax.legend(loc="upper right")
        fig.savefig(root_dir + f"/figs/informer_epoch{epoch}_pred.png")
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
            torch.save(model, "best_inf.pth")
            print("Saving best model")
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

    root_dir = "/kaggle/working/datas"  # specify where results will be saved

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    fig_dir = "/kaggle/working/datas/figs"

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    sns.set_style("whitegrid")
    sns.set_palette(["#57068c", "#E31212", "#01AD86"])
    print("pytorch version: ", torch.__version__)
    # test
    # best_config = {'d_model': 216, 'n_heads': 2, 'e_layers': 3, 'd_layers': 3, 'd_ff': 1024, 'window_size': 192, 'dropout': 0.2, 'lr': 0.0001, 'optim_step': 5, 'lr_decay': 0.9, 'factor': 9, 'batch_size': 128}
    # future
    best_config = {
        "d_model": 216,
        "n_heads": 2,
        "e_layers": 3,
        "d_layers": 3,
        "d_ff": 512,
        "window_size": 64,
        "dropout": 0.2,
        "lr": 0.0001,
        "optim_step": 5,
        "lr_decay": 0.9,
        "factor": 9,
        "batch_size": 128,
    }

    train_proportion = 0.9
    test_proportion = 0  # test_size is fixed to same as NASA's range and train+val will fill out the rest of the time points if use_nasa_test_range = True.
    val_proportion = 0.1

    print(f"Config: {best_config}")

    lr = best_config["lr"]
    optim_step = best_config["optim_step"]
    lr_decay = best_config["lr_decay"]
    window_size = best_config["window_size"]
    batch_size = best_config["batch_size"]
    enc_channel_in = 1
    dec_channel_in = 1
    channel_out = 1
    seq_len = best_config["window_size"]
    label_len = best_config["window_size"] - 1
    out_len = 1  # best_config['window_size']
    factor = best_config["factor"]
    d_model = best_config["d_model"]
    n_heads = best_config["n_heads"]
    e_layers = best_config["e_layers"]
    d_layers = best_config["d_layers"]
    d_ff = best_config["d_ff"]
    dropout = best_config["dropout"]

    train_val_loader, train_loader, val_loader, test_loader, scaler = get_data_loaders(
        train_proportion,
        test_proportion,
        val_proportion,
        window_size=window_size,
        pred_size=1,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        test_mode=True,
        use_nasa_test_range=use_nasa_test_range,
    )

    if requires_training:
        if use_pre_trained:
            model = torch.load(pre_trained_file_name)
        else:
            model = Informer(
                enc_channel_in,
                dec_channel_in,
                channel_out,
                seq_len,
                label_len,
                out_len,
                factor,
                d_model,
                n_heads,
                e_layers,
                d_layers,
                d_ff,
                dropout,
                attn="prob",
                embed="fixed",
                freq="m",
                activation="gelu",
                output_attention=False,
                distil=True,
                device=torch.device("cuda:0"),
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

        epochs = 150
        train_losses = []
        test_losses = []
        tolerance = 10
        best_test_loss = float("inf")
        Early_Stopping = early_stopping(patience=12)
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0

            for i, (data, targets) in enumerate(train_val_loader):
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()
                output, truth = process_one_batch(data, targets)
                loss = criterion(output[:, -1, :], targets[:, -1, :])
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            if (epoch % 10 == 0) & (use_nasa_test_range != "non_nasa_no_test"):
                print(f"Saving prediction for epoch {epoch}")
                predict_model(model, test_loader, window_size, epoch, plot=False)

            train_losses.append(total_loss * batch_size / len(train_val_loader.dataset))
            test_loss = evaluate(model, test_loader, criterion, scaler)
            test_losses.append(test_loss / len(test_loader.dataset))

            if epoch == 1:  ###DEBUG
                print(
                    f"Total of {len(train_val_loader.dataset)} samples in training set and {len(test_loader.dataset)} samples in test set",
                    flush=True,
                )

            print(
                f"Epoch: {epoch}, train_loss: {total_loss*batch_size/len(train_val_loader.dataset)}, test_loss: {test_loss/len(test_loader.dataset)}, lr: {scheduler.get_last_lr()}",
                flush=True,
            )

            Early_Stopping(model, test_loss / len(test_loader))
            if Early_Stopping.early_stop:
                break

            if epoch % 1 == 0:
                scheduler.step()

        ## Plot losses
        xs = np.arange(len(train_losses))
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        ax.plot(xs, train_losses)
        fig.savefig(root_dir + "/figs/informer_train_loss.png")
        plt.close(fig)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        ax.plot(xs, test_losses)
        fig.savefig(root_dir + "/figs/informer_test_loss.png")
        plt.close(fig)

    ### Predict
    if not requires_training:
        model = torch.load(pre_trained_file_name)
    else:
        model = torch.load("best_inf.pth")
    model.eval()
    test_rollout = torch.Tensor(0)
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():
        for i, (data, targets) in enumerate(test_loader):
            if i == 0:
                enc_in = data
                dec_in = targets
                test_rollout = targets
            else:
                enc_in = test_rollout[:, -window_size:, :]
                dec_in = torch.zeros([enc_in.shape[0], 1, enc_in.shape[-1]]).float()
                dec_in = torch.cat(
                    [enc_in[:, : (window_size - 1), :], dec_in], dim=1
                ).float()

            enc_in, dec_in, targets = (
                enc_in.to(device),
                dec_in.to(device),
                targets.to(device),
            )
            output = model(enc_in, dec_in)

            test_rollout = torch.cat(
                [test_rollout, output[:, -1:, :].detach().cpu()], dim=1
            )
            test_result = torch.cat(
                (test_result, output[:, -1, :].view(-1).detach().cpu()), 0
            )
            truth = torch.cat((truth, targets[:, -1, :].view(-1).detach().cpu()), 0)

    ### Check MSE, MAE
    test_result = test_result.numpy().reshape(-1, 1)
    test_result = scaler.inverse_transform(test_result)
    truth = truth.numpy().reshape(-1, 1)
    truth = scaler.inverse_transform(truth)

    RMSE = mean_squared_error(truth, test_result) ** 0.5
    MAE = mean_absolute_error(truth, test_result)
    print(f"RMSE: {RMSE}, MAE: {MAE}")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
    ax.plot(test_result, label="forecast")
    ax.plot(truth, label="truth")
    ax.plot(test_result - truth, ls="--", label="residual")
    # ax.grid(True, which='both')
    ax.axhline(y=0)
    ax.legend(loc="upper right")
    fig.savefig(root_dir + "/figs/informer_inversed_pred.png")

    plt.close(fig)

    test_result_df = pd.DataFrame(test_result)
    test_result_df["truth"] = truth
    test_result_df.to_csv(root_dir + "/informer_prediction.csv")
