#!/bin/bash python
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import random
from utils import *
import torch.nn as nn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_future_preds", type=int)
    parser.add_argument("--pre_trained_file_name")
    args = parser.parse_args()

    num_future_preds = args.num_future_preds
    pre_trained_file_name = args.pre_trained_file_name

    torch.cuda.manual_seed(1008)
    torch.cuda.manual_seed_all(1008)
    np.random.seed(1008)
    random.seed(1008)
    torch.manual_seed(1008)

    root_dir = "/kaggle/working/datas"  # specify where results will be saved
    sns.set_style("whitegrid")
    sns.set_palette(["#57068c", "#E31212", "#01AD86"])
    print("pytorch version: ", torch.__version__)
    train_proportion = 0.7
    test_proportion = 0
    val_proportion = 0.3

    window_size = 192
    batch_size = 128
    train__val_loader, train_loader, val_loader, test_loader, scaler = get_data_loaders(
        train_proportion,
        test_proportion,
        val_proportion,
        window_size=window_size,
        pred_size=1,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True,
        test_mode=True,
    )

    model = torch.load(pre_trained_file_name)

    ### Predict
    model.eval()
    future_rollout = torch.Tensor(0)
    future_result = torch.Tensor(0)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():
        for i, (data, targets) in enumerate(test_loader):
            if i == len(test_loader.dataset) - 1:
                data_in = data
                future_rollout = targets
                data_in = data_in.to(device)
                output = model(data_in)
                future_rollout = torch.cat(
                    [future_rollout, output[:, -1:, :].detach().cpu()], dim=1
                )
                future_result = torch.cat(
                    (future_result, output[:, -1, :].view(-1).detach().cpu()), 0
                )

        for _ in range(num_future_preds):  ### number of forecast steps
            data_in = future_rollout[:, -window_size:, :]
            data_in = data_in.to(device)
            output = model(data_in)
            future_rollout = torch.cat(
                [future_rollout, output[:, -1:, :].detach().cpu()], dim=1
            )
            future_result = torch.cat(
                (future_result, output[:, -1, :].view(-1).detach().cpu()), 0
            )

    ### Plot prediction
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
    ax.plot(future_result, label="future_forecast")
    ax.axhline(y=0)
    ax.legend(loc="upper right")
    fig.savefig(root_dir + "/figs/lstm_future_pred.png")
    plt.close(fig)

    ### Check MSE, MAE
    future_result = future_result.numpy().reshape(-1, 1)
    future_result = scaler.inverse_transform(future_result)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
    ax.plot(future_result, label="future_forecast")
    ax.axhline(y=0)
    ax.legend(loc="upper right")
    fig.savefig(root_dir + "/figs/lstm_future_inversed_pred.png")
    plt.close(fig)
    ### Save model result

    future_result_df = pd.DataFrame(future_result)
    future_result_df.to_csv(root_dir + "/lstm_future_prediction.csv")
