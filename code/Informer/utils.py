#!/bin/bash python
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def to_windowed(data, window_size, pred_size):
    out = []
    for i in range(len(data) - window_size):
        feature = np.array(data[i : i + (window_size)])
        target = np.array(data[i + pred_size : i + window_size + pred_size])
        out.append((feature, target))

    return np.array(out)  # , np.array(targets)


def train_test_val_split(
    x_vals,
    x_marks=None,
    train_proportion=0.6,
    val_proportion=0.2,
    test_proportion=0.2,
    window_size=12,
    pred_size=1,
    use_nasa_test_range=True,
):
    scaler = StandardScaler()
    x_vals = scaler.fit_transform(x_vals.reshape(-1, 1)).reshape(-1)
    x_vals = to_windowed(x_vals, window_size, pred_size)
    total_len = len(x_vals)
    if use_nasa_test_range is True or use_nasa_test_range == "nasa_test":
        test_len = 132  ### 274 specified to train second round with data range conform to NASA's
        val_proportion = val_proportion / (val_proportion + train_proportion)
        val_len = int((total_len - test_len) * val_proportion)
        val_len = 264
        train_len = total_len - val_len - test_len
    elif use_nasa_test_range is False:
        train_len = int(total_len * train_proportion)
        val_len = total_len - train_len
    else:
        train_len = int(total_len * train_proportion)
        val_len = int(total_len * val_proportion)
        test_len = total_len - train_len - val_len

    train = x_vals[0:train_len]
    train = torch.from_numpy(train).float()

    val = x_vals[train_len : (train_len + val_len)]
    val = torch.from_numpy(val).float()

    if not use_nasa_test_range:
        # if not using a test set, set it same as val to avoid error
        test = val
    else:
        test = x_vals[(train_len + val_len) :]
        test = torch.from_numpy(test).float()

    if x_marks is not None:
        train_marks = x_marks[0:train_len]
        val_marks = x_marks[train_len : (train_len + val_len)]
        test_marks = x_marks[(train_len + val_len) :]

        train_marks_window = to_windowed(train_marks, window_size, pred_size)
        val_marks_window = to_windowed(val_marks, window_size, pred_size)
        test_marks_window = to_windowed(test_marks, window_size, pred_size)

        train_marks_window = torch.from_numpy(train_marks_window).float()
        val_marks_window = torch.from_numpy(val_marks_window).float()
        test_marks_window = torch.from_numpy(test_marks_window).float()

        return (
            train,
            val,
            test,
            train_marks_window,
            val_marks_window,
            test_marks_window,
            scaler,
        )
    else:
        return train, val, test, scaler


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x):
        self.x = x
        # self.x_mark = x_mark

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx][0].view(-1, 1), self.x[idx][1].view(-1, 1))


def get_data_loaders(
    train_proportion=0.6,
    test_proportion=0.2,
    val_proportion=0.2,
    window_size=10,
    pred_size=1,
    batch_size=10,
    num_workers=1,
    pin_memory=True,
    test_mode=False,
    use_nasa_test_range=True,
):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(505)

    data_path = "/kaggle/input/sunspot-research/sunspot monthly recent.csv"
    try:
        sp = pd.read_csv(data_path, header=0, parse_dates=["date"])
    except Exception:
        sp = pd.read_csv(
            "../../data/sunspot monthly recent.csv", header=0, parse_dates=["date"]
        )

    target = sp["sn_cnt_mth"].values

    train_data, val_data, test_data, scaler = train_test_val_split(
        target,
        train_proportion=train_proportion,
        val_proportion=val_proportion,
        test_proportion=test_proportion,
        window_size=window_size,
        pred_size=pred_size,
        use_nasa_test_range=use_nasa_test_range,
    )
    if test_mode:
        train_val_data = torch.cat((train_data, val_data), 0)
        dataset_train_val, dataset_train, dataset_val, dataset_test = (
            CustomDataset(train_val_data),
            CustomDataset(train_data),
            CustomDataset(val_data),
            CustomDataset(test_data),
        )
        train_val_loader = torch.utils.data.DataLoader(
            dataset_train_val,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=8,
        )
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=1,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=8,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=1,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=8,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=1,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=8,
        )
        return train_val_loader, train_loader, val_loader, test_loader, scaler
    if not test_mode:
        dataset_train, dataset_test, dataset_val = (
            CustomDataset(train_data),
            CustomDataset(test_data),
            CustomDataset(val_data),
        )
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=128,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=128,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=batch_size,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=128,
        )

        return train_loader, val_loader, test_loader
