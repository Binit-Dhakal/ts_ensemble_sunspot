#!/bin/bash python
import torch 
from torch.utils import tensorboard
import torch.optim as optim
import time
import random
import os
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler, ASHAScheduler
from sklearn.metrics import mean_squared_error
from informer import Informer
from utils import *


def evaluate(model,data_loader,criterion,window_size,scaler):
    model.eval()    
    test_rollout = torch.Tensor(0)   
    test_result = torch.Tensor(0)  
    truth = torch.Tensor(0)
    total_loss = 0.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    with torch.no_grad():
        for i, (data,targets) in enumerate(data_loader):
            if i == 0:
                enc_in = data
                dec_in = targets
                test_rollout = targets
            else:
                enc_in = test_rollout[:,-window_size:,:]
                dec_in = torch.zeros([enc_in.shape[0], 1, enc_in.shape[-1]]).float()
                dec_in = torch.cat([enc_in[:,:(window_size-1),:], dec_in], dim=1).float()
                #dec_in = enc_in[:,:(window_size-1),:]
            enc_in, dec_in, targets = enc_in.to(device), dec_in.to(device), targets.to(device)
            output = model(enc_in, dec_in)

            total_loss += criterion(output[:,-1:,:], targets[:,-1:,:]).detach().cpu().numpy()

            test_rollout = torch.cat([test_rollout,output[:,-1:,:].detach().cpu()],dim = 1)
            test_result = torch.cat((test_result, output[:,-1,:].view(-1).detach().cpu()), 0)
            truth = torch.cat((truth, targets[:,-1,:].view(-1).detach().cpu()), 0)
    
    test_result = test_result.numpy()
    test_result = scaler.inverse_transform(test_result)
    truth = truth.numpy()
    truth = scaler.inverse_transform(truth)
    RMSE = mean_squared_error(truth, test_result)**0.5

    return total_loss, RMSE


def train(config, checkpoint_dir):
    torch.cuda.manual_seed(1008)
    torch.cuda.manual_seed_all(1008)  
    np.random.seed(1008)  
    random.seed(1008) 
    torch.manual_seed(1008)
    
    train_proportion = 0.6
    test_proportion = 0.2
    val_proportion = 0.2

    lr = config['lr']
    lr_decay = config['lr_decay']
    window_size = config['window_size']
    optim_step = config['optim_step']
    batch_size = config['batch_size']
    enc_in = 1
    dec_in = 1
    c_out = 1
    seq_len = config['window_size']
    label_len = config['window_size']-1
    out_len = 1 #config['window_size']
    factor = config['factor']
    d_model = config['d_model']
    n_heads = config['n_heads']
    e_layers = config['e_layers']
    d_layers = config['d_layers']
    d_ff = config['d_ff']
    dropout = config['dropout']


    model = Informer(enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor, d_model, n_heads, e_layers, d_layers, d_ff, 
                dropout, attn='prob', embed='fixed', freq='d', activation='gelu', 
                output_attention = False, distil=True,
                device=torch.device('cuda:0'))
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            print('Using multiple GPUs with data parallel')
            torch.distributed.init_process_group()
            model = nn.parallel.DistributedDataParallel(model)
    print(f'Using device : {device}')
    model.to(device)
    epochs = 150        
    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, optim_step, gamma=lr_decay)
    
    train_loader, _, val_loader, scaler = get_data_loaders(train_proportion, test_proportion, val_proportion,\
         window_size=window_size, pred_size =1, batch_size=batch_size, num_workers = 2, pin_memory = True, test_mode = True)

    assert device == "cuda:0"
    for epoch in range(1, epochs + 1):
        model.train() 
        total_loss = 0.

        for (data, targets) in train_loader:
            enc_in, dec_in = data.clone().to(device), targets.clone().to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            dec_zeros = torch.zeros([dec_in.shape[0], 1, dec_in.shape[-1]]).float().to(device)
            dec_in = torch.cat([dec_in[:,:(window_size-1),:], dec_zeros], dim=1).float().to(device)
            output = model(enc_in, dec_in)
            loss = criterion(output[:,-1,:], targets[:,-1,:])
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        val_loss, RMSE = evaluate(model, val_loader, criterion, window_size,scaler)
        tune.report(rmse=RMSE)
        scheduler.step() 


if __name__ == "__main__":
    config = {
        'd_model':tune.choice([128,216,512,1024]),
        'n_heads':tune.choice([2,4,8]),
        'e_layers':tune.choice([2,3,4,5]),
        'd_layers':tune.choice([2,3,4,5]),
        'd_ff':tune.choice([128,216,512,1024]),
        'window_size':tune.choice([192])
        'dropout':tune.choice([0.1,0.2]),
        'lr': tune.choice([1e-3,1e-4,1e-5,1e-6]),
        'optim_step': tune.choice([2,5,10,15,20]), 
        'lr_decay': tune.choice([0.95,0.9,0.85,0.8,0.75,0.7]),
        'factor': tune.choice([3,6,9]),
        'batch_size': tune.choice([16,32,64,128])
}
    ray.init(ignore_reinit_error=False, include_dashboard=True, dashboard_host= '0.0.0.0')
    sched = ASHAScheduler(
            max_t=80,
            grace_period=10,
            reduction_factor=2)
    analysis = tune.run(tune.with_parameters(train), config=config, num_samples=1000 ,metric='rmse', mode='min',\
         scheduler=sched, resources_per_trial=tune.PlacementGroupFactory([{"CPU": 12, "GPU": 0.5}]),max_concurrent_trials = 4, queue_trials = True, max_failures=200, local_dir="/scratch/yd1008/ray_results",)

    # analysis = tune.run(tune.with_parameters(train), config=config, metric='val_loss', mode='min',\
    #      scheduler=sched, resources_per_trial={"gpu": 0.5},max_concurrent_trials=6, max_failures=1000,queue_trials = True,local_dir="/scratch/yd1008/ray_results",)

    best_trail = analysis.get_best_config(mode='min')
    print('The best configs are: ',best_trail)
    ray.shutdown()
