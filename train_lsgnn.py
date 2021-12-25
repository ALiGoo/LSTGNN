import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import *

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from model.lsgnn import LSGNN
from model.lsgnn_simple import LSGNN

log_path = "logs/lsgnn_simple"
# batch_size = 40 * 24 # lsgnn
batch_size = 55 * 24 # lsgnn simple

# load data
loader = joblib.load("process_data/loader_lsgnn_simple.pt")
station_info = pd.read_hdf('process_data/features_201307_201709.h5', key='info')
graph_in = loader['graph_in']
graph_out = loader['graph_out']

training_loader = DataLoader(loader['training_loader'], batch_size=batch_size, num_workers=16)
validation_loader = DataLoader(loader['validation_loader'], batch_size=batch_size, num_workers=16)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# in_channels = 8 # lsgnn
in_channels = 16 # lsgnn simple
out_channels = 64
# additinal_channels=57 # lsgnn
additinal_channels=45 # lsgnn simple
# num_layers = int(np.floor(np.log2(batch_size)))
num_layers = 9
station_num = len(station_info)
learning_rate = 1e-3
edge_index_short = create_short_edge(batch_size, station_num)
edge_index_log = create_log_edge(batch_size, station_num)

"""
建立model與optimizer與loss function
"""
edge_index_log = edge_index_log.to(device)
edge_index_short = edge_index_short.to(device)
graph_in = graph_in.to(device)
graph_out = graph_out.to(device)
model = LSGNN(
    in_channels=in_channels,
    out_channels=out_channels,
    additinal_channels=additinal_channels,
    num_layers=num_layers,
    station_num=station_num,
    edge_index_short=edge_index_short,
    edge_index_log=edge_index_log,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
writer = SummaryWriter(log_path)

# Load Weight
# model.load_state_dict(torch.load("data/model/epoch100.pt", map_location=device))

# Training
for epoch in tqdm(range(1, 100 + 1, 1)):
    mean_loss = {
        "train_rmse": [],
        "train_rmsle": [],
        "train_len": 0,
        "val_rmse": [],
        "val_rmsle": [],
        "val_len": 0,
    }

    # training
    model.train()
    for inputs in tqdm(training_loader):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        inputs.x = model(inputs, graph_in, graph_out)
        inputs.y = inputs.y[inputs.is_alive == 1]

        # backward
        loss = torch.sqrt(criterion(inputs.x, inputs.y))
        loss.backward()
        optimizer.step()

        # Store Loss
        mean_loss["train_rmse"].append(
            torch.sqrt(criterion(inputs.x, inputs.y)).item() * inputs.y.size(0)
        )
        mean_loss["train_rmsle"].append(
            torch.sqrt(criterion(torch.log1p(inputs.x), torch.log1p(inputs.y))).item()
            * inputs.y.size(0)
        )
        mean_loss["train_len"] += inputs.y.size(0)

    # validation
    model.eval()
    with torch.no_grad():
        for inputs in tqdm(validation_loader):
            inputs = inputs.to(device)
            inputs.x = model(inputs, graph_in, graph_out)
            inputs.y = inputs.y[inputs.is_alive == 1]

            # Store Loss
            mean_loss["val_rmse"].append(
                torch.sqrt(criterion(inputs.x, inputs.y)).item() * inputs.y.size(0)
            )
            mean_loss["val_rmsle"].append(
                torch.sqrt(
                    criterion(torch.log1p(inputs.x), torch.log1p(inputs.y))
                ).item()
                * inputs.y.size(0)
            )
            mean_loss["val_len"] += inputs.y.size(0)

    train_rmse = np.sum(np.array(mean_loss["train_rmse"])) / mean_loss["train_len"]
    train_rmsle = np.sum(np.array(mean_loss["train_rmsle"])) / mean_loss["train_len"]
    val_rmse = np.sum(np.array(mean_loss["val_rmse"])) / mean_loss["val_len"]
    val_rmsle = np.sum(np.array(mean_loss["val_rmsle"])) / mean_loss["val_len"]

    writer.add_scalar("RMSE/Train", train_rmse, epoch)
    writer.add_scalar("RMSE/Validation", val_rmse, epoch)
    writer.add_scalar("RMSLE/Train", train_rmsle, epoch)
    writer.add_scalar("RMSLE/Validation", val_rmsle, epoch)
    torch.save(model.state_dict(), f"{log_path}/checkpoint_epoch{epoch}.pt")