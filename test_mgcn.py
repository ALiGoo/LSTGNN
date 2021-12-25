import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import *

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.mgcn import MGCN

batch_size = 100 * 24

# load data
loader = joblib.load("process_data/loader_mgcn.pt")
graph = loader['graph']
station_info = pd.read_hdf('process_data/features_201307_201709.h5', key='info')

testing_loader = DataLoader(loader['testing_loader'], batch_size=batch_size, num_workers=16)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hidden_channels = 64
additinal_channels = 57
seq_len = 6
station_num = len(station_info)
graph = graph.to(device)


"""
建立model與optimizer與loss function
"""
model = MGCN(
    hidden_channels = hidden_channels,
    additinal_channels = additinal_channels,
    seq_len = seq_len,
    station_num = station_num,
).to(device)

# Load Weight
model.load_state_dict(torch.load("logs/mgcn/finetune/checkpoint_epoch100.pt", map_location=device))

predicts = []
model.eval()
with torch.no_grad():
    for inputs in tqdm(testing_loader):
        inputs = inputs.to(device)
        inputs.x = model(inputs, graph, pre_trained=False)
        inputs.x = inputs.x[inputs.is_alive == 1]
        predicts.append(inputs.x.to('cpu').numpy())
predicts = np.concatenate(predicts, axis=0)
np.save("logs/mgcn/predicts.npy", predicts)