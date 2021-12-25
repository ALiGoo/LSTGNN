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

batch_size = 120 * 24

# load data
loader = joblib.load("process_data/loader_lsgnn_simple.pt")
station_info = pd.read_hdf('process_data/features_201307_201709.h5', key='info')
graph_in = loader['graph_in']
graph_out = loader['graph_out']

testing_loader = DataLoader(loader['testing_loader'], batch_size=batch_size, num_workers=16)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# in_channels = 8 # lsgnn
in_channels = 16 # lsgnn simple
out_channels = 64
# additinal_channels=57 # lsgnn
additinal_channels=45 # lsgnn simple
# num_layers = int(np.floor(np.log2(batch_size)))
num_layers = 9
station_num = len(station_info)
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

# Load Weight
model.load_state_dict(torch.load("logs/lsgnn_simple/checkpoint_epoch66.pt", map_location=device))

predicts = []
model.eval()
with torch.no_grad():
    for inputs in tqdm(testing_loader):
        inputs = inputs.to(device)
        inputs.x = model(inputs, graph_in, graph_out)
        predicts.append(inputs.x.to('cpu').numpy())
predicts = np.concatenate(predicts, axis=0)
np.save("logs/lsgnn_simple/predicts.npy", predicts)