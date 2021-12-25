import torch
import numpy as np

def create_short_edge(batch_size, station_num):
    edge = []
    for i in range(batch_size):
        for j in range(4):
            edge.append(torch.tensor([[i - j], [i]], dtype=torch.long))
    edge = torch.cat(edge, 1)
    edge = edge[:,edge[0] >= 0]

    edge_index_short = []
    for i in range(station_num):
        node = torch.arange(i, batch_size * station_num, station_num)
        edge_index_short.append(torch.stack([node[edge[0]], node[edge[1]]]))
    edge_index_short = torch.cat(edge_index_short, 1)
    
    return edge_index_short

def create_log_edge(batch_size, station_num):
    edge = []
    for i in range(1, batch_size + 1, 1):
        for j in range(int(np.floor(np.log2(i))) + 1):
            edge.append(
                torch.tensor(
                    [[i - pow(2, np.floor(np.log2(i) - j))], [i - 1]], dtype=torch.long
                )
            )
    edge = torch.cat(edge, 1)

    edge_index_log = []
    for i in range(station_num):
        node = torch.arange(i, batch_size * station_num, station_num)
        edge_index_log.append(torch.stack([node[edge[0]], node[edge[1]]]))
    edge_index_log = torch.cat(edge_index_log, 1)
    
    return edge_index_log

def create_seq_log_edge(b, n, seq_len):
    edge = []
    for i in range(1, seq_len + 1, 1):
        for j in range(int(np.floor(np.log2(i))) + 1):
            edge.append(
                torch.tensor(
                    [[i - pow(2, np.floor(np.log2(i) - j))], [i - 1]], dtype=torch.long
                )
            )
    edge = torch.cat(edge, 1)

    edge_index = []
    for i in range(b*n):
        node = torch.arange(i, seq_len*b*n, b*n)
        edge_index.append(torch.stack([node[edge[0]], node[edge[1]]]))
    edge_index = torch.cat(edge_index, 1)
    return edge_index