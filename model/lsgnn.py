import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import glorot

class MultiGraphConv(nn.Module):
    def __init__(self, nodes, channels, num_graphs):
        super(MultiGraphConv, self).__init__()
        self.weight_fusion = nn.Parameter(torch.Tensor(num_graphs, nodes, nodes))
        self.weight = nn.Parameter(torch.Tensor(nodes, nodes))
        self.bias = nn.Parameter(torch.zeros(channels))
    
        self.reset_parameters()
    def reset_parameters(self):
        glorot(self.weight_fusion)
        glorot(self.weight)
    
    def forward(self, x, graph):
        graph = (graph * self.weight_fusion).sum(dim=0)
        x = (graph @ self.weight @ x) + self.bias
        return x


# Short Term Convolution
class STConv(nn.Module):
    def __init__(self, in_channels, out_channels, station_num, num_graphs,downsample=False):
        super(STConv, self).__init__()

        self.conv_in = MultiGraphConv(station_num, in_channels, num_graphs)
        self.conv_out = MultiGraphConv(station_num, in_channels, num_graphs)
        self.conv_temporal = GCNConv(in_channels*2, out_channels*2)
        self.residual = None
        if downsample:
            self.residual = nn.Linear(in_channels, out_channels)
    

    def forward(self, x, graph_in, graph_out, edge_index_short):
        residual = x
        x = F.relu(torch.cat([self.conv_in(x, graph_in), self.conv_out(x, graph_out)], axis=-1), inplace=True)
        b, n, c = x.size()
        x = F.glu(self.conv_temporal(x.reshape(-1, c), edge_index_short), dim=-1).reshape(b, n, -1)
        # Residual
        if self.residual:
            b, n, c = residual.size()
            residual = F.relu(self.residual(residual.reshape(-1,c)), inplace=True).reshape(b, n, -1)
        x = x + residual

        return x

## Long Term Convolution
class LTConv(nn.Module):
    def __init__(self, channels, num_layers):
        super(LTConv, self).__init__()

        self.conv = nn.ModuleList([GCNConv(channels, channels*2) for i in range(num_layers)])

    def forward(self, x, edge_index):
        for m in self.conv:
            residual = x
            x = F.glu(m(x, edge_index), dim=-1)
            x = x + residual
        return x

## Long Short Term GNN

class LSGNN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        additinal_channels,
        num_layers,
        station_num,
        edge_index_short,
        edge_index_log,
        num_graphs=7,
    ):
        super(LSGNN, self).__init__()

        #hyperparameter
        self.in_channels = in_channels

        # edge
        self.edge_index_short = edge_index_short
        self.edge_index_log = edge_index_log

        # layer
        self.stconv_now = nn.ModuleList(
            [
                STConv(in_channels, out_channels, station_num, num_graphs, downsample=True),
                STConv(out_channels, out_channels, station_num, num_graphs),
            ]
        )
        self.stconv_period = nn.ModuleList(
            [
                STConv(in_channels*2, out_channels, station_num, num_graphs, downsample=True),
                STConv(out_channels, out_channels, station_num, num_graphs),
            ]
        )
        self.stconv_trend = nn.ModuleList(
            [
                STConv(in_channels, out_channels, station_num, num_graphs, downsample=True),
                STConv(out_channels, out_channels, station_num, num_graphs),
            ]
        )
        self.fusion = nn.Linear(out_channels*3,out_channels)
        self.sgconv = LTConv(out_channels, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(out_channels + additinal_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),
            nn.ReLU(inplace=True),
        )


    def forward(self, inputs, graph_in, graph_out,inference=True):
        # Spatial Temporal Convolution
        x, x_period, x_trend = (
            inputs.x[:, :, : self.in_channels],
            inputs.x[:, :, self.in_channels : -self.in_channels],
            inputs.x[:, :, -self.in_channels:],
        )
        edge_index = self.edge_index_short[:, self.edge_index_short[1] < x.size(0) * x.size(1)]
        for m_now, m_period, m_trend in zip(
            self.stconv_now, self.stconv_period, self.stconv_trend
        ):
            x = m_now(x, graph_in, graph_out, edge_index)
            x_period = m_period(x_period, graph_in, graph_out, edge_index)
            x_trend = m_trend(x_trend, graph_in, graph_out, edge_index)


        # Fusion
        x = torch.cat([x, x_period, x_trend], dim=-1)
        _, _, c = x.size()
        x = F.relu(self.fusion(x.reshape(-1,c)), inplace=True)


        # Check if the last batch
        edge_index = self.edge_index_log[:, self.edge_index_log[1] < x.size(0)]

        # log sparse convolution
        x = self.sgconv(x, edge_index)

        # Inference
        if inference:
            x = torch.cat([x, inputs.x_fc], -1)
            x = x[inputs.is_alive == 1]
            x = self.fc(x)

        return x