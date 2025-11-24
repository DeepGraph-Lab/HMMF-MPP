import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GATConv
# from torch_geometric.nn import global_mean_pool as gmp, global_add_pool as gap
from torch_geometric.nn import global_add_pool as gmp
from torch_geometric.nn import Sequential
from torch.nn import Linear

import torch.nn as nn
from torch_geometric.nn import GENConv, TransformerConv, LEConv, GCNConv, \
    GraphConv, GeneralConv, SSGConv, SGConv


class fpNet_w1_d1(nn.Module):
    def __init__(self, fp_input, hidden_dim=512, dropout=0.2):
        super(fpNet_w1_d1, self).__init__()
        self.fp_input = fp_input
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.fp_predict = nn.Sequential(
            nn.Linear(fp_input, hidden_dim)
        )

    def forward(self, fp_x):
        fp_feature = self.fp_predict(fp_x)
        return fp_feature


class fpNet_w1_d2(nn.Module):
    def __init__(self, fp_input, hidden_dim=512, dropout=0.2):
        super(fpNet_w1_d2, self).__init__()
        self.fp_input = fp_input
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.fp_predict = nn.Sequential(
            nn.Linear(fp_input, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, fp_x):
        fp_feature = self.fp_predict(fp_x)
        return fp_feature


class fpNet_w1_d3(nn.Module):
    def __init__(self, fp_input, hidden_dim=512, dropout=0.2):
        super(fpNet_w1_d3, self).__init__()
        self.fp_input = fp_input
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.fp_predict = nn.Sequential(
            nn.Linear(fp_input, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, fp_x):
        fp_feature = self.fp_predict(fp_x)
        return fp_feature


class fpNet_w2_d2(nn.Module):
    def __init__(self, fp_input, hidden_dim=512, dropout=0.2):
        super(fpNet_w2_d2, self).__init__()
        self.fp_input = fp_input
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.fp_predict = nn.Sequential(
            nn.Linear(fp_input, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, fp_x):
        fp_feature = self.fp_predict(fp_x)
        return fp_feature


class fpNet_w2_d3(nn.Module):
    def __init__(self, fp_input, hidden_dim=512, dropout=0.2):
        super(fpNet_w2_d3, self).__init__()
        self.fp_input = fp_input
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.fp_predict = nn.Sequential(
            nn.Linear(fp_input, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, fp_x):
        fp_feature = self.fp_predict(fp_x)
        return fp_feature


class fpNet_w3_d2(nn.Module):
    def __init__(self, fp_input, hidden_dim=512, dropout=0.2):
        super(fpNet_w3_d2, self).__init__()
        self.fp_input = fp_input
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.fp_predict = nn.Sequential(
            nn.Linear(fp_input, hidden_dim * 3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 3),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim)
        )

    def forward(self, fp_x):
        fp_feature = self.fp_predict(fp_x)
        return fp_feature


class fpNet_w3_d3(nn.Module):
    def __init__(self, fp_input, hidden_dim=512, dropout=0.2):
        super(fpNet_w3_d3, self).__init__()
        self.fp_input = fp_input
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.fp_predict = nn.Sequential(
            nn.Linear(fp_input, hidden_dim * 3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 3),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim * 3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 3),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim)
        )

    def forward(self, fp_x):
        fp_feature = self.fp_predict(fp_x)
        return fp_feature


class GINNet(torch.nn.Module):
    def __init__(self, x_input, hidden_dim, edge_dim, eps=0., train_eps=True):
        super(GINNet, self).__init__()

        self.x_input = x_input
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        self.conv1 = GINEConv(nn=Linear(x_input, hidden_dim), eps=eps, train_eps=train_eps)
        self.conv2 = GINEConv(nn=Linear(hidden_dim, hidden_dim * 2), eps=eps, train_eps=train_eps)
        self.conv3 = GINEConv(nn=Linear(hidden_dim * 2, hidden_dim), eps=eps, train_eps=train_eps)

        self.fc1 = Linear(edge_dim, x_input)
        self.fc2 = Linear(edge_dim, hidden_dim)
        self.fc3 = Linear(edge_dim, hidden_dim * 2)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr1 = self.fc1(edge_attr)
        edge_attr2 = self.fc2(edge_attr)
        edge_attr3 = self.fc3(edge_attr)

        x = self.conv1(x, edge_index, edge_attr1)
        x = F.elu(x)

        x = self.conv2(x, edge_index, edge_attr2)
        x = F.elu(x)

        x = self.conv3(x, edge_index, edge_attr3)
        x = F.elu(x)

        x_mean = gmp(x, batch)

        return x_mean


class GATNet(torch.nn.Module):
    def __init__(self, x_input, hidden_dim, edge_dim, heads=4):
        super(GATNet, self).__init__()
        self.x_input = x_input
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.heads = heads

        self.conv1 = GATConv(x_input, hidden_dim, heads=heads, edge_dim=edge_dim, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim * 2, heads=heads, edge_dim=edge_dim, concat=False)
        self.conv3 = GATConv(hidden_dim * 2, hidden_dim, heads=heads, edge_dim=edge_dim, concat=False)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = F.elu(x)

        x_mean = gmp(x, batch)

        return x_mean


class GENNet(torch.nn.Module):
    def __init__(self, x_input, hidden_dim, edge_dim):
        super(GENNet, self).__init__()
        self.x_input = x_input
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        self.conv1 = GENConv(x_input, hidden_dim, edge_dim=edge_dim)
        self.conv2 = GENConv(hidden_dim, hidden_dim * 2, edge_dim=edge_dim)
        self.conv3 = GENConv(hidden_dim * 2, hidden_dim, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = F.elu(x)

        x_mean = gmp(x, batch)

        return x_mean


class TransformerNet(torch.nn.Module):
    def __init__(self, x_input, hidden_dim, edge_dim, heads=4):
        super(TransformerNet, self).__init__()
        self.x_input = x_input
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.heads = heads

        self.conv1 = TransformerConv(x_input, hidden_dim, heads=heads, edge_dim=edge_dim, concat=False)
        self.conv2 = TransformerConv(hidden_dim, hidden_dim * 2, heads=heads, edge_dim=edge_dim, concat=False)
        self.conv3 = TransformerConv(hidden_dim * 2, hidden_dim, heads=heads, edge_dim=edge_dim, concat=False)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = F.elu(x)

        x_mean = gmp(x, batch)

        return x_mean


class LENet(torch.nn.Module):
    def __init__(self, x_input, hidden_dim, edge_dim):
        super(LENet, self).__init__()
        self.x_input = x_input
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        self.conv1 = LEConv(x_input, hidden_dim)
        self.conv2 = LEConv(hidden_dim, hidden_dim * 2)
        self.conv3 = LEConv(hidden_dim * 2, hidden_dim)

        self.fc1 = Linear(edge_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr = self.fc1(edge_attr).squeeze(-1)

        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = F.elu(x)

        x_mean = gmp(x, batch)

        return x_mean


class GCNNet(torch.nn.Module):
    def __init__(self, x_input, hidden_dim, edge_dim):
        super(GCNNet, self).__init__()
        self.x_input = x_input
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        self.conv1 = GCNConv(x_input, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.conv3 = GCNConv(hidden_dim * 2, hidden_dim)

        self.fc1 = Linear(edge_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr = self.fc1(edge_attr).squeeze(-1)

        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = F.elu(x)

        x_mean = gmp(x, batch)

        return x_mean


class GraphNet(torch.nn.Module):
    def __init__(self, x_input, hidden_dim, edge_dim):
        super(GraphNet, self).__init__()
        self.x_input = x_input
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        self.conv1 = GraphConv(x_input, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim * 2)
        self.conv3 = GraphConv(hidden_dim * 2, hidden_dim)

        self.fc1 = Linear(edge_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr = self.fc1(edge_attr).squeeze(-1)

        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = F.elu(x)

        x_mean = gmp(x, batch)

        return x_mean


class GeneralNet(torch.nn.Module):
    def __init__(self, x_input, hidden_dim, edge_dim):
        super(GeneralNet, self).__init__()
        self.x_input = x_input
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        self.conv1 = GeneralConv(x_input, hidden_dim, in_edge_channels=edge_dim)
        self.conv2 = GeneralConv(hidden_dim, hidden_dim * 2, in_edge_channels=edge_dim)
        self.conv3 = GeneralConv(hidden_dim * 2, hidden_dim, in_edge_channels=edge_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = F.elu(x)

        x_mean = gmp(x, batch)

        return x_mean


class SSGNet(torch.nn.Module):
    def __init__(self, x_input, hidden_dim, edge_dim):
        super(SSGNet, self).__init__()
        self.x_input = x_input
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        self.conv1 = SSGConv(x_input, hidden_dim, alpha=0.1)
        self.conv2 = SSGConv(hidden_dim, hidden_dim * 2, alpha=0.1)
        self.conv3 = SSGConv(hidden_dim * 2, hidden_dim, alpha=0.1)

        self.fc1 = Linear(edge_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr = self.fc1(edge_attr).squeeze(-1)

        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = F.elu(x)

        x_mean = gmp(x, batch)

        return x_mean


class SGNet(torch.nn.Module):
    def __init__(self, x_input, hidden_dim, edge_dim):
        super(SGNet, self).__init__()
        self.x_input = x_input
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        self.conv1 = SGConv(x_input, hidden_dim)
        self.conv2 = SGConv(hidden_dim, hidden_dim * 2)
        self.conv3 = SGConv(hidden_dim * 2, hidden_dim)

        self.fc1 = Linear(edge_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr = self.fc1(edge_attr).squeeze(-1)

        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)

        x = self.conv3(x, edge_index, edge_attr)
        x = F.elu(x)

        x_mean = gmp(x, batch)

        return x_mean

