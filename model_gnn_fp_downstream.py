import torch.nn as nn
import torch.nn.functional as F
import torch

from model_pathway import fpNet_w1_d1, fpNet_w1_d2, fpNet_w1_d3, fpNet_w2_d2, \
    fpNet_w2_d3, fpNet_w3_d2, fpNet_w3_d3
from model_pathway import GINNet, GATNet, GENNet, TransformerNet, LENet, GCNNet, \
    GraphNet, GeneralNet, SSGNet, SGNet
from model_gating import Gating, gumbel_topk, load_balance_loss
# from torch_geometric.nn import global_mean_pool as gmp
from torch_geometric.nn import global_add_pool as gmp


### Dual MoE
class DPMoE(nn.Module):
    def __init__(self, x_input, fp_input, edge_dim, output=1, p1_num=2, p2_num=2, DPMoE_variant='weighted_sum', hidden_dim=512, dropout=0.2):
        super(DPMoE, self).__init__()
        self.x_input = x_input
        self.fp_input = fp_input
        self.edge_dim = edge_dim
        self.output = output
        self.p1_num = p1_num
        self.p2_num = p2_num
        self.hidden_dim = hidden_dim
        self.dropout = dropout


        self.pathway1 = torch.nn.ModuleList()
        self.pathway1.append(GINNet(x_input, hidden_dim, edge_dim))
        self.pathway1.append(GATNet(x_input, hidden_dim, edge_dim))
        self.pathway1.append(GENNet(x_input, hidden_dim, edge_dim))
        self.pathway1.append(TransformerNet(x_input, hidden_dim, edge_dim))
        self.pathway1.append(LENet(x_input, hidden_dim, edge_dim))
        self.pathway1.append(GCNNet(x_input, hidden_dim, edge_dim))
        self.pathway1.append(GraphNet(x_input, hidden_dim, edge_dim))
        self.pathway1.append(GeneralNet(x_input, hidden_dim, edge_dim))
        self.pathway1.append(SSGNet(x_input, hidden_dim, edge_dim))
        self.pathway1.append(SGNet(x_input, hidden_dim, edge_dim))


        self.pathway2 = torch.nn.ModuleList()
        self.pathway2.append(fpNet_w1_d1(fp_input))
        self.pathway2.append(fpNet_w1_d2(fp_input))
        self.pathway2.append(fpNet_w1_d3(fp_input))
        self.pathway2.append(fpNet_w2_d2(fp_input))
        self.pathway2.append(fpNet_w2_d3(fp_input))
        self.pathway2.append(fpNet_w3_d2(fp_input))
        self.pathway2.append(fpNet_w3_d3(fp_input))


        self.gating1 = Gating(x_input, len(self.pathway1))
        self.gating2 = Gating(fp_input, len(self.pathway2))

        if DPMoE_variant == 'gumble_adaptive_lb_loss':
            self.lam1 = nn.Parameter(torch.tensor(0.01))
            self.lam2 = nn.Parameter(torch.tensor(0.01))


        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.pre = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output)
        )


    def forward_weighted_sum(self, data):
        x, edge_index, batch, y ,edge_attr, w = data.x, data.edge_index, data.batch, data.y, data.edge_attr, data.w


        path1_logits = self.gating1(gmp(x,batch))
        path1_logits_weighted = F.softmax(path1_logits, dim=1)
        path1_expert_features = [expert(x, edge_index, edge_attr, batch) for expert in self.pathway1]
        path1_expert_features = torch.stack(path1_expert_features, dim=1)
        path1_logits_weighted_unsqueeze = path1_logits_weighted.unsqueeze(-1)
        out1 = (path1_logits_weighted_unsqueeze * path1_expert_features).sum(dim=1)
        out1 = F.normalize(out1)


        pubchem = data.pubchem.reshape(data.num_graphs, -1)
        maccs = data.maccs.reshape(data.num_graphs, -1)
        erg = data.erg.reshape(data.num_graphs, -1)
        ecfp = data.ecfp.reshape(data.num_graphs, -1)
        fp_x = torch.cat((pubchem, maccs, erg, ecfp), dim=1)

        path2_logits = self.gating2(fp_x)
        path2_logits_weighted = F.softmax(path2_logits, dim=1)
        path2_expert_features = [expert(fp_x) for expert in self.pathway2]
        path2_expert_features = torch.stack(path2_expert_features, dim=1)
        path2_logits_weighted_unsqueeze = path2_logits_weighted.unsqueeze(-1)
        out2 = (path2_logits_weighted_unsqueeze * path2_expert_features).sum(dim=1)
        out2 = F.normalize(out2)


        alpha = torch.sigmoid(self.attention(torch.cat([out1, out2], dim=-1)))
        out = alpha * out1 + (1 - alpha) * out2
        out = self.pre(out)
        out = out.reshape(-1)

        non_999_indices = y != 999
        out_loss = out[non_999_indices]
        y_loss = y[non_999_indices]
        w_loss = w[non_999_indices]

        return out, y, w, out_loss, y_loss, w_loss


    def forward_gumble_lb_loss(self, data):
        x, edge_index, batch, y ,edge_attr, w = data.x, data.edge_index, data.batch, data.y, data.edge_attr, data.w


        path1_logits = self.gating1(gmp(x,batch))
        path1_logits_gumbel = gumbel_topk(path1_logits, k=self.p1_num)
        path1_expert_features = [expert(x, edge_index, edge_attr, batch) for expert in self.pathway1]
        path1_expert_features = torch.stack(path1_expert_features, dim=1)
        path1_logits_gumbel_unsqueeze = path1_logits_gumbel.unsqueeze(-1)
        out1 = (path1_logits_gumbel_unsqueeze * path1_expert_features).sum(dim=1)
        out1 = F.normalize(out1)


        pubchem = data.pubchem.reshape(data.num_graphs, -1)
        maccs = data.maccs.reshape(data.num_graphs, -1)
        erg = data.erg.reshape(data.num_graphs, -1)
        ecfp = data.ecfp.reshape(data.num_graphs, -1)
        fp_x = torch.cat((pubchem, maccs, erg, ecfp), dim=1)

        path2_logits = self.gating2(fp_x)
        path2_logits_gumbel = gumbel_topk(path2_logits, k=self.p2_num)
        path2_expert_features = [expert(fp_x) for expert in self.pathway2]
        path2_expert_features = torch.stack(path2_expert_features, dim=1)
        path2_logits_gumbel_unsqueeze = path2_logits_gumbel.unsqueeze(-1)
        out2 = (path2_logits_gumbel_unsqueeze * path2_expert_features).sum(dim=1)
        out2 = F.normalize(out2)


        alpha = torch.sigmoid(self.attention(torch.cat([out1, out2], dim=-1)))
        out = alpha * out1 + (1 - alpha) * out2
        out = self.pre(out)
        out = out.reshape(-1)

        non_999_indices = y != 999
        out_loss = out[non_999_indices]
        y_loss = y[non_999_indices]
        w_loss = w[non_999_indices]

        lb_loss_path1 = load_balance_loss(path1_logits, path1_logits_gumbel)
        lb_loss_path2 = load_balance_loss(path2_logits, path2_logits_gumbel)
        lb_loss = lb_loss_path1+lb_loss_path2

        return out, y, w, out_loss, y_loss, w_loss, lb_loss


    def forward_gumble_adaptive_lb_loss(self, data):
        x, edge_index, batch, y ,edge_attr, w = data.x, data.edge_index, data.batch, data.y, data.edge_attr, data.w


        path1_logits = self.gating1(gmp(x,batch))
        path1_logits_gumbel = gumbel_topk(path1_logits, k=self.p1_num)
        path1_expert_features = [expert(x, edge_index, edge_attr, batch) for expert in self.pathway1]
        path1_expert_features = torch.stack(path1_expert_features, dim=1)
        path1_logits_gumbel_unsqueeze = path1_logits_gumbel.unsqueeze(-1)
        out1 = (path1_logits_gumbel_unsqueeze * path1_expert_features).sum(dim=1)
        out1 = F.normalize(out1)


        pubchem = data.pubchem.reshape(data.num_graphs, -1)
        maccs = data.maccs.reshape(data.num_graphs, -1)
        erg = data.erg.reshape(data.num_graphs, -1)
        ecfp = data.ecfp.reshape(data.num_graphs, -1)
        fp_x = torch.cat((pubchem, maccs, erg, ecfp), dim=1)

        path2_logits = self.gating2(fp_x)
        path2_logits_gumbel = gumbel_topk(path2_logits, k=self.p2_num)
        path2_expert_features = [expert(fp_x) for expert in self.pathway2]
        path2_expert_features = torch.stack(path2_expert_features, dim=1)
        path2_logits_gumbel_unsqueeze = path2_logits_gumbel.unsqueeze(-1)
        out2 = (path2_logits_gumbel_unsqueeze * path2_expert_features).sum(dim=1)
        out2 = F.normalize(out2)


        alpha = torch.sigmoid(self.attention(torch.cat([out1, out2], dim=-1)))
        out = alpha * out1 + (1 - alpha) * out2
        out = self.pre(out)
        out = out.reshape(-1)

        non_999_indices = y != 999
        out_loss = out[non_999_indices]
        y_loss = y[non_999_indices]
        w_loss = w[non_999_indices]

        lb_loss_path1 = load_balance_loss(path1_logits, path1_logits_gumbel)
        lb_loss_path2 = load_balance_loss(path2_logits, path2_logits_gumbel)
        lb_loss = self.lam1*lb_loss_path1 + self.lam2*lb_loss_path2

        return out, y, w, out_loss, y_loss, w_loss, lb_loss
