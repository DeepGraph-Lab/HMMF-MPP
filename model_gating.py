import torch.nn as nn
import torch.nn.functional as F
import torch


class Gating(nn.Module):
    def __init__(self, input, output, hidden_dim=512, dropout=0.2, num_layers=2):
        super(Gating, self).__init__()
        self.input = input
        self.output = output
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.lins = torch.nn.ModuleList()
        if num_layers == 1:
            self.lins.append(torch.nn.Linear(input, hidden_dim))
        else:
            self.lins.append(torch.nn.Linear(input, hidden_dim))
            for _ in range(num_layers - 2):
                self.lins.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.lins.append(torch.nn.Linear(hidden_dim, output))

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, input_feature):
        x = input_feature
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lins[-1](x)
        return x


def gumbel_softmax_sample(logits, tau=1.0, eps=1e-10):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
    y = logits + gumbel_noise
    return F.softmax(y / tau, dim=-1)


def gumbel_topk(logits, k=1, tau=1.0):
    """ Gumbel-Softmax + Top-k + Straight-Through """
    y_soft = gumbel_softmax_sample(logits, tau=tau)
    # Obtain hard one-hot mask of top-k
    topk_idx = torch.topk(y_soft, k, dim=-1).indices
    y_hard = torch.zeros_like(y_soft).scatter_(-1, topk_idx, 1.0)
    # Straight-Through: forward hard, backward soft
    y = y_hard + (y_soft - y_soft.detach())
    return y


def load_balance_loss(gate_logits, gate_weights):
    """
    gate_logits: [batch, num_experts], origin gating logits
    gate_weights: [batch, num_experts], top-k Gumbel-Softmax output (hard/soft)
    """
    # 1. Theoretical probability: batch average of softmax logits
    probs = F.softmax(gate_logits, dim=-1).mean(0)  # [num_experts]

    # 2. Actual usage frequency: top-k gating is activated on average in batches
    usage = gate_weights.float().mean(0)  # [num_experts]

    # 3. Load balancing loss
    loss = (probs * usage).sum() * gate_logits.size(1)
    return loss
