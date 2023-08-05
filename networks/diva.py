import argparse

import os
import time
import random
import numpy as np
import sklearn.metrics as metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datetime import timedelta

from torch.nn import functional as F
import torch.distributions as dist


# if __package__ is None or __package__ == '':
#     # uses current directory visibility
#     from utils import init_weights, makedir, paint, AverageMeter
#     from datasets import SensorDataset
# else:
#     # uses current package visibility
#     from .utils import init_weights, makedir, paint, AverageMeter
#     from .datasets import SensorDataset


# Decoders
class px(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(px, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(zd_dim + zx_dim + zy_dim, x_dim, bias=False), nn.BatchNorm1d(x_dim))

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        self.fc1[1].weight.data.fill_(1)
        self.fc1[1].bias.data.zero_()

    def forward(self, zd, zx, zy):
        zdzxzy = torch.cat((zd, zx, zy), dim=1)
        decoded_x = self.fc1(zdzxzy)

        return decoded_x

class pzd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(pzd, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(d_dim, zd_dim, bias=False), nn.BatchNorm1d(zd_dim), nn.LeakyReLU())
        self.fc21 = nn.Sequential(nn.Linear(zd_dim, zd_dim))
        self.fc22 = nn.Sequential(nn.Linear(zd_dim, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        self.fc1[1].weight.data.fill_(1)
        self.fc1[1].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, d):
        hidden = self.fc1(d)
        zd_loc = self.fc21(hidden)
        zd_scale = self.fc22(hidden) + 1e-7

        return zd_loc, zd_scale

class pzy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(pzy, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(y_dim, zy_dim, bias=False), nn.BatchNorm1d(zy_dim), nn.LeakyReLU())
        self.fc21 = nn.Sequential(nn.Linear(zy_dim, zy_dim))
        self.fc22 = nn.Sequential(nn.Linear(zy_dim, zy_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        self.fc1[1].weight.data.fill_(1)
        self.fc1[1].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, y):
        hidden = self.fc1(y)
        zy_loc = self.fc21(hidden)
        zy_scale = self.fc22(hidden) + 1e-7

        return zy_loc, zy_scale

# Encoders
class qzd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qzd, self).__init__()

        self.fc1 = nn.Linear(in_features=x_dim, out_features=zd_dim)
        self.bn1 = nn.BatchNorm1d(zd_dim, affine=True)
        # self.in1 = nn.InstanceNorm2d(zd_dim, affine=True)
        self.act = nn.LeakyReLU()

        # input shape needs to be calculated
        self.fc11 = nn.Sequential(nn.Linear(zd_dim, zd_dim))
        self.fc12 = nn.Sequential(nn.Linear(zd_dim, zd_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.weight.data.fill_(1)
        self.fc1.bias.data.zero_()
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
        # self.in1.weight.data.fill_(1)
        # self.in1.bias.data.zero_()

        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        # import pdb; pdb.set_trace()
        h = self.fc1(x)
        h_bn = self.bn1(h)
        # h_in = self.in1(h)
        # h = torch.cat((h_bn, h_in), dim=1)
        h = self.act(h_bn)

        zd_loc = self.fc11(h)
        zd_scale = self.fc12(h) + 1e-7

        return zd_loc, zd_scale


class qzx(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qzx, self).__init__()

        self.fc1 = nn.Linear(in_features=x_dim, out_features=zx_dim)
        self.bn1 = nn.BatchNorm1d(zx_dim, affine=True)
        # self.in1 = nn.InstanceNorm1d(zx_dim, affine=True)
        self.act = nn.LeakyReLU()

        # input shape needs to be calculated
        self.fc11 = nn.Sequential(nn.Linear(zx_dim, zx_dim))
        self.fc12 = nn.Sequential(nn.Linear(zx_dim, zx_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.weight.data.fill_(1)
        self.fc1.bias.data.zero_()
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
        # self.in1.weight.data.fill_(1)
        # self.in1.bias.data.zero_()

        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        h = self.fc1(x)
        h_bn = self.bn1(h)
        # h_in = self.in1(h)
        # h = torch.cat((h_bn, h_in), dim=1)
        h = self.act(h_bn)

        zx_loc = self.fc11(h)
        zx_scale = self.fc12(h) + 1e-7

        return zx_loc, zx_scale

class qzy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qzy, self).__init__()

        self.fc1 = nn.Linear(in_features=x_dim, out_features=zy_dim)
        self.bn1 = nn.BatchNorm1d(zy_dim, affine=True)
        # self.in1 = nn.InstanceNorm1d(zy_dim, affine=True)
        self.act = nn.LeakyReLU()

        # input shape needs to be calculated
        self.fc11 = nn.Sequential(nn.Linear(zy_dim, zy_dim))
        self.fc12 = nn.Sequential(nn.Linear(zy_dim, zy_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.weight.data.fill_(1)
        self.fc1.bias.data.zero_()
        self.bn1.weight.data.fill_(1)
        self.bn1.bias.data.zero_()
        # self.in1.weight.data.fill_(1)
        # self.in1.bias.data.zero_()

        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()

    def forward(self, x):
        h = self.fc1(x)
        h_bn = self.bn1(h)
        # h_in = self.in1(h)
        # h = torch.cat((h_bn, h_in), dim=1)
        h = self.act(h_bn)

        zy_loc = self.fc11(h)
        zy_scale = self.fc12(h) + 1e-7

        return zy_loc, zy_scale

# Auxiliary tasks
class qd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qd, self).__init__()

        self.fc1 = nn.Linear(zd_dim, d_dim)
        self.activation = nn.LeakyReLU()

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zd):
        h = self.activation(zd)
        loc_d = self.fc1(h)

        return loc_d


class qy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qy, self).__init__()

        self.fc1 = nn.Linear(zy_dim, y_dim)
        self.activation = nn.LeakyReLU()

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.zero_()

    def forward(self, zy):
        h = self.activation(zy)
        loc_y = self.fc1(h)

        return loc_y

class DIVA(nn.Module):
    def __init__(self, args):
        super(DIVA, self).__init__()
        self.device = args.device

        # self.model = args.model
        # self.dataset = args.dataset
        # self.experiment = args.experiment

        # makedir(self.path_checkpoints)
        # makedir(self.path_logs)
        # makedir(self.path_visuals)

        ## original DIVA components
        self.zd_dim = args.zd_dim
        self.zx_dim = args.zx_dim
        self.zy_dim = args.zy_dim
        self.d_dim = args.num_domains
        self.x_dim = args.x_dim
        self.y_dim = args.num_classes

        # decoders
        self.px = px(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.pzd = pzd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.pzy = pzy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        # encoders
        self.qzd = qzd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.qzx = qzx(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.qzy = qzy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        # auxiliary classifier for d & y
        self.qd = qd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.qy = qy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        self.aux_loss_multiplier_d = args.aux_loss_multiplier_d
        self.aux_loss_multiplier_y = args.aux_loss_multiplier_y

        self.beta_d = args.beta_d
        self.beta_x = args.beta_x
        self.beta_y = args.beta_y

        if self.device == torch.device('cuda') or self.device == torch.device('cuda:0') or self.device == torch.device('cuda:1'):
            self.cuda(device=self.device)
        else:
            self.cpu()

    def forward(self, d, x, y):
        ## original DIVA components
        # Encode
        zd_q_loc, zd_q_scale = self.qzd(x)
        zx_q_loc, zx_q_scale = self.qzx(x)
        zy_q_loc, zy_q_scale = self.qzy(x)

        # Reparameterization trick
        qzd = dist.Normal(zd_q_loc, zd_q_scale)
        zd_q = qzd.rsample()
        qzx = dist.Normal(zx_q_loc, zx_q_scale)
        zx_q = qzx.rsample()
        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        zy_q = qzy.rsample()

        # Decode
        x_recon = self.px(zd_q, zx_q, zy_q)

        # Prior
        zd_p_loc, zd_p_scale = self.pzd(d)

        if self.device == torch.device('cuda') or self.device == torch.device('cuda:0') or self.device == torch.device('cuda:1'):
            zx_p_loc, zx_p_scale = torch.zeros(zd_p_loc.size()[0], zd_p_loc.size()[1]).cuda(device=self.device),\
                                   torch.ones(zd_p_loc.size()[0], zd_p_loc.size()[1]).cuda(device=self.device)
        else:
            zx_p_loc, zx_p_scale = torch.zeros(zd_p_loc.size()[0], zd_p_loc.size()[1]).cpu(),\
                                   torch.ones(zd_p_loc.size()[0], zd_p_loc.size()[1]).cpu()

        zy_p_loc, zy_p_scale = self.pzy(y)

        # Reparameterization trick
        pzd = dist.Normal(zd_p_loc, zd_p_scale)
        pzx = dist.Normal(zx_p_loc, zx_p_scale)
        pzy = dist.Normal(zy_p_loc, zy_p_scale)

        # Auxiliary losses
        d_hat = self.qd(zd_q)
        y_hat = self.qy(zy_q)

        return x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q

    def loss_function(self, d, x, y):
        x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q = self.forward(d, x, y)

        CE_x = F.mse_loss(x, x_recon, reduction='sum')

        zd_p_minus_zd_q = torch.sum(pzd.log_prob(zd_q) - qzd.log_prob(zd_q))
        KL_zx = torch.sum(pzx.log_prob(zx_q) - qzx.log_prob(zx_q))
        zy_p_minus_zy_q = torch.sum(pzy.log_prob(zy_q) - qzy.log_prob(zy_q))

        _, d_target = d.max(dim=1)

        CE_d = F.cross_entropy(d_hat, d_target.type(torch.int64), reduction='sum')

        _, y_target = y.max(dim=1)
        CE_y = F.cross_entropy(y_hat, y_target.type(torch.int64), reduction='sum')

        # reconstruction loss (zx, zd, zy) + prior_zd loss + prior_zx loss + prior_zy loss + domain_loss + class_loss
        loss_invariant = CE_x \
                         - self.beta_d * zd_p_minus_zd_q \
                         - self.beta_x * KL_zx \
                         - self.beta_y * zy_p_minus_zy_q \
                         + self.aux_loss_multiplier_d * CE_d \
                         + self.aux_loss_multiplier_y * CE_y

        return loss_invariant, \
               CE_x, self.aux_loss_multiplier_d * CE_d, self.aux_loss_multiplier_y * CE_y, \
               - self.beta_d * zd_p_minus_zd_q, - self.beta_x * KL_zx, - self.beta_y * zy_p_minus_zy_q

    def get_zy(self, x):
        with torch.no_grad():
            zy_q_loc, _ = self.qzy.forward(x)
            zy = zy_q_loc
        return zy

    def classifier(self, x):
        """
        classify an image (or a batch of images)
        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        with torch.no_grad():
            zd_q_loc, _ = self.qzd.forward(x)
            zd = zd_q_loc
            alpha_d = F.softmax(self.qd(zd), dim=1)
            d = alpha_d.argmax(dim=1, keepdim=True)

            zy_q_loc, _ = self.qzy.forward(x)
            zy = zy_q_loc
            alpha_y = F.softmax(self.qy(zy), dim=1)
            y = alpha_y.argmax(dim=1, keepdim=True)

        return d, y, alpha_d, alpha_y

    # @property
    # def path_checkpoints(self):
    #     return f"./models/{self.model}/{self.dataset}/{self.experiment}/checkpoints/"

    # @property
    # def path_logs(self):
    #     return f"./models/{self.model}/{self.dataset}/{self.experiment}/logs/"

    # @property
    # def path_visuals(self):
    #     return f"./models/{self.model}/{self.dataset}/{self.experiment}/visuals/"