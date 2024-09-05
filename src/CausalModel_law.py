import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist

from torch import nn
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.poutine.trace_messenger import TraceMessenger

pyro.enable_validation(True)
pyro.set_rng_seed(1)
pyro.enable_validation(True)

import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def quickprocess(var):
    if var is None:
        return var
    var = var.view(-1).to(device)
    return var

def to_onehot(var, num_classes=-1):
    var_onehot = F.one_hot(var, num_classes)
    dim = num_classes if num_classes != -1 else var_onehot.shape[1]
    return var_onehot, dim

def onehot_to_int(var):
    var_int = torch.argmax(var, dim=1)
    return var_int

class CausalModel_law(PyroModule):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.one_hot = 0

    def forward(self, data):
        dim_race = 1
        data_race, data_UGPA, data_LSAT, data_ZFYA = data['race'], data['UGPA'], data['LSAT'], data['ZFYA']
        data_race, data_UGPA, data_LSAT, data_ZFYA = quickprocess(data_race), quickprocess(data_UGPA), quickprocess(data_LSAT), quickprocess(data_ZFYA)
        if data_LSAT is not None:
            data_LSAT = torch.floor(data_LSAT)
        if self.one_hot:
            dim_race = 3

        self.pi = pyro.param(self.model_name + "_" + "pi", torch.tensor([0.4, 0.3, 0.3])).to(device)  # S~Cate(pi)

        self.b_g = pyro.param(self.model_name + "_" + "b_g", torch.tensor(0.)).to(device)
        self.w_g_k = pyro.param(self.model_name + "_" + "w_g_k", torch.tensor(0.)).to(device)
        self.w_g_r = pyro.param(self.model_name + "_" + "w_g_r", torch.zeros(dim_race, 1)).to(device)
        self.sigma_g = pyro.param(self.model_name + "_" + "sigma_g", torch.tensor(1.)).to(device)

        self.b_l = pyro.param(self.model_name + "_" + "b_l", torch.tensor(0.)).to(device)
        self.w_l_k = pyro.param(self.model_name + "_" + "w_l_k", torch.tensor(0.)).to(device)
        self.w_l_r = pyro.param(self.model_name + "_" + "w_l_r", torch.zeros(dim_race, 1)).to(device)

        self.w_f_k = pyro.param(self.model_name + "_" + "w_f_k", torch.tensor(0.)).to(device)
        self.w_f_r = pyro.param(self.model_name + "_" + "w_f_r", torch.zeros(dim_race, 1)).to(device)

        n = len(data_race)
        with pyro.plate('observe_data', size=n, device=device):
            knowledge = pyro.sample('knowledge', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).to(device)  # prior, n
            race = pyro.sample('obs_race', pyro.distributions.Categorical(self.pi), obs=data_race)  # S ~ Categorical(pi)
            race_out = race
            if self.one_hot:
                race_out, dim_race = to_onehot(data_race.long(), 3)
                race_out = race_out.float()
            gpa_mean = self.b_g + self.w_g_k * knowledge + (race_out.view(-1,dim_race) @ self.w_g_r).view(-1)
            sat_mean = torch.exp(self.b_l + self.w_l_k * knowledge + (race_out.view(-1,dim_race) @ self.w_l_r).view(-1))
            fya_mean = self.w_f_k * knowledge + (race_out.view(-1,dim_race) @ self.w_f_r).view(-1)

            gpa_obs = pyro.sample("obs_UGPA", dist.Normal(gpa_mean, torch.abs(self.sigma_g)), obs=data_UGPA)
            sat_obs = pyro.sample("obs_LSAT", dist.Poisson(sat_mean), obs=data_LSAT)
            fya_obs = pyro.sample("obs_ZFYA", dist.Normal(fya_mean, 1), obs=data_ZFYA)

        data_return = {'knowledge': knowledge, 'LSAT': sat_obs, 'ZFYA': fya_obs, 'UGPA': gpa_obs, 'race': race}
        return data_return
