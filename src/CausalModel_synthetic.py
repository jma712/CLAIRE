import torch
import numpy as np

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

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

import math

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def quickprocess(var):
    if var is None:
        return var
    var = var.view(-1).to(device)
    return var

class CausalModel_synthetic(PyroModule):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def forward_true(self, data):
        data_s, data_x0, data_x1, data_x2, data_y = data['S'], data['X_0'], data['X_1'], data['X_2'], data['Y']  # n x 1
        data_s, data_x0, data_x1, data_x2, data_y = quickprocess(data_s), quickprocess(data_x0), quickprocess(data_x1), quickprocess(data_x2), quickprocess(data_y)

        self.pi = torch.tensor([0.5, 0.4, 0.05, 0.05]).to(device)
        self.sigma_s = 0.1 * torch.tensor([0.5, 1.0, 1.5, 2.0])
        self.w_x1_u = pyro.param(self.model_name + "_" + "w_x1_u", torch.tensor([0.]).to(device))  # parameters should be initiated here  # 1
        self.w_x1_s = pyro.param(self.model_name + "_" + "w_x1_s", torch.zeros(size=(4,)).to(device))  # [0.1, 0.2, 1, 2]
        self.sigma_x1 = pyro.param(self.model_name + "_" + "sigma_x1", self.sigma_s.to(device))  # [0.5, 1.0, 1.5, 2.0]
        self.w_y_x1 = pyro.param(self.model_name + "_" + "w_y_x1", torch.tensor([0.]).to(device))  # 1
        self.w_y_s = pyro.param(self.model_name + "_" + "w_y_s", torch.tensor([0.]).to(device))  # 0
        self.w_y_x0 = pyro.param(self.model_name + "_" + "w_y_x0", torch.tensor([0.]).to(device))  # 1
        self.sigma_y = pyro.param(self.model_name + "_" + "sigma_y", self.sigma_s.to(device))
        self.w_x2_y = pyro.param(self.model_name + "_" + "w_x2_y", torch.tensor([0.]).to(device))  # 1
        self.w_x2_s = pyro.param(self.model_name + "_" + "w_x2_s", torch.tensor([0.]).to(device))  # 0
        self.sigma_x2 = pyro.param(self.model_name + "_" + "sigma_x2", self.sigma_s.to(device))

        n = len(data_s)
        with pyro.plate('observe_data', size=n, device=device):
            s = pyro.sample('obs_s', pyro.distributions.Categorical(self.pi), obs=data_s)  # S ~ Categorical(pi)
            u = pyro.sample('U', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).to(device)  # latent variable: U ~ N(0,1)
            x_0 = pyro.sample('obs_x0', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.), obs=data_x0)
            x1_mean = self.w_x1_s[s.long()] * s + self.w_x1_u * u
            x_1 = pyro.sample('obs_x1', pyro.distributions.Normal(x1_mean, self.sigma_x1[s.long()]), obs=data_x1)
            y_mean = self.w_y_x1 * x_1 + self.w_y_x0 * x_0 + self.w_y_s * s
            y = pyro.sample('obs_y', pyro.distributions.Normal(y_mean, self.sigma_y[s.long()]), obs=data_y)
            x2_mean = self.w_x2_y * y + self.w_x2_s * s
            x_2 = pyro.sample('obs_x2', pyro.distributions.Normal(x2_mean, self.sigma_x2[s.long()]), obs=data_x2)

        data_return = {'U': u, 'Y': y, 'X_0': data_x0, 'X_1': x_1, 'X_2': x_2, 'S': s}
        return data_return

    def forward_false_1(self, data):
        data_s, data_x0, data_x1, data_x2, data_y = data['S'], data['X_0'], data['X_1'], data['X_2'], data['Y']  # n x 1
        data_s, data_x0, data_x1, data_x2, data_y = data_s.view(-1), data_x0.view(-1), data_x1.view(-1), data_x2.view(-1), data_y.view(-1)
        data_s, data_x0, data_x1, data_x2, data_y = data_s.to(device), data_x0.to(device), data_x1.to(device), data_x2.to(device), data_y.to(device)

        self.pi = torch.tensor([0.5, 0.4, 0.05, 0.05]).to(device)
        self.sigma_s = 0.1 * torch.tensor([0.5, 1.0, 1.5, 2.0])
        self.w_x1_u = pyro.param(self.model_name + "_" + "w_x1_u", torch.tensor([0.]).to(device))  # parameters should be initiated here
        self.w_x1_s = pyro.param(self.model_name + "_" + "w_x1_s", torch.zeros(size=(4,)).to(device))
        self.sigma_x1 = pyro.param(self.model_name + "_" + "sigma_x1", self.sigma_s.to(device))
        self.w_y_x1 = pyro.param(self.model_name + "_" + "w_y_x1", torch.tensor([0.]).to(device))
        self.w_y_s = pyro.param(self.model_name + "_" + "w_y_s", torch.tensor([0.]).to(device))
        self.w_y_x0 = pyro.param(self.model_name + "_" + "w_y_x0", torch.tensor([0.]).to(device))
        self.w_y_x2 = pyro.param(self.model_name + "_" + "w_y_x2", torch.tensor([0.]).to(device))
        self.sigma_y = pyro.param(self.model_name + "_" + "sigma_y", self.sigma_s.to(device))
        self.w_s_x2 = pyro.param(self.model_name + "_" + "w_s_x2", torch.tensor([0.]).to(device))
        self.sigma_s = pyro.param(self.model_name + "_" + "sigma_s", torch.tensor([1.]).to(device))
        self.sigma_x2 = pyro.param(self.model_name + "_" + "sigma_x2", torch.tensor([1.]).to(device))

        n = len(data_s)
        with pyro.plate('observe_data', size=n, device=device):
            s = pyro.sample('obs_s', pyro.distributions.Categorical(self.pi), obs=data_s)  # S ~ Categorical(pi)
            u = pyro.sample('U', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).to(device)  # latent variable: U ~ N(0,1)
            x_0 = pyro.sample('obs_x0', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.), obs=data_x0)
            x1_mean = self.w_x1_s[s.long()] * s + self.w_x1_u * u
            x_1 = pyro.sample('obs_x1', pyro.distributions.Normal(x1_mean, self.sigma_x1[s.long()]), obs=data_x1)
            x_2 = pyro.sample('obs_x2', pyro.distributions.Normal(torch.tensor(0.).to(device), self.sigma_x2), obs=data_x2)
            y_mean = self.w_y_x1 * x_1 + self.w_y_x0 * x_0 + self.w_y_s * s + self.w_y_x2 * x_2
            y = pyro.sample('obs_y', pyro.distributions.Normal(y_mean, self.sigma_y[s.long()]), obs=data_y)

        data_return = {'U': u, 'Y': y}
        return data_return

    def forward_false_2(self, data):
        data_s, data_x0, data_x1, data_x2, data_y = data['S'], data['X_0'], data['X_1'], data['X_2'], data['Y']  # n x 1
        data_s, data_x0, data_x1, data_x2, data_y = data_s.view(-1), data_x0.view(-1), data_x1.view(-1), data_x2.view(-1), data_y.view(-1)
        data_s, data_x0, data_x1, data_x2, data_y = data_s.to(device), data_x0.to(device), data_x1.to(device), data_x2.to(device), data_y.to(device)

        self.pi = torch.tensor([0.5, 0.4, 0.05, 0.05]).to(device)
        self.sigma_s = 0.1 * torch.tensor([0.5, 1.0, 1.5, 2.0])
        self.w_x1_u = pyro.param(self.model_name + "_" + "w_x1_u", torch.tensor([0.]).to(device))  # parameters should be initiated here
        self.sigma_x1 = pyro.param(self.model_name + "_" + "sigma_x1", torch.tensor([1.]).to(device))
        self.w_y_x1 = pyro.param(self.model_name + "_" + "w_y_x1", torch.tensor([0.]).to(device))
        self.w_y_s = pyro.param(self.model_name + "_" + "w_y_s", torch.tensor([0.]).to(device))
        self.w_y_x0 = pyro.param(self.model_name + "_" + "w_y_x0", torch.tensor([0.]).to(device))
        self.sigma_y = pyro.param(self.model_name + "_" + "sigma_y", self.sigma_s.to(device))
        self.w_x2_y = pyro.param(self.model_name + "_" + "w_x2_y", torch.tensor([0.]).to(device))
        self.w_x2_s = pyro.param(self.model_name + "_" + "w_x2_s", torch.tensor([0.]).to(device))
        self.sigma_x2 = pyro.param(self.model_name + "_" + "sigma_x2", self.sigma_s.to(device))

        n = len(data_s)
        with pyro.plate('observe_data', size=n, device=device):
            s = pyro.sample('obs_s', pyro.distributions.Categorical(self.pi), obs=data_s)  # S ~ Categorical(pi)
            u = pyro.sample('U', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).to(device)  # latent variable: U ~ N(0,1)
            x_0 = pyro.sample('obs_x0', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.), obs=data_x0)
            x_1 = pyro.sample('obs_x1', pyro.distributions.Normal(self.w_x1_u * u, self.sigma_x1), obs=data_x1)
            y_mean = self.w_y_x1 * x_1 + self.w_y_x0 * x_0 + self.w_y_s * s
            y = pyro.sample('obs_y', pyro.distributions.Normal(y_mean, self.sigma_y[s.long()]), obs=data_y)
            x_2 = pyro.sample('obs_x2', pyro.distributions.Normal(self.w_x2_y * y + self.w_x2_s * s, self.sigma_x2[s.long()]), obs=data_x2)

        data_return = {'U': u, 'Y': y}
        return data_return

    def forward(self, data):
        if self.model_name == 'true':
            return self.forward_true(data)
        elif self.model_name == 'false_1':
            return self.forward_false_1(data)
        elif self.model_name == 'false_2':
            return self.forward_false_2(data)