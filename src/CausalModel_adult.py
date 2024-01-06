import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist

from torch import nn
import torch.nn.functional as F
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.poutine.trace_messenger import TraceMessenger

pyro.enable_validation(True)
pyro.set_rng_seed(1)
pyro.enable_validation(True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def quickprocess(var):
    if var is None:
        return var
    var = var.view(-1).to(device)
    return var

def argmax_withNan(x, dim=1):
    return None if x is None else torch.argmax(x, dim=dim)

class CausalModel_adult(PyroModule):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def forward(self, data):
        data_Race, data_Sex, data_MaritalStatus, data_Occupation, data_EducationNum, data_HoursPerWeek, data_Income = data['Race'], \
                 data['Sex'], data['MaritalStatus'], data['Occupation'], data['EducationNum'], data['HoursPerWeek'], data['Income']
        data_Race = data_Race.view(-1, 1)

        self.pi_Race = pyro.param(self.model_name + "_" + "pi", torch.tensor([0.4, 0.3, 0.3]).to(device))
        self.pi_Sex = pyro.param(self.model_name + "_" + "pi", torch.tensor([0.5, 0.5]).to(device))

        # marital status: ~ categorical, logits = wx + b
        m_size = 7 if data_MaritalStatus is None else data_MaritalStatus.shape[1]
        self.w_mar_race = pyro.param("w_mar_race", torch.zeros(data_Race.shape[1], m_size).to(device))  # d x d'
        self.w_mar_sex = pyro.param("w_mar_sex", torch.zeros(data_Sex.shape[1], m_size).to(device))

        # education: Normal (better after standardization) ~ N(mean, 1), mean = wx + eps_edu
        e_size = 1 if data_EducationNum is None else data_EducationNum.shape[1]
        self.w_edu_race = pyro.param("w_edu_race", torch.zeros(data_Race.shape[1], 1).to(device))  # d x 1
        self.w_edu_sex = pyro.param("w_edu_sex", torch.zeros(data_Sex.shape[1], 1).to(device))  # d x 1

        # hour per week
        h_size = 1 if data_HoursPerWeek is None else data_HoursPerWeek.shape[1]
        self.w_hour_race = pyro.param("w_hour_race", torch.zeros(data_Race.shape[1], 1).to(device))  # d x 1
        self.w_hour_sex = pyro.param("w_hour_sex", torch.zeros(data_Sex.shape[1], 1).to(device))
        self.w_hour_mar = pyro.param("w_hour_mar", torch.zeros(m_size, 1).to(device))
        self.w_hour_edu = pyro.param("w_hour_edu", torch.zeros(e_size, 1).to(device))

        # occupation
        o_size = 14 if data_Occupation is None else data_Occupation.shape[1]
        self.w_occ_race = pyro.param("w_occ_race", torch.zeros(data_Race.shape[1], o_size).to(device))  # d x d'
        self.w_occ_sex = pyro.param("w_occ_sex", torch.zeros(data_Sex.shape[1], o_size).to(device))
        self.w_occ_mar = pyro.param("w_occ_mar", torch.zeros(m_size, o_size).to(device))
        self.w_occ_edu = pyro.param("w_occ_edu", torch.zeros(e_size, o_size).to(device))

        # income
        self.w_income_race = pyro.param("w_income_race", torch.zeros(data_Race.shape[1], 2).to(device))  # d x 2
        self.w_income_sex = pyro.param("w_income_sex", torch.zeros(data_Sex.shape[1], 2).to(device))
        self.w_income_mar = pyro.param("w_income_mar", torch.zeros(m_size, 2).to(device))
        self.w_income_edu = pyro.param("w_income_edu", torch.zeros(e_size, 2).to(device))
        self.w_income_hour = pyro.param("w_income_hour", torch.zeros(h_size, 2).to(device))
        self.w_income_occ = pyro.param("w_income_occ", torch.zeros(o_size, 2).to(device))

        n = len(data_Race)

        with pyro.plate('observe_data', size=n, device=device):
            Race = pyro.sample('obs_Race', pyro.distributions.Categorical(self.pi_Race), obs=data_Race.view(-1)).view(-1, 1) # S ~ Categorical(pi)
            Sex = pyro.sample('obs_Sex', pyro.distributions.Categorical(self.pi_Sex), obs=torch.argmax(data_Sex, dim=1)).to(device)  # n, raw data
            Sex = F.one_hot(Sex, num_classes=data_Sex.shape[1]).float()  # raw -> one hot, n x d

            eps_MaritalStatus = pyro.sample('eps_MaritalStatus', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).view(-1, 1)
            MaritalStatus_logit = torch.softmax((torch.tile(eps_MaritalStatus, (1, m_size)) + torch.matmul(Race, self.w_mar_race) +
                                                torch.matmul(Sex, self.w_mar_sex)), dim=1)  # n x d_mar
            MaritalStatus = pyro.sample('obs_MaritalStatus', pyro.distributions.Categorical(MaritalStatus_logit),
                                        obs=argmax_withNan(data_MaritalStatus, dim=1)).to(device)   # n
            MaritalStatus = F.one_hot(MaritalStatus, num_classes=m_size).float()  # n x d

            eps_EducationNum = pyro.sample('eps_EducationNum', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).view(-1, 1)
            EducationNum_mean = torch.matmul(Race, self.w_edu_race) + torch.matmul(Sex, self.w_edu_sex) + eps_EducationNum  # n x 1
            EducationNum = pyro.sample("obs_EducationNum", pyro.distributions.Normal(EducationNum_mean.view(-1), 1.0), obs=quickprocess(data_EducationNum)).view(-1,1)  # n x 1

            eps_HoursPerWeek = pyro.sample('eps_HoursPerWeek', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).view(-1, 1)
            HoursPerWeek_mean = torch.matmul(Race, self.w_hour_race) + torch.matmul(Sex, self.w_hour_sex) + \
                                torch.matmul(MaritalStatus, self.w_hour_mar) + torch.matmul(EducationNum, self.w_hour_edu) + eps_HoursPerWeek
            HoursPerWeek = pyro.sample("obs_HoursPerWeek", pyro.distributions.Normal(HoursPerWeek_mean.view(-1), 1.0), obs=quickprocess(data_HoursPerWeek)).view(-1,1)  # n x 1

            eps_Occupation = pyro.sample('eps_Occupation', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).view(-1, 1)
            Occupation_logit = torch.softmax((torch.tile(eps_Occupation, (1, o_size)) +
                                              torch.matmul(Race, self.w_occ_race) + torch.matmul(Sex, self.w_occ_sex) +
                                              torch.matmul(EducationNum, self.w_occ_edu) + torch.matmul(MaritalStatus, self.w_occ_mar)), dim=1)  # n x d
            Occupation = pyro.sample('obs_Occupation', pyro.distributions.Categorical(Occupation_logit),
                                        obs=argmax_withNan(data_Occupation, dim=1)).to(device)  # n x 1
            Occupation = F.one_hot(Occupation, num_classes=o_size).float()  # n x d

            eps_Income = pyro.sample('eps_Income', pyro.distributions.Normal(torch.tensor(0.).to(device), 1.)).view(-1, 1)  # n x 1
            Income_logit = torch.softmax((torch.tile(eps_Income, (1, 2)) + torch.matmul(Race, self.w_income_race) + torch.matmul(Sex, self.w_income_sex) +
                                         torch.matmul(MaritalStatus, self.w_income_mar) + torch.matmul(EducationNum, self.w_income_edu) +
                                         torch.matmul(HoursPerWeek, self.w_income_hour) + torch.matmul(Occupation, self.w_income_occ)), dim=1)  # n x 2
            Income = pyro.sample('obs_Income', pyro.distributions.Categorical(Income_logit),
                                        obs=argmax_withNan(data_Income, dim=1)).to(device).view(-1, 1).float()  # n x 1

        data_return = {'eps_MaritalStatus': eps_MaritalStatus, 'eps_EducationNum': eps_EducationNum,
                       'eps_Occupation': eps_Occupation, 'eps_HoursPerWeek': eps_HoursPerWeek,
                       'eps_Income': eps_Income, 'MaritalStatus': MaritalStatus, 'Occupation': Occupation,
                       'EducationNum': EducationNum, 'HoursPerWeek': HoursPerWeek, 'Income': Income,
                       'Race': Race, 'Sex': Sex
                       }
        return data_return