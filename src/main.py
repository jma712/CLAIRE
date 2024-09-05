import time
import argparse
import numpy as np
import pandas as pd
import csv
import torch
from torchvision import datasets
from torch import nn, optim, autograd
from CausalModel_synthetic import *
from CausalModel_law import *
from CausalModel_adult import *
import matplotlib.pyplot as plt
import baselines
from Claire_model import *
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, f1_score
import copy
import utils
import pickle
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Counterfactual fairness with no explicit prior knowledge')
parser.add_argument('--nocuda', type=int, default=0, help='Disables CUDA training.')
parser.add_argument('--dataset', default='law', help='Dataset name')  # 'law', 'adult', 'synthetic
parser.add_argument('--epochs', type=int, default=1001
                    , metavar='N',
                    help='number of epochs to train (default: 10)')  # law: 1001, adult: 251
parser.add_argument('--epochs_vae', type=int, default=1401, metavar='N',  # synthetic/law: 1401
                    help='number of epochs to train (default: 10)')
parser.add_argument('--num_iterations_cm', type=int, default=10000, metavar='N',  # synthetic: 10000, law/adult: 12000
                    help='number of epochs to train causal model (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--h_dim', type=int, default=5, metavar='N',
                    help='dimension of hidden variables')
parser.add_argument('--vae_h_dim', type=int, default=2, metavar='N',
                    help='dimension of hidden variables')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='learning rate for optimizer')
parser.add_argument('--lr_cm', type=float, default=0.001,
                    help='learning rate for causal mode')
parser.add_argument('--l2_regularizer_weight', type=float, default=1e-3,
                    help='l2_regularizer_weight')
parser.add_argument('--alpha', type=float, default=10.0,
                    help='mmd weight in VAE loss')
parser.add_argument('--lamda', type=float, default=1e-3,
                    help='irm weight')
parser.add_argument('--irm', type=float, default=0.1,  # 0.1
                    help='irm term in loss')
parser.add_argument('--beta', type=float, default=0.5,  # 0.5
                    help='cf loss weight')
parser.add_argument('--K', type=int, default=10,
                    help='sample num')
parser.add_argument('--decoder_type', default='together', help='decoder type in vae')  # 'together', 'separate'
parser.add_argument('--train_cm', type=int, default=0, help='1: train or 0: load')  # in causal model
parser.add_argument('--train_new_vae', type=int, default=0, help='1: train or 0: load')  # in CLAIRE
parser.add_argument('--train_new_claire_pred', type=int, default=0, help='1: train or 0: load')  # in CLAIRE

args = parser.parse_args()

# select gpu if available
args.cuda = not args.nocuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")
args.device = device

print('using device: ', device)

# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def load_data(path, dataset):
    if dataset == 'synthetic':
        with open(path, 'rb') as f:
            data_save = pickle.load(f)
        n = data_save['data']['X_0'].shape[0]
    elif dataset == 'law':
        data_save = utils.load_data(path, dataset)
        n = data_save['data']['race'].shape[0]
    elif dataset == 'adult':
        data_save = utils.load_data(path, dataset)
        n = data_save['data']['Race'].shape[0]
    trn_idx_list, val_idx_list, tst_idx_list = utils.split_data(n, exp_num=10, rates=[0.6, 0.2, 0.2], type='random', sorted=True)
    data_save['index'] = {'trn_idx_list': trn_idx_list, 'val_idx_list': val_idx_list, 'tst_idx_list': tst_idx_list}

    return data_save

def to_tensor(data_save, cuda=True):
    data = data_save['data']
    index = data_save['index']
    for k in data:
        data[k] = torch.FloatTensor(data[k])
        if cuda:
            data[k] = data[k].to(device)
    for k in index:
        index[k] = torch.LongTensor(index[k])
        if cuda:
            index[k] = index[k].to(device)
    if args.dataset == 'synthetic':
        param = data_save['params']
        return {'data': data, 'params': param, 'index': index}
    return {'data': data, 'index': index}

def baseline_cfp_u(data_save, model_name, args, exp_i, S_name=None, s_cf=None, path_cm_root=None, causal_model_types=['prob'], causal_model_weights=[1.0]):
    trn_idx = data_save['index']['trn_idx_list'][exp_i]
    val_idx = data_save['index']['val_idx_list'][exp_i]
    tst_idx = data_save['index']['tst_idx_list'][exp_i]

    # ========  train the causal model ================
    if args.dataset == 'synthetic':
        model = CausalModel_synthetic(model_name)
        model = model.to(device)
    elif args.dataset == 'law':
        model = CausalModel_law(model_name)
        model = model.to(device)
        data_vae = torch.cat([data_save['data']['LSAT'], data_save['data']['UGPA'], data_save['data']['ZFYA']], dim=1)
        dim_x = data_vae.shape[1]
        cm_vae_true = Causal_model_vae(args, dim_x, len(s_cf), 1)
        cm_vae_true = cm_vae_true.to(device)
    elif args.dataset == 'adult':
        model = CausalModel_adult(model_name)
        model = model.to(device)
        data_vae = torch.cat([data_save['data']['Sex'], data_save['data']['MaritalStatus'], data_save['data']['Occupation'],
             data_save['data']['EducationNum'], data_save['data']['HoursPerWeek'], data_save['data']['Income']], dim=1)
        dim_x = data_vae.shape[1]
        cm_vae_true = Causal_model_vae(args, dim_x, len(s_cf), 5)

    fair_latent_list = []  # |cm| x n x d

    # get true causal model: prob
    if 'prob' in causal_model_types:
        path_cm_model = path_cm_root + f'{args.dataset}_' + model_name + '.pt'
        path_cm_param = path_cm_root + f'{args.dataset}_' + model_name + '_param' + '.pt'
        model, guide = train_cm_prob(model, data_save, trn_idx, path_cm_model, path_cm_param, train_flag=args.train_cm)
        fair_latent = get_latent_var_prob(guide, args.dataset, num_of_samples=1)
        fair_latent_list.append(fair_latent)

    # get true causal model: vae
    if 'vae' in causal_model_types:
        path_cm_vae_true_model = path_cm_root + f'{args.dataset}_' + 'vae_' + model_name + '.pt'
        train_cm_vae(cm_vae_true, data_vae, data_save, S_name, s_cf, trn_idx, val_idx, tst_idx, path_cm_vae_true_model,
                     train_flag=args.train_cm)
        fair_latent = cm_vae_true.get_latent_var(data_vae)
        fair_latent_list.append(fair_latent)

    fair_latent_agg = agg_fair_latent(fair_latent_list, weight=causal_model_weights)
    x_fair, data_y = get_fair_var(fair_latent_agg, data_save, args.dataset, model_name)
    x_fair_trn, x_fair_tst, data_y_trn = x_fair[trn_idx.cpu().detach().numpy()], x_fair[tst_idx.cpu().detach().numpy()], data_y[trn_idx.cpu().detach().numpy()]

    if args.dataset != 'adult':
        clf = LinearRegression()
    else:
        clf = LogisticRegression(class_weight='balanced')
    clf.fit(x_fair_trn, data_y_trn.reshape(-1))  # train
    y_pred_tst = clf.predict(x_fair_tst)  # n_tst

    data_return_all = {'predictor': clf, 'y_pred': y_pred_tst, 'causal_model': model, 'guide': guide}

    return data_return_all

def get_fair_var(latent_var, data_save, dataset, model_name=None):
    if dataset == 'synthetic':
        if model_name == 'true':
            x_fair = latent_var  # n x 1
        elif model_name == 'false_1':
            x_fair = np.concatenate([latent_var, data_save['data']['X_2'].cpu().detach().numpy()], axis=1)  # n x 2
        elif model_name == 'false_2':
            x_fair = np.concatenate([latent_var, data_save['data']['X_1'].cpu().detach().numpy()], axis=1)  # n x 2
        data_y = data_save['data']['Y'].cpu().detach().numpy()
    elif dataset == 'law':
        x_fair = latent_var
        data_y = data_save['data']['ZFYA'].cpu().detach().numpy()
    elif dataset == 'adult':
        # ['Sex', 'MaritalStatus', 'Occupation', 'EducationNum', 'HoursPerWeek']
        x_fair = np.concatenate([data_save['data']['Sex'].cpu().detach().numpy(), latent_var], axis=1)
        data_y = data_save['data']['Income'].cpu().detach().numpy()
    return x_fair, data_y

def agg_fair_latent(fair_latent_list, weight):
    fair_agg = np.zeros_like(fair_latent_list[0])
    for i in range(len(fair_latent_list)):
        fair_agg += (weight[i] * fair_latent_list[i])
    fair_agg /= sum(weight)
    return fair_agg

def get_latent_var_prob(guide, dataset, num_of_samples=1):
    data_return_list = []
    for i in range(num_of_samples):
        data_return = guide()
        data_return_list.append(data_return)

    # ========= train a predictor ==============
    # only use "fair" latent variables for prediction
    if dataset == 'synthetic':
        latent_fair = data_return_list[0]['U'].view(-1, 1).cpu().detach().numpy()  # n x 1
    elif dataset == 'law':
        latent_fair = data_return_list[0]['knowledge'].view(-1, 1).cpu().detach().numpy()
    elif dataset == 'adult':
        latent_fair = torch.cat([data_return_list[0]['eps_MaritalStatus'].view(-1, 1),
                            data_return_list[0]['eps_Occupation'].view(-1, 1), data_return_list[0]['eps_EducationNum'].view(-1, 1),
                            data_return_list[0]['eps_HoursPerWeek'].view(-1, 1), data_return_list[0]['eps_Income'].view(-1, 1)],
                           dim=1).cpu().detach().numpy()
    return latent_fair

def evaluate_pred_clf(y_true, y_pred, metrics):
    # y_true, y_pred: n_sample
    eval_result = {}
    if 'F1-score' in metrics:
        eval_result['F1-score'] = f1_score(y_true, y_pred)
    if 'Accuracy' in metrics:
        eval_result['Accuracy'] = accuracy_score(y_true, y_pred)
    if 'RMSE' in metrics:
        eval_result['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    if 'MAE' in metrics:
        eval_result['MAE'] = mean_absolute_error(y_true, y_pred)
    return eval_result

def get_intervene_data_prob(cm_model, guide, data_select, index_select, latent_names, nondes_names, des_name, s_name, s_cf):
    n_select = len(index_select)
    # get latent variables for select data
    data_guide = guide()
    data_latent = {name: data_guide[name].to(device)[index_select] for name in latent_names}  # {'latent': sampled}
    data_nondes = {name: data_select['data'][name] for name in nondes_names}  # non-descendant observed variables
    data_nondes.update(data_latent)
    data_des = {name: None for name in des_name}

    # intervention  # only one sensitive feature is allowed
    data_cf_tst_allS = []  # size = |S|
    for si in range(len(s_cf)):
        data_s = {name: torch.full(size=(n_select,), fill_value=s_cf[si], dtype=torch.float).to(device) for name in
                  s_name}  # data type????
        data_do = data_s.copy()
        data_do.update(data_nondes)
        data_do.update(data_des)
        intervened_model = pyro.poutine.do(cm_model, data=data_s)  # do(S=s_cf[si])
        data_cf_tst = intervened_model(data_do)  # {'var name': value for test}
        data_cf_tst_allS.append(data_cf_tst)
    return data_cf_tst_allS

def get_intervene_data_vae(cm_vae, s_cf, dataset, data_vae, S_name, type='kv', num_sample=1):  # type=kv, raw
    x_cf_list = []  # num_sample x S x n x d, tensor;
    n = data_vae.shape[0]
    for sample_i in range(num_sample):
        x_cf_allS = []
        for i in range(len(s_cf)):
            data_s = torch.full(size=(n, 1), fill_value=s_cf[i]).to(device)
            return_result = cm_vae(data_vae, data_s)  # n x d
            data_cf, latent_var = return_result['reconstruct'], return_result['h_sample']  # x_cf, y_cf

            if type == 'kv':
                data_cf_kv = map_cf_out(data_cf, latent_var, dataset)
                x_cf_allS.append(data_cf_kv)
            else:
                x_cf = data_cf[:, :-1]  # exclude y
                x_cf_allS.append(x_cf)
        x_cf_list.append(x_cf_allS)
    return x_cf_list

def map_cf_out(data_cf, latent_var, dataset):
    # data_cf: n x dim_x
    data = dict()
    if dataset == 'law':
        data['LSAT'] = data_cf[:, 0].view(-1)
        data['UGPA'] = data_cf[:, 1].view(-1)
        n = len(data['LSAT'])
        data['knowledge'] = latent_var.view(n, -1)
        if data['knowledge'].shape[1] == 1:
            data['knowledge'] = data['knowledge'].view(-1)
        data['ZFYA'] = data_cf[:, 2].view(-1)
    if dataset == 'adult':
        data['Sex'] = data_cf[:, :2]
        data['MaritalStatus'] = data_cf[:, 2:9]
        data['Occupation'] = data_cf[:, 9:23]
        data['EducationNum'] = data_cf[:, 23].view(-1, 1)
        data['HoursPerWeek'] = data_cf[:, 24].view(-1, 1)
        data['Income'] = data_cf[:, 25].view(-1, 1)

        data['eps_MaritalStatus'] = latent_var[:, 0].view(-1,1)
        data['eps_EducationNum'] = latent_var[:, 1].view(-1,1)
        data['eps_Occupation'] = latent_var[:, 2].view(-1,1)
        data['eps_HoursPerWeek'] = latent_var[:, 3].view(-1,1)
        data['eps_Income'] = latent_var[:, 4].view(-1,1)

    return data

def agg_cf_allS(data_cf_allS_allCM, S_name, causal_model_weights):
    num_cm = len(data_cf_allS_allCM)
    num_s = len(data_cf_allS_allCM[0])
    data_agg = [{var: 0.0 for var in data_cf_allS_allCM[0][si]} for si in range(num_s)]  # S, {var}

    for si in range(num_s):  # S
        data_agg[si][S_name] = data_cf_allS_allCM[0][si][S_name]  # {'race': xxx}
        for var in data_cf_allS_allCM[0][si]:
            if var != S_name:
                for i in range(num_cm):
                    data_agg[si][var] += (causal_model_weights[i] * data_cf_allS_allCM[i][si][var])
                data_agg[si][var] /= sum(causal_model_weights)
    return data_agg

def get_data_vae(data_save, dataset):
    if dataset == 'law':
        data_vae = torch.cat([data_save['data']['LSAT'], data_save['data']['UGPA'], data_save['data']['ZFYA']], dim=1)
    elif dataset == 'adult':
        data_vae = torch.cat(
            [data_save['data']['Sex'], data_save['data']['MaritalStatus'], data_save['data']['Occupation'],
             data_save['data']['EducationNum'], data_save['data']['HoursPerWeek'], data_save['data']['Income']], dim=1)
    return data_vae

def evaluate_fairness(causal_models, causal_model_types, causal_model_weights, data_select, index_select, model_dict, latent_names=[], nondes_names=[], des_name=[], s_name=[], s_cf=[], num_of_samples=100, p_mmd=0.003, s_wass=0.005):
    n_select = len(index_select)
    model_fairness = {model_name: {} for model_name in model_dict}

    for i in range(num_of_samples):
        data_cf_allS_allCM = []  # |cm| x |S|, {'var': value}
        for ci in range(len(causal_models)):
            if causal_model_types[ci] == 'prob':
                cm_model, guide = causal_models[ci]
                data_cf_prob = get_intervene_data_prob(cm_model, guide, data_select, index_select, latent_names, nondes_names, des_name, s_name, s_cf)
                data_cf_allS_allCM.append(data_cf_prob)

            elif causal_model_types[ci] == 'vae':
                cm_vae = causal_models[ci]
                data_vae_select = get_data_vae(data_select, args.dataset)
                data_cf_vae = get_intervene_data_vae(cm_vae, s_cf, args.dataset, data_vae_select, s_name[0], type='kv', num_sample=1)[0]
                data_cf_allS_allCM.append(data_cf_vae)

        data_cf_tst_allS = agg_cf_allS(data_cf_allS_allCM, s_name[0], causal_model_weights)  # S, {var}

        # model fairness performance
        for model_name in model_dict:
            clf = model_dict[model_name]
            y_pred_cf = get_cf_pred(model_name, clf, data_cf_tst_allS, s_cf)  # list, size = S, each elem: n_select
            eval_fair_result = eval_fairness(y_pred_cf, p_mmd=p_mmd, s_wass=s_wass)
            #print('======= ' + model_name + " ============")
            for k in eval_fair_result:  # mmd, wass
                # print(k, ": ", eval_fair_result[k])
                if k not in model_fairness[model_name]:  # first time
                    model_fairness[model_name][k] = {}
                for s_pair in eval_fair_result[k]:  # '0_1', 'average'
                    if s_pair not in model_fairness[model_name][k]:
                        model_fairness[model_name][k][s_pair] = []
                    model_fairness[model_name][k][s_pair].append(eval_fair_result[k][s_pair])

    # calculate average fairness
    for model_name in model_dict:
        print('======= ' + model_name + " ============")
        perform = ''
        for k in model_fairness[model_name]:  # mmd, wass
            perform += (k + ': ')
            for s_pair in model_fairness[model_name][k]:  # '0_1', 'average'
                fair_all_exp = model_fairness[model_name][k][s_pair]
                model_ave = sum(fair_all_exp) / len(fair_all_exp)  # size = num of samples
                # print(model_name, k, s_pair, model_ave)
                perform += s_pair +": "+ ("%.3f" % model_ave) + " "
            perform += ' | '
        print(perform)

    return

def eval_fairness(y_pred_cf_raw, type=1,  p_mmd=0.003, s_wass=0.005):
    y_pred_cf = y_pred_cf_raw.copy()
    MMD_dict = {}
    wass_dict = {}
    for i in range(len(y_pred_cf)):  # each S
        for j in range(i+1, len(y_pred_cf)):
            mmd = utils.mmd_linear(torch.FloatTensor(y_pred_cf[i].reshape(-1,1)).to(args.device), torch.FloatTensor(y_pred_cf[j].reshape(-1,1)).to(args.device), p=p_mmd)
            wass, _ = utils.wasserstein(torch.FloatTensor(y_pred_cf[i].reshape(-1,1)).to(args.device), torch.FloatTensor(y_pred_cf[j].reshape(-1,1)).to(args.device), args.device, cuda=True, scal=s_wass)

            MMD_dict[str(i)+'_'+str(j)] = mmd.item()
            wass_dict[str(i)+'_'+str(j)] = wass.item()

    MMD_list = [MMD_dict[k] for k in MMD_dict]
    wass_list = [wass_dict[k] for k in wass_dict]
    MMD_dict['Average'] = (sum(MMD_list) / len(MMD_list))
    wass_dict['Average'] = (sum(wass_list) / len(wass_list))
    eval_result = {'MMD': MMD_dict, 'Wass': wass_dict}
    return eval_result

# Given one sampling of test data with all S
def get_cf_pred(model_name, clf, data_cf_tst, s_cf):
    y_pred_cf = []  # S x n
    for si in range(len(s_cf)):
        if args.dataset == 'synthetic':
            if model_name == 'true':
                x_fair = data_cf_tst[si]['U'].view(-1, 1).cpu().detach().numpy()  # n x 1
            elif model_name == 'false_1':
                x_fair = torch.cat([data_cf_tst[si]['U'].view(-1, 1), data_cf_tst[si]['X_2'].view(-1, 1)], dim=1).cpu().detach().numpy()
            elif model_name == 'false_2':
                x_fair = torch.cat([data_cf_tst[si]['U'].view(-1, 1), data_cf_tst[si]['X_1'].view(-1, 1)], dim=1).cpu().detach().numpy()
            elif model_name == 'Claire':
                x_fair = torch.cat([data_cf_tst[si]['X_0'].view(-1, 1), data_cf_tst[si]['X_1'].view(-1, 1), data_cf_tst[si]['X_2'].view(-1, 1)], dim=1)

        elif args.dataset == 'law':
            if model_name == 'full':
                x_fair = torch.cat([data_cf_tst[si]['LSAT'].view(-1, 1), data_cf_tst[si]['UGPA'].view(-1, 1),
                                    data_cf_tst[si]['race'].view(-1, 1)], dim=1).cpu().detach().numpy()
            elif model_name == 'CFP-U':
                x_fair = data_cf_tst[si]['knowledge'].view(-1, 1).cpu().detach().numpy()  # n x 1
            elif model_name == 'unaware':
                x_fair = torch.cat([data_cf_tst[si]['LSAT'].view(-1, 1), data_cf_tst[si]['UGPA'].view(-1, 1)], dim=1).cpu().detach().numpy()
            elif model_name == 'constant':
                tst_size = len(data_cf_tst[si]['race'])
                y_pred_si = np.full((tst_size, 1), clf)
            elif model_name == 'Claire':
                x_fair = torch.cat([data_cf_tst[si]['LSAT'].view(-1, 1), data_cf_tst[si]['UGPA'].view(-1, 1)], dim=1)

        elif args.dataset == 'adult':
            if model_name == 'full':
                x_fair = torch.cat([data_cf_tst[si]['Sex'], data_cf_tst[si]['MaritalStatus'],
                     data_cf_tst[si]['Occupation'], data_cf_tst[si]['EducationNum'],
                     data_cf_tst[si]['HoursPerWeek'], data_cf_tst[si]['Race']], dim=1).cpu().detach().numpy()
            elif model_name == 'CFP-U':
                #x_fair = data_latent_list.cpu().detach().numpy()  # n x 1
                x_fair = torch.cat([data_cf_tst[si]['Sex'], data_cf_tst[si]['eps_MaritalStatus'],data_cf_tst[si]['eps_EducationNum'],
                                    data_cf_tst[si]['eps_Occupation'],data_cf_tst[si]['eps_HoursPerWeek'],
                                    data_cf_tst[si]['eps_Income']],dim=1).cpu().detach().numpy()  # n x 1
            elif model_name == 'unaware':
                x_fair = torch.cat([data_cf_tst[si]['Sex'], data_cf_tst[si]['MaritalStatus'],
                                    data_cf_tst[si]['Occupation'], data_cf_tst[si]['EducationNum'],
                                    data_cf_tst[si]['HoursPerWeek']], dim=1).cpu().detach().numpy()
            elif model_name == 'constant':
                tst_size = len(data_cf_tst[si]['Sex'])
                y_pred_si = np.full((tst_size, 1), clf)
            elif model_name == 'Claire':
                x_fair = torch.cat([data_cf_tst[si]['Sex'], data_cf_tst[si]['MaritalStatus'],
                                    data_cf_tst[si]['Occupation'], data_cf_tst[si]['EducationNum'],
                                    data_cf_tst[si]['HoursPerWeek']], dim=1)
        if model_name != 'constant':
            #print("=== model name: ", model_name)
            if model_name == 'Claire':
                y_pred_si = clf(x_fair, None).cpu().detach().numpy()
            else:
                if args.dataset == 'adult':  # y_prob
                    y_pred_si = clf.predict_proba(x_fair)[:, 1]
                else: # y_hat
                    y_pred_si = clf.predict(x_fair)
        y_pred_cf.append(y_pred_si)  # n x 1
    return y_pred_cf

def norm_mse(pred, true):
    mean = torch.mean(true, dim=0)
    std = torch.std(true, dim=0)
    norm_pred = (pred - mean) / std
    norm_true = (true - mean) / std
    loss_mse = torch.mean(torch.pow(norm_pred - norm_true, 2), dim=0)
    loss_mse = loss_mse.sum()
    return loss_mse

def loss_function_vae(return_result, data, data_s, s_cf):
    h = return_result['mu_h']
    mu_h = h
    logvar_h = return_result['logvar_h']
    data_reconst = return_result['reconstruct']

    # representation balancing
    num_ij = 0
    mmd_loss = 0.0
    for i in range(len(s_cf)):
        idx_si = torch.where(data_s.view(-1) == i)  # index
        for j in range(i+1, len(s_cf)):
            idx_sj = torch.where(data_s.view(-1) == j)  # index
            num_ij += 1
            mmd_loss += utils.mmd_linear(h[idx_si], h[idx_sj], p=10)
    mmd_loss /= num_ij

    # reconstruct loss
    loss_r = norm_mse(data_reconst, data)

    KL = torch.mean(-0.5 * torch.sum(1 + logvar_h - mu_h.pow(2) - logvar_h.exp(), dim=1), dim=0)

    loss = loss_r + args.alpha * mmd_loss + KL

    eval_result = {'loss': loss, 'loss_r': loss_r, 'loss_mmd': mmd_loss, 'loss_kl': KL}
    return eval_result

def test_vae(model, idx_select, data, data_s, s_cf):
    model.eval()
    return_result = model(data[idx_select], data_s[idx_select])
    eval_result = loss_function_vae(return_result, data[idx_select], data_s[idx_select], s_cf)
    return eval_result

def train_vae(epochs, model, optimizer, data, data_s, s_cf, trn_idx, val_idx, tst_idx):
    time_begin = time.time()

    model.train()
    print("start training!")

    for epoch in range(epochs):
        optimizer.zero_grad()
        return_result = model(data[trn_idx], data_s[trn_idx])
        eval_result = loss_function_vae(return_result, data[trn_idx], data_s[trn_idx], s_cf)
        loss, loss_r, loss_mmd, loss_kl = eval_result['loss'], eval_result['loss_r'], eval_result['loss_mmd'], eval_result['loss_kl']

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            model.eval()
            eval_result_tst = test_vae(model, tst_idx, data, data_s, s_cf)

            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss.item()),
                  'loss_test: {:.4f}'.format(eval_result_tst['loss'].item()),
                  'time: {:.4f}s'.format(time.time() - time_begin))
            model.train()

    return model

def compute_loss_claire(model, x_cf_list, s_cf, idx_si_trn_list, data_x_trn, data_y_trn, data_s_trn, trn_idx, epoch, penalty_anneal_iters, type_y='pred'):
    loss_y_pred = nn.MSELoss(reduction='mean').to(device) if type_y == 'pred' \
        else nn.CrossEntropyLoss(weight=torch.FloatTensor([0.25, 0.75]).to(device))  # balance for classification
    loss_irm = torch.tensor(0.).to(device)

    for i in range(len(s_cf)):
        idx_si = idx_si_trn_list[i]
        x_e = data_x_trn[idx_si]
        y_e = data_y_trn[idx_si]
        env_e = data_s_trn[idx_si]

        y_pred = model(x_e, env_e)

        # invariant
        penalty_e = utils.penalty(y_pred, y_e)
        penalty_weight = (args.lamda
                      if epoch >= penalty_anneal_iters else 1.0)
        loss_irm += penalty_weight * penalty_e  # IRM term

    # irm
    loss_irm = loss_irm / len(s_cf)

    # invariant
    if penalty_weight > 1.0:
        # Rescale the entire loss to keep gradients in a reasonable range
        loss_irm /= penalty_weight

    y_pred_trn = model(data_x_trn, data_s_trn)

    loss_cf = torch.tensor(0.0).to(device)
    num_sij = 0
    n_trn = len(trn_idx)

    for sample_i in range(len(x_cf_list)):  # sample x S x n x d
        for i in range(len(s_cf)):
            x_si_trn = x_cf_list[sample_i][i][trn_idx]
            data_si = torch.full(size=(n_trn, 1), fill_value=s_cf[i]).to(device)
            y_pred_si = model(x_si_trn, data_si)
            loss_cf += torch.mean(torch.pow(y_pred_si - y_pred_trn, 2))
            num_sij += 1

    loss_cf /= num_sij

    # prediction loss
    if type_y == 'pred':
        loss_y = loss_y_pred(y_pred_trn.view(-1), data_y_trn.view(-1))
    else:
        y_pred_logits_trn = torch.cat([1 - y_pred_trn, y_pred_trn], dim=1)
        loss_y = loss_y_pred(y_pred_logits_trn, data_y_trn.view(-1).long())

    loss = loss_y + args.beta * loss_cf + args.irm * loss_irm

    return {'loss': loss, 'loss_y': loss_y, 'loss_cf': loss_cf, 'loss_irm': loss_irm}

def test(model, data_x_select, data_s_select, data_y_select, metrics={'RMSE','MAE'}):
    model.eval()
    y_pred = model(data_x_select, data_s_select)
    if args.dataset == 'adult':
        y_pred = torch.round(y_pred)
    eval_result = evaluate_pred_clf(data_y_select.view(-1).cpu().detach().numpy(), y_pred.view(-1).cpu().detach().numpy(), metrics=metrics)
    return eval_result

def train(epochs, model, data_save, x_cf_list, s_cf, trn_idx, val_idx, tst_idx, optimizer, type_y='pred'):
    time_begin = time.time()
    model.train()
    print("start training!")

    penalty_anneal_iters = 100
    if args.dataset == 'synthetic':
        data_x = torch.cat([data_save['data']['X_0'], data_save['data']['X_1'], data_save['data']['X_2']], dim=1)
        data_y = data_save['data']['Y']
        data_s = data_save['data']['S']
    elif args.dataset == 'law':
        data_x = torch.cat([data_save['data']['LSAT'], data_save['data']['UGPA']], dim=1)
        data_y = data_save['data']['ZFYA']
        data_s = data_save['data']['race']
    elif args.dataset == 'adult':
        data_x = torch.cat([data_save['data']['Sex'], data_save['data']['MaritalStatus'], data_save['data']['Occupation'], data_save['data']['EducationNum'], data_save['data']['HoursPerWeek']], dim=1)
        data_y = data_save['data']['Income']
        data_s = data_save['data']['Race']
    data_x_trn, data_y_trn, data_s_trn, data_x_tst, data_y_tst, data_s_tst = data_x[trn_idx], data_y[trn_idx], \
                                                                             data_s[trn_idx], data_x[tst_idx], \
                                                                             data_y[tst_idx], data_s[tst_idx]
    idx_si_trn_list = []
    for i in range(len(s_cf)):
        idx_si = torch.where(data_s_trn.view(-1) == s_cf[i])[0]
        idx_si_trn_list.append(idx_si)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_return = compute_loss_claire(model, x_cf_list, s_cf, idx_si_trn_list, data_x_trn, data_y_trn, data_s_trn, trn_idx, epoch,
                            penalty_anneal_iters, type_y)
        loss = loss_return['loss']

        # backward propagation
        loss.backward(retain_graph = True)  # retain_graph = True
        optimizer.step()

        if epoch % 100 == 0:
            model.eval()
            metrics = ['Accuracy', 'F1-score'] if args.dataset == 'adult' else ['RMSE', 'MAE']
            eval_result_tst = test(model, data_x_tst, data_s_tst, data_y_tst, metrics=metrics)

            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_return['loss'].item()),
                  metrics[0]+': {:.4f}'.format(eval_result_tst[metrics[0]].item()),
                  metrics[1] + ': {:.4f}'.format(eval_result_tst[metrics[1]].item()),
                  'time: {:.4f}s'.format(time.time() - time_begin))
            model.train()
    return

def run_baselines_real(data_save, exp_i, metrics_set, type='linear'):
    # data_save contains numpy
    if args.dataset == 'law':
        x = np.concatenate([data_save['data']['LSAT'], data_save['data']['UGPA']], axis=1)
        y = data_save['data']['ZFYA']
        env = data_save['data']['race']
    elif args.dataset == 'adult':
        x = np.concatenate([data_save['data']['Sex'], data_save['data']['MaritalStatus'], data_save['data']['Occupation'],
                            data_save['data']['EducationNum'], data_save['data']['HoursPerWeek']], axis=1)
        y = data_save['data']['Income']
        env = data_save['data']['Race']

    trn_idx = data_save['index']['trn_idx_list'][exp_i]
    val_idx = data_save['index']['val_idx_list'][exp_i]
    tst_idx = data_save['index']['tst_idx_list'][exp_i]

    y_pred_tst_constant = baselines.run_constant(x, y.reshape(-1), trn_idx, tst_idx, type)
    eval_constant = evaluate_pred_clf(y[tst_idx].reshape(-1), y_pred_tst_constant, metrics_set)
    print("=========== evaluation for constant predictor ================: ", eval_constant)

    y_pred_tst_full, clf_full = baselines.run_full(x, y.reshape(-1), env, trn_idx, tst_idx, type)
    eval_full = evaluate_pred_clf(y[tst_idx].reshape(-1), y_pred_tst_full, metrics_set)
    print("=========== evaluation for full predictor ================: ", eval_full)

    y_pred_tst_unaware, clf_unaware = baselines.run_unaware(x, y.reshape(-1), trn_idx, tst_idx, type)
    eval_unaware = evaluate_pred_clf(y[tst_idx].reshape(-1), y_pred_tst_unaware, metrics_set)
    print("=========== evaluation for unaware predictor ================: ", eval_unaware)

    return {'clf_full': clf_full, 'clf_unaware': clf_unaware, 'constant_y': y_pred_tst_constant[0]}

def generate_cf_with_cm(cm, guide, data_save, index_select, latent_names, nondes_names, des_name, x_names, s_cf, s_name, num_sample=1):
    # intervention
    x_cf_all = []  # size = num_sample x |S| x n_select x d
    n_select = len(index_select)
    data_select = utils.select_index(data_save, index_select)
    for i in range(num_sample):
        x_cf_allS = []  # size = |S| x n_select
        data_guide = guide()
        data_latent = {name: data_guide[name][index_select] for name in latent_names}  # {'latent': sampled}
        data_nondes = {name: data_select['data'][name] for name in nondes_names}  # non-descendant observed variables
        data_nondes.update(data_latent)
        data_des = {name: None for name in des_name}   # only this part can be modified by intervention
        for si in range(len(s_cf)):
            data_s = {name: torch.full(size=(n_select,), fill_value=s_cf[si], dtype=torch.float).to(device) for name in s_name}  # data type????
            data_do = data_s.copy()
            data_do.update(data_nondes)
            data_do.update(data_des)
            intervened_model = pyro.poutine.do(cm, data=data_s)  # do(S=s_cf[si])
            data_cf = intervened_model(data_do)  # {'var name': value for test}
            x_cf = []
            for x_name in x_names:
                x_cf.append(data_cf[x_name].view(-1, 1))  # [nx1, nx1, ...]
            x_cf = torch.cat(x_cf, dim=1)
            x_cf_allS.append(x_cf)
        x_cf_all.append(x_cf_allS)
    return x_cf_all

def data_standardize(data_save, col_names):
    data_save_norm = copy.deepcopy(data_save)
    for col in col_names:
        x = data_save['data'][col]
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        data_save_norm['data'][col] = (x - x_mean) / (x_std + 0.000001)
    return data_save_norm

def train_cm_prob(model, data_save, trn_idx, path_cm_model, path_cm_param, train_flag=True):
    n_trn = len(trn_idx)
    if train_flag:  # train
        guide = AutoDiagonalNormal(model)
        adam = pyro.optim.Adam({"lr": args.lr_cm})
        svi = SVI(model, guide, adam, loss=Trace_ELBO())
        pyro.clear_param_store()
        # train model
        for j in range(args.num_iterations_cm):
            # calculate the loss and take a gradient step
            loss = svi.step(data_save['data'])  # all data is used here
            if j % 100 == 0:
                print("[iteration %04d] loss: %.4f" % (j + 1, loss / n_trn))

        for name, value in pyro.get_param_store().items():
            print(name, pyro.param(name))

        # save
        
        save_flag = True
        if save_flag:
            torch.save({"model": model.state_dict(), "guide": guide}, path_cm_model)
            pyro.get_param_store().save(path_cm_param)
            print('saved causal model in: ', path_cm_model)
    else:  # load
        saved_model_dict = torch.load(path_cm_model)
        model.load_state_dict(saved_model_dict['model'])
        guide = saved_model_dict['guide']
        pyro.get_param_store().load(path_cm_param)

        print('loaded causal model from: ', path_cm_model)
    return model, guide

def train_cm_vae(model, data_vae, data_save, S_name, s_cf, trn_idx, val_idx, tst_idx, path_cae_model, train_flag=True):
    if train_flag:
        # train vae
        optimizer_vae = optim.Adam(model.parameters(), lr=args.lr)
        train_vae(args.epochs_vae, model, optimizer_vae, data_vae, data_save['data'][S_name], s_cf, trn_idx, val_idx, tst_idx)
        #
        save_vae_flag = True
        if save_vae_flag:
            torch.save(model.state_dict(), path_cae_model)
            print('saved VAE model in: ', path_cae_model)

    else:  # load
        model.load_state_dict(torch.load(path_cae_model))
        print('loaded VAE model from: ', path_cae_model)
    return

def exp_synthetic(path, exp_num=3):
    data_save = load_data(path, args.dataset)
    data_save = to_tensor(data_save, args.cuda)
    path_cm_root = f'../models_save/'
    num_of_samples = 1

    S_name = 'S'
    latent_names = 'U'
    nondes_names = ['X_0']
    des_name = ['Y', 'X_1', 'X_2']
    s_name = ['S']
    type_y = 'pred'
    s_cf = [0, 1, 2, 3]

    causal_model_types = ['prob']
    causal_model_weights = [1.0]
    metrics_set = set(['RMSE', 'MAE'])

    for exp_i in range(exp_num):
        # 2. baseline: CFP-U ==========================================================================
        trn_idx = data_save['index']['trn_idx_list'][exp_i]
        val_idx = data_save['index']['val_idx_list'][exp_i]
        tst_idx = data_save['index']['tst_idx_list'][exp_i]

        # CFP-U, True
        return_cfp_u = baseline_cfp_u(data_save, 'true', args, exp_i, path_cm_root=path_cm_root)
        clf_cfp_u, y_pred_test, cm_true, guide_true = return_cfp_u['predictor'], return_cfp_u['y_pred'], return_cfp_u['causal_model'], return_cfp_u['guide']
        # CFP-U, False-1
        return_cfp_u_false1 = baseline_cfp_u(data_save, 'false_1', args, exp_i, path_cm_root=path_cm_root)
        clf_cfp_u_false1, y_pred_test_false1 = return_cfp_u_false1['predictor'], return_cfp_u_false1['y_pred']
        # CFP-U, False-2
        return_cfp_u_false2 = baseline_cfp_u(data_save, 'false_2', args, exp_i, path_cm_root=path_cm_root)
        clf_cfp_u_false2, y_pred_test_false2 = return_cfp_u_false2['predictor'], return_cfp_u_false2['y_pred']
        #
        y_true_tst = data_save['data']['Y'][tst_idx]
        #
        print('========= Prediction: true ============')
        eval_result_true = evaluate_pred_clf(y_true_tst.view(-1).cpu().detach().numpy(), y_pred_test, metrics_set)
        print(eval_result_true)
        print('========= Prediction: false 1 ============')
        eval_result_false1 = evaluate_pred_clf(y_true_tst.view(-1).cpu().detach().numpy(), y_pred_test_false1, metrics_set)
        print(eval_result_false1)
        print('========= Prediction: false 2 ============')
        eval_result_false2 = evaluate_pred_clf(y_true_tst.view(-1).cpu().detach().numpy(), y_pred_test_false2, metrics_set)
        print(eval_result_false2)

        # 3. our method  ==========================================
        print("========== train CLAIRE-VAE ==========")
        #s_cf = [0.0, 1.0, 2.0, 3.0]
        # 1. ====== Counterfactual auto encoder: X, Y => H (+S) => X', Y'
        data_vae = torch.cat([data_save['data']['X_0'], data_save['data']['X_1'], data_save['data']['X_2'], data_save['data']['Y']], dim=1)
        dim_x = data_vae.shape[1]

        model_cae = Claire_vae(args, dim_x, len(s_cf))
        model_cae = model_cae.to(device)

        path_cae_model = f'../models_save/' + f'{args.dataset}_' + 'claire_vae' + '.pt'
        if args.train_new_vae:
            # train vae
            optimizer_vae = optim.Adam(model_cae.parameters(), lr=args.lr)
            train_result_vae = train_vae(args.epochs_vae, model_cae, optimizer_vae, data_vae, data_save['data'][S_name],
                                         s_cf, trn_idx, val_idx, tst_idx)
            #
            save_vae_flag = True
            if save_vae_flag:
                torch.save(model_cae.state_dict(), path_cae_model)
                print('saved CLAIRE-VAE model in: ', path_cae_model)

        else:  # load
            model_cae.load_state_dict(torch.load(path_cae_model))
            print('loaded CLAIRE-VAE model from: ', path_cae_model)

        print("========== CLAIRE-VAE Obtained! ==========")
        # generate counterfactuals
        x_cf_list = []  # num_sample x S x n x d, tensor;
        n = data_save['data'][S_name].shape[0]
        num_sample = 25
        for sample_i in range(num_sample):
            x_cf_allS = []
            for i in range(len(s_cf)):
                data_s = torch.full(size=(n, 1), fill_value=s_cf[i]).to(device)
                return_result = model_cae(data_vae, data_s)
                data_cf = return_result['reconstruct']  # x_cf, y_cf
                x_cf = data_cf[:, :-1]  # exclude y
                x_cf_allS.append(x_cf)
            x_cf_list.append(x_cf_allS)

        # num_sample x |S| x n x d
        n = data_vae.shape[0]
        # 2. ====== Predictor: X,S => Y_hat
        print('========= train CLAIRE-Predictor ============')
        model_claire = Claire_model(dim_x - 1, args, type='pred').to(device)

        path_claire_model = path_cm_root + f'{args.dataset}_' + 'CLAIRE' + '.pt'
        if args.train_new_claire_pred:
            optimizer = optim.Adam(model_claire.parameters(), lr=args.lr)
            train(args.epochs, model_claire, data_save, x_cf_list, s_cf, trn_idx, val_idx, tst_idx, optimizer, type_y)
            #
            save_claire_flag = True
            if save_claire_flag:
                torch.save(model_claire.state_dict(), path_claire_model)
                print('saved CLAIRE model in: ', path_claire_model)

        else:  # load
            model_claire.load_state_dict(torch.load(path_claire_model))
            print('loaded CLAIRE model from: ', path_claire_model)

        # 3. ====== evaluate prediction performance
        data_x = torch.cat([data_save['data']['X_0'], data_save['data']['X_1'], data_save['data']['X_2']], dim=1)
        data_y = data_save['data']['Y']
        data_s = data_save['data']['S']
        data_x_tst, data_y_tst, data_s_tst = data_x[tst_idx], data_y[tst_idx], data_s[tst_idx]

        model_claire.eval()
        y_pred_test = model_claire(data_x_tst, data_s_tst)
        eval_result_claire = evaluate_pred_clf(data_y[tst_idx].view(-1).cpu().detach().numpy(),
                                               y_pred_test.cpu().detach().numpy(), metrics_set)
        print('========== ' + 'CLAIRE' + '============')
        for k in eval_result_claire:
            print(k, ': ', eval_result_claire[k])

        # evaluate fairness
        # =========  evaluate fairness ============
        model_dict = {'true': clf_cfp_u, 'false_1': clf_cfp_u_false1, 'false_2': clf_cfp_u_false2, 'Claire': model_claire}
        data_select = utils.select_index(data_save, tst_idx)
        eval_fair_result = evaluate_fairness([(cm_true, guide_true)], causal_model_types, causal_model_weights, data_select,
                                             tst_idx, model_dict, latent_names=latent_names, nondes_names=nondes_names,
                                             des_name=des_name, s_name=s_name, s_cf=s_cf, num_of_samples=num_of_samples,
                                             p_mmd=1, s_wass=1)
        print(eval_fair_result)
    return

def exp_real(path, exp_num=2):
    data_save_orin = load_data(path, args.dataset)  # environment selection
    standardize = False
    if args.dataset == 'law':
        X_name = ['LSAT', 'UGPA']
        Y_name = 'ZFYA'
        S_name = 'race'
        type_clf = 'linear'
        type_y = 'pred'
        metrics_set = set({"RMSE", "MAE"})
        latent_names = ['knowledge']
        nondes_names = []
        des_name = ['LSAT', 'ZFYA', 'UGPA']
        s_name = ['race']
        s_cf = [0.0, 1.0, 2.0]
    elif args.dataset == 'adult':
        X_name = ['Sex', 'MaritalStatus', 'Occupation', 'EducationNum', 'HoursPerWeek']
        Y_name = 'Income'
        S_name = 'Race'
        type_clf = 'logistic'
        type_y = 'class'
        metrics_set = set({"Accuracy", 'F1-score'})
        latent_names = ['eps_MaritalStatus', 'eps_EducationNum', 'eps_Occupation', 'eps_HoursPerWeek', 'eps_Income']
        nondes_names = ['Sex']
        des_name = ['MaritalStatus', 'Occupation', 'EducationNum', 'HoursPerWeek', 'Income']
        s_name = ['Race']
        s_cf = [0.0, 1.0, 2.0]
    if standardize:
        data_save_orin = data_standardize(data_save_orin, X_name)

    path_cm_root = f'../models_save/'
    num_of_samples = 1  # you can set it larger to get more samples
    causal_model_types = ['prob', 'vae']

    for exp_i in range(exp_num):
        # baseline (constant, unaware, full), prediction
        baseline_return = run_baselines_real(data_save_orin, exp_i, metrics_set, type_clf)

        import copy
        data_save = copy.deepcopy(data_save_orin)
        data_save = to_tensor(data_save, args.cuda)

        trn_idx, val_idx, tst_idx = data_save['index']['trn_idx_list'][exp_i], data_save['index']['val_idx_list'][exp_i], data_save['index']['tst_idx_list'][exp_i]

        model_name = 'true'
        # load true causal model
        if args.dataset == 'law':
            cm_prob_true = CausalModel_law(model_name)

            # vae
            x_names = ['LSAT', 'UGPA']
            data_vae = torch.cat([data_save['data']['LSAT'], data_save['data']['UGPA'], data_save['data']['ZFYA']], dim=1)
            dim_x = data_vae.shape[1]
            cm_vae_true = Causal_model_vae(args, dim_x, len(s_cf), 1)
        elif args.dataset == 'adult':
            cm_prob_true = CausalModel_adult(model_name)
            #
            x_names = ['Sex', 'MaritalStatus', 'Occupation', 'EducationNum', 'HoursPerWeek']
            data_vae = torch.cat([data_save['data']['Sex'], data_save['data']['MaritalStatus'], data_save['data']['Occupation'],
                 data_save['data']['EducationNum'], data_save['data']['HoursPerWeek'], data_save['data']['Income']], dim=1)
            dim_x = data_vae.shape[1]
            cm_vae_true = Causal_model_vae(args, dim_x, len(s_cf), 5)

        cm_prob_true = cm_prob_true.to(device)
        cm_vae_true = cm_vae_true.to(device)
        model_name = 'true'

        # get true causal model: prob
        print('===== getting true causal models =======')
        if 'prob' in causal_model_types:
            path_cm_model = path_cm_root + f'{args.dataset}_' + model_name + '.pt'
            path_cm_param = path_cm_root + f'{args.dataset}_' + model_name + '_param' + '.pt'
            cm_prob_true, guide_true = train_cm_prob(cm_prob_true, data_save, trn_idx, path_cm_model, path_cm_param,
                                                     train_flag=args.train_cm)

        # get true causal model: vae
        if 'vae' in causal_model_types:
            path_cm_vae_true_model = path_cm_root + f'{args.dataset}_' + 'vae_true' + '.pt'
            train_cm_vae(cm_vae_true, data_vae, data_save, S_name, s_cf, trn_idx, val_idx, tst_idx, path_cm_vae_true_model, train_flag=args.train_cm
                        )
        print('===== true causal models obtained =======')
        causal_models = [(cm_prob_true, guide_true), cm_vae_true]
        causal_model_weights = [0.5, 0.5]

        # baseline: CFP-U, True
        return_cfp_u = baseline_cfp_u(data_save, 'true', args, exp_i, S_name, s_cf, path_cm_root, causal_model_types, causal_model_weights)
        clf_cfp_u, y_pred_test, cm_true_cfp, guide_true_cfp = return_cfp_u['predictor'], return_cfp_u['y_pred'], return_cfp_u['causal_model'], return_cfp_u['guide']
        y_true_tst = data_save['data'][Y_name][tst_idx]
        # Performance
        print('========= Prediction: CFP-U (True) ============')
        eval_result_true = evaluate_pred_clf(y_true_tst.view(-1).cpu().detach().numpy(), y_pred_test, metrics_set)
        print(eval_result_true)

        # 3. our method  ==========================================
        # 1. ====== Counterfactual auto encoder: X, Y => H (+S) => X', Y'
        print('========= CLAIRE-VAE ============')
        dim_x = data_vae.shape[1]

        model_cae = Claire_vae(args, dim_x, len(s_cf))
        model_cae = model_cae.to(device)

        path_cae_model = f'../models_save/' + f'{args.dataset}_' + 'claire_vae' + '.pt'
        if args.train_new_vae:
            # train vae
            optimizer_vae = optim.Adam(model_cae.parameters(), lr=args.lr)
            train_result_vae = train_vae(args.epochs_vae, model_cae, optimizer_vae, data_vae, data_save['data'][S_name], s_cf,
                                         trn_idx, val_idx, tst_idx)
            #
            save_vae_flag = True
            if save_vae_flag:
                torch.save(model_cae.state_dict(), path_cae_model)
                print('saved CLAIRE-VAE model in: ', path_cae_model)

        else:  # load
            model_cae.load_state_dict(torch.load(path_cae_model))
            print('loaded CLAIRE-VAE model from: ', path_cae_model)

        # generate counterfactuals
        # model_cae.eval()
        x_cf_list = []  # num_sample x S x n x d, tensor;
        n = data_save['data'][S_name].shape[0]
        num_sample = 25
        for sample_i in range(num_sample):
            x_cf_allS = []
            for i in range(len(s_cf)):
                data_s = torch.full(size=(n, 1), fill_value=s_cf[i]).to(device)
                return_result = model_cae(data_vae, data_s)
                data_cf = return_result['reconstruct']  # x_cf, y_cf
                x_cf = data_cf[:, :-1]  # exclude y
                x_cf_allS.append(x_cf)
            x_cf_list.append(x_cf_allS)

        # num_sample x |S| x n x d
        n = data_vae.shape[0]
        # 2. ====== Predictor: X,S => Y_hat
        print('========= CLAIRE-Predictor ============')
        model_claire = Claire_model(dim_x - 1, args, type_y).to(device)

        path_claire_model = path_cm_root + f'{args.dataset}_' + 'CLAIRE' + '.pt'
        if args.train_new_claire_pred:
            optimizer = optim.Adam(model_claire.parameters(), lr=args.lr)
            train(args.epochs, model_claire, data_save, x_cf_list, s_cf, trn_idx, val_idx, tst_idx, optimizer, type_y)
            #
            save_claire_flag = True
            if save_claire_flag:
                torch.save(model_claire.state_dict(), path_claire_model)
                print('saved CLAIRE model in: ', path_claire_model)

        else:  # load
            model_claire.load_state_dict(torch.load(path_claire_model))
            print('loaded CLAIRE model from: ', path_claire_model)

        # 3. ====== evaluate prediction performance
        if args.dataset == 'law':
            data_x = torch.cat([data_save['data']['LSAT'], data_save['data']['UGPA']], dim=1)
        elif args.dataset == 'adult':
            data_x = torch.cat([data_save['data']['Sex'], data_save['data']['MaritalStatus'], data_save['data']['Occupation'], data_save['data']['EducationNum'], data_save['data']['HoursPerWeek']], dim=1)
        data_y = data_save['data'][Y_name]
        data_s = data_save['data'][S_name]

        data_x_tst, data_y_tst, data_s_tst = data_x[tst_idx], data_y[tst_idx], data_s[tst_idx]

        model_claire.eval()
        y_pred_test = model_claire(data_x_tst, data_s_tst)
        if type_y == 'class':
            y_pred_test = torch.round(y_pred_test)
        eval_result_claire = evaluate_pred_clf(data_y[tst_idx].view(-1).cpu().detach().numpy(),
                                               y_pred_test.cpu().detach().numpy(), metrics_set)  
        print('========== ' + 'CLAIRE' + '============')
        for k in eval_result_claire:
            print(k, ': ', eval_result_claire[k])

        # evaluate fairness
        model_dict = {'constant': baseline_return['constant_y'], 'full': baseline_return['clf_full'],
                      'unaware': baseline_return['clf_unaware'], 'CFP-U': clf_cfp_u, 'Claire': model_claire}
        data_select = utils.select_index(data_save, tst_idx)
        eval_fair_result = evaluate_fairness(causal_models, causal_model_types, causal_model_weights, data_select, tst_idx, model_dict,
                                             latent_names=latent_names, nondes_names=nondes_names, des_name=des_name,
                                             s_name=s_name, s_cf=s_cf, num_of_samples=num_of_samples)
        print(eval_fair_result)



if __name__ == '__main__':
    exp_num = 1  # set it larger when you need more runs

    if args.dataset == 'synthetic':
        path = '../dataset/synthetic_data.pickle'
        exp_synthetic(path, exp_num=exp_num)
    elif args.dataset == 'law':
        path = '../dataset/law_data.csv'
        exp_real(path, exp_num=exp_num)
    elif args.dataset == 'adult':
        path = '../dataset/adult.data'
        exp_real(path, exp_num=exp_num)