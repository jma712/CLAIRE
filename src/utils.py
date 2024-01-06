import numpy as np
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import pandas as pd
import csv
import random
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE as tsn
from sklearn import metrics

font_sz = 24
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
matplotlib.rcParams.update({'font.size': font_sz})

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('using device: ', device)

def select_index(data_save, idx_select):
    data_select = {}
    data_select['data'] = {k: data_save['data'][k][idx_select] for k in data_save['data']}
    if 'params' in data_save:
        data_select['params'] = data_save['params']
    data_select['index'] = data_save['index']
    return data_select

def standerlize(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x = (x - x_mean) / (x_std + 0.000001)
    return x

# def evaluate_fairness(y_cf1, y_cf2):  # y_cf1: n x samplenum
#     mmd = mmd2_lin(y_cf1, y_cf2, 0.3)
#     wass, _ = wasserstein(y_cf1, y_cf2, device, cuda=True)
#
#     return mmd, wass

def data_statistics(path, name):
    print('checking dataset: ', name)
    if name == 'law':
        csv_data = pd.read_csv(path)
        print("instance num and feature num: ", csv_data.shape)

        # sensitive attributes
        race = csv_data['race'].value_counts()
        print(race)

        sex = csv_data['sex'].value_counts()
        print(sex)
    if name == 'adult':
        csv_data = pd.read_csv(path)
        print("instance num and feature num: ", csv_data.shape)

        # sensitive attributes
        race = csv_data['race'].value_counts()
        print(race)

        sex = csv_data['sex'].value_counts()
        print(sex)
    return

def draw_freq(data, x_label=None, bool_discrete = False, title="Title"):
    '''
    :param data: (n,) array or sequence of (n,) arrays
    :param x_label:
    :param bool_discrete:
    :return:
    '''
    fig = plt.figure()
    plt.hist(data, bins=50)

    plt.xlabel(x_label)
    plt.ylabel("Frequency")

    ax = fig.add_subplot(1, 1, 1)

    # Find at most 10 ticks on the y-axis
    if not bool_discrete:
        max_xticks = 10
        xloc = plt.MaxNLocator(max_xticks)
        ax.xaxis.set_major_locator(xloc)

    plt.title(title)
    plt.show()

def load_data(path, name):
    print('loading dataset: ', name)
    if name == 'law':
        csv_data = pd.read_csv(path)

        # index selection
        selected_races = ['White', 'Black', 'Asian']
        print("select races: ", selected_races)
        select_index = np.array(csv_data[(csv_data['race'] == selected_races[0]) | (csv_data['race'] == selected_races[1]) |
                                         (csv_data['race'] == selected_races[2])].index, dtype=int)
        # shuffle
        np.random.shuffle(select_index)

        LSAT = csv_data[['LSAT']].to_numpy()[select_index]  # n x 1
        UGPA = csv_data[['UGPA']].to_numpy()[select_index]  # n x 1
        x = csv_data[['LSAT','UGPA']].to_numpy()[select_index]  # n x d
        ZFYA = csv_data[['ZFYA']].to_numpy()[select_index]  # n x 1
        sex = csv_data[['sex']].to_numpy()[select_index] - 1  # n x 1

        n = ZFYA.shape[0]
        rr = csv_data['race']
        env_race = csv_data['race'][select_index].to_list()  # n, string list
        env_race_id = np.array([selected_races.index(env_race[i]) for i in range(n)]).reshape(-1, 1)

        # env_race_onehot = np.zeros((n, len(selected_races)))
        # for i in range(n):
        #     env_race_onehot[i][selected_races.index(env_race[i])] = 1.   # n x len(selected_races)
        #     #env_sex_onehot[i][sexes.index(env_sex[i])] = 1.
        # #env = np.concatenate([env_race_onehot, env_sex_onehot], axis=1)  # n x (No. races + No. sex)
        # env = env_race_onehot

        data_save = {'data': {'LSAT': LSAT, 'UGPA': UGPA, 'ZFYA': ZFYA, 'race': env_race_id, 'sex': sex}}

        # plot
        flag_plot = False
        if flag_plot:
            font_sz = 20
            fig, ax = plt.subplots(nrows=2, ncols= 2 * len(selected_races), figsize=(12, 6), sharey=True)

            for race_idx in range(len(selected_races)):
                race_cur = selected_races[race_idx]
                data_i = csv_data[(csv_data["race"] == race_cur)]

                ax[race_idx][0].scatter(data_male['LSAT'],
                                data_male['ZFYA'], 3, marker='o', label=race_cur, color='black')
                ax[race_idx][1].scatter(data_male['UGPA'],
                                data_male['ZFYA'], 5, marker='o', label=race_cur, color='blue')

                ax[race_idx][0].set(xlabel="LSAT", ylabel="FYA", title=race_cur)
                ax[race_idx][1].set(xlabel="UGPA", ylabel="FYA", title=race_cur)

            plt.show()

    if name == 'adult':
        csv_data = pd.read_csv(path)
        csv_data.replace(' ?', np.NaN, inplace=True)  # Replacing all the missing values with NaN

        csv_data.columns = ["Age", "WorkClass", "fnlwgt", "Education", "EducationNum", "MaritalStatus", "Occupation", "Relationship", "Race", "Sex",
                            "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"]  # for ease of human interpretation

        column_of_interest = ['Race', 'Sex', 'MaritalStatus', 'Occupation', 'EducationNum', 'HoursPerWeek', 'Income']
        column_of_interest_cat = ['Sex', 'MaritalStatus', 'Occupation']
        column_of_interest_num = ['EducationNum', 'HoursPerWeek']
        csv_data = csv_data[column_of_interest]

        # drop NaNs
        csv_data.dropna(axis=0, how='any', inplace=True)  # Dropping all the missing values (hence reduced training set)

        # index selection
        selected_races = ['White', 'Black', 'Asian-Pac-Islander']  # races = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
        print("select races: ", selected_races)
        csv_data['Race'] = csv_data['Race'].str.strip()
        csv_data = csv_data.loc[(csv_data['Race'] == selected_races[0]) | (csv_data['Race'] == selected_races[1]) | (csv_data['Race'] == selected_races[2])]

        Income = csv_data["Income"].map({' <=50K': 0, ' >50K': 1}).to_numpy().reshape(-1, 1)  # just to give binary labels
        csv_data.drop(["Income"], axis=1, inplace=True)

        n = len(csv_data.index)
        select_index = np.arange(n)
        np.random.shuffle(select_index)
        #
        env_race = csv_data['Race'].values[select_index]  # n, string list
        env_race_id = np.array([selected_races.index(env_race[i]) for i in range(n)]).reshape(-1, 1)

        data_save = {'data': {'Race': env_race_id, 'Income': Income[select_index]}}

        # categorical features: one hot
        for cat in column_of_interest_cat:
            encoding_pipeline = Pipeline([
                ('encode_cat', ce.OneHotEncoder(cols=cat, return_df=True))
            ])
            feat_onehot = encoding_pipeline.fit_transform(csv_data[[cat]]).values
            data_save['data'][cat] = feat_onehot[select_index]

        # numbers
        for nu in column_of_interest_num:
            data_save['data'][nu] = csv_data[nu].to_numpy()[select_index].reshape(-1, 1)

    return data_save # x, y, env


def get_index_for_env(env_cur, distinct_envs, args):
    idx_each_env = []  # idx_each_env[i]: index of those "env = distinct_envs[i]" in env_cur
    for e in distinct_envs:
        idx_e = []
        if args.cuda:
            e=e.to(device)
        for i in range(env_cur.shape[0]):
            if torch.equal(env_cur[i], e):
                idx_e.append(i)

        idx_e = torch.LongTensor(idx_e)
        if args.cuda:
            idx_e = idx_e.to(device)

        idx_each_env.append(idx_e)
    return idx_each_env

def make_environments(x, y, env, args):
    '''
    separate the data by different environments
    :param x: tensor, shape: n x d
    :param y: tensor, shape: n x 1
    :param env: tensor, shape: n x d_e
    :return: a list, size = Num of environments, each elem is a dict {x, y, env}
    '''
    env_data = []
    distinct_envs = torch.unique(env, dim=0)  # unique rows
    for e in distinct_envs:
        #idx_e = torch.where(env == e)
        idx_e = []
        for i in range(env.shape[0]):
            if torch.equal(env[i], e):
                idx_e.append(i)

        idx_e = torch.LongTensor(idx_e)
        if args.cuda:
            idx_e = idx_e.to(device)

        env_data.append({
            'x': x[idx_e],
            'y': y[idx_e],
            'env': env[idx_e],
            'orin_index': idx_e
        })
    return env_data

def split_data(n, exp_num=10, rates=[0.6, 0.2, 0.2], labels=None, type='random', sorted=False, label_number=1000000):
    idx_train_list = []
    idx_val_list = []
    idx_test_list = []

    trn_rate, val_rate, tst_rate = rates[0], rates[1], rates[2]

    if type == 'ratio':  # follow the original ratio of label distribution, only applicable to binary classification!
        label_idx_0 = np.where(labels == 0)[0]
        label_idx_1 = np.where(labels == 1)[0]

        for i in range(exp_num):
            random.shuffle(label_idx_0)
            random.shuffle(label_idx_1)

            idx_train = np.append(label_idx_0[:min(int(trn_rate * len(label_idx_0)), label_number // 2)],
                                  label_idx_1[:min(int(trn_rate * len(label_idx_1)), label_number // 2)])
            idx_val = np.append(label_idx_0[int(trn_rate * len(label_idx_0)):int((trn_rate + val_rate) * len(label_idx_0))],
                                label_idx_1[int(trn_rate * len(label_idx_1)):int((trn_rate + val_rate) * len(label_idx_1))])
            idx_test = np.append(label_idx_0[int((trn_rate + val_rate) * len(label_idx_0)):], label_idx_1[int((trn_rate + val_rate) * len(label_idx_1)):])

            np.random.shuffle(idx_train)
            np.random.shuffle(idx_val)
            np.random.shuffle(idx_test)

            if sorted:
                idx_train.sort()
                idx_val.sort()
                idx_test.sort()

            idx_train_list.append(idx_train.copy())
            idx_val_list.append(idx_val.copy())
            idx_test_list.append(idx_test.copy())

    elif type == 'random':
        for i in range(exp_num):
            idx_all = np.arange(n)
            idx_train = np.random.choice(n, size=int(trn_rate * n), replace=False)
            idx_left = np.setdiff1d(idx_all, idx_train)
            idx_val = np.random.choice(idx_left, int(val_rate * n), replace=False)
            idx_test = np.setdiff1d(idx_left, idx_val)

            #sorted=True
            if sorted:
                idx_train.sort()
                idx_val.sort()
                idx_test.sort()

            idx_train_list.append(idx_train.copy())
            idx_val_list.append(idx_val.copy())
            idx_test_list.append(idx_test.copy())
    # elif type == "balanced":

    return idx_train_list, idx_val_list, idx_test_list

def split_trn_tst_random(trn_rate, tst_rate, n):
    trn_id_list = random.sample(range(n), int(n * trn_rate))
    not_trn = list(set(range(n)) - set(trn_id_list))
    tst_id_list = random.sample(not_trn, int(n * tst_rate))
    val_id_list = list(set(not_trn) - set(tst_id_list))
    trn_id_list.sort()
    val_id_list.sort()
    tst_id_list.sort()
    return trn_id_list, val_id_list, tst_id_list

def mean_nll(logits, y):
    #return nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss_mse = nn.MSELoss(reduction='mean').to(device)
    return loss_mse(logits.view(-1), y.view(-1))

def penalty(logits, y):
    scale = torch.tensor(1.).to(device).requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

# def mmd2_lin(Xt, Xc,p):
#     ''' Linear MMD '''
#     mean_control = torch.mean(Xc,0)
#     mean_treated = torch.mean(Xt,0)
#
#     mmd = torch.sum((2.0*p*mean_treated - 2.0*(1.0-p)*mean_control) ** 2)
#
#     return mmd

def mmd_linear(X, Y, p=0.003):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Returns:
        [scalar] -- [MMD value]
    """
    delta = (X.mean(0) - Y.mean(0))/p
    mmd = delta.dot(delta.T)
    return mmd


def mmd(x, y, gamma=0.3, kernel='rbf'):
    # http://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm

    n, d = x.shape
    m, d2 = y.shape
    assert d == d2

    # gamma
    # if gamma is None:
    #     dists = torch.pdist(torch.cat([x, y], dim=0))
    #     gamma = dists[:100].median() / 2

    xy = torch.cat([x.detach(), y.detach()], dim=0)
    dists = torch.cdist(xy, xy, p=2.0)
    # we are a bit sloppy here as we just keep the diagonal and everything twice
    # note that sigma should be squared in the RBF to match the Gretton et al heuristic
    k = torch.exp((-1 / (2 * gamma ** 2)) * dists ** 2) + torch.eye(n + m).to(x.device) * 1e-5
    k_x = k[:n, :n]
    k_y = k[n:, n:]
    k_xy = k[:n, n:]
    # The diagonals are always 1 (up to numerical error, this is (3) in Gretton et al.)
    # note that their code uses the biased (and differently scaled mmd)
    mmd = k_x.sum() / (n * (n - 1)) + k_y.sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
    # n, d = x.shape
    # m, d2 = y.shape
    # assert d == d2
    #
    # xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    # rx = (xx.diag().unsqueeze(0).expand_as(xx))
    # ry = (yy.diag().unsqueeze(0).expand_as(yy))
    #
    # dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    # dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    # dxy = rx.t() + ry - 2. * zz  # Used for C in (1)
    #
    # XX, YY, XY = (torch.zeros(xx.shape).to(device),
    #               torch.zeros(xx.shape).to(device),
    #               torch.zeros(xx.shape).to(device))
    #
    # if kernel == "multiscale":
    #
    #     bandwidth_range = [0.2, 0.5, 0.9, 1.3]
    #     for a in bandwidth_range:
    #         XX += a ** 2 * (a ** 2 + dxx) ** -1
    #         YY += a ** 2 * (a ** 2 + dyy) ** -1
    #         XY += a ** 2 * (a ** 2 + dxy) ** -1
    #
    # if kernel == "rbf":
    #     # MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
    #     bandwidth_range = [10, 15, 20, 50]
    #     for a in bandwidth_range:
    #         XX += torch.exp(-gamma * 0.5 * dxx / a)
    #         YY += torch.exp(-gamma * 0.5 * dyy / a)
    #         XY += torch.exp(-gamma * 0.5 * dxy / a)
    #
    # mmd = XX.mean() + YY.mean() - 2 * XY.mean()  # torch.mean(XX + YY - 2. * XY)

    return mmd

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    """Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    norm : float
        The l_p norm to be used.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)


def wasserstein(x, y, device, p=0.5, lam=10, its=10, sq=False, backpropT=False, cuda=False, scal=0.005):
    """return W dist between x and y"""
    '''distance matrix M'''
    # from scipy.stats import wasserstein_distance
    # xx = x.view(-1).cpu().detach().numpy()
    # yy = y.view(-1).cpu().detach().numpy()
    # wd = wasserstein_distance(xx, yy)
    # wd = torch.tensor(wd, dtype=torch.float)
    # if cuda:
    #     wd = wd.to(device)
    # return wd, np.NaN

    nx = x.shape[0]
    ny = y.shape[0]

    #x = x.squeeze()
    #y = y.squeeze()

    #    pdist = torch.nn.PairwiseDistance(p=2)

    M = pdist(x, y)  # distance_matrix(x,y,p=2)

    '''estimate lambda and delta'''
    M_mean = torch.mean(M)
    M_drop = F.dropout(M, 0.5 / (nx * ny))
    delta = torch.max(M_drop).detach()
    eff_lam = (lam / M_mean).detach()

    '''compute new distance matrix'''
    Mt = M
    row = delta * torch.ones(M[0:1, :].shape).to(device)
    col = torch.cat([delta * torch.ones(M[:, 0:1].shape).to(device), torch.zeros((1, 1)).to(device)], 0)
    if cuda:
        #row = row.cuda()
        #col = col.cuda()
        row = row.to(device)
        col = col.to(device)
    Mt = torch.cat([M, row], 0)
    Mt = torch.cat([Mt, col], 1)

    '''compute marginal'''
    a = torch.cat([p * torch.ones((nx, 1)) / nx, (1 - p) * torch.ones((1, 1))], 0)
    b = torch.cat([(1 - p) * torch.ones((ny, 1)) / ny, p * torch.ones((1, 1))], 0)

    '''compute kernel'''
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1) * 1e-6
    if cuda:
        #temp_term = temp_term.cuda()
        #a = a.cuda()
        #b = b.cuda()
        temp_term = temp_term.to(device)
        a = a.to(device)
        b = b.to(device)
    K = torch.exp(-Mlam) + temp_term
    U = K * Mt
    ainvK = K / a

    u = a

    for i in range(its):
        u = 1.0 / (ainvK.matmul(b / torch.t(torch.t(u).matmul(K))))
        if cuda:
            #u = u.cuda()
            u = u.to(device)
    v = b / (torch.t(torch.t(u).matmul(K)))
    if cuda:
        #v = v.cuda()
        v = v.to(device)

    upper_t = u * (torch.t(v) * K).detach()

    E = upper_t * Mt
    D = 2 * torch.sum(E)
    D /= scal

    if cuda:
        #D = D.cuda()
        D = D.to(device)

    return D, Mlam

def safe_sqrt(x, lbound=1e-10):
    ''' Numerically safe version of pytorch sqrt '''
    return torch.sqrt(torch.clamp(x, lbound, np.inf))