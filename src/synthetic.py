'''
generate synthetic data
'''
import numpy as np
import pickle

# seed
np.random.seed(1)

def generate_synthetic(n, save_path=None):
    pi = [0.5, 0.4, 0.05, 0.05]
    W_s = [0.1, 0.2, 1.0, 2.0]
    sigma_u = 1.0
    sigma_0 = 1.0
    sigma_s = 0.1 * np.array([0.5, 1.0, 1.5, 2.0])
    sigma_s_1 = sigma_s
    sigma_s_y = sigma_s
    sigma_s_2 = sigma_s

    num_s = len(pi)

    U = np.random.normal(loc=0, scale=sigma_u, size=(n, 1))
    S = np.random.choice(num_s, (n, 1), p=pi)  # np.random.randint(num_s, size=(n, 1))
    X_0 = np.random.normal(loc=0, scale=sigma_0, size=(n, 1))  # N x dim

    idx = []
    n_list = []
    for i in range(num_s):
        idx.append(np.where(S.reshape(-1) == i)[0])
        n_list.append(np.count_nonzero(S.reshape(-1) == i))
    print('number of different classes: ', n_list)

    # different sensitive subgroups
    X_1 = np.zeros((n, 1))
    Y = np.zeros((n, 1))
    X_2 = np.zeros((n, 1))
    for i in range(num_s):
        size_i = len(idx[i])
        epsilon_1 = np.random.normal(loc=0, scale=sigma_s_1[i], size=(size_i, 1))
        epsilon_y = np.random.normal(loc=0, scale=sigma_s_y[i], size=(size_i, 1))
        epsilon_2 = np.random.normal(loc=0, scale=sigma_s_2[i], size=(size_i, 1))
        X_1[idx[i]] = W_s[i] * S[idx[i]] + U[idx[i]] + epsilon_1
        Y[idx[i]] = X_1[idx[i]] + X_0[idx[i]] + epsilon_y
        X_2[idx[i]] = Y[idx[i]] + epsilon_2

    data = {
        'S': S,
        'U': U,
        'X_0': X_0,
        'X_1': X_1,
        'X_2': X_2,
        'Y': Y
    }

    params = {
        'pi': pi,
        'W_s': W_s,
        'sigma_u': sigma_u,
        'sigma_0': sigma_0,
        'sigma_s_1': sigma_s_1,
        'sigma_s_y': sigma_s_y,
        'sigma_s_2': sigma_s_2
    }

    data_save = {'data': data, 'params': params}
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(data_save, f)
        print('data saved! ', save_path)

    return data_save


if __name__ == '__main__':
    save_path = '../dataset/synthetic_data.pickle'
    data_save = generate_synthetic(5000, save_path)

