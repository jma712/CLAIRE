import random
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

class Claire_model(nn.Module):
    def __init__(self, x_dim, args, type='pred'):
        super(Claire_model, self).__init__()
        self.x_dim = x_dim
        self.h_dim = args.h_dim
        self.type = type
        self.mlp = nn.Sequential(nn.Linear(x_dim, self.h_dim), nn.LeakyReLU(),
                                 #nn.Linear(self.h_dim, self.h_dim), nn.LeakyReLU(),
                                 #nn.Linear(self.h_dim, self.h_dim), nn.LeakyReLU()
                                 )  # if you need more layers
        self.pred = nn.Sequential(nn.Linear(self.h_dim, 1))

    def forward(self, x, env):
        rep = self.mlp(x)
        y_pred = self.pred(rep)
        if self.type == 'class':
            y_pred = torch.sigmoid(y_pred)
        return y_pred

class Claire_vae(nn.Module):
    def __init__(self, args, dim_x, num_s):
        super(Claire_vae, self).__init__()
        self.device = args.device
        self.decoder_type = args.decoder_type
        self.num_s = num_s
        self.dim_x = dim_x
        self.dim_h = args.vae_h_dim

        self.mu_h = nn.Sequential(nn.Linear(dim_x, self.dim_h), nn.LeakyReLU(), nn.Linear(self.dim_h, self.dim_h))
        self.logvar_h = nn.Sequential(nn.Linear(dim_x, self.dim_h), nn.LeakyReLU(), nn.Linear(self.dim_h, self.dim_h))

        if self.decoder_type == 'together':  # train a decoder: H + S -> X (non-sensitive features)
            self.decoder_elem = nn.Sequential(nn.Linear(self.dim_h + 1, self.dim_h), nn.LeakyReLU(), nn.Linear(self.dim_h, dim_x))
        elif self.decoder_type == 'separate':  # separately train a decoder for each S: H -> X
            self.decoder_elem = [nn.Sequential(nn.Linear(self.dim_h, dim_h), nn.LeakyReLU(), nn.Linear(self.dim_h, dim_x)) for i in range(num_s)]

    def encoder(self, data):
        mu_h = self.mu_h(data)  # x, y
        logvar_h = self.logvar_h(data)

        return mu_h, logvar_h

    def get_embeddings(self, data):
        mu_h, logvar_h = self.encoder(data)
        return mu_h, logvar_h

    def reparameterize(self, mu, logvar):
        if self.training:
            # do this only while training
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decoder(self, h_sample, s):
        if self.decoder_type == 'together':  #
            input_dec = torch.cat([h_sample, s], dim=1)
            data_reconst = self.decoder_elem(input_dec)
        elif self.decoder_type == 'separate':
            data_reconst = torch.zeros((len(h_sample), self.dim_x+1)).to(args.device)
            for i in range(self.num_s):
                idx_si = torch.where(s == i)
                input_dec_i = h_sample[idx_si]
                data_reconst_i = self.decoder_elem[i](input_dec_i)
                data_reconst[idx_si] = data_reconst_i
        return data_reconst

    def forward(self, data, s):  # data: n x d, s: n x 1
        mu_h, logvar_h = self.encoder(data)
        h_sample = self.reparameterize(mu_h, logvar_h)

        data_reconst = self.decoder(h_sample, s)

        result_all = {'reconstruct': data_reconst, 'mu_h': mu_h, 'logvar_h': logvar_h}
        return result_all

class Causal_model_vae(nn.Module):
    def __init__(self, args, dim_x, num_s, dim_h):
        super(Causal_model_vae, self).__init__()
        self.device = args.device
        self.decoder_type = args.decoder_type
        self.num_s = num_s
        self.dim_x = dim_x
        self.dim_h = dim_h

        self.mu_h = nn.Sequential(nn.Linear(dim_x, self.dim_h), nn.LeakyReLU(), nn.Linear(self.dim_h, self.dim_h))
        self.logvar_h = nn.Sequential(nn.Linear(dim_x, self.dim_h), nn.LeakyReLU(), nn.Linear(self.dim_h, self.dim_h))

        if self.decoder_type == 'together':  # train a decoder: H + S -> X (non-sensitive features)
            self.decoder_elem = nn.Sequential(nn.Linear(self.dim_h + 1, self.dim_h), nn.LeakyReLU(), nn.Linear(self.dim_h, dim_x))
        elif self.decoder_type == 'separate':  # separately train a decoder for each S: H -> X
            self.decoder_elem = [nn.Sequential(nn.Linear(self.dim_h, dim_h), nn.LeakyReLU(), nn.Linear(self.dim_h, dim_x)) for i in range(num_s)]

    def encoder(self, data):
        mu_h = self.mu_h(data)  # x, y
        logvar_h = self.logvar_h(data)

        return mu_h, logvar_h

    def get_embeddings(self, data):
        mu_h, logvar_h = self.encoder(data)
        return mu_h, logvar_h

    def reparameterize(self, mu, logvar):
        if self.training:
            # do this only while training
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decoder(self, h_sample, s):
        if self.decoder_type == 'together':  #
            input_dec = torch.cat([h_sample, s], dim=1)
            data_reconst = self.decoder_elem(input_dec)
        elif self.decoder_type == 'separate':
            data_reconst = torch.zeros((len(h_sample), self.dim_x+1)).to(args.device)
            for i in range(self.num_s):
                idx_si = torch.where(s == i)
                input_dec_i = h_sample[idx_si]
                data_reconst_i = self.decoder_elem[i](input_dec_i)
                data_reconst[idx_si] = data_reconst_i
        return data_reconst

    def forward(self, data, s):  # data: n x d, s: n x 1
        mu_h, logvar_h = self.encoder(data)
        h_sample = self.reparameterize(mu_h, logvar_h)

        data_reconst = self.decoder(h_sample, s)

        result_all = {'reconstruct': data_reconst, 'mu_h': mu_h, 'logvar_h': logvar_h, 'h_sample': h_sample}
        return result_all

    def get_latent_var(self, data):
        mu_h, logvar_h = self.encoder(data)
        h_sample = self.reparameterize(mu_h, logvar_h)
        h_sample = h_sample.view(-1, self.dim_h).cpu().detach().numpy()
        return h_sample