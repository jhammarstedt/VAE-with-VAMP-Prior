import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from utils import *
from nn import *
import matplotlib.pyplot as plt
import math


class VanillaVAE(nn.Module):
    def __init__(self, input_size: int, hidden_dims: int, latent_dims: int, args: dict):
        super(VanillaVAE, self).__init__()
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.latent_dims =latent_dims
        self.input_type = args['input_type']
        self.priors = args['prior']
        self.psudo_input_size = args['psudo_inp']


        # encoder p(z|x) - encode our input into the latent space with hopes to get as good representation as possible in less dimensions
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims),  # here we lower the dimensions to match our latent space
            nn.Tanh(),
        )

        # The expected value and variance of z given the input are computed through NNs
        if self.input_type == 'continuous':
            self.enc_mu = nn.Linear(hidden_dims, latent_dims)
            self.enc_logvar = nn.Sequential(
                nn.Linear(hidden_dims, latent_dims),
                nn.Hardtanh(min_val=-6., max_val=2.)
            )
        elif self.input_type == 'binary':
            self.enc_mu = nn.Linear(hidden_dims, latent_dims)

        # decoder q(x|z)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, input_size),  # here we increase the dimensions to match the original image
            nn.Tanh()
        )

        if self.prior == 'vamp':
            self.K = 200  # nbr of psudo inputs/components
            self.pseudo_input = torch.eye(self.K, self.K, requires_grad=False)  # initializing psudo inputs to just be identity

            # mapper maps from nbr of components to input size- in case of mnist 200-> 784
            self.psudo_mapper = PsudoInpMapping(in_size=self.K, out_size=self.input_size)  # ? do we need to initialize the network like they do in the paper



            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.psudo_mapper.to(device)
            self.pseudo_input = self.pseudo_input.to(device)

        def he_init(m):
            s = np.sqrt(2. / m.in_features)
            m.weight.data.normal_(0, s)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

    def encode(self, x):
        x = self.encoder(x)  # encode the data to reduce dimensionality
        mu = self.enc_mu(x)  # get the expected value in the latent space w.r.t x
        logvar = self.enc_logvar(x)  # get the variance of the latent space w.r.t x
        return mu, logvar

    def decode(self, x):
        return self.decoder(x)

    def reparametrization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_loss(self, data, beta=0.7, choice_of_prior='standard'):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        
        Lower_bound = - log_p(z|x) - KL(q_z||p_z) where log_p(z|x) is the reconstruction error
        In variational autoencoders, the loss function is composed of a reconstruction term 
        (that makes the encoding-decoding scheme efficient) and a regularisation term (that makes the latent space regular).
        """
        dim = 1
        # Running the vanillaVAE
        reconstruction, true_input, z_mu, z_lvar, z_sample = self.forward(data)
        # plt.imshow(data[0].reshape(28, 28))
        # plt.show()
        # plt.imshow(reconstruction[0].detach().numpy().reshape(28, 28))
        # plt.show()

        # compute reconstruction error
        # start off with MSE but we'll fix that later
        loss = nn.MSELoss(reduction='sum')
        recon_error = loss(reconstruction, true_input)  # temp


        p_z = self.get_z_prior(z_sample=z_sample, dim=dim)
        # q_z = torch.mean(-0.5 * torch.pow(z_sample, 2), dim=dim) #get the true distribtion

        q_z = torch.sum(-0.5 * (z_lvar + torch.pow(z_sample - z_mu, 2) / torch.exp(z_lvar)),
                         dim=dim)  # the approximated dist, should it be mean or not?

        # p_z_ = log_Normal_standard(z_sample, dim=1)
        # q_z_ = log_Normal_diag(z_sample, z_mu, z_lvar, dim=1)
        '''
        They are logged already and can therefore just be subtracted, 
        and in accordance with module 10 we take the expected value and 
        are only left with the logs in the KL'''
        KL = - (p_z - q_z)

        loss = recon_error + beta * KL
        loss = torch.mean(loss)
        recon_error = torch.mean(recon_error)
        KL = torch.mean(KL)

        return loss, recon_error, KL

    def vamp_prior(self, z):
        K = self.psudo_input_size  # nbr of psudo inputs/components

        psudo_input = self.psudo_mapper(self.pseudo_input)  # learn how to get best mapping
        prior_mean, prior_logvar = self.encode(psudo_input)  # running the encoding with the psi params

        # ! --- Need to change, their code
        # expand z
        z_expand = z.unsqueeze(1)
        means = prior_mean.unsqueeze(0)
        logvars = prior_logvar.unsqueeze(0)

        a = torch.sum(-0.5 * (logvars + torch.pow(z_expand - means, 2) / torch.exp(logvars)),
                      dim=2) - math.log(K)

        #a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(K)  # MB x C
        a_max, _ = torch.max(a, 1)  # MB x 1

        # calculte log-sum-exp
        log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB
        # ! ---

        return log_prior


    def get_z_prior(self, z_sample, dim):
        if self.prior == 'standard':
            log_p = torch.mean(-0.5 * torch.pow(z_sample, 2),
                               dim=dim)  # get the prior that we are pulling the posterior towards by KL
        elif self.prior == 'vamp':
            log_p = self.vamp_prior(z_sample)
            # implement vamp prior
        else:
            raise TypeError("Need to specify the type of prior")

        return log_p

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
        return self.decode(z), x, mu, logvar, z  # also need to return samples of z


class PsudoInpMapping(nn.Module):
    def __init__(self, in_size, out_size):
        super(PsudoInpMapping, self).__init__()

        self.mapper = nn.Linear(int(in_size), int(out_size), bias=False)
        self.activate = nn.Hardtanh(min_val=0.0, max_val=1.0)
        pseudoinputs_mean = 0.05
        pseudoinputs_std = 0.01
        self.mapper.weight.data.normal_(pseudoinputs_mean, pseudoinputs_std)
        # self.mapper.apply(normal_init(mapper.))
        # normal_init(self.mapper.linear, pseudoinputs_mean, pseudoinputs_std)

    def forward(self, x):
        X = self.mapper(x)
        X = self.activate(X)  # activate with Hardtanh
        return X
