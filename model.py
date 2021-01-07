import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from utils import *
import matplotlib.pyplot as plt

class VanillaVAE(nn.Module):
    def __init__(self, input_size: int, hidden_dims: int, latent_dims: int):
        super(VanillaVAE, self).__init__()

        #encoder p(z|x) - encode our input into the latent space with hopes to get as good representation as possible in less dimensions
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_dims), 
            nn.Tanh(),
            nn.Linear(hidden_dims, hidden_dims), #here we lower the dimensions to match our latent space
            nn.Tanh(),
        )
        
        # The expected value and variance of z given the input are computed through NNs
        self.enc_mu = nn.Linear(hidden_dims, latent_dims) 
        self.enc_logvar = nn.Sequential(
            nn.Linear(hidden_dims, latent_dims),
            nn.Hardtanh(min_val=-6.,max_val=2.)
        )

        #decoder q(x|z) 
        self.decoder = nn.Sequential(
            nn.Linear(latent_dims, hidden_dims),
            nn.Tanh(),
            nn.Linear(hidden_dims, input_size), #here we increase the dimensions to match the original image
            nn.Tanh()
        )

        def he_init(m):
            s = np.sqrt(2. / m.in_features)
            m.weight.data.normal_(0, s)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

        
    def encode(self, x):
        x = self.encoder(x) # encode the data to reduce dimensionality 
        mu = self.enc_mu(x) # get the expected value in the latent space w.r.t x
        logvar = self.enc_logvar(x) #get the variance of the latent space w.r.t x
        return mu, logvar

    def decode(self, x):
        return self.decoder(x) 

    def reparametrization(self, mu, logvar):
        # std = torch.exp(0.5 * logvar)
        # eps = torch.randn_like(std)
        # return eps * std + mu
        eps = torch.FloatTensor(logvar.size()).normal_()
        eps = Variable(eps)
        return eps.mul(logvar).add_(mu)

    def get_loss(self,data,beta=0.01,choice_of_prior='standard'):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        
        Lower_bound = - log_p(z|x) - KL(q_z||p_z) where log_p(z|x) is the reconstruction error
        In variational autoencoders, the loss function is composed of a reconstruction term 
        (that makes the encoding-decoding scheme efficient) and a regularisation term (that makes the latent space regular).
        """
        dim = 1
        #Running the vanillaVAE
        reconstruction,true_input,z_mu,z_lvar,z_sample = self.forward(data)
        # plt.imshow(data[0].reshape(28, 28))
        # plt.show()
        # plt.imshow(reconstruction[0].detach().numpy().reshape(28, 28))
        # plt.show()
        #compute reconstruction error
        #start off with MSE but we'll fix that later
        loss = nn.MSELoss()
        recon_error = loss(reconstruction,true_input) # temp

        #get prior
        prior_type= 'standard' # should be and argument in the beginning

        q_z = self.get_z_prior(z= z_sample,type_of_prior=prior_type,dim=dim) #get the true distribtion
        p_z = torch.mean(-0.5 * (z_lvar+torch.pow(z_sample-z_mu,2)/torch.exp(z_lvar)),dim=dim) #the approximated dist, should it be mean or not?

        q_z_ = log_Normal_standard(z_sample, dim=1)
        p_z_ = log_Normal_diag(z_sample, z_mu, z_lvar, dim=1)
        '''
        They are logged already and can therefore just be subtracted, 
        and in accordance with module 10 we take the expected value and 
        are only left with the logs in the KL'''
        KL = - (p_z_ - q_z_)


        #kld_loss = torch.mean(-0.5 * torch.sum(1 + z_lvar - z_mu ** 2 - z_lvar.exp(), dim = 1), dim = 0)
        loss = recon_error + beta*KL
        #loss = recon_error + beta *KL #the loss is the lower bound we will later use 
        loss = torch.mean(loss)
        recon_error = torch.mean(recon_error)
        KL = torch.mean(KL)
        print(KL)
        if KL < -100:
            KL = KL
        return loss, recon_error, KL


    def get_psudo_inputs(self):
        pass

    
    def get_z_prior(self,type_of_prior,z,dim):
        if type_of_prior == 'standard':
            log_p = torch.mean(-0.5 * torch.pow(z, 2 ),dim=dim) #standard normal prior
        elif type_of_prior == 'Vamp':
                     
            #implement vamp prior
            pass
        else:
            raise TypeError("Need to specify the type of prior")

        return log_p

        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
        return self.decode(z), x, mu, logvar, z #also need to return samples of z