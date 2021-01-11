from nn import *
from scipy.special import logsumexp
import math


class VAE_model(nn.Module):
    def __init__(self, input_size: int, args: dict):
        super(VAE_model, self).__init__()
        self.input_size = input_size
        self.hidden_dims = args['hidden_dims']
        self.latent_dims = args['latent_dims']
        self.input_type = args['input_type']
        self.prior = args['prior']
        self.psudo_input_size = args['psudo_inp']

        # encoder p(z|x) - encode our input into the latent space with hopes to get as good representation as possible in less dimensions
        self.encoder = nn.Sequential(
            nn.Linear(input_size, self.hidden_dims),
            nn.Tanh(),
            nn.Linear(self.hidden_dims, self.hidden_dims),  # here we lower the dimensions to match our latent space
            nn.Tanh(),
        )

        # The expected value and variance of z given the input are computed through NNs
        if self.input_type == 'continuous':
            self.enc_mu = nn.Linear(self.hidden_dims, self.latent_dims)
            self.enc_logvar = nn.Sequential(
                nn.Linear(self.hidden_dims, self.latent_dims),
                nn.Hardtanh(min_val=-6., max_val=2.)
            )
        elif self.input_type == 'binary':
            self.enc_mu = nn.Linear(self.hidden_dims, self.latent_dims)

        # decoder q(x|z)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dims, self.hidden_dims),
            nn.Tanh(),
            nn.Linear(self.hidden_dims, input_size),  # here we increase the dimensions to match the original image
            nn.Tanh()
        )

        self.p_x_mean = NonLinear(input_size=self.input_size,output_size=784, activation=nn.Sigmoid())
        self.p_x_logvar = NonLinear(input_size=self.input_size, output_size=np.prod(self.input_size),
                                    activation=nn.Hardtanh(min_val=-4.5, max_val=0))

        if self.prior == 'vamp':
            self.K = 200  # nbr of psudo inputs/components
            self.pseudo_input = torch.eye(self.K, self.K,
                                          requires_grad=False)  # initializing psudo inputs to just be identity

            # mapper maps from nbr of components to input size- in case of mnist 200-> 784
            self.psudo_mapper = PsudoInpMapping(in_size=self.K,
                                                out_size=self.input_size)  # ? do we need to initialize the network like they do in the paper

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
        z = self.decoder(x)
        x_mean = self.p_x_mean(z)
        x_mean = torch.clamp(x_mean, min=0. + 1. / 512., max=1. - 1. / 512.)
        x_logvar = self.p_x_logvar(z)

        return x_mean, x_logvar, z

    def reparametrization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    # ! Their code
    # !#######################################################
    def log_Logistic_256(self,x, mean, logvar, average=False, reduce=True, dim=None):
        bin_size = 1. / 256.
        # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
        scale = torch.exp(logvar)
        x = (torch.floor(x / bin_size) * bin_size - mean) / scale
        cdf_plus = torch.sigmoid(x + bin_size / scale)
        cdf_minus = torch.sigmoid(x)

        # calculate final log-likelihood for an image
        log_logist_256 = - torch.log(cdf_plus - cdf_minus + 1.e-7)

        return torch.sum(log_logist_256, dim)

    # THE MODEL: GENERATIVE DISTRIBUTION
    def p_x(self, z):
        z = self.decoder(z)

        x_mean = self.p_x_mean(z)
        x_mean = torch.clamp(x_mean, min=0. + 1. / 512., max=1. - 1. / 512.)
        x_logvar = self.p_x_logvar(z)
        return x_mean, x_logvar

    # !#######################################################

    def get_loss(self, data, beta=0.7, warmup=True):
        """
        Computes the VAE loss function.

        Lower_bound = - log_p(z|x) - KL(q_z||p_z) where log_p(z|x) is the reconstruction error
        In variational autoencoders, the loss function is composed of a reconstruction term 
        (that makes the encoding-decoding scheme efficient) and a regularisation term (that makes the latent space regular).
        """
        x_mean,x_logvar,reconstruction, true_input, z_mu, z_lvar, z_sample = self.forward(data)

        # compute reconstruction error
        loss = nn.MSELoss(reduction='mean')
        recon_error = loss(reconstruction, true_input)  # temp


        p_z = self.get_z_prior(z_sample=z_sample, dim=1)
        q_z = torch.sum(-0.5 * (z_lvar + torch.pow(z_sample - z_mu, 2) / torch.exp(z_lvar)),
                        dim=1)  # Get the approximated distribution


        '''
        They are logged already and can therefore just be subtracted, 
        and in accordance with module 10 we take the expected value and 
        are only left with the logs in the KL'''
        KL = - (p_z - q_z)
        if warmup: #skipping beta for the first epoch
            loss = -recon_error + KL
        else:
            loss = -recon_error + beta * KL
        loss = torch.mean(loss)
        recon_error = torch.mean(recon_error)
        KL = torch.mean(KL)

        return loss, recon_error, KL

    def compute_LL(self, test_data, ll_no_samples=5000, ll_batch_size=100):
        """
        computes the log-liklihood
        :param test_data: test data
        :param ll_no_samples: no of samples for the log likelihood estimation
        :param ll_batch_size: batch size for the log likelihood estimation
        :return:
        """

        no_runs = int(ll_no_samples / ll_batch_size) if ll_no_samples > ll_batch_size else 1
        data_N = test_data.size(0)

        likelihood_mc = np.zeros((data_N, 1))
        for i, data_item in enumerate(test_data):
            data_item = data_item.unsqueeze(0)

            results = np.zeros((no_runs, 1))
            for j in range(no_runs):
                # x = x_single.expand(S, data_item.size(1))
                tmp_loss, _, _ = self.get_loss(data_item)
                results[j] = (-tmp_loss.cpu().data.numpy())

            # calculate max
            results = np.reshape(results, (results.shape[0] * results.shape[1], 1))
            likelihood_x = logsumexp(results)
            likelihood_mc[i] = (likelihood_x - np.log(no_runs))

        return -np.mean(likelihood_mc)

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
                      dim=2) - math.log(K)  # ? Why log(K)?

        # a = log_Normal_diag(z_expand, means, logvars, dim=2) - math.log(K)  # MB x C
        a_max, _ = torch.max(a, 1)  # MB x 1

        # calculte log-sum-exp
        log_prior = a_max + torch.log(torch.sum(torch.exp(a - a_max.unsqueeze(1)), 1))  # MB
        # ! ---

        return log_prior

    def GM_prior(self, z):
        """Here we implement the guassian mixture prior"""
        K = self.psudo_input_size  # same idea as vamp
        raise NotImplementedError()

    def get_z_prior(self, z_sample, dim):
        if self.prior == 'standard':
            log_p = torch.mean(-0.5 * torch.pow(z_sample, 2),
                               dim=dim)  # get the prior that we are pulling the posterior towards by KL
        elif self.prior == 'vamp':
            log_p = self.vamp_prior(z_sample)
        elif self.prior == 'GM':
            log_p = self.GM_prior(z_sample)
        else:
            raise TypeError("Need to specify the type of prior")

        return log_p

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_dim)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z = z.to(device)
        return self.decode(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
        x_mean, x_logvar, recon = self.decode(z)
        return x_mean, x_logvar, recon, x, mu, logvar, z  # also need to return samples of z


class PsudoInpMapping(nn.Module):
    def __init__(self, in_size, out_size):
        super(PsudoInpMapping, self).__init__()

        self.mapper = nn.Linear(int(in_size), int(out_size), bias=False)
        self.activate = nn.Hardtanh(min_val=0.0, max_val=1.0)
        pseudoinputs_mean = 0.05
        pseudoinputs_std = 0.01
        self.mapper.weight.data.normal_(pseudoinputs_mean, pseudoinputs_std)

    def forward(self, x):
        X = self.mapper(x)
        X = self.activate(X)  # activate with Hardtanh
        return X
