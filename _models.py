from operator import le
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.distributions as D

# ---------------------------
# - Variational Autoencoder -
# ---------------------------
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2) # mu + log_var
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            torch.nn.Sigmoid())
        
    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z, library_size=None):
        x_hat = self.decoder(z)
        if library_size is not None:
            x_hat = x_hat * library_size
        return x_hat
    
    def forward(self, x, library_size=None):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z, library_size)
        return x_hat, mu, log_var

    def loss(self, x, x_hat, mu, log_var):
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss, kl_div

    # sample around cell
    def sample(self, x, n=10, scale=1., library_size=None):
        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        eps = scale * torch.randn(n, self.latent_dim)
        z = mu + eps * std
        x_hat = self.decode(z, library_size)
        return x_hat, z
    
# ---------------------------
# -  Negative Binomial VAE  -
# ---------------------------
class NBVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, r_init=2, scaling_type="library", extra_outputs=0):
        super(NBVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.extra_outputs = extra_outputs
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2) # mu + log_var
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim))
        
        self.nb = NBLayer(input_dim-extra_outputs, r_init=r_init, scaling_type=scaling_type)
        
    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        z = z.view(-1, self.latent_dim)
        x_hat = self.decoder(z)

        if self.extra_outputs:
            x_hat[:, :-self.extra_outputs] = self.nb(x_hat[:, :-self.extra_outputs])
        else:
            x_hat = self.nb(x_hat)
            
        return x_hat
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    def loss(self, x, x_hat, mu, log_var, scaling, deg_list=None, deg_weight=20):
        recon_loss = self.nb.loss(x, scaling, x_hat)
        if deg_list is not None:
            recon_loss[:,deg_list] *= deg_weight
        recon_loss = recon_loss.sum()
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss, kl_div

    # sample around cell
    def sample(self, x, n=10, scale=1.):
        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        eps = scale * torch.randn(n, self.latent_dim)
        z = mu + eps * std
        x_hat = self.decode(z)
        return x_hat, z
    

# -----------------------------
# -  Deep Generative Decoder  -
# -----------------------------
class DGD(nn.Module):
    def __init__(self, dim_list, r_init=2, scaling_type='library', n_conditional_vars=0, extra_outputs=0):
        super(DGD, self).__init__()
        self.extra_outputs = extra_outputs
        self.n_conditional_vars = n_conditional_vars
        dim_list = [dim_list[0] + n_conditional_vars] + dim_list[1:]


        # create fully connected layers
        layers = []
        for i in range(len(dim_list) - 1):
            layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            if i != len(dim_list) - 2:  # Exclude the last layer from adding activation
                layers.append(nn.ReLU(inplace=True))

        self.fc_layers = nn.Sequential(*layers)
        self.nb = NBLayer(dim_list[-1]-extra_outputs, r_init=r_init, scaling_type=scaling_type)


    def forward(self, z, conditional_vars=None):
        # add condition to representation
        if conditional_vars is not None:
            z = torch.cat((z, conditional_vars), dim=-1)

        recon = self.fc_layers(z)
        recon = self.nb(recon)
        return recon 
    

class DGDTaskFromLatent(nn.Module):
    def __init__(self, dim_list, r_init=2, scaling_type='library', n_conditional_vars=0, extra_outputs=0):
        super(DGDTaskFromLatent, self).__init__()
        self.extra_outputs = extra_outputs
        self.n_conditional_vars = n_conditional_vars
        dim_list = [dim_list[0] + n_conditional_vars] + dim_list[1:]

        # reconstruction layers
        layers = []
        for i in range(len(dim_list) - 1):
            layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            if i != len(dim_list) - 2:
                layers.append(nn.ELU(inplace=True))
        self.fc_layers = nn.Sequential(*layers)
        self.nb = NBLayer(dim_list[-1], r_init=r_init, scaling_type=scaling_type)

        # task layers
        self.fc_layers_task = nn.Sequential(
            nn.Linear(dim_list[0] - n_conditional_vars, 10),
            nn.ELU(),
            nn.Linear(10, extra_outputs//2),
            nn.ELU(),
            nn.Linear(extra_outputs//2, extra_outputs))


    def forward(self, z, conditional_vars=None):
        # add condition to representation
        if conditional_vars is not None:
            z = torch.cat((z, conditional_vars), dim=-1)

        recon = self.fc_layers(z)
        recon = self.nb(recon)

        task = self.fc_layers_task(z[:, :-self.n_conditional_vars])
        return torch.cat((recon, task), dim=-1)
    

# ---------
# - NBVAE -
# ---------
class NBVAE(nn.Module):
    def __init__(self, dim_list, r_init=2, scaling_type="library", extra_outputs=0, n_conditional_vars=0):
        super(NBVAE, self).__init__()
        self.extra_outputs = extra_outputs
        
        self.n_conditional_vars = n_conditional_vars
        dim_list = [dim_list[0] + n_conditional_vars] + dim_list[1:]

        # encoder layers
        enc_layers = []
        for i in range(len(dim_list) - 1, 0, -1):
            enc_layers.append(nn.Linear(dim_list[i], dim_list[i - 1]))
            if i == len(dim_list) - 1:
                enc_layers[-1] = nn.Linear(dim_list[i] + n_conditional_vars, dim_list[i-1])
            if i == 1:
                enc_layers[-1] = nn.Linear(dim_list[i], (dim_list[i - 1] - n_conditional_vars) * 2)
            if i != 1:
                enc_layers.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*enc_layers)

        # decoder layers
        dec_layers = []
        for i in range(len(dim_list) - 1):
            dec_layers.append(nn.Linear(dim_list[i], dim_list[i + 1]))
            if i != len(dim_list) - 2:  # Exclude the last layer from adding activation
                dec_layers.append(nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(*dec_layers)

        # nb layer
        self.nb = NBLayer(dim_list[-1]-extra_outputs, r_init=r_init, scaling_type=scaling_type)

    def encode(self, x, conditional_vars=None):
        if conditional_vars is not None:
            x = torch.cat((x, conditional_vars), dim=-1)

        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z, conditional_vars=None):
        if conditional_vars is not None:
            z = torch.cat((z, conditional_vars), dim=-1)

        x_hat = self.decoder(z)

        if self.extra_outputs:
            x_hat[:, :-self.extra_outputs] = self.nb(x_hat[:, :-self.extra_outputs])
        else:
            x_hat = self.nb(x_hat)
        return x_hat
    
    def forward(self, x, conditional_vars=None):
        mu, log_var = self.encode(x, conditional_vars)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z, conditional_vars)
        return x_hat, mu, log_var

    def loss(self, x, x_hat, mu, log_var, scaling, deg_list=None, deg_weight=20):
        recon_loss = self.nb.loss(x, scaling, x_hat)
        if deg_list is not None:
            recon_loss[:,deg_list] *= deg_weight
        recon_loss = recon_loss.sum()
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss, kl_div

    # sample around cell
    def sample(self, x, n=10, scale=1.):
        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        eps = scale * torch.randn(n, self.latent_dim)
        z = mu + eps * std
        x_hat = self.decode(z)
        return x_hat, z
    
# NOTE: 
# The following building blocks until EOD are from Schuster and Krogh (2023)
# ---------------------------
class RepresentationLayer(torch.nn.Module):
    '''
    Implements a representation layer, that accumulates pytorch gradients.

    Representations are vectors in nrep-dimensional real space. By default
    they will be initialized as a tensor of dimension nsample x nrep from a
    normal distribution (mean and variance given by init).

    One can also supply a tensor to initialize the representations (values=tensor).
    The representations will then have the same dimension and will assumes that
    the first dimension is nsample (and the last is nrep).

    forward() takes a sample index and returns the representation.

    Representations are "linear", so a representation layer may be followed
    by an activation function.

    To update representations, the pytorch optimizers do not always work well,
    so the module comes with it's own SGD update (self.update(lr,mom,...)).

    If the loss has reduction="sum", things work well. If it is ="mean", the
    gradients become very small and the learning rate needs to be rescaled
    accordingly (batchsize*output_dim).

    Do not forget to call the zero_grad() before each epoch (not inside the loop
    like with the weights).

    '''
    def __init__(self,
                nrep,        # Dimension of representation
                nsample,     # Number of training samples
                init=(0.,1.),# Normal distribution mean and stddev for
                                # initializing representations
                values=None  # If values is given, the other parameters are ignored
                ):
        super(RepresentationLayer, self).__init__()
        self.dz = None
        if values is None:
            self.nrep=nrep
            self.nsample=nsample
            self.mean, self.stddev = init[0],init[1]
            self.init_random(self.mean,self.stddev)
        else:
            # Initialize representations from a tensor with values
            self.nrep = values.shape[-1]
            self.nsample = values.shape[0]
            self.mean, self.stddev = None, None
            # Is this the way to copy values to a parameter?
            self.z = torch.nn.Parameter(torch.zeros_like(values), requires_grad=True)
            with torch.no_grad():
                self.z.copy_(values)

    def init_random(self,mean,stddev):
        # Generate random representations
        self.z = torch.nn.Parameter(torch.normal(mean,stddev,size=(self.nsample,self.nrep), requires_grad=True))

    def forward(self, idx=None):
        if idx is None:
            return self.z
        else:
            return self.z[idx]

    # Index can be whatever it can be for a torch.tensor (e.g. tensor of idxs)
    def __getitem__(self,idx):
        return self.z[idx]

    def fix(self):
        self.z.requires_grad = False

    def unfix(self):
        self.z.requires_grad = True

    def zero_grad(self):  # Used only if the update function is used
        if self.z.grad is not None:
            self.z.grad.detach_()
            self.z.grad.zero_()

    def update(self,idx=None,lr=0.001,mom=0.9,wd=None):
        if self.dz is None:
            self.dz = torch.zeros(self.z.size()).to(self.z.device)
        with torch.no_grad():
            # Update z
            # dz(k,j) = sum_i grad(k,i) w(i,j) step(z(j))
            self.dz[idx] = self.dz[idx].mul(mom) - self.z.grad[idx].mul(lr)
            if wd is not None:
                self.dz[idx] -= wd*self.z[idx]
            self.z[idx] += self.dz[idx]

    def rescale(self):
        #z_flat = self.z.cpu().detach().numpy().flatten()
        #m = np.mean(z_flat)
        #sd = np.std(z_flat)
        z_flat = torch.flatten(self.z.cpu().detach())
        sd, m = torch.std_mean(z_flat)
        with torch.no_grad():
            self.z -= m
            self.z /= sd

class gaussian():
    '''
    This is a simple Gaussian prior used for initializing mixture model means
    '''
    def __init__(self,mean,stddev):
        self.mean = mean
        self.stddev = stddev
        self.g = torch.distributions.normal.Normal(mean,stddev)
    def sample(self,n,dim):
        return self.g.sample((n, dim))
    def log_prob(self,x):
        return self.g.log_prob(x)

class uniform():
    '''
    This is a simple uniform prior used for initializing mixture model means
    '''
    def __init__(self,low,high):
        self.g = torch.distributions.uniform.Uniform(low, high)
    def sample(self,n,dim):
        return self.g.sample((n, dim))
    def log_prob(self, x):
        return self.g.log_prob(x)

class softball():
    '''
    Almost uniform prior for the m-dimensional ball.
    Logistic function makes a soft (differentiable) boundary.
    Returns a prior function and a sample function.
    The prior takes a tensor with a batch of z
    vectors (last dim) and returns a tensor of prior log-probabilities.
    The sample function returns n samples from the prior (approximate
    samples uniform from the m-ball). NOTE: APPROXIMATE SAMPLING.
    '''
    def __init__(self,dim,radius,a=1):
        self.dim = dim
        self.radius = radius
        self.a = a
        self.norm = math.lgamma(1+dim*0.5)-dim*(math.log(radius)+0.5*math.log(math.pi))
    def sample(self,n,dim):
        # Return n random samples
        # Approximate: We sample uniformly from n-ball
        with torch.no_grad():
            # Gaussian sample
            sample = torch.randn((n,self.dim))
            # n random directions
            sample.div_(sample.norm(dim=-1,keepdim=True))
            # n random lengths
            local_len = self.radius*torch.pow(torch.rand((n,1)),1./self.dim)
            sample.mul_(local_len.expand(-1,self.dim))
        return sample
    def log_prob(self,z):
        # Return log probabilities of elements of tensor (last dim assumed to be z vectors)
        #y = 0.5 * torch.erf(self.a*(self.radius-z)/math.sqrt(2)) # might be * k?
        #return self.norm - y.sum()
        return (self.norm-torch.log(1+torch.exp(self.a*(z.norm(dim=-1)/self.radius-1))))

class GaussianMixture(nn.Module):
    def __init__(self, Nmix, dim, type="isotropic",
               alpha=1, mean_prior=None, logbeta_prior=None, mean_init=(0.,1.), sd_init=(0.5,0.5), weight_prior=None
               ):
        '''
        A mixture of multi-variate Gaussians

        Nmix is the number of components in the mixture
        dim is the dimension of the space
        type can be "fixed", "isotropic" or "diagonal", which refers to the covariance matrices
        mean_prior is a prior class with a log_prob and sample function
            - Standard normal if not specified.
            - Other option is ('softball',<radius>,<hardness>)
        If there is no mean_prior specified, a default Gaussian will be chosen with
            - mean_init[0] as mean and mean_init[1] as standard deviation
        logbeta_prior is a prior class for the negative log variance of the mixture components
            - logbeta = log (1/sigma^2)
            - If it is not specified, we make this prior a Gaussian from sd_init parameters
            - For the sake of interpretability, the sd_init parameters represent the desired mean and (approximately) sd of the standard deviation
            - the difference btw giving a prior beforehand and giving only init values is that with a given prior, the logbetas will be sampled from it, otherwise they will be initialized the same
        alpha determines the Dirichlet prior on mixture codgd(efficients
        Mixture coefficients are initialized uniformly
        Other parameters are sampled from prior
        '''
        super(GaussianMixture, self).__init__()
        self.dim = dim
        self.Nmix = Nmix
        #self.init = init

        # Means with shape: Nmix,dim
        self.mean = nn.Parameter(torch.empty(Nmix,dim),requires_grad=True)
        if mean_prior is None:
            self.mean_prior = gaussian(mean_init[0],mean_init[1])
        else:
            self.mean_prior = mean_prior

        if weight_prior is None:
            #self.weight_prior = uniform(0.,1.)
            self.weight_prior = gaussian(1/self.Nmix,0.5)
        else:
            self.weight_prior = weight_prior

        # Dirichlet prior on mixture
        self.alpha = alpha
        self.dirichlet_constant = math.lgamma(Nmix*alpha)-Nmix*math.lgamma(alpha)

        # Log inverse variance with shape (Nmix,dim) or (Nmix,1)
        self.sd_init = sd_init
        self.betafactor = dim*0.5 # rename this!
        self.bdim=1 # If 'diagonal' the dimension of lobbeta is = dim
        if type == 'fixed':
            # No gradient needed for training
            # This is a column vector to be correctly broadcastet in std dev tensor
            self.logbeta = nn.Parameter(torch.empty(Nmix,self.bdim),requires_grad=False)
            self.logbeta_prior = None
        else:
            if type == 'diagonal':
                self.betafactor = 0.5
                self.bdim = dim
            elif type != 'isotropic':
                raise ValueError("type must be 'isotropic' (default), 'diagonal', or 'fixed'")
            
            self.logbeta = nn.Parameter(torch.empty(Nmix,self.bdim),requires_grad=True)
            self.logbeta_prior = logbeta_prior
  
        # Mixture coefficients. These are weights for softmax
        self.weight = nn.Parameter(torch.empty(Nmix),requires_grad=True)
        self.init_params()

        # -dim*0.5*log(2pi)
        self.pi_term = - 0.5*self.dim*math.log(2*math.pi)
 
    def init_params(self):
        with torch.no_grad():
            # Means are sampled from the prior
            self.mean.copy_(self.mean_prior.sample(self.Nmix,self.dim))
            if self.logbeta_prior is None:
                self.logbeta.fill_(-2*math.log(self.sd_init[0]))
                self.logbeta_prior = gaussian(-2*math.log(self.sd_init[0]),self.sd_init[1])
            else:
                # Betas are sampled from prior
                self.logbeta.copy_(-torch.log(self.logbeta_prior.sample(self.Nmix,self.bdim)))

            # Weights are initialized to 1, corresponding to uniform mixture coeffs
            self.weight.fill_(1)

    def forward(self,x,label=None):

        # The beta values are obtained from logbeta
        halfbeta = 0.5*torch.exp(self.logbeta)

        # y = logp =  - 0.5*log (2pi) -0.5*beta(x-mean[i])^2 + 0.5*log(beta)
        # sum terms for each component (sum is over last dimension)
        # y is one-dim with length Nmix
        # x is unsqueezed to (nsample,1,dim), so broadcasting of mean (Nmix,dim) works
        y = self.pi_term - (x.unsqueeze(-2)-self.mean).square().mul(halfbeta).sum(-1) + self.betafactor*self.logbeta.sum(-1)
        # For each component multiply by mixture probs
        y += torch.log_softmax(self.weight,dim=0)
        y = torch.logsumexp(y, dim=-1)
        y = y + self.prior() # += gives cuda error

        return y

    def log_prob(self,x): # Add label?
        self.forward(x)

    def mixture_probs(self):
        return torch.softmax(self.weight,dim=-1)

    def covariance(self):
        return torch.exp(-self.logbeta)
    
    def prior(self):
        ''' Calculate log prob of prior on mean, logbeta, and mixture coeff '''
        # Mixture
        p = self.dirichlet_constant 
        if self.alpha!=1:
            p = p + (self.alpha-1.)*(self.mixture_probs().log().sum())
        # Means
        p = p+self.mean_prior.log_prob(self.mean).sum()
        # logbeta
        if self.logbeta_prior is not None:
            p =  p+self.logbeta_prior.log_prob(self.logbeta).sum()
        return p

    def Distribution(self):
        with torch.no_grad():
            mix = D.Categorical(probs=torch.softmax(self.weight,dim=-1))
            comp = D.Independent(D.Normal(self.mean,torch.exp(-0.5*self.logbeta)), 1)
            return D.MixtureSameFamily(mix, comp)

    def sample(self,nsample):
        with torch.no_grad():
            gmm = self.Distribution()
            return gmm.sample(torch.tensor([nsample]))

    def component_sample(self,nsample):
        '''Returns a sample from each component. Tensor shape (nsample,nmix,dim)'''
        with torch.no_grad():
            comp = D.Independent(D.Normal(self.mean,torch.exp(-0.5*self.logbeta)), 1)
            return comp.sample(torch.tensor([nsample]))


class NBLayer(nn.Module):
    '''
    Schuster and Krogh (2023)
    A negative binomial of scaled values of m and learned parameters for r.
    mhat = m/M, where M is the scaling factor

    The scaled value mhat is typically the output value from the NN

    If rhat=None, it is assumed that the last half of mhat contains rhat.

    m = M*mhat
    '''
    def __init__(self, out_dim, r_init, scaling_type='library',reduction='none'):
        super(NBLayer, self).__init__()

        # initialize parameter for r
        # real-valued positive parameters are usually used as their log equivalent
        self.log_r = torch.nn.Parameter(torch.full(fill_value=math.log(r_init), size=(1,out_dim)), requires_grad=True)
        #self.log_r = torch.nn.Parameter(torch.full(fill_value=math.log(r_init), size=(1,out_dim)), requires_grad=False)
        # determine the type of activation based on scaling type
        if scaling_type in ['library','total_count']:
            self.activation = 'sigmoid'
        elif scaling_type in ['mean','median']:
            self.activation = 'softplus'
        else:
            raise ValueError('Unknown scaling type specified. Please use one of: "library", "total_count", "mean", or "median".')
        self.reduction = reduction
    
    def forward(self, x):
        if self.activation == 'sigmoid':
            return torch.sigmoid(x)
        else:
            #return F.relu(x)
            return F.softplus(x)

    # Convert to m from scaled variables
    def rescale(self,M,mhat):
        return M*mhat

    def loss(self,x,M,mhat):
        if self.reduction == 'none':
            return -logNBdensity(x,self.rescale(M,mhat),torch.exp(self.log_r))
        elif self.reduction == 'sum':
            return -logNBdensity(x,self.rescale(M,mhat),torch.exp(self.log_r)).sum()

    # The logprob of the tensor
    def logprob(self,x,M,mhat):
        return logNBdensity(x,self.rescale(M,mhat),torch.exp(self.log_r))

    def sample(self,nsample,M,mhat):
        # Note that torch.distributions.NegativeBinomial returns FLOAT and not int
        with torch.no_grad():
            # m = pr/(1-p), so p = m/(m+r)
            m = self.rescale(M,mhat)
            probs = m/(m+torch.exp(self.log_r))
            nb = torch.distributions.NegativeBinomial(torch.exp(self.log_r), probs=probs)
            return nb.sample([nsample]).squeeze()

def logNBdensity(k,m,r):
  ''' 
  Negative Binomial NB(k;m,r), where m is the mean and r is "number of failures"
  r can be real number (and so can k)
  k, and m are tensors of same shape
  r is tensor of shape (1, n_genes)
  Returns the log NB in same shape as k
  '''
  # remember that gamma(n+1)=n!
  eps = 1.e-10
  c = 1./(r+m+eps)
  # Under-/over-flow protection is added
  x = torch.lgamma(k+r) - torch.lgamma(r) - torch.lgamma(k+1) + k*torch.log(m*c+eps) + r*torch.log(r*c)
  return x


