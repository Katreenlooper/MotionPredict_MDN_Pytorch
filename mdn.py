"""A module for a mixture density network layer

For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import sys # for testing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import math

ONE_OVER_SQRT_2PI = torch.rsqrt(torch.tensor(2*math.pi))

class MDN(nn.Module):
    """ 
    A Mixture Density Network layer

    Process:
        Input:x -> Some model (body) -> Characteristic vector:z (feature)
                -> MDN (head)        -> Probabilistic  vector:p (output)
    
    Symbols: 
        B - Batch size
        G - Number of Gaussian components
        D - Input's dimensions
        F - Feature's dimensions
        C - Output's dimensions (Gaussian distribution's dimensions)
    
    Arguments:
        dim_fea  (int): the feature's dimensions
        dim_prob (int): the output's dimenssions
        num_gaus (int): the number of Gaussians per output dimension
    
    Input:
        minibatch (BxF)
    Output:
        (alp, mu, sigma) (BxG, BxGxC, BxGxC)
        alp   - (alpha) component's weight
        mu    - mean value
        sigma - standard deviation
    """
    def __init__(self, dim_fea, dim_prob, num_gaus):
        super(MDN, self).__init__()
        self.dim_fea = dim_fea
        self.dim_prob = dim_prob
        self.num_gaus = num_gaus
        self.alp = nn.Sequential(
            nn.Linear(dim_fea, num_gaus),
            nn.Softmax(dim=1) # If 1, go along each row
        )
        self.mu    = nn.Linear(dim_fea, dim_prob*num_gaus)
        self.sigma = nn.Linear(dim_fea, dim_prob*num_gaus)

    def forward(self, minibatch):
        alp = self.alp(minibatch)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaus, self.dim_prob)
        sigma = torch.exp(self.sigma(minibatch))
        sigma = sigma.view(-1, self.num_gaus, self.dim_prob)
        return alp, mu, sigma


def cal_GauProb(mu, sigma, x):
    """
    Return the probability of "data" given MoG parameters "mu" and "sigma".
    
    Arguments:
        mu    (BxGxC) - The means of the Gaussians. 
        sigma (BxGxC) - The standard deviation of the Gaussians.
        x     (BxC)   - A batch of data points.

    Return:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding mu/sigma index.
    """
    x = x.unsqueeze(1).expand_as(mu) # BxC -> Bx1xC -> BxGxC
    prob = ONE_OVER_SQRT_2PI * torch.exp(-((x - mu) / sigma)**2 / 2) / sigma
    return torch.prod(prob, dim=2) # overall probability for all output's dimensions in each component


def loss_NLL_MDN(alp, mu, sigma, data):
    """
    Calculates the error, given the MoG parameters and the data
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    prob = alp * cal_GauProb(mu, sigma, data)
    # nll = -torch.sum(torch.log(prob), dim=1)
    nll = -torch.log(torch.sum(prob, dim=1))
    return torch.mean(nll)


def sample(alp, mu, sigma):
    """
    Draw samples from a MoG.
    """
    categorical = Categorical(alp) # aka. generalized Bernoulli
    alps = list(categorical.sample().data) # take a sample for each batch
    sample = sigma.data.new(sigma.size(0), sigma.size(2)).normal_()
    for i, idx in enumerate(alps):
        sample[i] = sample[i].mul(sigma[i,idx]).add(mu[i,idx])
    return sample
