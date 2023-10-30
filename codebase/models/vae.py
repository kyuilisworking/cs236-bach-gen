# Copyright (c) 2021 Rui Shu

import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        """
        So there are neural nets for mean and variance of Q(z|x^i), approximated by Q(z|x) - the variance results in a diagonal comatrix
        so both have the same output dimension. Once we have the mean and variance, we can sample from it, using sample_gaussian.

        So from the x's, get the z's. Then from z's, pass the batch of z's throgh the decoder to get the batch of logits for p(x|z).
        Then, use log_bernoulli_with_logits 
        
        Steps (Just for organization)
        1) Get the q_mean and q_var of Q(z|x; phi) from the encoder by passing in the x's (the first dim is mean, second dim is variance)
        2) Sample z from Q(z|x; phi) by using sample_gaussian, passing in q_mean, q_var
        3) Get logits from the decoder for p(x|z; theta)
        4) Calculate recon_loss = -1 * log_bernoulli_with_logits(x, logits) to calculate the likelihood of each x
        5) Use kl_div = kl_normal(q_mean, q_var, self.z_prior_m, self.z_prior_v) to get the KL divergence
        6) Average recon_loss and kl_div to get the averages.
        7) Return result
        """
        q_m, q_v = self.enc(x)
        z = ut.sample_gaussian(q_m, q_v)
        logits = self.dec(z)
            
        rec = -1 * ut.log_bernoulli_with_logits(x, logits).mean()
        kl = ut.kl_normal(q_m, q_v, self.z_prior_m, self.z_prior_v).mean()
        nelbo = rec + kl

        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################
        
        batch = x.shape[0]

        # sample from z
        q_m, q_v = self.enc(x)
        q_m = ut.duplicate(q_m, iw)
        q_v = ut.duplicate(q_v, iw)
        z = ut.sample_gaussian(q_m, q_v)

        # calculate log p_theta(x_z)
        logits = self.dec(z)
        log_prob_x_z = ut.log_bernoulli_with_logits(ut.duplicate(x, iw), logits).reshape(iw, batch).transpose(1, 0)
        
        # calculate log p(z)
        z_prior_m = self.z_prior_m.view(-1, 1).expand(batch * iw, self.z_dim)
        z_prior_v = self.z_prior_v.view(-1, 1).expand(batch * iw, self.z_dim)
        log_prob_z = ut.log_normal(z, z_prior_m, z_prior_v).reshape(iw, batch).transpose(1, 0)

        # calculate log q(z|x)
        log_q_z_x = ut.log_normal(z, q_m, q_v).reshape(iw, batch).transpose(1, 0)

        # calculate niwae
        niwae = -ut.log_mean_exp(log_prob_x_z + log_prob_z - log_q_z_x, dim=1).mean()

        _, kl, rec = self.negative_elbo_bound(x)

        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
