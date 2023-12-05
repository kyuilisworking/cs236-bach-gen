# Copyright (c) 2021 Rui Shu

import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F
from codebase.models.nns.params import EncoderConfig, DecoderConfig


class LstmVAE(nn.Module):
    def __init__(
        self,
        encoderConfig: EncoderConfig,
        decoderConfig: DecoderConfig,
        nn="v3_lstm",
        name="lstm_vae",
    ):
        super().__init__()
        self.name = name
        nn = getattr(nns, nn)
        self.encoderConfig = encoderConfig
        self.decoderConfig = decoderConfig
        self.enc = nn.BidirectionalLstmEncoder(encoderConfig)
        if decoderConfig.decoder_type == "categorical":
            self.dec = nn.CategoricalLstmDecoder(decoderConfig)
        else:
            self.dec = nn.HierarchicalLstmDecoder(decoderConfig)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x, *, anneal_pct):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, sequence_length, dim): Observations
            anneal_pct: scalar between 0 and 1, used to anneal the KL divergence

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        q_m, q_v = self.enc(x)
        z = ut.sample_gaussian(q_m, q_v)
        logits = self.dec(z, x)
        logits = logits.reshape(-1, logits.size(-1))

        target = torch.argmax(x, dim=-1).reshape(-1)
        rec = F.cross_entropy(logits, target)
        kl = ut.kl_normal(q_m, q_v, self.z_prior_m, self.z_prior_v).mean()
        nelbo = rec + anneal_pct * kl

        return nelbo, kl, rec

    def loss(self, x, *, anneal_pct=1.0):
        nelbo, kl, rec = self.negative_elbo_bound(x, anneal_pct=anneal_pct)
        loss = nelbo

        summaries = dict(
            (
                ("train/loss", nelbo),
                ("gen/elbo", -nelbo),
                ("gen/kl_z", kl),
                ("gen/rec", rec),
            )
        )

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.encoderConfig.z_dim),
            self.z_prior[1].expand(batch, self.encoderConfig.z_dim),
        )

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z, temperature=1.0):
        return self.dec.sample(z, temperature)
