from sae_lens import TrainingSAE
from sae_lens.training.training_sae import JumpReLU
from torch import nn
import numpy as np
import torch

from adaptive_jumprelu.activations import GaussianJumpReLU


class AdaptiveThresholdJumpReLU(TrainingSAE):

    def initialize_weights_jumprelu(self):
        # same as the superclass, except we use a log_threshold parameter instead of threshold
        self.log_threshold_mapping = nn.Linear(
            self.cfg.d_in, self.cfg.d_sae, dtype=self.dtype, device=self.device
        )
        self.log_threshold_mapping.weight.data = (
            self.log_threshold_mapping.weight.data
            * np.log(self.cfg.jumprelu_init_threshold)
        )
        self.initialize_weights_basic()

    def encode_with_hidden_pre_jumprelu(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = sae_in @ self.W_enc + self.b_enc
        threshold = self.log_threshold_mapping(sae_in).exp()
        feature_acts = JumpReLU.apply(hidden_pre, threshold, self.bandwidth)
        return feature_acts, hidden_pre


class AdaptiveBandwidthJumpReLU(TrainingSAE):

    estimated_stdev: float | None = None
    n_activations_for_stdev: int | None = None

    def encode_with_hidden_pre_jumprelu(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = sae_in @ self.W_enc + self.b_enc
        threshold = self.log_threshold.exp()

        hidden_pre_stdev = self.calculate_hidden_pre_stdev(sae_in)

        # Modified Silverman's rule of thumb for d-dimensional data
        # h = σ * (4/(d+2))^(1/(d+4)) * n^(-1/(d+4))
        d: int = hidden_pre.shape[-1]  # number of dimensions
        bandwidth = (
            hidden_pre_stdev
            * ((4 / (d + 2)) ** (1 / (d + 4)))
            * (self.n_activations_for_stdev ** (-1 / (d + 4)))
        )

        feature_acts = GaussianJumpReLU.apply(hidden_pre, threshold, bandwidth)
        return feature_acts, hidden_pre

    def calculate_hidden_pre_stdev(
        self,
        input_stdev: float | torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the standard deviation of hidden_pre given input standard deviation.

        Args:
            input_stdev: Standard deviation of sae_in (either scalar or per-dimension)

        Returns:
            torch.Tensor: Standard deviation for each dimension of hidden_pre
        """
        if isinstance(input_stdev, float):
            input_stdev = torch.full(
                (self.W_enc.shape[0],), input_stdev, device=self.W_enc.device
            )

        # For each output dimension j, calculate sqrt(sum(W_ij² * σ_i²))
        # Using einsum for efficient computation
        hidden_pre_var = torch.einsum("ij,i->j", self.W_enc.pow(2), input_stdev.pow(2))
        hidden_pre_stdev = torch.sqrt(hidden_pre_var)

        return hidden_pre_stdev
