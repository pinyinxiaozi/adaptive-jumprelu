from typing import Optional
from sae_lens import TrainingSAE
from sae_lens.training.training_sae import JumpReLU, TrainStepOutput
from torch import nn
import numpy as np
import torch

from adaptive_jumprelu.activations import GaussianJumpReLU, GaussianStep


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

        bandwidth = self.get_bandwidth()

        feature_acts = GaussianJumpReLU.apply(hidden_pre, threshold, bandwidth)
        return feature_acts, hidden_pre

    def training_forward_pass(
        self,
        sae_in: torch.Tensor,
        current_l1_coefficient: float,
        dead_neuron_mask: Optional[torch.Tensor] = None,
    ) -> TrainStepOutput:

        # do a forward pass to get SAE out, but we also need the
        # hidden pre.
        feature_acts, hidden_pre = self.encode_with_hidden_pre_fn(sae_in)
        sae_out = self.decode(feature_acts)

        # MSE LOSS
        per_item_mse_loss = self.mse_loss_fn(sae_out, sae_in)
        mse_loss = per_item_mse_loss.sum(dim=-1).mean()

        losses: dict[str, float | torch.Tensor] = {}

        assert self.cfg.architecture == "jumprelu"
        bandwidth = self.get_bandwidth()
        threshold = torch.exp(self.log_threshold)
        l0 = torch.sum(GaussianStep.apply(hidden_pre, threshold, bandwidth), dim=-1)  # type: ignore
        l0_loss = (current_l1_coefficient * l0).mean()
        loss = mse_loss + l0_loss
        losses["l0_loss"] = l0_loss
        losses["mse_loss"] = mse_loss

        # not technically a loss, but useful for logging to wandb
        losses["bandwidth"] = bandwidth
        losses["mean_threshold"] = threshold.mean()

        return TrainStepOutput(
            sae_in=sae_in,
            sae_out=sae_out,
            feature_acts=feature_acts,
            hidden_pre=hidden_pre,
            loss=loss,
            losses=losses,
        )

    def get_bandwidth(self) -> float:
        hidden_pre_stdev = self.calculate_hidden_pre_stdev()

        # Modified Silverman's rule of thumb for d-dimensional data
        # h = σ * (4/(d+2))^(1/(d+4)) * n^(-1/(d+4))
        d = self.cfg.d_sae  # number of dimensions
        bandwidth = (
            hidden_pre_stdev
            * ((4 / (d + 2)) ** (1 / (d + 4)))
            * (self.n_activations_for_stdev ** (-1 / (d + 4)))
        )
        return bandwidth

    def calculate_hidden_pre_stdev(
        self,
    ) -> torch.Tensor:
        """
        Calculate the standard deviation of hidden_pre given input standard deviation.

        Returns:
            torch.Tensor: Standard deviation for each dimension of hidden_pre
        """
        input_stdev = self.estimated_stdev
        if isinstance(input_stdev, float):
            input_stdev = torch.full(
                (self.W_enc.shape[0],), input_stdev, device=self.W_enc.device
            )

        # For each output dimension j, calculate sqrt(sum(W_ij² * σ_i²))
        # Using einsum for efficient computation
        hidden_pre_var = torch.einsum("ij,i->j", self.W_enc.pow(2), input_stdev.pow(2))
        hidden_pre_stdev = torch.sqrt(hidden_pre_var)

        return hidden_pre_stdev
