from sae_lens.training.sae_trainer import SAETrainer
import torch


class StdevEstimationSAETrainer(SAETrainer):

    @torch.no_grad()
    def _estimate_norm_scaling_factor_if_needed(self) -> None:
        super()._estimate_norm_scaling_factor_if_needed()
        stdev, num_stdev_acts = self.activation_store.estimate_stdev_data()
        self.estimated_stdev = stdev
        self.n_activations_for_stdev = num_stdev_acts

        self.sae.estimated_stdev = stdev
        self.sae.n_activations_for_stdev = num_stdev_acts
