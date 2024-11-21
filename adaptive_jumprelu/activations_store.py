import numpy as np
from sae_lens import ActivationsStore
import torch
from tqdm import tqdm


class StdevEstimationActivationsStore(ActivationsStore):

    estimated_stdev: float | None = None
    n_activations_for_stdev: int | None = None

    @torch.no_grad()
    def estimate_stdev_data(
        self, n_batches_for_stdev_estimate: int = int(1e3)
    ) -> tuple[float, int]:
        if self.estimated_stdev is not None:
            return self.estimated_stdev, self.n_activations_for_stdev

        # Collect variances across all dimensions
        all_activations: list[torch.Tensor] = []
        total_activations = 0

        for _ in tqdm(
            range(n_batches_for_stdev_estimate), desc="Estimating dataset stdev"
        ):
            acts = self.next_batch()
            # Flatten all dimensions except the last one (feature dimension)
            batch_size = acts.shape[:-1].numel()
            flattened_acts = acts.reshape(-1, acts.shape[-1])
            all_activations.append(flattened_acts)
            total_activations += batch_size

        # Concatenate all batches
        all_acts = torch.cat(all_activations, dim=0)

        # Calculate standard deviation across all samples and features
        # This gives us a better estimate of the overall scale of the data
        self.estimated_stdev = float(torch.std(all_acts, unbiased=True).mean().item())
        self.n_activations_for_stdev = total_activations

        return self.estimated_stdev, self.n_activations_for_stdev
