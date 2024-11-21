import numpy as np
from sae_lens import ActivationsStore
import torch
from tqdm import tqdm
from typing import Tuple


class StdevEstimationActivationsStore(ActivationsStore):

    estimated_stdev: float | None = None
    n_activations_for_stdev: int | None = None

    @torch.no_grad()
    def estimate_stdev_data(
        self,
        n_batches_for_stdev_estimate: int = int(1e3),
        chunk_size: int = 100,
    ) -> Tuple[float, int]:
        """Estimate standard deviation using chunked processing and running statistics.

        Args:
            n_batches_for_stdev_estimate: Total number of batches to process
            chunk_size: Number of batches to process at once

        Returns:
            Tuple of (estimated standard deviation, total number of activations processed)
        """
        if self.estimated_stdev is not None:
            return self.estimated_stdev, self.n_activations_for_stdev

        # Initialize running statistics
        running_mean: torch.Tensor = 0
        running_sq_mean: torch.Tensor = 0
        total_activations: int = 0

        # Process in chunks
        n_chunks = (n_batches_for_stdev_estimate + chunk_size - 1) // chunk_size

        for chunk_idx in tqdm(range(n_chunks), desc="Estimating dataset stdev"):
            # Determine batches for this chunk
            start_batch = chunk_idx * chunk_size
            end_batch = min(start_batch + chunk_size, n_batches_for_stdev_estimate)
            current_chunk_size = end_batch - start_batch

            # Collect activations for current chunk
            chunk_activations: list[torch.Tensor] = []
            chunk_total: int = 0

            for _ in range(current_chunk_size):
                acts = self.next_batch()
                batch_size = acts.shape[:-1].numel()
                flattened_acts = acts.reshape(-1, acts.shape[-1])
                chunk_activations.append(flattened_acts)
                chunk_total += batch_size

            # Process chunk
            chunk_acts = torch.cat(chunk_activations, dim=0)

            # Update running statistics using Welford's online algorithm
            chunk_mean = chunk_acts.mean()
            chunk_sq_mean = (chunk_acts**2).mean()

            # Update running statistics
            delta = chunk_mean - running_mean
            running_mean += delta * (chunk_total / (total_activations + chunk_total))

            delta_sq = chunk_sq_mean - running_sq_mean
            running_sq_mean += delta_sq * (
                chunk_total / (total_activations + chunk_total)
            )

            total_activations += chunk_total

            # Free memory
            del chunk_activations
            del chunk_acts
            torch.cuda.empty_cache()

        # Calculate final standard deviation
        variance = running_sq_mean - (running_mean**2)
        self.estimated_stdev = float(torch.sqrt(variance).mean().item())
        self.n_activations_for_stdev = total_activations

        return self.estimated_stdev, self.n_activations_for_stdev
