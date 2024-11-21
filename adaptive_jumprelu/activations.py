from typing import Any
import torch
import math


def gaussian_pdf(x: torch.Tensor) -> torch.Tensor:
    return (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * x * x)


class GaussianStep(torch.autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor, threshold: torch.Tensor, bandwidth: float
    ) -> torch.Tensor:
        return (x > threshold).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[None, torch.Tensor, None]:  # type: ignore[override]
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        threshold_grad = torch.sum(
            -(1.0 / bandwidth)
            * gaussian_pdf((x - threshold) / bandwidth)
            * grad_output,
            dim=0,
        )
        return None, threshold_grad, None


class GaussianJumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor, threshold: torch.Tensor, bandwidth: float
    ) -> torch.Tensor:
        return (x * (x > threshold)).to(x)

    @staticmethod
    def setup_context(
        ctx: Any, inputs: tuple[torch.Tensor, torch.Tensor, float], output: torch.Tensor
    ) -> None:
        x, threshold, bandwidth = inputs
        del output
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:  # type: ignore[override]
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold) * grad_output  # We don't apply STE to x input
        threshold_grad = torch.sum(
            -(threshold / bandwidth)
            * gaussian_pdf((x - threshold) / bandwidth)
            * grad_output,
            dim=0,
        )
        return x_grad, threshold_grad, None
