"""Neural marginal post-processors for member-wise PV forecasts.

The input is one normalized control proxy followed by the sorted normalized
ensemble members for a single farm and forecast hour. Each row has one observed
hourly AC-power target; members are features, not independent observations.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional


def standard_normal_cdf(value):
    """Evaluate the Gaussian CDF with operations supported by autograd."""
    return 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))


def standard_normal_density(value):
    """Evaluate the standard Gaussian probability density."""
    return torch.exp(-0.5 * value.square()) / math.sqrt(2.0 * math.pi)


def censored_normal_crps(target, location, scale):
    """Score a Gaussian censored at zero and one, preserving PyTorch gradients.

    The closed-form expression corresponds to scoringRules::crps_cnorm with
    lower=0 and upper=1. It includes both boundary point masses.
    """
    lower = -location / scale
    upper = (1.0 - location) / scale
    standardized_target = (target - location) / scale
    clipped_target = torch.minimum(torch.maximum(standardized_target, lower), upper)

    # The lower and upper terms account for probability censored at each bound.
    lower_probability = standard_normal_cdf(lower)
    upper_probability = standard_normal_cdf(-upper)
    lower_term = -lower * lower_probability.square()
    lower_term = lower_term - 2.0 * standard_normal_density(lower) * lower_probability
    upper_term = upper * upper_probability.square()
    upper_term = upper_term - 2.0 * standard_normal_density(upper) * upper_probability

    # Integrate the squared difference between the CDF and the observed step.
    middle_term = clipped_target * (2.0 * standard_normal_cdf(clipped_target) - 1.0)
    middle_term = middle_term + 2.0 * standard_normal_density(clipped_target)
    integral = standard_normal_cdf(upper * math.sqrt(2.0))
    integral = integral - standard_normal_cdf(lower * math.sqrt(2.0))
    score = middle_term + lower_term + upper_term - integral / math.sqrt(math.pi)
    score = score + torch.abs(standardized_target - clipped_target)
    return scale * score


def quantile_huber_loss(target, predictions, levels, epsilon):
    """Average the paper's smooth asymmetric quantile loss over cases and levels."""
    # Enforce the input contract: The quantile Huber threshold must be positive.
    if epsilon <= 0:
        raise ValueError("The quantile Huber threshold must be positive")

    # Smooth only the region around zero; retain the asymmetric quantile weight.
    residual = target.unsqueeze(-1) - predictions
    magnitude = residual.abs()
    quadratic = magnitude.square() / (2.0 * epsilon)
    linear = magnitude - epsilon / 2.0
    smooth_absolute = torch.where(magnitude <= epsilon, quadratic, linear)
    weight = torch.where(residual >= 0.0, levels, 1.0 - levels)
    return (weight * smooth_absolute).mean()


def make_backbone(input_size, hidden_sizes, activation):
    """Construct an MLP shared by the paper's neural post-processors."""
    activations = {
        "relu": nn.ReLU,
        "softplus": nn.Softplus,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }
    # Enforce the input contract: Unsupported activation: `activation`.
    if activation not in activations:
        raise ValueError(f"Unsupported activation: {activation}")
    # Enforce the input contract: At least one positive hidden-layer width is required.
    if not hidden_sizes or any(width < 1 for width in hidden_sizes):
        raise ValueError("At least one positive hidden-layer width is required")

    layers = []
    previous = input_size

    # Each hidden layer has its own activation, as in the feed-forward models.
    for width in hidden_sizes:
        layers.append(nn.Linear(previous, width))
        layers.append(activations[activation]())
        previous = width

    return nn.Sequential(*layers), previous


class CensoredGaussianNetwork(nn.Module):
    """Predict the location and positive scale of a doubly censored Gaussian."""

    def __init__(self, input_size, hidden_sizes=(15, 10, 10), minimum_scale=1e-6):
        super().__init__()
        self.backbone, width = make_backbone(input_size, hidden_sizes, "relu")
        self.output = nn.Linear(width, 2)
        self.minimum_scale = float(minimum_scale)
        # Enforce the input contract: The minimum Gaussian scale must be positive.
        if self.minimum_scale <= 0:
            raise ValueError("The minimum Gaussian scale must be positive")

    def forward(self, features):
        """Return an unconstrained location and a differentiably positive scale."""
        parameters = self.output(self.backbone(features))
        location = parameters[:, 0]
        scale = functional.softplus(parameters[:, 1]) + self.minimum_scale
        return location, scale


class QuantileNetwork(nn.Module):
    """Predict one unconstrained quantile per output neuron."""

    def __init__(self, input_size, hidden_sizes, activation, quantile_count):
        super().__init__()
        self.backbone, width = make_backbone(input_size, hidden_sizes, activation)
        self.output = nn.Linear(width, quantile_count)

    def forward(self, features):
        """Return raw quantiles so crossing and physical bounds can be audited."""
        return self.output(self.backbone(features))


class BernsteinQuantileNetwork(nn.Module):
    """Predict a nondecreasing Bernstein-polynomial quantile function."""

    def __init__(self, input_size, hidden_sizes, activation, degree, levels):
        super().__init__()
        # Enforce the input contract: The Bernstein degree must be positive.
        if degree < 1:
            raise ValueError("The Bernstein degree must be positive")
        self.backbone, width = make_backbone(input_size, hidden_sizes, activation)
        self.output = nn.Linear(width, degree + 1)

        # Bernstein basis weights are fixed by the chosen probability levels.
        levels = torch.as_tensor(levels, dtype=torch.float32)
        basis = []
        # Build one basis polynomial for each Bernstein coefficient.
        for index in range(degree + 1):
            coefficient = math.comb(degree, index)
            basis.append(coefficient * levels.pow(index) * (1.0 - levels).pow(degree - index))
        self.register_buffer("basis", torch.stack(basis, dim=1))

    def forward(self, features):
        """Use positive coefficient increments to prevent quantile crossing."""
        raw = self.output(self.backbone(features))
        first = raw[:, :1]
        increments = functional.softplus(raw[:, 1:])
        coefficients = torch.cat([first, first + torch.cumsum(increments, dim=1)], dim=1)
        return coefficients @ self.basis.T


class NonCrossingQuantileNetwork(nn.Module):
    """Map positive hidden activations through triangular nonnegative increments.

    The triangular cumulative construction follows the NCQRNN principle. Its
    particular parameterization is documented as an implementation choice;
    the source article's complete equations are not reproduced in Mayer et al.
    """

    def __init__(self, input_size, hidden_sizes, activation, quantile_count, noncrossing_width):
        super().__init__()
        # Enforce the input contract: The non-crossing layer must have at least one neuron per quantile.
        if noncrossing_width < quantile_count:
            raise ValueError("The non-crossing layer must have at least one neuron per quantile")
        self.backbone, width = make_backbone(input_size, hidden_sizes, activation)
        self.hidden = nn.Linear(width, noncrossing_width)
        self.base = nn.Linear(width, 1)
        # Small positive initial increments keep the initial quantile range usable.
        self.raw_weights = nn.Parameter(torch.full((quantile_count, noncrossing_width), -7.0))

        # Allow each output increment to depend on its own and earlier hidden groups.
        groups = torch.arange(noncrossing_width) * quantile_count // noncrossing_width
        rows = torch.arange(quantile_count).unsqueeze(1)
        self.register_buffer("triangular_mask", (groups.unsqueeze(0) <= rows).float())

    def forward(self, features):
        """Accumulate nonnegative increments above one unconstrained base value."""
        backbone = self.backbone(features)
        hidden = functional.softplus(self.hidden(backbone))
        weights = functional.softplus(self.raw_weights) * self.triangular_mask
        increments = hidden @ weights.T
        return self.base(backbone) + torch.cumsum(increments, dim=1)
