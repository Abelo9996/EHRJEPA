r"""SIGReg: Sketched Isotropic Gaussian Regularization (LeJEPA, arXiv:2511.08544).

The anti-collapse term. Instead of penalising the embedding covariance directly,
SIGReg asks a distributional question: *is the batch of embeddings a sample from
an isotropic Gaussian?* By the Cramer-Wold theorem a distribution in
:math:`\mathbb{R}^D` is determined by its one-dimensional projections, so the
:math:`D`-dimensional test is *sketched* into :math:`K` random directions and each
projection is tested against :math:`\mathcal{N}(0, 1)` on its own.

Definition 2 of the paper:

.. math::

    \mathrm{SIGReg}_T(\mathbb{A}, \{f_\theta(x_n)\}_{n=1}^N)
      \;\triangleq\; \frac{1}{|\mathbb{A}|}
        \sum_{a \in \mathbb{A}} T\!\left(\{a^\top f_\theta(x_n)\}_{n=1}^N\right),

with :math:`\mathbb{A}` a fresh set of unit directions drawn every step (the
paper's default is 1024 "slices"; it reports 512 as competitive) and :math:`T`
the **Epps-Pulley** characteristic-function test of normality:

.. math::

    \mathrm{EP} \;=\; N \int_{-\infty}^{\infty}
        \left| \hat{\varphi}_X(t) - \varphi(t) \right|^2 w(t)\, dt,
    \qquad
    \hat{\varphi}_X(t) = \frac{1}{N}\sum_{n=1}^{N} e^{\mathrm{i} t x_n},
    \qquad
    \varphi(t) = e^{-t^2/2},

so that, writing the empirical characteristic function in real form,

.. math::

    \left| \hat{\varphi}_X(t) - \varphi(t) \right|^2 =
      \Big(\tfrac{1}{N}\!\sum_n \cos(t x_n) - e^{-t^2/2}\Big)^2
      + \Big(\tfrac{1}{N}\!\sum_n \sin(t x_n)\Big)^2 .

The weighting is Gaussian, :math:`w(t) = e^{-t^2/\sigma^2}` with
:math:`\sigma = 1`, and the integral is evaluated by the trapezoid rule on 17
points over :math:`[-5, 5]` -- the paper's grid. The statistic is
:math:`O(N)` in both time and memory: nothing here is pairwise.

Two deliberate choices, both exposed:

* **The** :math:`N` **prefactor is off by default** (``scale_by_n=False``). It is
  part of the Epps-Pulley definition -- it is what makes the statistic converge to
  a fixed null distribution -- but as a *loss* it would multiply the
  regularization by the number of rows, which for 8192 token rows swamps the
  prediction term at the paper's own :math:`\lambda = 0.05`. Turning it on
  reproduces the test statistic exactly.
* **Nothing is standardized inside SIGReg.** No centering, no whitening, no
  per-dimension normalization. The whole point is that the embeddings must become
  isotropic Gaussian *on their own*; subtracting the mean would hand the model
  the part of the objective it is supposed to learn.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["SIGReg", "epps_pulley", "random_directions", "sigreg"]

#: The paper's integration grid: 17 trapezoid points over [-5, 5].
DEFAULT_N_GRID = 17
DEFAULT_T_MAX = 5.0
#: Gaussian weighting w(t) = exp(-t^2 / sigma^2).
DEFAULT_SIGMA = 1.0
#: The paper's default number of sketching directions.
DEFAULT_N_DIRECTIONS = 1024


def random_directions(
    dim: int,
    n_directions: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> Tensor:
    """``(dim, n_directions)`` of independent uniform points on the unit sphere."""
    raw = torch.randn(dim, n_directions, device=device, dtype=dtype, generator=generator)
    return raw / raw.norm(dim=0, keepdim=True).clamp_min(torch.finfo(dtype).tiny)


def epps_pulley(
    projections: Tensor,
    n_grid: int = DEFAULT_N_GRID,
    t_max: float = DEFAULT_T_MAX,
    sigma: float = DEFAULT_SIGMA,
    scale_by_n: bool = False,
    t_chunk: int = 4,
) -> Tensor:
    """Epps-Pulley statistic of each column of ``projections`` against ``N(0, 1)``.

    ``projections`` is ``(n_samples, n_directions)``. Returns one statistic per
    direction, shape ``(n_directions,)``. Evaluated by the trapezoid rule over
    ``n_grid`` points on ``[-t_max, t_max]``; ``t_chunk`` only bounds peak memory.
    """
    if projections.ndim != 2:
        raise ValueError(f"expected (n_samples, n_directions), got {tuple(projections.shape)}")
    n_samples = projections.shape[0]
    if n_samples == 0:
        return projections.new_zeros(projections.shape[1])
    device, dtype = projections.device, projections.dtype
    grid = torch.linspace(-t_max, t_max, n_grid, device=device, dtype=dtype)
    integrand = torch.empty(projections.shape[1], n_grid, device=device, dtype=dtype)
    for lo in range(0, n_grid, max(1, t_chunk)):
        t = grid[lo : lo + max(1, t_chunk)]
        angles = projections.unsqueeze(-1) * t  # (N, K, c)
        real = angles.cos().mean(dim=0) - torch.exp(-0.5 * t * t)  # (K, c)
        imag = angles.sin().mean(dim=0)
        integrand[:, lo : lo + t.shape[0]] = (real * real + imag * imag) * torch.exp(
            -(t * t) / (sigma * sigma)
        )
    stat = torch.trapezoid(integrand, grid, dim=-1)
    return stat * n_samples if scale_by_n else stat


def sigreg(
    embeddings: Tensor,
    n_directions: int = DEFAULT_N_DIRECTIONS,
    n_grid: int = DEFAULT_N_GRID,
    t_max: float = DEFAULT_T_MAX,
    sigma: float = DEFAULT_SIGMA,
    scale_by_n: bool = False,
    generator: torch.Generator | None = None,
    t_chunk: int = 4,
) -> Tensor:
    """Average Epps-Pulley statistic over ``n_directions`` fresh random directions.

    ``embeddings`` is ``(n_samples, dim)``. The result is a scalar tensor that
    carries gradient into ``embeddings``.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"expected (n_samples, dim), got {tuple(embeddings.shape)}")
    if embeddings.shape[0] == 0:
        return embeddings.new_zeros(())
    directions = random_directions(
        embeddings.shape[1],
        n_directions,
        device=embeddings.device,
        dtype=embeddings.dtype,
        generator=generator,
    )
    return epps_pulley(
        embeddings @ directions,
        n_grid=n_grid,
        t_max=t_max,
        sigma=sigma,
        scale_by_n=scale_by_n,
        t_chunk=t_chunk,
    ).mean()


class SIGReg(nn.Module):
    """Stateless module wrapper around :func:`sigreg`, plus row subsampling.

    ``max_rows`` caps how many embedding rows enter the statistic (the token-level
    call sees up to ``batch * seq_len`` rows, which is far more than the test
    needs). Rows are drawn without replacement, fresh every call.
    """

    def __init__(
        self,
        n_directions: int = DEFAULT_N_DIRECTIONS,
        max_rows: int = 8192,
        n_grid: int = DEFAULT_N_GRID,
        t_max: float = DEFAULT_T_MAX,
        sigma: float = DEFAULT_SIGMA,
        scale_by_n: bool = False,
        t_chunk: int = 4,
    ) -> None:
        super().__init__()
        self.n_directions = n_directions
        self.max_rows = max_rows
        self.n_grid = n_grid
        self.t_max = t_max
        self.sigma = sigma
        self.scale_by_n = scale_by_n
        self.t_chunk = t_chunk

    def subsample(self, rows: Tensor, generator: torch.Generator | None = None) -> Tensor:
        if self.max_rows <= 0 or rows.shape[0] <= self.max_rows:
            return rows
        pick = torch.randperm(rows.shape[0], device=rows.device, generator=generator)
        return rows[pick[: self.max_rows]]

    def forward(self, rows: Tensor, generator: torch.Generator | None = None) -> Tensor:
        return sigreg(
            self.subsample(rows, generator),
            n_directions=self.n_directions,
            n_grid=self.n_grid,
            t_max=self.t_max,
            sigma=self.sigma,
            scale_by_n=self.scale_by_n,
            generator=generator,
            t_chunk=self.t_chunk,
        )
