"""--subspace: run the committee on the first layer's measured input subspace (MLPs).

Every row of the black box's input-Jacobian lies in rowspace(W1), a k-dim subspace of the
d-dim input (k = rows of W1 < d). So:
  1. measure: central-difference output Jacobians at a few centres (2*d queries each, all
     counted), stacked, SVD -> U (d x k), the top-k right singular vectors;
  2. reduce: f(x) = g(U^T x) with g's first layer W1 U (k x k). The committee learns g on
     z in R^k, and every committee query z is the real black box at x = U z;
  3. map back: W1_hat = W1' U^T, everything else unchanged.

Teacher parameters are read only to build g (the identical function teacher(U z), checked
numerically, like --teacher-full builds its sub-network oracle) and for evaluation metrics.
"""
import copy

import torch

from nets import MLP


@torch.no_grad()
def measure(teacher, k, centers, eps, dtype=torch.float64, chunk=1024):
    """Top-k right singular subspace of the stacked finite-difference Jacobians of `teacher`,
    queried in `dtype`. Returns (U [d, k] float64 orthonormal, singular values, queries)."""
    net = copy.deepcopy(teacher).to(dtype).eval()
    d = centers.shape[1]
    jac, n = [], 0
    for c in centers.to(dtype):
        cols = []
        for a in range(0, d, chunk):
            b = min(d, a + chunk)
            E = torch.zeros(b - a, d, dtype=dtype, device=c.device)
            E[torch.arange(b - a), torch.arange(a, b)] = eps
            y = net(torch.cat([c + E, c - E]))
            n += 2 * (b - a)
            cols.append(((y[:b - a] - y[b - a:]) / (2 * eps)).T)
        jac.append(torch.cat(cols, 1))
    S = torch.cat(jac).double()
    if S.shape[0] < k:
        raise SystemExit(f"[subspace] {S.shape[0]} Jacobian rows cannot span {k} dims; add --subspace-centers")
    _, s, vh = torch.linalg.svd(S, full_matrices=False)
    return vh[:k].T.contiguous(), s, n


@torch.no_grad()
def reduce(teacher, U):
    """MLP computing z -> teacher(U z): first layer W1 U, all later layers unchanged."""
    p = next(teacher.parameters())
    red = MLP([U.shape[1]] + list(teacher.dims[1:]), getattr(teacher.act, "negative_slope", 0.01),
              teacher.act_name).to(device=p.device, dtype=p.dtype)
    red.load_state_dict({k: v for k, v in teacher.state_dict().items() if k != "layers.0.weight"}, strict=False)
    red.layers[0].weight.copy_((teacher.layers[0].weight.double() @ U.to(p.device)).to(p.dtype))
    z = torch.randn(1000, U.shape[1], generator=torch.Generator().manual_seed(0), dtype=torch.float64).to(p.device)
    ref = teacher((z @ U.T.to(p.device)).to(p.dtype)).double()
    err = ((red(z.to(p.dtype)).double() - ref).abs().max() / ref.abs().max()).item()
    if err > 1e-4:
        raise SystemExit(f"[subspace] reduced black box differs from teacher(U z): {err:.2e}")
    return red.eval(), err


def map_back(net, U, full_dims):
    """Full-input net (float64) with W1_hat = W1' U^T."""
    full = MLP(full_dims, getattr(net.act, "negative_slope", 0.01), net.act_name).double()
    sd = {k: v.detach().double().cpu() for k, v in net.state_dict().items()}
    sd["layers.0.weight"] = sd["layers.0.weight"] @ U.T.cpu()
    full.load_state_dict(sd)
    return full


def W1_outside(U, teacher):
    """EVALUATION ONLY: relative norm of the teacher's W1 outside span(U)."""
    W1 = teacher.layers[0].weight.detach().double()
    U = U.to(W1.device)
    return ((W1.T - U @ (U.T @ W1.T)).norm() / W1.norm()).item()
