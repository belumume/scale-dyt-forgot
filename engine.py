"""DyT stress-test core engine ("The Scale DyT Forgot").

Stress-tests "Transformers without Normalization" (Zhu, Chen, He, Yu, Yuille,
LeCun; arXiv 2503.10622): DyT(x) = tanh(alpha * x) * gamma + beta, dropped in
wherever a Pre-LN transformer used LayerNorm. alpha is a single LEARNED
SCALAR per norm site, frozen after training.

The claim under test is not "does DyT train well" (the paper already shows
that, on the input distribution it was trained on). It is: LayerNorm is
exactly scale-invariant by a one-line algebraic identity --

    LN(c * x) = (c*x - mean(c*x)) / std(c*x) = sign(c) * LN(x)   for c != 0

(mean and std both scale by |c| and cancel; gamma/beta are per-channel and
never see x directly). DyT has no such identity: tanh(alpha * c * x) is only
close to tanh(alpha * x) while alpha*x stays in tanh's near-linear region
near 0. Once c pushes alpha*x into the saturating tails, DyT clips to
+-gamma+beta regardless of what the un-scaled activation actually was --
losing exactly the per-token information a normalization layer exists to
preserve. alpha cannot compensate because it is fixed after training; it
never sees the shifted distribution.

This module builds:
  1. Three interchangeable norm sites (LayerNorm / DyT / a candidate fix,
     RMS-DyT) sharing one interface, plus a numeric invariance-gap check
     that is exact algebra for two of them and empirical for DyT.
  2. A fully synthetic causal TinyTransformer trained on the induction/copy
     task (Olsson et al. 2022, "In-context Learning and Induction Heads",
     Anthropic transformer-circuits thread): a random sequence is repeated
     once, and the model must predict, for each token in the second half,
     the token that followed that same token's earlier occurrence. Needs
     no download, no external data -- everything is generated in-process.
  3. A rescale stress harness: multiply every norm site's input by an
     independent per-token lognormal(0, sigma) factor times a global
     multiplier, and measure the accuracy hit. This models real-world
     activation-scale heterogeneity (mixed sequence lengths, outlier
     tokens, quantization/precision drift, domain shift) that a frozen
     scalar alpha was never trained against.
  4. A depth sweep testing a prediction implied by the paper's own Fig 8-9
     analysis (trained alpha values adapt to the depth-dependent scale of
     their OWN training-time activations): if alpha ends up precisely
     tuned to sit just inside its linear region for the depth it was
     trained at, a frozen alpha has progressively less headroom before
     saturation the deeper the network, and errors introduced at any one
     site compound through the residual stream at every site downstream.
     The prediction is that the LN-vs-DyT fragility gap under an identical
     stress test widens with depth.
  5. A candidate fix (not in the paper): divide by the per-token RMS before
     the learned-alpha tanh, i.e. DyT(x) = tanh(alpha * x / rms(x)) *
     gamma + beta. Algebraically this restores the same cancellation
     LayerNorm relies on (rms(c*x) = |c| * rms(x) for any per-token
     positive-scalar rescale, including the per-token heterogeneous case),
     while keeping DyT's tanh nonlinearity and learned-scalar shape. The
     honest cost: it reintroduces the per-token reduction op DyT's whole
     pitch was designed to eliminate.

Design notes:
  - Attention uses torch.nn.functional.scaled_dot_product_attention
    (built into torch, no extra dependency) with is_causal=True. Everything
    else is plain nn.Module / nn.Parameter -- standard autograd, no manual
    backprop; a tiny attention transformer does not benefit from
    hand-derived gradients the way the flagship fractal MLP does.
  - alpha0 = 0.5 matches the paper's stated default for language-model-
    scale Pre-LN transformers; the paper's own appendix uses a smaller
    alpha0 specifically for the final projection norm, which we do not
    replicate here (one alpha0 for every site, for simplicity -- fully
    configurable via Config.dyt_alpha0 if a reader wants to explore that).
  - Everything stays fp32. Logits are clamped before cross-entropy. Every
    eval loop treats a non-finite forward pass as a soft 0.0 accuracy
    rather than raising -- a model that diverges under stress IS a
    legitimate experimental outcome here, not a bug to crash the notebook
    on. assert_finite() is the strict counterpart, used where a NaN really
    would mean a bug (the smoke test's own training loss).
  - Device fallback: default_config(device) hands back a GPU-scale Config
    when CUDA is present and a smaller CPU-scale one otherwise, so the
    notebook still runs end-to-end with zero errors if a session ever
    loses its GPU.
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class Config:
    # architecture
    vocab_size: int = 24
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    seq_len: int = 16  # HALF length; full sequence is 2 * seq_len
    norm: str = "layernorm"  # "layernorm" | "dyt" | "rms_dyt"
    dyt_alpha0: float = 0.5
    dropout: float = 0.0
    eps: float = 1e-5
    # training
    steps: int = 600
    batch_size: int = 64
    # 1e-2 (not the more conventional 3e-3) after local tuning: at 3e-3 the
    # induction-head "phase transition" (Olsson et al.) is unreliable
    # within a tiny-model step budget -- DyT/RMS-DyT stalled near chance,
    # LayerNorm was inconsistent across seeds. 1e-2 converges reliably for
    # all three norm kinds at the step counts below; grad_clip keeps it
    # stable (no divergence observed across the norms/seeds tested).
    lr: float = 1e-2
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eval_every: int = 50
    eval_batches: int = 8

    def __post_init__(self):
        if self.vocab_size < self.seq_len:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) must be >= seq_len "
                f"({self.seq_len}) -- each half of the induction sequence "
                "needs distinct tokens for the target to be unambiguous."
            )


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_config(device: torch.device | str) -> Config:
    """The notebook's GPU story is the *number of runs* (the multi-seed x
    multi-depth x multi-norm training matrix, 27 trainings, plus a
    91-point rescale grid over 3 models), NOT a large single model. So the
    GPU config keeps a small, reliably-trainable model and spends the GPU
    on running many of them.

    The dims below (d_model=32, seq_len=10, vocab=16) are the regime local
    CPU testing VERIFIED trains all three norm kinds above chance and
    reproduces the full result (LayerNorm/RMS-DyT flat under rescale, DyT
    collapsing). A larger model (d_model=64, seq_len=16) was tried and left
    plain DyT stuck at chance within 900 steps -- undertrained, and exactly
    the kind of silent failure that would gut the notebook's central
    comparison. steps=1500 (up from the 900 the story was verified at) and
    batch_size=64 both move in the safe direction (more optimization, less
    gradient noise) to reduce the seed-to-seed variance induction-head
    formation is known for.

    FIRST MOLAB ACTION for this entry (see NOTES.md): train all three
    norms at each depth in DEPTH_LIST across SEED_LIST and confirm every
    baseline lands clearly above chance BEFORE trusting the depth sweep or
    the rescale grid. If a larger model is wanted for the submission's
    optics, scale up HERE and re-verify convergence on the GPU (seconds per
    run there), never assume it transfers.

    The CPU branch shrinks steps/batch so a local smoke test -- or a molab
    session that ever loses its GPU -- still runs end-to-end without
    hanging (no errors = no DQ).
    """
    device = torch.device(device)
    if device.type == "cuda":
        return Config(
            vocab_size=16,
            d_model=32,
            n_heads=2,
            n_layers=2,
            seq_len=10,
            steps=1500,
            batch_size=64,
            eval_every=150,
            eval_batches=8,
        )
    return Config(
        vocab_size=16,
        d_model=32,
        n_heads=2,
        n_layers=2,
        seq_len=10,
        steps=150,
        batch_size=24,
        eval_every=25,
        eval_batches=4,
    )


def chance_accuracy(cfg: Config) -> float:
    return 1.0 / cfg.vocab_size


def assert_finite(t: torch.Tensor, name: str) -> None:
    """Raise immediately and clearly the moment a tensor goes non-finite,
    instead of letting a NaN propagate silently into a plot -- or into a
    disqualifying uncaught exception three cells later."""
    if not torch.isfinite(t).all():
        bad = int((~torch.isfinite(t)).sum().item())
        raise FloatingPointError(f"{name}: {bad}/{t.numel()} non-finite values")


# --------------------------------------------------------------------------
# Norm sites: LayerNorm, DyT, and the candidate RMS-DyT fix, one interface.
# --------------------------------------------------------------------------


class _NormSite(nn.Module):
    """Shared interface: forward(x, rescale) = governing_quantity(x *
    rescale) * gamma + beta. `governing_quantity` is the pre-affine value
    whose distribution each norm's scale-robustness claim is actually
    about -- subclasses implement it, forward() is generic."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    @staticmethod
    def _rescaled(x: torch.Tensor, rescale) -> torch.Tensor:
        return x if rescale is None else x * rescale

    def governing_quantity(self, x: torch.Tensor, rescale=None) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, rescale=None) -> torch.Tensor:
        return self.governing_quantity(x, rescale) * self.gamma + self.beta


class LayerNormSite(_NormSite):
    """Standard LayerNorm. governing_quantity is xhat = (x-mean)/std,
    which is EXACTLY invariant (up to sign) to any nonzero scalar
    multiplier applied to x: mean and std both scale by |c| and cancel."""

    def governing_quantity(self, x: torch.Tensor, rescale=None) -> torch.Tensor:
        x = self._rescaled(x, rescale)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps)


class DyTSite(_NormSite):
    """Zhu et al. 2503.10622: DyT(x) = tanh(alpha*x)*gamma + beta. alpha is
    ONE learned scalar per site, frozen after training -- nothing in the
    forward pass lets it react to an input scale it never saw at train
    time. governing_quantity is tanh(pre_tanh(x)); pre_tanh(x) = alpha*x
    is the number whose magnitude decides whether tanh saturates."""

    def __init__(self, dim: int, alpha0: float, eps: float):
        super().__init__(dim, eps)
        self.alpha = nn.Parameter(torch.tensor(float(alpha0)))

    def pre_tanh(self, x: torch.Tensor, rescale=None) -> torch.Tensor:
        x = self._rescaled(x, rescale)
        return self.alpha * x

    def governing_quantity(self, x: torch.Tensor, rescale=None) -> torch.Tensor:
        return torch.tanh(self.pre_tanh(x, rescale))


class RMSDyTSite(DyTSite):
    """Candidate fix tried in this notebook, not in the paper: divide by
    the per-token RMS before the learned-alpha tanh. Algebraically,
    rms(c*x) = |c| * rms(x) for ANY per-token positive scalar c -- global
    or per-token heterogeneous -- so pre_tanh cancels the same way
    LayerNorm's xhat does:

        pre_tanh(c*x) = alpha*c*x / (|c|*rms(x)) = sign(c) * pre_tanh(x)

    This keeps DyT's tanh + learned-scalar shape while restoring the exact
    invariance property LayerNorm has by construction. The honest cost:
    computing rms(x) is a per-token reduction over d_model, exactly the
    operation DyT's efficiency pitch was designed to eliminate."""

    def pre_tanh(self, x: torch.Tensor, rescale=None) -> torch.Tensor:
        x = self._rescaled(x, rescale)
        rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
        return self.alpha * x / rms


def make_norm(kind: str, dim: int, cfg: Config) -> _NormSite:
    if kind == "layernorm":
        return LayerNormSite(dim, cfg.eps)
    if kind == "dyt":
        return DyTSite(dim, cfg.dyt_alpha0, cfg.eps)
    if kind == "rms_dyt":
        return RMSDyTSite(dim, cfg.dyt_alpha0, cfg.eps)
    raise ValueError(f"unknown norm kind {kind!r}; expected layernorm | dyt | rms_dyt")


def diagnostic_quantity(norm: _NormSite, x: torch.Tensor, rescale=None) -> torch.Tensor:
    """The value to histogram / compare for invariance: DyT and RMS-DyT ->
    the pre-tanh argument (the number that saturates); LayerNorm -> xhat
    itself (there is no saturating nonlinearity downstream of it to
    inspect, so xhat IS the quantity whose stability under rescale is the
    entire claim)."""
    if isinstance(norm, DyTSite):
        return norm.pre_tanh(x, rescale)
    return norm.governing_quantity(x, rescale)


def ln_invariance_gap(model_or_layer, x: torch.Tensor, multiplier: float) -> float:
    """max|diagnostic_quantity(multiplier*x) - diagnostic_quantity(x)| for
    the given norm site, or the first norm site found inside a
    TinyTransformer. Zero up to float round-off for LayerNorm AND for
    RMS-DyT's pre-tanh argument (both cancel any positive scalar
    multiplier by construction). Grows with |log(multiplier)| for plain
    DyT once alpha*x leaves tanh's near-linear region and saturates."""
    site = (
        model_or_layer
        if isinstance(model_or_layer, _NormSite)
        else norm_sites(model_or_layer)[0]
    )
    with torch.no_grad():
        a = diagnostic_quantity(site, x)
        b = diagnostic_quantity(site, x, rescale=multiplier)
        return float((a - b).abs().max().item())


# --------------------------------------------------------------------------
# TinyTransformer: causal, Pre-Norm, configurable norm site.
# --------------------------------------------------------------------------


class _CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, D // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each [B, n_heads, T, head_dim]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.drop(self.proj(out))


class _Block(nn.Module):
    """Pre-Norm transformer block: x + attn(norm(x)); x + mlp(norm(x)).
    Matches the paper's setup (DyT replaces every LayerNorm in a Pre-LN
    transformer, including the final one before the output head)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = make_norm(cfg.norm, cfg.d_model, cfg)
        self.attn = _CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm2 = make_norm(cfg.norm, cfg.d_model, cfg)
        hidden = 4 * cfg.d_model
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, rescale=None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x, rescale))
        x = x + self.mlp(self.norm2(x, rescale))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(2 * cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(_Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = make_norm(cfg.norm, cfg.d_model, cfg)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # standard GPT-style tying

    def forward(self, idx: torch.Tensor, rescale=None) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x, rescale)
        x = self.norm_f(x, rescale)
        return self.head(x)


def norm_sites(model: TinyTransformer) -> list[_NormSite]:
    return [m for m in model.modules() if isinstance(m, _NormSite)]


# --------------------------------------------------------------------------
# The rescale stress: global multiplier x per-token heterogeneity.
# --------------------------------------------------------------------------


def make_rescale(
    batch: int, seq_len_full: int, multiplier: float, sigma: float, seed: int, device
) -> torch.Tensor | None:
    """Per-token heterogeneous activation rescale, shape [batch,
    seq_len_full, 1] (broadcasts over d_model). Each token's own factor is
    multiplier * LogNormal(0, sigma); the SAME per-token draw is applied at
    every norm site within one forward pass (one "distortion profile" per
    token, reused through depth -- a physically sensible model of e.g. an
    unusually long document or an outlier token whose scale is elevated
    throughout, rather than independently re-rolled noise at every layer).
    sigma=0 collapses every token to the same global multiplier; multiplier
    == 1 and sigma == 0 returns None (a plain forward pass, no rescale).
    """
    if multiplier == 1.0 and sigma == 0.0:
        return None
    g = torch.Generator(device="cpu").manual_seed(seed)
    if sigma > 0:
        z = torch.randn(batch, seq_len_full, 1, generator=g)
        per_token = torch.exp(sigma * z)
    else:
        per_token = torch.ones(batch, seq_len_full, 1)
    return (multiplier * per_token).to(device=device, dtype=torch.float32)


# --------------------------------------------------------------------------
# Synthetic induction/copy task (Olsson et al. 2022).
# --------------------------------------------------------------------------


def induction_eval_start(cfg: Config) -> int:
    """Index into the (idx, targets) arrays (length 2*seq_len - 1) where
    induction-testable predictions begin. Position seq_len - 1 is the seam
    between the two halves and is NOT decidable from induction (the token
    there is the last, unique element of the first half, so it has no
    earlier occurrence to look up yet) -- excluding it keeps the accuracy
    metric honestly about induction rather than a memorized wrap-around.
    From position seq_len onward, every target is well-defined: the
    current token's only earlier occurrence sits at (t - seq_len) in the
    first half, and the token that followed it there is exactly the
    ordinary next-token target, because the second half is a verbatim
    repeat of the first."""
    return cfg.seq_len


def make_induction_batch(
    batch_size: int, vocab_size: int, half_len: int, seed: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Doubled-random-sequence induction task. Each half is a random
    permutation of `half_len` DISTINCT tokens drawn from [0, vocab_size),
    so every token's earlier-occurrence lookup is unambiguous. full
    sequence = [half, half]; idx/targets are the standard next-token
    shift. Returns (idx, targets), each [batch_size, 2*half_len - 1]."""
    if vocab_size < half_len:
        raise ValueError("vocab_size must be >= half_len for unambiguous induction")
    g = torch.Generator(device="cpu").manual_seed(seed)
    halves = torch.stack(
        [torch.randperm(vocab_size, generator=g)[:half_len] for _ in range(batch_size)]
    )
    full = torch.cat([halves, halves], dim=1)  # [B, 2*half_len]
    idx = full[:, :-1].to(device)
    targets = full[:, 1:].to(device)
    return idx, targets


def _accuracy(logits: torch.Tensor, targets: torch.Tensor, from_pos: int) -> float:
    pred = logits[:, from_pos:, :].argmax(dim=-1)
    return float((pred == targets[:, from_pos:]).float().mean().item())


def _forward_loss(
    model: TinyTransformer, idx: torch.Tensor, targets: torch.Tensor, rescale=None
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model(idx, rescale=rescale)
    logits = logits.clamp(
        -30.0, 30.0
    )  # protect cross-entropy under saturated activations
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return loss, logits


@torch.no_grad()
def _eval_accuracy(
    model: TinyTransformer,
    cfg: Config,
    seed: int,
    device,
    rescale_kwargs: dict | None = None,
) -> float:
    model.eval()
    from_pos = induction_eval_start(cfg)
    accs = []
    for b in range(cfg.eval_batches):
        idx, targets = make_induction_batch(
            cfg.batch_size,
            cfg.vocab_size,
            cfg.seq_len,
            seed=seed * 7_777 + b,
            device=device,
        )
        rescale = None
        if rescale_kwargs is not None:
            rescale = make_rescale(
                idx.shape[0],
                idx.shape[1],
                seed=seed * 13 + b,
                device=device,
                **rescale_kwargs,
            )
        logits = model(idx, rescale=rescale)
        accs.append(
            _accuracy(logits, targets, from_pos)
            if torch.isfinite(logits).all()
            else 0.0
        )
    model.train()
    return float(np.mean(accs))


def train_model(cfg: Config, seed: int, device) -> tuple[TinyTransformer, dict]:
    """Train one TinyTransformer on the induction task. Returns (model,
    history); history holds 'step' / 'loss' / 'induction_acc' trajectories
    at eval_every cadence, plus 'final_acc' and 'diverged'. A non-finite
    training loss aborts the run early and is recorded, not raised -- a
    model that diverges under a given (cfg, seed) IS a legitimate
    experimental outcome for the depth sweep, not a bug to crash on."""
    torch.manual_seed(seed)
    model = TinyTransformer(cfg).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    history: dict = {"step": [], "loss": [], "induction_acc": [], "diverged": False}

    for step in range(cfg.steps):
        idx, targets = make_induction_batch(
            cfg.batch_size,
            cfg.vocab_size,
            cfg.seq_len,
            seed=seed * 100_000 + step,
            device=device,
        )
        loss, _ = _forward_loss(model, idx, targets)
        if not torch.isfinite(loss):
            history["diverged"] = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            history["step"].append(step)
            history["loss"].append(float(loss.item()))
            history["induction_acc"].append(_eval_accuracy(model, cfg, seed, device))

    history["final_acc"] = (
        history["induction_acc"][-1] if history["induction_acc"] else 0.0
    )
    return model, history


def eval_under_rescale(
    model: TinyTransformer,
    cfg: Config,
    scale_multiplier: float,
    per_token_sigma: float,
    seed: int,
    device,
) -> float:
    """Accuracy of an already-trained model when every norm site sees its
    input pre-multiplied by an independent per-token lognormal(0, sigma)
    factor times scale_multiplier (sigma=0 -> pure global rescale;
    multiplier=1, sigma=0 -> identity, the model's ordinary training-scale
    accuracy)."""
    return _eval_accuracy(
        model,
        cfg,
        seed,
        device,
        rescale_kwargs={"multiplier": scale_multiplier, "sigma": per_token_sigma},
    )


def fragility_gap(acc_baseline: float, acc_rescaled: float) -> float:
    """Accuracy drop under rescale, floored at 0 (a model that gets
    marginally LUCKIER under noise is not "more fragile than itself")."""
    return max(0.0, acc_baseline - acc_rescaled)


# --------------------------------------------------------------------------
# The precomputed grid the interactive 2D pad indexes -- never a live
# forward pass on slider drag.
# --------------------------------------------------------------------------


@dataclasses.dataclass
class RescaleGrid:
    """One full sweep over (log2_multiplier, sigma), computed once. Every
    interactive panel reads out of these arrays; nothing here re-fires on
    a slider tick."""

    log2_multipliers: np.ndarray  # [M]
    sigmas: np.ndarray  # [S]
    accuracy: dict[str, np.ndarray]  # name -> [M, S]
    hist: dict[str, np.ndarray]  # name -> [M, S, n_bins]; governing-quantity histogram
    hist_edges: np.ndarray  # [n_bins + 1]
    saturation_threshold: float


def rescale_response_grid(
    models: dict[str, TinyTransformer],
    cfg: Config,
    log2_multipliers: np.ndarray,
    sigmas: np.ndarray,
    seed: int,
    device,
    hist_bins: int = 40,
    hist_range: tuple[float, float] = (-8.0, 8.0),
    saturation_threshold: float = 3.0,
) -> RescaleGrid:
    """Compute accuracy(mult, sigma) and a governing-quantity histogram for
    every (model, mult, sigma) triple ONCE, over a fixed shared set of eval
    batches and shared rescale-noise seeds across models (so the same
    (mult, sigma) grid point applies IDENTICAL stress to every model --
    paired comparison, not independently-noisy runs). This is the single
    expensive pass; the notebook's interactive panels only ever index the
    result arrays. The histogram uses the FIRST norm site (block 0's
    pre-attention norm, i.e. the input layer) as the representative site;
    deeper sites show a similar or amplified pattern, which is exactly
    what the depth-fragility sweep measures in aggregate.
    """
    from_pos = induction_eval_start(cfg)
    edges = np.linspace(*hist_range, hist_bins + 1)
    eval_batches = [
        make_induction_batch(
            cfg.batch_size,
            cfg.vocab_size,
            cfg.seq_len,
            seed=seed * 7_777 + b,
            device=device,
        )
        for b in range(cfg.eval_batches)
    ]

    n_m, n_s = len(log2_multipliers), len(sigmas)
    accuracy = {name: np.zeros((n_m, n_s), dtype=np.float32) for name in models}
    hist = {name: np.zeros((n_m, n_s, hist_bins), dtype=np.float32) for name in models}

    for name, model in models.items():
        model.eval()
        first_site = norm_sites(model)[0]
        with torch.no_grad():
            for i, lm in enumerate(log2_multipliers):
                mult = float(2.0**lm)
                for j, sig in enumerate(sigmas):
                    accs = []
                    sample_quantity = None
                    for b, (idx, targets) in enumerate(eval_batches):
                        rescale = make_rescale(
                            idx.shape[0],
                            idx.shape[1],
                            mult,
                            float(sig),
                            seed=seed * 13 + b,
                            device=device,
                        )
                        logits = model(idx, rescale=rescale)
                        if not torch.isfinite(logits).all():
                            accs.append(0.0)
                            continue
                        accs.append(_accuracy(logits, targets, from_pos))
                        if b == 0:  # one batch is plenty for a histogram
                            x0 = (
                                model.tok_emb(idx)
                                + model.pos_emb(
                                    torch.arange(idx.shape[1], device=device)
                                )[None]
                            )
                            q = diagnostic_quantity(first_site, x0, rescale)
                            sample_quantity = q.detach().float().cpu().numpy().ravel()
                    accuracy[name][i, j] = float(np.mean(accs)) if accs else 0.0
                    if sample_quantity is not None:
                        counts, _ = np.histogram(sample_quantity, bins=edges)
                        hist[name][i, j] = counts.astype(np.float32)
        model.train()

    return RescaleGrid(
        log2_multipliers=np.asarray(log2_multipliers, dtype=np.float32),
        sigmas=np.asarray(sigmas, dtype=np.float32),
        accuracy=accuracy,
        hist=hist,
        hist_edges=edges,
        saturation_threshold=saturation_threshold,
    )


# --------------------------------------------------------------------------
# Depth sweep: does the fragility gap widen with depth (paper Fig 8-9)?
# --------------------------------------------------------------------------


def depth_fragility_sweep(
    cfg_base: Config,
    depths: list[int],
    norms: list[str],
    seeds: list[int],
    stress_multiplier: float,
    stress_sigma: float,
    device,
    progress=None,
) -> list[dict]:
    """Train one model per (depth, norm, seed), evaluate baseline (no
    rescale) and stressed (stress_multiplier, stress_sigma) accuracy, and
    return one flat result row per run. `progress`, if given, is called as
    progress(done, total) after each run -- a plain callable, so this
    function stays independent of marimo and independently testable.

    This is what the depth-prediction panel tests: a well-trained alpha
    adapted (via training) to sit inside its linear region for its OWN
    training-time activation scale at that depth. That predicts -- but
    does not by itself measure -- that a frozen alpha has progressively
    less headroom before saturation the deeper the network it was trained
    in, and that a saturation event at any one site compounds through the
    residual stream at every later site. The empirical question this
    function answers is whether the LN-vs-DyT gap under an IDENTICAL
    stress test actually widens with depth, or stays flat.
    """
    results = []
    total = len(depths) * len(norms) * len(seeds)
    done = 0
    for depth in depths:
        for norm in norms:
            for seed in seeds:
                cfg = dataclasses.replace(cfg_base, n_layers=depth, norm=norm)
                model, hist = train_model(cfg, seed=seed, device=device)
                baseline = hist["final_acc"]
                stressed = eval_under_rescale(
                    model,
                    cfg,
                    stress_multiplier,
                    stress_sigma,
                    seed=seed,
                    device=device,
                )
                results.append(
                    dict(
                        depth=depth,
                        norm=norm,
                        seed=seed,
                        baseline_acc=baseline,
                        stressed_acc=stressed,
                        gap=fragility_gap(baseline, stressed),
                        diverged=bool(hist["diverged"]),
                    )
                )
                done += 1
                if progress is not None:
                    progress(done, total)
    return results


def main():
    torch.manual_seed(0)
    print("=== DyT stress-test engine smoke test (tiny, local, CPU-safe) ===")

    print("\n-- scale invariance, untrained sites, d=32 --")
    demo_cfg = Config(vocab_size=16, seq_len=10, dyt_alpha0=0.5)
    x = torch.randn(4, 6, 32)
    for kind in ("layernorm", "dyt", "rms_dyt"):
        site = make_norm(kind, 32, demo_cfg)
        gaps = [ln_invariance_gap(site, x, m) for m in (1.0, 2.0, 8.0)]
        print(f"  {kind:10s} gap @ mult 1/2/8: {[f'{g:.4f}' for g in gaps]}")

    device = pick_device()
    cfg = dataclasses.replace(
        default_config(device),
        n_layers=2,
        steps=40,
        batch_size=16,
        eval_batches=2,
        eval_every=10,
    )
    print(f"\n-- tiny training smoke test on {device} --")
    t0 = time.time()
    model, hist = train_model(cfg, seed=0, device=device)
    dt = time.time() - t0
    assert_finite(torch.tensor(hist["loss"]), "loss trajectory")
    print(
        f"  {cfg.steps} steps in {dt:.2f}s | final_acc={hist['final_acc']:.3f} "
        f"(chance={chance_accuracy(cfg):.3f}) | diverged={hist['diverged']}"
    )

    print("\n-- eval_under_rescale --")
    base_acc = eval_under_rescale(model, cfg, 1.0, 0.0, seed=0, device=device)
    stressed_acc = eval_under_rescale(model, cfg, 8.0, 0.5, seed=0, device=device)
    print(
        f"  baseline acc={base_acc:.3f} | stressed (x8, sigma=0.5) acc={stressed_acc:.3f}"
    )

    print("\n-- mini rescale_response_grid --")
    grid = rescale_response_grid(
        {"demo": model},
        cfg,
        log2_multipliers=np.array([-1.0, 0.0, 1.0]),
        sigmas=np.array([0.0, 0.5]),
        seed=0,
        device=device,
        hist_bins=10,
    )
    print(
        f"  accuracy shape={grid.accuracy['demo'].shape} hist shape={grid.hist['demo'].shape}"
    )

    print("\n-- mini depth_fragility_sweep (2 depths x 2 norms x 1 seed) --")
    rows = depth_fragility_sweep(
        dataclasses.replace(cfg, steps=25),
        depths=[2, 4],
        norms=["layernorm", "dyt"],
        seeds=[0],
        stress_multiplier=8.0,
        stress_sigma=0.5,
        device=device,
    )
    for r in rows:
        print(
            f"  depth={r['depth']} norm={r['norm']:9s} "
            f"baseline={r['baseline_acc']:.3f} stressed={r['stressed_acc']:.3f} gap={r['gap']:.3f}"
        )

    print("\nsmoke test OK -- no NaNs, no crashes.")
    print(
        "GPU-scale timings here are a FLOPs-based estimate (see NOTES.md); "
        "this script deliberately stays tiny -- run the notebook on molab "
        "for the real multi-seed x multi-depth matrix."
    )


if __name__ == "__main__":
    main()
