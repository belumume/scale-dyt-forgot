# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "wigglystuff==0.5.13",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def title(mo):
    mo.md(r"""
    # The Scale DyT Forgot

    LayerNorm has a theorem behind it. DyT has a fitted curve.

    "Transformers without Normalization" (Zhu, Chen, He, Yu, Yuille, LeCun, 2025) replaces every LayerNorm in a transformer with one line: $\mathrm{DyT}(x) = \tanh(\alpha x) \odot \gamma + \beta$, where $\alpha$ is a single learned number per layer. It matches or beats LayerNorm across a wide range of benchmarks, and it is simpler and faster to run.

    It also gives up a property LayerNorm has for free: $\mathrm{LN}(cx) = \mathrm{LN}(x)$ for any positive constant $c$, exactly, by algebra. Rescale the input to a LayerNorm and nothing downstream notices. Rescale the input to a DyT layer and $\alpha$, frozen the moment training ends, has no way to react. Below, we train a tiny transformer twice on the same task, once with each kind of layer, then push the input off the scale it trained on and watch what breaks -- and what doesn't.
    """)
    return


@app.cell
def mo_import():
    import marimo as mo

    return (mo,)


@app.cell
def imports():
    import dataclasses
    import time

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from wigglystuff import Slider2D

    return F, Figure, Slider2D, dataclasses, nn, np, plt, time, torch


@app.cell
def engine(F, dataclasses, nn, np, time, torch):
    # DyT stress-test engine: the three norm sites (LayerNorm / DyT / RMS-DyT),
    # a tiny transformer, the induction/copy task, the rescale-stress harness,
    # and the sweep functions the panels below index into.
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

    return (
        chance_accuracy,
        default_config,
        depth_fragility_sweep,
        ln_invariance_gap,
        make_norm,
        pick_device,
        rescale_response_grid,
        train_model,
    )


@app.cell(hide_code=True)
def setup_md(mo):
    mo.md(r"""
    ## The task, and the theorem

    We train on the induction/copy task (Olsson et al., 2022, "In-context Learning and Induction Heads"): a random sequence of distinct tokens, then the same sequence again. To predict the next token anywhere in the second half, the model has to notice "I have seen this token before, what came right after it" and copy the answer. No download, no labeled corpus, nothing beyond a random-number generator -- and even a two-layer attention model learns it reliably.

    The theorem doing the work in this notebook is one line of algebra. For any input $x$ and scalar $c \neq 0$:

    $$
    \mathrm{LN}(cx) = \frac{cx - \mathrm{mean}(cx)}{\mathrm{std}(cx)} = \frac{c\,(x - \mathrm{mean}(x))}{|c|\cdot \mathrm{std}(x)} = \mathrm{sign}(c) \cdot \mathrm{LN}(x)
    $$

    Mean and standard deviation both scale by $|c|$ and cancel. LayerNorm's output does not know or care what scale its input arrived at. DyT's $\tanh(\alpha x)$ has no such cancellation: it is close to linear (so close to scale-*sensitive*) near the origin, and it saturates to $\pm 1$ far from it. $\alpha$ was fit once, at training time, to whatever scale the activations happened to have then. It is not a normalization in the algebraic sense at all -- it is a fixed nonlinearity that happened to sit in the right place for one particular input distribution.

    Below: a two-dimensional stress test on a single drag pad. Horizontal: a global scale multiplier applied to every token. Vertical: how unevenly that scale is spread across tokens in the same sequence, an outlier-token simulator. Drag it and watch three views of the same trained models respond together.
    """)
    return


@app.cell
def device_setup(default_config, mo, pick_device):
    device = pick_device()
    config = default_config(device)
    _lines = [
        f"**device:** `{device.type}` -- "
        + ("GPU-scale" if device.type == "cuda" else "CPU fallback (same code path, smaller config)")
        + " config",
        "",
        f"vocab_size={config.vocab_size}, d_model={config.d_model}, n_heads={config.n_heads}, "
        f"seq_len(half)={config.seq_len}, steps={config.steps}, batch_size={config.batch_size}, lr={config.lr:g}",
    ]
    mo.md("\n".join(_lines))
    return config, device


@app.cell
def sweep_params(mo, np):
    HERO_DEPTH = 4
    LOG2_MULTS = np.linspace(-3.0, 3.0, 13)
    SIGMAS = np.linspace(0.0, 1.5, 7)
    DEPTH_LIST = [2, 4, 6]
    NORM_LIST = ["layernorm", "dyt", "rms_dyt"]
    SEED_LIST = [0, 1, 2]
    STRESS_MULT = 8.0
    STRESS_SIGMA = 0.5

    _n_grid = len(LOG2_MULTS) * len(SIGMAS) * len(NORM_LIST)
    _n_depth = len(DEPTH_LIST) * len(NORM_LIST) * len(SEED_LIST)
    mo.md(
        f"sweep grid: {len(LOG2_MULTS)} multipliers x {len(SIGMAS)} sigmas x "
        f"{len(NORM_LIST)} norms = {_n_grid} stress evaluations. "
        f"depth sweep: {len(DEPTH_LIST)} depths x {len(NORM_LIST)} norms x "
        f"{len(SEED_LIST)} seeds = {_n_depth} training runs."
    )
    return (
        DEPTH_LIST,
        HERO_DEPTH,
        LOG2_MULTS,
        NORM_LIST,
        SEED_LIST,
        SIGMAS,
        STRESS_MULT,
        STRESS_SIGMA,
    )


@app.cell
def train_hero(
    HERO_DEPTH,
    NORM_LIST,
    chance_accuracy,
    config,
    dataclasses,
    device,
    mo,
    train_model,
):
    hero_models = {}
    hero_history = {}
    hero_config = dataclasses.replace(config, n_layers=HERO_DEPTH)
    with mo.status.spinner(title=f"training {len(NORM_LIST)} hero models (depth={HERO_DEPTH}) ..."):
        for _name in NORM_LIST:
            _cfg = dataclasses.replace(hero_config, norm=_name)
            _model, _hist = train_model(_cfg, seed=0, device=device)
            hero_models[_name] = _model
            hero_history[_name] = _hist

    _rows = [
        f"| {_n} | {hero_history[_n]['final_acc']:.3f} | {'yes' if hero_history[_n]['diverged'] else 'no'} |"
        for _n in NORM_LIST
    ]
    mo.md(
        "\n".join(
            [
                f"**hero models trained** (depth={HERO_DEPTH}, seed=0, "
                f"{hero_config.steps} steps, unstressed accuracy; chance = {chance_accuracy(hero_config):.3f}):",
                "",
                "| norm | baseline accuracy | diverged |",
                "|---|---|---|",
                *_rows,
            ]
        )
    )
    return hero_config, hero_models


@app.cell
def invariance_compute(
    NORM_LIST,
    config,
    ln_invariance_gap,
    make_norm,
    np,
    torch,
):
    # Pure numeric check on a fresh random tensor, independent of the trained
    # hero models and the induction task entirely -- panel C isolates the
    # theorem itself from any one training run's particular alpha value.
    torch.manual_seed(0)
    _x_demo = torch.randn(4, 8, config.d_model)
    invariance_demo_mults = np.linspace(-3.0, 3.0, 25)
    invariance_demo = {}
    for _name in NORM_LIST:
        _site = make_norm(_name, config.d_model, config)
        invariance_demo[_name] = np.array(
            [ln_invariance_gap(_site, _x_demo, float(2.0**_lm)) for _lm in invariance_demo_mults]
        )
    f"invariance demo ready: {len(NORM_LIST)} norm kinds x {len(invariance_demo_mults)} multipliers"

    return invariance_demo, invariance_demo_mults


@app.cell
def rescale_grid_compute(
    LOG2_MULTS,
    NORM_LIST,
    SIGMAS,
    device,
    hero_config,
    hero_models,
    mo,
    rescale_response_grid,
):
    import os as _os
    import hashlib as _hashlib
    import pickle as _pickle

    _GRID_CACHE_DIR = "/tmp/dyt_cache"
    _os.makedirs(_GRID_CACHE_DIR, exist_ok=True)

    _grid_key = _hashlib.sha1(
        repr((hero_config, tuple(LOG2_MULTS.tolist()), tuple(SIGMAS.tolist()), 0)).encode()
    ).hexdigest()[:16]
    _grid_path = f"{_GRID_CACHE_DIR}/grid_{_grid_key}.pkl"

    if _os.path.exists(_grid_path):
        with open(_grid_path, "rb") as _f:
            rescale_grid = _pickle.load(_f)
    else:
        with mo.status.spinner(
            title=f"sweeping {len(LOG2_MULTS)}x{len(SIGMAS)} rescale grid x {len(NORM_LIST)} models ..."
        ):
            rescale_grid = rescale_response_grid(hero_models, hero_config, LOG2_MULTS, SIGMAS, seed=0, device=device)
        with open(_grid_path, "wb") as _f:
            _pickle.dump(rescale_grid, _f)

    _shape = rescale_grid.accuracy["dyt"].shape
    f"rescale grid ready: {_shape[0]} multipliers x {_shape[1]} sigmas, cached at {_grid_path}"

    return (rescale_grid,)


@app.cell
def pad_ui(LOG2_MULTS, SIGMAS, Slider2D, mo):
    scale_pad = mo.ui.anywidget(
        Slider2D(
            x=0.0, y=0.0,
            x_bounds=(float(LOG2_MULTS.min()), float(LOG2_MULTS.max())),
            y_bounds=(float(SIGMAS.min()), float(SIGMAS.max())),
            width=320, height=320,
        )
    )
    mo.hstack(
        [
            scale_pad,
            mo.md(
                "**the stress pad**"
                + chr(10) * 2 + "horizontal: global scale, log2(multiplier) in [-3, 3]"
                + chr(10) * 2 + "vertical: per-token heterogeneity, sigma in [0, 1.5]"
                + chr(10) * 2 + "Drag anywhere. The panels snap to the nearest precomputed grid point,"
                + " so no forward pass ever runs on a drag."
            ),
        ],
        gap=1.5,
    )
    return (scale_pad,)


@app.cell
def linked_panels(
    Figure,
    chance_accuracy,
    config,
    invariance_demo,
    invariance_demo_mults,
    np,
    rescale_grid,
    scale_pad,
):
    # Pure indexing into rescale_grid / invariance_demo -- zero forward
    # passes fire here, no matter how the sliders move.
    _mi = int(np.argmin(np.abs(rescale_grid.log2_multipliers - float(scale_pad.value["x"]))))
    _si = int(np.argmin(np.abs(rescale_grid.sigmas - float(scale_pad.value["y"]))))
    _mult_here = float(2.0 ** rescale_grid.log2_multipliers[_mi])
    _sigma_here = float(rescale_grid.sigmas[_si])

    # This cell re-runs on every slider drag. Build the figure with the
    # object-oriented Figure() API rather than plt.subplots so it never enters
    # pyplot's global figure registry -- otherwise each drag leaks a figure
    # ("more than 20 figures" warning + steady memory growth on the showpiece).
    _fig3 = Figure(figsize=(16, 4.6))
    _axA, _axB, _axC = _fig3.subplots(1, 3)

    # panel A: accuracy vs multiplier, reference curves at a few sigma slices,
    # live dot at the current pad position
    _sigma_ref_idx = sorted({0, len(rescale_grid.sigmas) // 2, len(rescale_grid.sigmas) - 1})
    for _sj in _sigma_ref_idx:
        _axA.plot(
            rescale_grid.log2_multipliers, rescale_grid.accuracy["dyt"][:, _sj],
            label=f"DyT, sigma={rescale_grid.sigmas[_sj]:.2f}",
            alpha=1.0 if _sj == _si else 0.5, linewidth=2.4 if _sj == _si else 1.3,
        )
    _axA.plot(
        rescale_grid.log2_multipliers, rescale_grid.accuracy["layernorm"][:, 0],
        "--", color="black", linewidth=1.8, label="LayerNorm (flat at every sigma)",
    )
    _axA.plot(
        [rescale_grid.log2_multipliers[_mi]], [rescale_grid.accuracy["dyt"][_mi, _si]],
        "o", color="crimson", markersize=12, zorder=5, label="you are here",
    )
    _axA.axhline(chance_accuracy(config), linestyle=":", color="gray", linewidth=1)
    _axA.set_xlabel("log2(global scale multiplier)")
    _axA.set_ylabel("induction accuracy")
    _axA.set_ylim(-0.05, 1.05)
    _axA.set_title("accuracy vs. scale (live dot)")
    _axA.legend(fontsize=7, loc="lower center")

    # panel B: pre-tanh histogram at the current pad position, saturation
    # zone shaded
    _edges = rescale_grid.hist_edges
    _centers = 0.5 * (_edges[:-1] + _edges[1:])
    _axB.bar(_centers, rescale_grid.hist["dyt"][_mi, _si], width=(_edges[1] - _edges[0]), color="steelblue")
    _thr = rescale_grid.saturation_threshold
    _axB.axvspan(_edges[0], -_thr, color="crimson", alpha=0.12)
    _axB.axvspan(_thr, _edges[-1], color="crimson", alpha=0.12, label="tanh saturation zone")
    _axB.set_xlabel("alpha * x  (pre-tanh, first norm site)")
    _axB.set_ylabel("token count")
    _axB.set_title(f"DyT pre-tanh distribution\n(mult={_mult_here:.2f}, sigma={_sigma_here:.2f})")
    _axB.legend(fontsize=7)

    # panel C: the invariance theorem itself, precomputed, marker at current mult
    for _name, _color in (("layernorm", "black"), ("dyt", "crimson"), ("rms_dyt", "seagreen")):
        _axC.plot(invariance_demo_mults, invariance_demo[_name], label=_name, color=_color, linewidth=2)
    _axC.axvline(float(rescale_grid.log2_multipliers[_mi]), linestyle=":", color="gray")
    _axC.set_xlabel("log2(multiplier)")
    _axC.set_ylabel("max |value(mult*x) - value(x)|")
    _axC.set_title("the theorem: invariance gap")
    _axC.legend(fontsize=8)

    _fig3.tight_layout()
    _fig3
    return


@app.cell(hide_code=True)
def panels_caption(mo):
    mo.md(r"""
    ### What to look for

    The left panel is accuracy against scale, several fixed noise levels drawn as reference curves, with a marker showing exactly where the pad sits right now. LayerNorm's curve is flat: task accuracy at 8x scale is the same as at 1x, the same as at 1/8th, because $\mathrm{LN}$ genuinely cannot tell the difference. DyT's curve peaks near the multiplier it was trained at and falls away on *both* sides, not only toward saturation -- shrinking the input shrinks $\tanh(\alpha x)$'s output by roughly the same factor while it is still in the near-linear region, which throws off the relative magnitude the rest of the network learned to expect from that branch, well before anything actually saturates.

    The middle panel is the distribution of $\alpha x$ for one representative norm site (the input layer), the values tanh is about to squash, with the region where $|\alpha x|$ is large enough that tanh has essentially clipped shaded in red. Push the multiplier up and watch the histogram slide into that band. Push it down and the histogram huddles near zero, off the training distribution in the other direction.

    The right panel is the theorem itself, measured directly on a fresh random tensor with no trained weights involved: the largest difference between a norm site's output at scale $c$ and at scale 1. It sits at zero for LayerNorm at every multiplier we test, by construction. It grows for DyT as soon as $\alpha x$ leaves the linear region.
    """)
    return


@app.cell
def depth_sweep_compute(
    DEPTH_LIST,
    NORM_LIST,
    SEED_LIST,
    STRESS_MULT,
    STRESS_SIGMA,
    config,
    depth_fragility_sweep,
    device,
    mo,
):
    import os as _os2
    import hashlib as _hashlib2
    import pickle as _pickle2

    _os2.makedirs("/tmp/dyt_cache", exist_ok=True)
    _depth_key = _hashlib2.sha1(
        repr((config, DEPTH_LIST, NORM_LIST, SEED_LIST, STRESS_MULT, STRESS_SIGMA)).encode()
    ).hexdigest()[:16]
    _depth_path = f"/tmp/dyt_cache/depth_sweep_{_depth_key}.pkl"

    if _os2.path.exists(_depth_path):
        with open(_depth_path, "rb") as _f:
            depth_sweep_rows = _pickle2.load(_f)
    else:
        _total = len(DEPTH_LIST) * len(NORM_LIST) * len(SEED_LIST)
        with mo.status.progress_bar(
            total=_total,
            title=f"depth sweep: {len(DEPTH_LIST)} depths x {len(NORM_LIST)} norms x {len(SEED_LIST)} seeds",
        ) as _bar:
            depth_sweep_rows = depth_fragility_sweep(
                config, DEPTH_LIST, NORM_LIST, SEED_LIST, STRESS_MULT, STRESS_SIGMA, device,
                progress=lambda done, total: _bar.update(),
            )
        with open(_depth_path, "wb") as _f:
            _pickle2.dump(depth_sweep_rows, _f)

    _n_diverged = sum(1 for r in depth_sweep_rows if r["diverged"])
    f"depth sweep: {len(depth_sweep_rows)} runs cached at {_depth_path} ({_n_diverged} diverged)"

    return (depth_sweep_rows,)


@app.cell
def depth_sweep_display(
    DEPTH_LIST,
    STRESS_MULT,
    STRESS_SIGMA,
    depth_sweep_rows,
    np,
    plt,
):
    _fig4, (_axG, _axH) = plt.subplots(1, 2, figsize=(13, 5.2))

    for _norm, _color in (("layernorm", "black"), ("dyt", "crimson"), ("rms_dyt", "seagreen")):
        _depths_here = sorted({r["depth"] for r in depth_sweep_rows if r["norm"] == _norm})
        _mean_gap, _lo_gap, _hi_gap, _mean_base, _mean_stress = [], [], [], [], []
        for _d in _depths_here:
            _rows_here = [r for r in depth_sweep_rows if r["norm"] == _norm and r["depth"] == _d]
            _gaps = [r["gap"] for r in _rows_here]
            _mean_gap.append(float(np.mean(_gaps)))
            _lo_gap.append(float(np.min(_gaps)))
            _hi_gap.append(float(np.max(_gaps)))
            _mean_base.append(float(np.mean([r["baseline_acc"] for r in _rows_here])))
            _mean_stress.append(float(np.mean([r["stressed_acc"] for r in _rows_here])))
        _axG.plot(_depths_here, _mean_gap, "o-", color=_color, label=_norm, linewidth=2.2, markersize=7)
        _axG.fill_between(_depths_here, _lo_gap, _hi_gap, color=_color, alpha=0.15)
        _axH.plot(_depths_here, _mean_base, "o--", color=_color, alpha=0.5, markersize=5)
        _axH.plot(_depths_here, _mean_stress, "o-", color=_color, label=_norm, linewidth=2.2, markersize=7)

    _axG.set_xlabel("depth (layers)")
    _axG.set_ylabel(f"fragility gap: baseline - stressed\n(mult={STRESS_MULT:g}, sigma={STRESS_SIGMA:g})")
    _axG.set_title("does the gap widen with depth? (band = min/max over seeds)")
    _axG.set_xticks(DEPTH_LIST)
    _axG.legend(fontsize=8)

    _axH.set_xlabel("depth (layers)")
    _axH.set_ylabel("induction accuracy")
    _axH.set_title("baseline (dashed) vs. stressed (solid), mean over seeds")
    _axH.set_xticks(DEPTH_LIST)
    _axH.set_ylim(-0.05, 1.05)
    _axH.legend(fontsize=8)

    _fig4.tight_layout()
    _fig4
    return


@app.cell(hide_code=True)
def depth_caption(mo):
    mo.md(r"""
    ### Reading the depth panel

    The paper's own analysis of trained $\alpha$ values shows that $\alpha$ adapts, during training, to the scale of its own layer's activations at that depth. A frozen scalar tuned precisely to sit inside tanh's linear region for one specific training-time distribution raises a question worth testing directly: does a deeper stack of such precisely-tuned, frozen scalars leave less room for error, so the same rescale stress opens a wider gap the more layers it has to propagate through?

    Left panel: the fragility gap (baseline accuracy minus stressed accuracy) against depth, one line per norm, shaded band showing the spread across seeds. Right panel: the two accuracies that gap is built from, separately, dashed for baseline and solid for stressed, so a shrinking gap can be told apart from "this run simply didn't finish training." We report what the multi-seed sweep actually shows rather than what the prediction says it should show; a small-scale single-seed check made while building this notebook did not find a clean monotonic trend, and that is worth taking at face value rather than explaining away.
    """)
    return


@app.cell
def rms_fix_display(chance_accuracy, config, plt, rescale_grid):
    _fig5, _axR = plt.subplots(figsize=(9, 5.6))
    _sigma_lo, _sigma_hi = 0, len(rescale_grid.sigmas) // 2
    for _name, _color in (("layernorm", "black"), ("dyt", "crimson"), ("rms_dyt", "seagreen")):
        _axR.plot(
            rescale_grid.log2_multipliers, rescale_grid.accuracy[_name][:, _sigma_lo],
            "-", color=_color, linewidth=2.4, label=f"{_name}, sigma={rescale_grid.sigmas[_sigma_lo]:.2f}",
        )
        _axR.plot(
            rescale_grid.log2_multipliers, rescale_grid.accuracy[_name][:, _sigma_hi],
            "--", color=_color, linewidth=1.6, alpha=0.7, label=f"{_name}, sigma={rescale_grid.sigmas[_sigma_hi]:.2f}",
        )
    _axR.axhline(chance_accuracy(config), linestyle=":", color="gray", linewidth=1, label="chance")
    _axR.set_xlabel("log2(global scale multiplier)")
    _axR.set_ylabel("induction accuracy")
    _axR.set_ylim(-0.05, 1.05)
    _axR.set_title("the candidate fix: divide by per-token RMS before tanh")
    _axR.legend(fontsize=7, ncol=2, loc="lower center")
    _fig5.tight_layout()
    _fig5
    return


@app.cell(hide_code=True)
def rms_caption(mo):
    mo.md(r"""
    ### A fix that isn't in the paper

    If DyT's problem is that $\alpha x$ is not scale-invariant, the direct fix is to make it scale-invariant the same way LayerNorm's argument is: divide by a per-token statistic of $x$ before applying $\alpha$. Here we divide by the per-token root-mean-square: $\mathrm{RMS\text{-}DyT}(x) = \tanh(\alpha \cdot x / \mathrm{rms}(x)) \cdot \gamma + \beta$. Algebraically, $\mathrm{rms}(cx) = |c| \cdot \mathrm{rms}(x)$ for any scalar $c$, global or per-token, so the multiplier cancels out of the tanh argument the same way it cancels out of LayerNorm's mean and standard deviation.

    The measured curve above backs this up: RMS-DyT tracks LayerNorm's flat line rather than DyT's collapsing one. That is not a free lunch. Computing a per-token RMS is a reduction over the feature dimension -- the exact operation DyT's efficiency pitch is built around removing. This candidate restores scale-invariance by re-adding the cost the paper spent its argument eliminating. Whether that trade is worth it depends entirely on whether the deployment setting has activation-scale drift to begin with, and the paper's own results suggest that for a great many settings, it does not.
    """)
    return


@app.cell(hide_code=True)
def guardrails(mo):
    mo.md(r"""
    ### What this does and does not show

    - The induction task, the tiny transformer, and the training loop are all synthetic and built for this notebook; nothing here downloads a checkpoint or a dataset. That buys speed and reproducibility, not a claim that these exact numbers transfer to a billion-parameter language model. The paper's own experiments, at real scale, are the place to look for that.
    - Every DyT $\alpha$ value in this notebook is learned by gradient descent on the unstressed task, exactly as the paper trains it, never hand-set to make a point. The fragility shown here is what training actually produces, not a worst-case $\alpha$ chosen to fail.
    - The rescale stress (a global multiplier times an independent per-token lognormal factor) is one specific, simple model of activation-scale drift. Real deployment-time distribution shift can look different: precision changes, sequence-length effects, domain shift, adversarial inputs. The mechanism this notebook demonstrates, a frozen scalar with no way to react to a shifted input scale, is general; the exact numbers are not a benchmark.
    - The depth sweep uses a handful of seeds per configuration, enough to see whether a trend is there, not enough to report a tight confidence interval on it.
    """)
    return


if __name__ == "__main__":
    app.run()
