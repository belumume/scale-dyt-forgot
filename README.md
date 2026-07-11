# The Scale DyT Forgot

A marimo notebook built on "Transformers without Normalization" (Zhu, Chen,
He, Yu, Yuille, LeCun, 2025, arXiv 2503.10622), for the alphaXiv x marimo molab Notebook
Competition #2.

LayerNorm has a theorem: LN(cx) = LN(x), exactly, by algebra. DyT has a
fitted scalar frozen at training time. This notebook trains tiny transformers
with both (plus a candidate fix), then drags their input off the scale they
trained on with a 2D stress pad and watches which ones survive. A multi-seed
depth sweep and an RMS-rescale fix round out the story.

Live notebook: https://molab.marimo.io/notebooks/nb_jNm948dpXRCvrK9demdeaf

Run locally:

    uvx marimo edit --sandbox notebook.py

A GPU makes the 27-run depth sweep fast; CPU runs the same code path.
`engine.py` is the standalone engine. `__marimo__/` carries rendered outputs
(added after the final full run).

## What this adds to the paper

The paper shows DyT is a drop-in replacement for LayerNorm, and it is. What it
does not price is the property LayerNorm carries for free: LN(cx) = LN(x) for any
positive c, exactly. tanh(alpha x) has no such cancellation, and alpha freezes the
moment training ends, so DyT only *matches* LayerNorm at the scale it trained at.

Measured here: two small transformers, one per norm, alpha learned the paper's own
way. At training scale DyT ties LayerNorm at 0.9996 accuracy. Push the input scale
8x and DyT slides to about 0.58 while LayerNorm does not move. A draggable 25x13
heatmap (input scale by per-token spread) shows accuracy, the pre-tanh values
sliding into tanh's flat region, and the invariance gap opening together.

The honest scope: this is out-of-distribution by construction, and the paper is
right that the distribution rarely moves in practice. The cost surfaces where scale
drifts for other reasons: quantized inference, precision drift, deployment shift.
A candidate fix (per-token RMS before the tanh) restores the flat curve and is
shipped in the notebook, along with the trade it re-introduces. It re-adds the
reduction DyT's efficiency argument existed to remove.

## The trilogy

Three notebooks, one per paper, three views of the same edge:

- **The Scale DyT Forgot** (this one): a frozen scalar pins where the stability boundary sits relative to your input scale.
- **The Coastline That Never Smooths**: that boundary is fractal. https://molab.marimo.io/notebooks/nb_SZ6ifRSGVy1kQY8b5yXrax
- **On the Wire**: training rides it. https://molab.marimo.io/notebooks/nb_QtxPsWxaZSpnRU2J8qWZWm
