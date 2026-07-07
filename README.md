# The Scale DyT Forgot

A marimo notebook built on "Transformers without Normalization" (Zhu, Chen,
He, Yu, Yuille, LeCun, 2025, arXiv 2503.10622), for the marimo x alphaXiv
Notebook Competition #2.

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
