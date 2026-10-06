"""Sparse-autoencoder latents as a second reference standard.

A ReLU SAE latent's pre-activation is linear in the residual stream, so its encoder
column is the exact direction the latent reads, the same kind of object as an MLP
neuron's input weight column. Loads the public jbloom GPT-2 small residual SAEs from the
Hugging Face cache (or the Hub).
"""

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

REPO = "jbloom/GPT2-Small-SAEs-Reformatted"


def load_sae(layer):
    """W_enc (d, n), b_enc (n,), b_dec (d,) and log10 feature density (n,) for one layer."""
    folder = f"blocks.{layer}.hook_resid_pre"
    weights = load_file(hf_hub_download(REPO, f"{folder}/sae_weights.safetensors"))
    sparsity = load_file(hf_hub_download(REPO, f"{folder}/sparsity.safetensors"))
    out = {k: weights[k].astype(np.float32) for k in ("W_enc", "b_enc", "b_dec")}
    out["log_density"] = next(iter(sparsity.values())).astype(np.float32)
    return out


def firing_counts(sae, stimulus, chunk=4096):
    """How many tokens of ``stimulus`` each latent fires on (pre-activation > 0)."""
    counts = np.zeros(sae["W_enc"].shape[1], dtype=np.int64)
    centred = stimulus - sae["b_dec"]
    for start in range(0, sae["W_enc"].shape[1], chunk):
        z = centred @ sae["W_enc"][:, start:start + chunk] + sae["b_enc"][start:start + chunk]
        counts[start:start + chunk] = (z > 0).sum(0)
    return counts


def draw_by_firing(counts, n, min_events=100, n_bins=4, seed=0):
    """n latents drawn evenly across quartiles of their observed firing count.

    The SAE's own density statistics come from OpenWebText and transfer poorly to our
    stimulus (correlation 0.5 in log space; latents rated 10^-2.8 fired 7 and 17 times in
    8,000 tokens). Eligibility is therefore set on the stimulus actually fitted: at least
    ``min_events`` firings, since E0.3 found one-direction recovery needs about 200.
    """
    rng = np.random.default_rng(seed)
    eligible = np.flatnonzero(counts >= min_events)
    ordered = eligible[np.argsort(counts[eligible], kind="stable")]
    groups = np.array_split(ordered, n_bins)
    per = [n // n_bins + (1 if k < n % n_bins else 0) for k in range(n_bins)]
    return np.concatenate([rng.choice(g, size=c, replace=False) for g, c in zip(groups, per)])
