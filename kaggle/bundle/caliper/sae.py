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


def draw_latents(sae, n, min_log_density=-3.0, n_bins=4, seed=0):
    """n latents drawn evenly across density quantiles of the eligible latents.

    A latent firing on a fraction 10^x of tokens fires about 8000 * 10^x times in an
    8,000-token stimulus, so latents below ``min_log_density`` are excluded as unfittable.
    The eligible set is split into ``n_bins`` equal-count density bins, the difficulty dial:
    sparse latents are the analogue of the sparse, heavy-tailed neurons that fail most.
    """
    rng = np.random.default_rng(seed)
    dens = sae["log_density"]
    eligible = np.flatnonzero(dens >= min_log_density)
    ordered = eligible[np.argsort(dens[eligible])]
    groups = np.array_split(ordered, n_bins)
    per = [n // n_bins + (1 if k < n % n_bins else 0) for k in range(n_bins)]
    chosen = np.concatenate([rng.choice(g, size=c, replace=False) for g, c in zip(groups, per)])
    return np.sort(chosen)
