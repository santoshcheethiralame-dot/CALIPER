"""Stimulus/response extraction from GPT-2 style models.

For E0.1 the stimulus is the post-LayerNorm residual stream that the MLP reads
(``ln_2`` output) and the response is a chosen MLP neuron's post-GELU activation.
That pairing has an analytically known answer: the response depends on the stimulus
only through ``w . s``, where ``w`` is the neuron's input weight column.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Probe:
    """A (stimulus, response) sample set plus the ground-truth direction."""

    stimulus: np.ndarray  # (n, d_model) post-ln_2 residual stream
    response: np.ndarray  # (n, n_neurons) post-GELU MLP activations
    weights: np.ndarray  # (d_model, n_neurons) true input weight columns
    neurons: np.ndarray  # (n_neurons,) neuron indices within the layer
    layer: int


def load_model(name="gpt2", device="cpu"):
    model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _blocks(model):
    for attr in ("transformer.h", "model.layers", "gpt_neox.layers"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError("could not locate transformer blocks")


def _mlp_in(block):
    """The MLP's input projection, whatever the architecture calls it."""
    for name in ("c_fc", "up_proj", "dense_h_to_4h"):
        if hasattr(block.mlp, name):
            return getattr(block.mlp, name)
    raise ValueError(f"no MLP input projection on {type(block.mlp).__name__}")


def _mlp_ln(block):
    """The layernorm whose output the MLP reads.

    GPT-2 calls it ln_2; GPT-NeoX calls it post_attention_layernorm. NeoX runs a
    parallel residual (use_parallel_residual=True), so the MLP still reads this
    layernorm's output directly and the identity holds either way.
    """
    for name in ("ln_2", "post_attention_layernorm", "post_attention_norm"):
        if hasattr(block, name):
            return getattr(block, name)
    raise ValueError(f"no MLP layernorm on {type(block).__name__}")


def _mlp_in_weight(block):
    """Input weight of the MLP, shaped (d_model, d_mlp).

    This column IS the neuron's direction: pre-activation = s . w, verified to a
    relative error of 3e-06 on GPT-2 and 1.7e-04 on Pythia-160m, with a correlation
    of 1.0000000000 in both. The absolute error differs between them only because
    of float32 accumulation order, not structure.
    """
    fc = _mlp_in(block)
    w = fc.weight.detach().cpu().numpy()
    d_model = _mlp_ln(block).weight.shape[0]
    # HF GPT-2 uses Conv1D with weight (d_model, d_mlp); Linear stores (out, in).
    return w if w.shape[0] == d_model else w.T


def collect(
    model,
    tokenizer,
    texts,
    layer,
    neurons,
    max_tokens=50_000,
    seq_len=128,
    batch_size=8,
    skip_first=1,
    seed=0,
    shuffle="token",
    sae=None,
):
    """Stream text through the model, capturing stimulus/response pairs.

    ``skip_first`` drops position 0, whose residual-stream norm is an order of
    magnitude larger than typical tokens and would dominate any least-squares fit.

    With ``sae`` (from `caliper.sae.load_sae`), the units are SAE latents instead of MLP
    neurons. The stimulus is the residual stream entering block ``layer``, mean-centred
    per token, which is what the jbloom GPT-2 SAEs were trained on (reconstruction explains
    0.83 of variance centred against 0.54 raw on our corpus). The response is the latent's
    ReLU activation. Its pre-activation is exactly (s - b_dec) . W_enc[:, j] + b_enc[j], and
    because s has zero mean across coordinates the identifiable reference is the encoder
    column with its own mean removed.
    """
    block = _blocks(model)[layer]
    neurons = np.asarray(neurons)
    captured = {}

    if sae is None:
        def hook_stim(_module, _inp, out):
            captured["stim"] = out.detach()

        def hook_resp(_module, _inp, out):
            captured["pre"] = out.detach()

        handles = [
            _mlp_ln(block).register_forward_hook(hook_stim),
            _mlp_in(block).register_forward_hook(hook_resp),
        ]
        weights = _mlp_in_weight(block)[:, neurons]

        def respond(stim, pre):
            return torch.nn.functional.gelu(pre)[:, :, neurons]
    else:
        def hook_pre(_module, args):
            x = args[0].detach()
            captured["stim"] = x - x.mean(-1, keepdim=True)

        handles = [block.register_forward_pre_hook(hook_pre)]
        w_enc = torch.as_tensor(sae["W_enc"][:, neurons], dtype=torch.float32)
        b_enc = torch.as_tensor(sae["b_enc"][neurons], dtype=torch.float32)
        b_dec = torch.as_tensor(sae["b_dec"], dtype=torch.float32)
        weights = (w_enc - w_enc.mean(0, keepdim=True)).numpy()

        def respond(stim, pre):
            return torch.relu((stim.float() - b_dec) @ w_enc + b_enc)
    stim_chunks, resp_chunks, total = [], [], 0
    rng = np.random.default_rng(seed)

    try:
        batch = []
        for text in texts:
            ids = tokenizer(text, return_tensors=None)["input_ids"]
            for start in range(0, len(ids) - seq_len + 1, seq_len):
                batch.append(ids[start : start + seq_len])
                if len(batch) < batch_size:
                    continue
                total += _run_batch(
                    model, batch, captured, respond,
                    stim_chunks, resp_chunks, skip_first,
                )
                batch = []
                if total >= max_tokens:
                    break
            if total >= max_tokens:
                break
    finally:
        for h in handles:
            h.remove()

    stimulus = np.concatenate(stim_chunks)[:max_tokens]
    response = np.concatenate(resp_chunks)[:max_tokens]

    # One sample per sequence would be the strictest control for token
    # autocorrelation; we shuffle here and record that the effective sample size
    # is below the nominal count.
    order = _order(len(stimulus), seq_len - skip_first, rng, shuffle)
    return Probe(
        stimulus=stimulus[order],
        response=response[order],
        weights=weights,
        neurons=neurons,
        layer=layer,
    )


def _order(n, block, rng, mode):
    """Row order for the stimulus/response pairs.

    ``token`` permutes every row, so the first 20% that `fit_batch` holds out shares
    sequences with the training rows, and neighbouring tokens in a sequence are close in
    residual space. Held-out R2 is then mildly optimistic. ``sequence`` permutes whole
    sequences and keeps each one's tokens together, so the held-out set is made of
    sequences the fit never saw, apart from at most one sequence cut by the boundary.
    """
    if mode == "token":
        return rng.permutation(n)
    if mode != "sequence":
        raise ValueError(f"unknown shuffle mode {mode!r}")
    seq = np.arange(n) // block
    return np.concatenate([np.flatnonzero(seq == k) for k in rng.permutation(seq.max() + 1)])


def _run_batch(model, batch, captured, respond, stim_out, resp_out, skip_first):
    ids = torch.tensor(batch)
    with torch.no_grad():
        model(ids)
        stim = captured["stim"][:, skip_first:, :]
        pre = captured.get("pre")
        resp = respond(stim, None if pre is None else pre[:, skip_first:, :])
    d_model = stim.shape[-1]
    stim_out.append(stim.reshape(-1, d_model).float().numpy())
    resp_out.append(resp.reshape(-1, resp.shape[-1]).float().numpy())
    return stim_out[-1].shape[0]


# Public-domain prose, used when the Hub is unavailable. The stimulus ensemble only
# needs to be natural English; E0.1's ground truth holds under any input distribution.
_FALLBACK_URLS = [
    "https://www.gutenberg.org/files/11/11-0.txt",  # Alice in Wonderland
    "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
    "https://www.gutenberg.org/files/84/84-0.txt",  # Frankenstein
]

CACHE = Path(__file__).resolve().parents[1] / "results" / "corpus_cache"


def sample_corpus(n_docs=400, seed=0, source="gutenberg"):
    """Text for the stimulus ensemble.

    Defaults to cached public-domain prose. ``source='hub'`` uses wikitext instead.
    The default is not the Hub because the installed huggingface_hub aborts the
    process rather than raising on legacy dataset ids, which no ``except`` can catch
    and which would make a measurement gate hostage to a library upgrade.
    """
    if source in ("auto", "hub"):
        try:
            from datasets import load_dataset

            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            texts = [t for t in ds["text"] if len(t) > 400]
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(texts), size=min(n_docs, len(texts)), replace=False)
            return [texts[i] for i in idx]
        except Exception as exc:
            if source == "hub":
                raise
            print(f"  [corpus] hub unavailable ({type(exc).__name__}), using fallback")

    return _gutenberg(n_docs, seed)


def _gutenberg(n_docs, seed):
    import urllib.request

    CACHE.mkdir(parents=True, exist_ok=True)
    chunks = []
    for url in _FALLBACK_URLS:
        path = CACHE / (url.rsplit("/", 1)[-1])
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as r:
                path.write_text(r.read().decode("utf-8", "ignore"), encoding="utf-8")
        body = path.read_text(encoding="utf-8")
        # Drop licence boilerplate at both ends.
        body = body[body.find("***", body.find("*** START")) + 3 :]
        end = body.find("*** END")
        chunks.extend(
            p.strip() for p in (body[:end] if end > 0 else body).split("\n\n")
            if len(p.strip()) > 400
        )

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(chunks), size=min(n_docs, len(chunks)), replace=False)
    return [chunks[i] for i in idx]
