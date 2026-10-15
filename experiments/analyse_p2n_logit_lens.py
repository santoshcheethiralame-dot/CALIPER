"""P2-N secondary: logit-lens accessibility (P2-L's check) on Qwen3.5, from the saved vectors.

    python experiments/analyse_p2n_logit_lens.py      # writes results/p2n_logit_lens.json

docs/preregistration-p2n-qwen35-current.md, Secondary: "Logit-lens accessibility (P2-L's check) computed
offline from the saved vectors, if the unembedding can be read." P2-L's statistic, unchanged
(analyse_p2l.score): z of the concept's first token in W_U (gamma * v / |v|). Qwen3.5 ties its unembedding
to the input embedding and its final RMSNorm scales by (1 + weight); both tensors are read by exact name
from the cached checkpoint (P2-L's generic picker would also match the multi-token-prediction head's
norm). Live labels: the filed substring label and the strict gate, as in P2-N's scoring.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import analyse_p2l as p2l  # noqa: E402

MODELS = {"qwen35_2b": "Qwen/Qwen3.5-2B", "qwen35_08b": "Qwen/Qwen3.5-0.8B"}
EMB, NORM = "model.language_model.embed_tokens.weight", "model.language_model.norm.weight"


def weights(repo):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoTokenizer
    path = Path(snapshot_download(repo, local_files_only=True))
    U = g = None
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, "pt") as s:
            keys = set(s.keys())
            if EMB in keys:
                U = s.get_tensor(EMB).float().numpy().astype(np.float16)
            if NORM in keys:
                g = 1.0 + s.get_tensor(NORM).float().numpy()
    assert U is not None and g is not None, "unembedding or final norm not found"
    return U, g.astype(np.float32), AutoTokenizer.from_pretrained(str(path))


def rows(model, label_dir):
    per = {(r["arm"], r["concept"]): r for r in json.load(open(ROOT / label_dir / f"{model}_s2.json"))["per_vector"]}
    out = []
    for arm, tag in p2l.S2_ARMS.items():
        z = np.load(ROOT / f"results/p2n/s2_{model}_{tag}_none.vectors.npz", allow_pickle=True)
        for c, v in zip(z["names"], z["vectors"]):
            r = per[(arm, str(c))]
            out.append((arm, str(c), v, r["live"], r["logit steering"]))
    return out


class ChunkedU:
    """U @ x in float32 over row chunks, so the float16 matrix is never expanded whole."""
    def __init__(self, U):
        self.U = U

    def __matmul__(self, x):
        return np.concatenate([self.U[i:i + 32768].astype(np.float32) @ x for i in range(0, len(self.U), 32768)])


def main():
    out = {}
    for model, repo in MODELS.items():
        U, g, tok = weights(repo)
        for label, d in (("filed substring label", "results/p2n"), ("strict gate", f"results/gate_strict/p2n_{model}")):
            rep = p2l.score(rows(model, d), ChunkedU(U), g, tok)
            out[f"{model}, {label}"] = rep
            print(model, label, {k: v for k, v in rep.items() if k in ("vectors", "live", "auc", "ci95")}, flush=True)
        del U
    json.dump(out, open(ROOT / "results/p2n_logit_lens.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
