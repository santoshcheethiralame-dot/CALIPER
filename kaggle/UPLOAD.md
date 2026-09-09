# Uploading the bundle to Kaggle

The repo is private, so Kaggle cannot `pip install` it anonymously — a 404. The code
travels as a **Kaggle Dataset** instead.

**File to upload:** `kaggle/caliper-bundle.zip` (504 KB)
**Rebuild it first if any source changed:** `python kaggle/build_bundle.py`, then re-zip.

---

## Datasets that already exist — do not overwrite these

| dataset | what it holds | used by |
|---|---|---|
| `caliper-s3` | one file, `kaggle_s3_positive_control.py`, versioned by date stamp | every Study 3 / Gemma / Qwen run sheet |
| `caliper-s1` | the older bundle zip built for S1 multiseed | `NEXT_SESSION_S1_MULTISEED.md` |

`caliper-bundle` is **new**. It is not a version of either of those — it carries the
`caliper` package plus the B-series scripts plus the cached corpus, which is a different
payload from both.

---

## First upload — creating the dataset

1. **kaggle.com** → **Datasets** in the left nav → **New Dataset** (top right).
2. Drag in `kaggle/caliper-bundle.zip`. **Kaggle unzips it automatically**, so the dataset
   ends up with `caliper/`, `experiments/` and `results/` at its top level — which is what
   the run sheets' Cell 1 expects. Do not unzip it yourself first.
3. Title: **`caliper-bundle`** exactly. The slug becomes
   `<your-username>/caliper-bundle`.
4. Leave it **Private**. Kaggle notebooks can read your own private datasets; nothing here
   needs to be public.
5. **Create**. Wait for it to finish processing — a notebook attached too early sees an
   empty directory.

## Attaching it to a notebook

1. New Notebook → right-hand panel → **Add Input**.
2. **Datasets** tab → **Your Datasets** → `caliper-bundle` → **Add**.
3. It mounts read-only at `/kaggle/input/caliper-bundle/`.

Then set the accelerator and internet per the run sheet:

| run sheet | accelerator | internet |
|---|---|---|
| `NEXT_SESSION_B0_DEVICE.md` | GPU T4 x2 | **off** — GPT-2 only, corpus is cached |
| `NEXT_SESSION_LADDER.md` | GPU T4 x2 | **on** — Pythia weights come from the Hub |

## Later updates

Dataset page → **New Version** → drag the rebuilt zip → give the version a note saying
what changed. Existing notebooks keep their pinned version until you refresh the input,
which is deliberate: a run sheet that names a version stays reproducible.

---

## Two things that have bitten this project

**The bundle going stale silently.** It was once two files behind *and missing
`batched.py` entirely*, which the gate run cannot start without — and nothing said so until
the run failed. `python kaggle/build_bundle.py` now prints exactly what it added, updated
and removed, and says whether an upload is needed. Run it before every upload.

**Outputs dying with the session.** Kaggle's `/kaggle/working` is ephemeral. This project
has already permanently lost the raw data for a set of GPU runs that way. Every run sheet
ends with a zip-and-download cell — do it *before* the session expires, not after the
numbers look good.

---

## Why internet can stay off for the corpus

`sample_corpus` defaults to cached public-domain prose, and the three Gutenberg texts ship
inside the bundle at `results/corpus_cache/`. `_gutenberg` calls
`CACHE.mkdir(parents=True, exist_ok=True)` and then checks `path.exists()` before
downloading — both are safe on a read-only mount, since `exist_ok=True` swallows the
`FileExistsError` and the files are only read. So the corpus never needs a network call.

Model weights are the only thing that does, which is why the ladder needs internet on and
B-0 does not.
