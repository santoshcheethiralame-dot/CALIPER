# Pre-registration: P2-H-1B, P2-H's protocol on Llama-3.2-1B-Instruct, run locally on CPU

**Filed 14 October 2026, before any P2-H-1B data.** Paper 2. A companion to P2-H
(`docs/preregistration-p2h-hahami-llama.md`, filed 13 Oct), which runs the same protocol on
Llama-3.1-8B-Instruct and is waiting on a Kaggle session (the weekly GPU quota is spent). Script:
`experiments/kaggle_p2h_hahami.py` v2026-10-13a, unchanged, run with the flags below. Scorer:
`experiments/analyse_p2h.py`, unchanged.

## Why

The second internal review of Paper 2 (ruling 9) found that the audit never observes the phenomenon
it audits, and made a positive control required for ICML. P2-H provides it on the model Hahami et
al. used. This run asks the same question on a smaller model of the same family that fits on the
laptop's CPU, so that Paper 2 has a second measurement of the protocol, of its manipulation checks
and of its released quirks, whatever the 8B session later shows. It is not a substitute for P2-H:
Hahami et al. report detection on Llama-3.1-8B, and detection may not exist at 1B.

## What changes from P2-H, and nothing else

| | P2-H | P2-H-1B |
|---|---|---|
| model | Llama-3.1-8B-Instruct (32 layers, d = 4096) | Llama-3.2-1B-Instruct (16 layers, d = 2048), Hugging Face hub |
| hardware, precision | Kaggle T4, fp16 (4-bit fp32 fallback) | laptop CPU, unquantised **fp32** (`--quant none --compute-dtype fp32`) |
| layers | 0 and 19 (0.6 of depth) | **0 and 10** (0.6 of 16 = 9.6, rounded) |
| strengths | 1, 2, 3, 4, 5 (run order 5, 3, 1, 2, 4) | the same, then **8 and 12** appended as secondary |
| fidelity check | cosine with Hahami et al.'s released 8B vectors | **not run** (`--no-fidelity`): their vectors are 4096-dimensional |

Everything else is P2-H's filing verbatim: the prompt, sentences, factual framings, the "avg"
vector recipe on CALIPER's 30 concepts, the injection span and the read/inject layer offset, the
released readout with the duplicated last token and the single-pass readout beside it, the three
arms (real, norm-matched random, impact-matched random), the strict liveness gate, manipulation
checks M1-M4, primary endpoints P0-P3 at layer 0 and strength 5 on the released readout, the
bootstrap over concepts, the two sensitivity analyses, the secondaries, and the interpretation table.

**Strengths 8 and 12.** A unit-vector strength is not a transferable dose (Paper 2, C4), and the
residual stream of the 1B model differs from the 8B model's. Strengths 8 and 12 are added so that a
P0 failure at 5 can be read against a stronger push; they are secondary, scored with P2-H's
endpoint code at each strength (the scorer reports every strength present), and never replace the
primary cell. They run last in the same file (`--alphas 5 3 1 2 4 8 12`), after every strength of
P2-H's design.

## Readings, scoped

P2-H's interpretation table applies to this model only:

- **P0 passes**: Llama-3.2-1B gives a YES the audit can dissect, and P1-P3 say what it is made of.
  Paper 2 then has a positive-control regime even if the 8B session never runs.
- **P0 fails**: the released protocol does not produce detection at 1B on CPU in fp32. This says
  nothing about 8B. If P0 also fails at strengths 8 and 12, it is reported as "no detection at any
  tested strength"; if it passes there, it is reported as a dose shift, with the strength stated.

## Commands

```
python experiments/kaggle_p2h_hahami.py --model-path <local snapshot of meta-llama/Llama-3.2-1B-Instruct> \n  --quant none --compute-dtype fp32 --layers 0 10 --alphas 5 3 1 2 4 8 12 --no-fidelity \n  --out results/p2h1b/p2h1b_llama1b.jsonl
```

Scored once: `python experiments/analyse_p2h.py --file results/p2h1b/p2h1b_llama1b.jsonl --out
results/p2h1b_analysis.json`.

## Exclusions and stopping

All 30 concepts, every arm, both layers; nothing excluded after data. Rows resume by key.

## Cost

About 12,600 forward passes of about 150 tokens,
on 12 CPU threads; measured on the first rows and reported. Local queue 15.
