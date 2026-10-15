# Paper checklist (NeurIPS form), Paper 2

Filled in under the programme's rigour standard (`docs/flagship-plan.md` §4a); the venue is TACL,
which does not require it. Section references are to `paper2/main.tex`. Items marked
**(after P2-H)** must be re-checked once P2-H, P2-H-1B and P2-E are scored and folded in.

1. **Claims.** Yes. C1 to C8 are each scoped to where they were measured; the filed-predictions table
   lists every prediction and its outcome, including the failed ones (S-2's checks, S-1, P2-G, P2-N's
   P2). **(after P2-H)** The readout claims change if the positive control detects.
2. **Limitations.** Yes: "Limitations" and the threats-to-validity appendix (lexical liveness, thirty
   concepts, 4-bit precision, prompts, one readout model, post-hoc analyses, one layer).
3. **Theory, assumptions and proofs.** Not applicable: no theorem is claimed.
4. **Reproducibility.** Yes. Every arm has a pre-registration and a run sheet; the appendices give the
   concepts, associate terms, prompts verbatim, both gate rules, and per-model precision and layer.
5. **Open access to data and code.** Yes: scripts, run sheets, every filing and every row are released
   with the paper.
6. **Experimental settings and details.** Yes: §3 (protocol) and the appendices (prompts, gate,
   liveness at every dose, models, precision, compute).
7. **Statistical significance.** Yes. AUCs carry stratified bootstrap intervals; paired comparisons
   use Wilcoxon signed-rank tests; excesses use a concept-level bootstrap; heterogeneity uses
   Cochran's $Q$; per-session tests are combined with Fisher's method, sidedness stated.
8. **Compute resources.** Yes: "Models, precision and compute" (Kaggle 2x T4 sessions within 12 h;
   P2-N 9.8 and 4.9 CPU-hours on one laptop). **(after P2-H)** Add P2-H-1B's CPU-hours and the P2-H
   and P2-E sessions.
9. **Code of ethics.** Yes. No human subjects or personal data.
10. **Broader impacts.** Yes: "Broader impact statement" (after the conclusion).
11. **Safeguards.** Not applicable: no model released; the vectors are for common concepts.
12. **Licences for existing assets.** Yes, stated in the compute appendix: Qwen2.5-7B and Qwen3.5
    (Apache-2.0), Qwen2.5-3B (Qwen Research License), Gemma 3 (Gemma Terms of Use), Llama-3.1-8B and
    Llama-3.2-1B (Llama 3.1 and 3.2 Community Licenses). Checked on the model cards on 15 Oct 2026.
    Open: the licence of the released code of Macar et al. is not recorded; check before submission.
13. **New assets.** Yes: the audit protocol and reporting checklist, released with the paper.
14. **Crowdsourcing and human subjects.** Not applicable.
15. **IRB approval.** Not applicable.
16. **Use of language models in the method.** The objects studied are language models. No language
    model is used to label or score any reported result: liveness is scored by word matching and the
    readout by first-token probabilities (judge-scored numbers were dropped, P2-J).
