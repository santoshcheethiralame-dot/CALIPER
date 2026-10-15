# Paper checklist (NeurIPS form), Paper 1

Filled in under the programme's rigour standard (`docs/flagship-plan.md` §4a), although TMLR does
not require it. Section references are to `paper1/main.tex`. Items marked **(after E-1)** must be
re-checked once E-1 (`docs/preregistration-e1-fair-comparison.md`) is scored and folded in.

1. **Claims.** Yes. The abstract and the contribution list claim only what §5 and the appendices
   show, each with its interval, and scope the setting (MLP units of small LayerNorm models, one
   estimator family, noiseless responses). Failed predictions are reported as failed (Appendix D).
   **(after E-1)** Re-check every abstract number against the new main experiment.
2. **Limitations.** Yes: "Limitations and scope" (§6) and Appendix B (threats to validity:
   construct, internal, external).
3. **Theory, assumptions and proofs.** Partly applicable. The one exact result (the pre-activation is
   $\mathbf{w}_j^\top\mathbf{s}+b_j$, and the component along $1/\gamma$ is unidentifiable) is derived
   in §3 and verified numerically (Table 1); no other theorem is claimed.
4. **Reproducibility.** Yes. Every arm's command, seeds, unit list and stopping rule are in its
   pre-registration; Appendix C lists every run; the estimator settings are in §4.
5. **Open access to data and code.** Yes: the OXIDE bench (Appendix E), released with code under
   MIT and unit lists, labels and fitted rows under CC BY 4.0; an anonymised copy accompanies the
   submission. The stimulus is public-domain text.
6. **Experimental settings and details.** Yes: §4 (estimator, splits, stopping, failure rule, checks)
   and Appendix C (per-arm settings and precision).
7. **Statistical significance.** Yes. Every AUC difference has a stratified bootstrap 95% CI, DeLong
   $p$ and, where filed, a paired permutation $p$; pooling uses HKSJ random effects with a prediction
   interval; the resampling unit is named in each case.
8. **Compute resources.** Yes: Appendix "Compute and software" (one laptop CPU, 169 CPU-hours over
   35 runs; one Kaggle GPU arm). **(after E-1)** Add E-1's hours.
9. **Code of ethics.** Yes. No human subjects, personal data or deployment.
10. **Broader impacts.** Yes: "Broader impact statement" (after the conclusion).
11. **Safeguards.** Not applicable: no model or data with misuse risk is released.
12. **Licences for existing assets.** Yes, stated in Appendix E: GPT-2 (MIT), GPT-Neo-125m (MIT),
    Pythia-160m, -410m and -1.4b (Apache-2.0), the GPT-2 small residual-stream SAEs used by T-SAE
    (`jbloom/GPT2-Small-SAEs-Reformatted`, MIT), and Project Gutenberg texts (public domain in the
    United States). Licences checked on the model cards on 15 Oct 2026.
13. **New assets.** Yes: OXIDE, documented in Appendix E, with its licence.
14. **Crowdsourcing and human subjects.** Not applicable.
15. **IRB approval.** Not applicable.
16. **Use of language models in the method.** Not applicable as a method component: the models
    studied are language models, but no language model is used to label, score or generate any
    result.

## Still to do before submission

- (Done 15 Oct) The licence list of item 12 is in Appendix E.
- Re-check items 1 and 8 after E-1.
