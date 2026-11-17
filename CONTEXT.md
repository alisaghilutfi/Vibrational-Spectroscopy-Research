# CONTEXT — Vibrational-Spectroscopy-Research

> **How to use this file**
> Read at session start. Update at session end.
> Instructions layer: strategic framing, findings, blockers, next steps — written
> by ProjectPlanner, read before acting.
> Session log: appended at session end by the executing agent, most recent first.

---

## Instructions layer

### Research position (Nov 2026)

**What this work is**
Deep learning for NRB removal in CARS spectroscopy. Four published papers
(2022–2024) across CNN fine-tuning, Bi-LSTM, hyperparameter optimisation, and
NRB-type effects. Paper 5 (Transformer/attention study) is in progress and
needs re-scoping before submission.

**Re-scoped thesis**
The "first Transformer for CARS" framing is invalid — several groups (Jena,
Denzler) have published attention/transformer models for CARS NRB removal
in 2025–2026. The correct angle is:
> *Sim-to-real generalisation study: attention vs. recurrence under synthetic-to-real
> shift in CARS Raman retrieval — with the publishable finding that best
> in-distribution MSE predicts worst generalisation.*

---

### Paper status

| # | Folder | Journal | Year | Status |
|---|---|---|---|---|
| 1 — NRB effect on CNN generalisation | `NRBs_on_extraction_of_Raman_signals_from_CARS_spectra/` | RSC Advances | 2022 | Published |
| 2 — Four architectures compared | `Four_DL_models_comparison/` | PCCP | 2023 | Published |
| 3 — Fine-tuned SpecNet | `Finetuned_SpecNet/` | Optics Continuum | 2022 | Published |
| 4 — Hyperparameter optimisation | `Hyperparameter_Optimization_of_CNN/` | Optics Continuum | 2024 | Published |
| 5 — Attention vs. recurrence study | `Transformer_CARS/` | TBD | — | In progress — needs re-scope |

**Paper 5 re-scope checklist**
- [ ] Retitle: *"Attention vs. recurrence under synthetic-to-real shift in CARS Raman retrieval"*
- [ ] Cite RamPINN, DA-DMD, iPINN; drop "first Transformer" claim
- [ ] Expand test set: add Milan public BCARS data, RamPINN 6-molecule set, toluene/DMSO (4 spectra is statistically insufficient)
- [ ] Report multiple seeds, peak-level metrics (F1, edge error), RamPINN/iPINN baselines
- [ ] State explicitly: MEM/KK reference is not ground truth — a low experimental MSE measures agreement with MEM, not accuracy
- [ ] Target venue: *Optics Express*, *Analyst*, or *J. Raman Spectroscopy*
- [ ] Release code, weights, synthetic generator on Hugging Face/GitHub on arXiv submission

---

### Key findings (empirically verified — do not re-derive without new evidence)

- **NRB shape matters more than model complexity:** polynomial NRB training gives the best generalisation to experimental spectra (Paper 1)
- **Bi-LSTM is the strongest architecture in this setup:** outperforms CNN, LSTM, and VECTOR on both synthetic and experimental spectra (Paper 2)
- **Transfer learning works:** fine-tuned SpecNet reaches 86% accuracy on semi-synthetic data and beats baseline SpecNet on 4 experimental spectra (Paper 3)
- **Hyperparameter search pays off:** optimised SpecNet at ~96% accuracy, MSE ~60x lower than baseline, resolving edge and intensity failure modes (Paper 4)
- **Paper 5 candidate finding (hypothesis — needs stronger evidence):** models with best in-distribution MSE can generalise worst to experimental spectra — a publishable negative result independent of architecture novelty

---

### Open hypotheses (claims requiring more evidence)

- "Attention generalises better than recurrence under synthetic-to-real shift" — currently based on 4 experimental spectra with MEM/KK references; needs multi-lab data and multiple seeds before it can be stated as a finding
- Physics-consistent constraints (KK loss, parametric peak decoder) reduce sim-to-real gap — not yet tested in this codebase

---

### Blockers

| Blocker | Impact | Status |
|---|---|---|
| "First Transformer" claim invalid | Paper 5 must be re-scoped before submission | Mitigated — negative-result angle is publishable |
| Experimental test set too small (4 spectra) | Statistical claims unsupportable | Action: add Milan + RamPINN public sets |
| MEM/KK reference is not ground truth | Limits what "accuracy" means in experimental results | Must be stated explicitly in Paper 5 |
| Experimental CARS spectra not in repo | Reproducibility | Contact Vartiainen for access; cite availability in Paper 5 |

---

### Research directions (ranked)

| Rank | Direction | Novelty | Feasibility |
|---|---|---|---|
| 1 | **Sim-to-real domain adaptation + calibrated uncertainty + KK/MEM-physics constraints** | Medium — unsupervised DA on unlabeled real spectra not yet addressed | High — builds directly on Papers 1–5 |
| 2 | **Open multi-lab CARS benchmark** (Hugging Face datasets + models, standard metrics: MSE, peak F1, edge error, calibration) | Medium-high — no multi-lab CARS benchmark exists | Very high |
| 3 | **Hyperspectral BCARS imaging** (3D conv / ViT over cubes, tissue/cell classification) | Medium-high | Medium — requires microscope access |
| 4 | **Coherent Raman foundation model** (self-supervised pretraining across Raman/CARS/SRS) | High for CARS | Medium-low — needs large multi-lab data ecosystem |

---

### Approved next steps (priority order)

1. **Re-scope and submit Paper 5** (Nov–Dec 2026)
   - Retitle, add benchmarks, report seeds + peak metrics
   - Post to arXiv; release code + weights on Hugging Face
   - Submit to *Optics Express*, *Analyst*, or *J. Raman Spectroscopy*

2. **Build open CARS benchmark** — start with Milan's public BCARS data; extend to
   RamPINN 6-molecule set and own synthetic generator

3. **Explore domain adaptation + physics-constrained retrieval** on the extended benchmark

---

## Session log

*(Appended at session end — most recent first)*

### 2026-11-17 — CONTEXT.md created

- Reviewed Strategic Roadmap (`CARS Deep-Learning Research_ Strategic Roadmap.md`)
  and repo structure
- Created CONTEXT.md using existing analytics CONTEXT.md patterns, adapted for
  research tone: findings/hypotheses/blockers/directions/funding tracker
- No code or model changes
