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

### 2026-11-17 (session 2) — Benchmark audit, HF search, Milan data investigation

#### What was done

**1. Notebook results summary filled in (`Transformer_CARS/benchmark_evaluation.ipynb`, cell `12-summary`)**

The placeholder table was replaced with the actual benchmark numbers:

Synthetic (300 spectra, seed=42):
- Bi-LSTM: MSE 3.05e-05 (best, 66x over SpecNet)
- Transformer-FH: 2.62e-03 | Transformer-K: 3.88e-03 | Transformer-PE: 4.86e-03
- All Transformers worse than SpecNet on synthetic data

Experimental (4 spectra):
- DMPC lipid membrane: Transformer-K best (6.94e-03); Bi-LSTM collapses to 2.65e-01 — 8700x degradation from its synthetic MSE
- MEC Yeast: Bi-LSTM best (3.02e-03)
- AMP-ADP-ATP: Transformer-FH best (2.45e-04)
- Protein droplet: Transformer-PE best (2.10e-04)
- Transformers win 3 of 4 experimental samples

Key finding stated in notebook: "Models optimized for in-distribution synthetic accuracy generalize worst under domain shift." This is the Paper 5 publishable claim.

Limitations checklist added: n=4 (no statistical claims), single seed, MEM/KK not ground truth, missing RamPINN/iPINN baselines, retitle required.

**2. Hugging Face MCP search for CARS/Raman datasets**

Findings:
- No CARS, BCARS, or NRB-removal datasets or models exist on HF. Milan's data is not there.
- Spontaneous Raman datasets found:
  - `HTW-KI-Werkstatt/RamanBench`: 94 Raman datasets unified, 108K downloads, arXiv:2605.02003, Parquet format, "other" license (per-source). Useful for domain adaptation pretraining.
  - `JunhanCai/GEMS-Raman-Dataset`: multi-source Raman for self-supervised + contrastive learning (GEMS framework), MIT license. Relevant to TRUST-CARS Aim 1.
  - `HTW-KI-Werkstatt/RamanSpectraEthanolicYeastFermentations`: yeast fermentation Raman, CC-BY 4.0. Adjacent to exp_2 (MEC Yeast).
- Opportunity confirmed: publishing a CARS benchmark on HF would be the first of its kind — no competition.

**3. Milan GitHub repo investigation (crimson-project-eu)**

Repos found:
- `NRB_removal` (MIT): CNN+GRU notebook + GAN code. Training data loaded from local `.pkl` — not published.
- `CARS_Simulation` (public): Their synthetic BCARS generator. MIT license. More sophisticated than the current generator:
  - 3 NRB/Stokes profile types (polynomial, Gaussian, sigmoidal product)
  - Noise range 10x wider (0.5%–5% vs 0.05%–0.3%)
  - Peak filtering step (removes weak/buried features)
  - 1000-point spectra, wavenumber 100–2000 cm-1
  - Returns BCARS + Raman + NRB (3 outputs)
- `deliverables`: project PDFs only

Conclusion: **experimental data is not publicly available on GitHub or HF.** The Nature paper's Data Availability section (behind paywall) likely points to a Zenodo record — blocked without LUT access.

#### Decisions made

- LUT access needed to retrieve Milan's experimental dataset DOI from Nature paper — deferred until LUT access restored or supervisor consulted.
- Three action items require supervisor consultation before proceeding:
  1. Milan experimental data access (via LUT/Vartiainen)
  2. HF model card drafting and weight publication (needs supervisor sign-off on what to release)
  3. Cross-generator generalization experiment (Milan's generator vs own — new result for Paper 5; needs supervisor alignment before adding new experiments)

#### Blockers added

| Blocker | Status |
|---|---|
| Milan experimental data: Nature paper paywalled, no LUT access | Deferred — consult Vartiainen |
| Milan experimental data: not on GitHub or HF, Zenodo DOI unknown | Deferred — same as above |
| HF weight publication, paper 5 scope changes, new experiments | Need supervisor consultation before proceeding |

#### Planned (approved in principle, pending supervisor consultation)

- **Cross-generator experiment**: train on own synthetic generator, test on Milan's generator spectra using existing weights. No retraining needed — CPU inference, ~20-30 min. New result that belongs in Paper 5 before submission.
- **HF model cards**: draft cards for all 5 models (SpecNet, Bi-LSTM, Transformer-K/PE/FH), upload `.h5` weights. Prerequisite for January outreach emails to Bocklitz and Polli.
- **Paper 5 outline**: section structure, new title, figure plan — ready to draft as a document.
- **HF CARS benchmark dataset repo**: publish 300 synthetic test spectra + 4 experimental CSVs + Milan data (once obtained). First CARS benchmark on HF.

#### Tools/resources noted for future use

- HF MCP (`mcp__claude_ai_Hugging_Face__*`): live search + file browsing of HF Hub. Authenticated as ali-saghi. Use for dataset search, model card publishing, file inspection.
- NotebookEdit: direct cell editing in `.ipynb` without opening Jupyter. Used to fill cell `12-summary`.
- Milan's synthetic generator: `crimson-project-eu/CARS_Simulation` — clone and compare with own generator before next experimental expansion.
- RamanBench (arXiv:2605.02003): 94 spontaneous Raman datasets, could serve as domain adaptation pretraining corpus for TRUST-CARS WP2.
- GEMS dataset (`JunhanCai/GEMS-Raman-Dataset`): MIT license, multi-source Raman, self-supervised framework — relevant to TRUST-CARS Aim 1.

---

### 2026-11-17 — CONTEXT.md created

- Reviewed Strategic Roadmap (`CARS Deep-Learning Research_ Strategic Roadmap.md`)
  and repo structure
- Created CONTEXT.md using existing analytics CONTEXT.md patterns, adapted for
  research tone: findings/hypotheses/blockers/directions/funding tracker
- No code or model changes
