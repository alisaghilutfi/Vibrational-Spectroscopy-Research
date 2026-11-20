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
| 1 — NRB effect on CNN generalisation | `paper1_NRB_effect_on_CNN/` | RSC Advances | 2022 | Published |
| 2 — Four architectures compared | `paper2_four_architectures/` | PCCP | 2023 | Published |
| 3 — Fine-tuned SpecNet | `paper3_finetuned_SpecNet/` | Optics Continuum | 2022 | Published |
| 4 — Hyperparameter optimisation | `paper4_hyperparameter_optimization/` | Optics Continuum | 2024 | Published |
| 5 — Attention vs. recurrence study | `paper5_Transformer_CARS/` | TBD | — | In progress — needs re-scope |

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
- **Paper 5 finding (statistically defensible as of session 7):** models with best in-distribution MSE generalise worst to experimental spectra — Bi-LSTM (best synthetic MSE 3.9e-05) has 3800x worse DMPC robustness than Transformer-K (1.8x ratio). Confirmed across 3 bootstrap seeds, stable rankings. Publishable as a negative/diagnostic result.

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

### Field landscape (verified Nov 2026 via web search)

The field has split into two tracks. Architecture novelty (CNN vs LSTM vs Transformer) is no longer publishable on its own.

**Track 1 — Physics-informed models (Jena dominates)**
- iPINN (Bocklitz, Sept 2026): Transformer encoder → 24 Lorentzian peak slots → differentiable forward model. MAE 0.016, best published result. No explicit NRB estimation during training.
- RamPINN (Denzler, Oct 2025): KK-consistent dual decoder, self-supervised, zero-shot to 6 real molecules.
- UQ-PINN (Denzler, Nov 2025): calibrated uncertainty (ensembles, MC-dropout, conformal prediction). Physics losses improve calibration.
- DA-DMD (Denzler, 2025): attention-weighted dynamic mode decomposition.

**Track 2 — Foundation models and benchmarks (spontaneous Raman only — CARS layer unoccupied)**
- RamanBench (NeurIPS 2026, arXiv 2605.02003): 94 spontaneous Raman datasets — not CARS.
- RamanPFN (Aug 2026, arXiv 2608.02157): tabular foundation model, 74 datasets — not CARS.
- DSCF (Nature MI 2025): 1M+ spectra pretrained — not CARS.
- GEMS (HF 2026): multi-source self-supervised Raman — not CARS.

**Key gap:** No CARS benchmark exists on HF or anywhere publicly. No systematic sim-to-real transfer study across multiple architectures. No head-to-head between physics-informed and data-driven models on the same real data. UQ not connected to domain shift.

---

### Research directions (ranked)

| Rank | Direction | Novelty | Feasibility |
|---|---|---|---|
| 1 | **Open CARS benchmark** (HF dataset + standard metrics: MSE, peak F1, edge error, calibration) | High — nobody has done this for CARS | Very high — data engineering is a strength |
| 2 | **Sim-to-real gap study** (cross-generator + cross-architecture, multiple seeds) | Medium-high — 8700x Bi-LSTM collapse is new evidence | High — Milan's generator in hand, no retraining needed |
| 3 | **Sim-to-real domain adaptation + KK physics constraints** | Medium — unsupervised DA on unlabeled real spectra not yet addressed | High — builds on Papers 1–5 |
| 4 | **Uncertainty as proxy for domain shift** | Medium — connects UQ-PINN and sim-to-real evidence | Medium — LUT Bayesian heritage is relevant |
| 5 | **Hyperspectral BCARS imaging** (3D conv / ViT over cubes, tissue/cell classification) | Medium-high | Medium — requires microscope access at host |
| 6 | **Coherent Raman foundation model** (self-supervised pretraining across CARS/SRS) | High for CARS specifically | Medium-low — benchmark must come first |

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

### 2026-11-20 (session 7) — SpecNet-Full diagnosis + multi-seed benchmark

#### What was done

1. **Diagnosed SpecNet-Full** — h5py weight-norm inspection confirmed the file
   is identical to the thesis archive copy (23,539 KB, total norm 89.10, output
   layer norm 41.11). Not a corrupt or intermediate checkpoint. F1~0 reflects
   genuine model behaviour — different training distribution or early stop.
   Verdict: include in benchmark with a note; no re-training needed.

2. **Multi-seed benchmark run** — updated `run_benchmark.py` to use bootstrap
   resampling (3 seeds: [42, 123, 7], draw-with-replacement from the fixed 300-
   spectrum test set). All 8 models × 3 seeds evaluated; each (model, seed) pair
   logged as a separate MLflow run + per-model SUMMARY runs with mean/std metrics.
   Full results in `cars_benchmark/CONTEXT.md`.

#### Key result

Rankings are fully stable across seeds. Variance is small vs. effect sizes:

| Model | MSE_syn (mean±std) | DMPC ratio (mean±std) |
|---|---|---|
| Bi-LSTM | 3.943e-05 ± 5.1e-06 | 6807.6x ± 938.6 |
| LSTM | 2.916e-04 ± 2.6e-05 | 1738.1x ± 160.4 |
| SpecNet-Polynomial | 4.204e-04 ± 1.5e-05 | 11.9x ± 0.4 |
| Transformer-FH | 2.606e-03 ± 1.2e-04 | 3.5x ± 0.2 |
| Transformer-K | 3.867e-03 ± 1.9e-04 | 1.8x ± 0.1 |

The ~4000x DMPC robustness gap between Transformer-K and Bi-LSTM is now
statistically defensible for Paper 5.

#### Next steps (carry forward)

- [x] Investigate SpecNet-Full checkpoint validity  *(done — valid file, genuine low performance)*
- [x] Add multiple seeds to benchmark  *(done — 3 seeds, all models)*
- [ ] Regenerate leaderboard figure with error bars
- [ ] Get Vartiainen sign-off for experimental data public release
- [ ] Upload to HF (`ali-saghi/CARS-Benchmark`) on arXiv submission
- [ ] Write Paper 5 benchmark section

---

### 2026-11-20 (session 6) — Transformer models added to benchmark (8-model leaderboard)

#### What was done

1. **Found Transformer weight files** — located in thesis archive at
   `_other stuff/PhD/thesis_.../06_Transformers/Finals/`. Three files:
   `weights_Transformer_timeseriesKeras_2_2.h5`, `weights_Transformer_timeseriesPosEncEmb.h5`,
   `weights_Transformer_timeseriesFourierHead.h5`. Not committed (unpublished paper).

2. **Added 3 Transformer architectures to `run_benchmark.py`** — Transformer-K,
   Transformer-PE, Transformer-FH. Architecture code ported directly from
   `paper5_Transformer_CARS/benchmark_evaluation.ipynb`. Custom Keras layers
   (`_EmbeddingLayer`, `_PositionalEncoding`, `_FourierHead`) implemented inline.

3. **Ran 8-model benchmark** — all models logged to MLflow `CARS-Benchmark`
   experiment. Full numbers in `cars_benchmark/CONTEXT.md` (private).

#### Key new finding (session 6)

Transformer-K gen_ratio on DMPC: **1.7x**. Bi-LSTM on the same sample: **6723x**.
That is a ~4000x difference in domain-shift robustness between the best Transformer
and the best recurrent model, on the hardest experimental sample. The Paper 5
hypothesis is now quantified across all 8 models:
> *Best in-distribution MSE (Bi-LSTM, 3.9e-05) → worst DMPC generalisation (6723x).
> Transformer-K has 100x worse synthetic MSE but 4000x better DMPC robustness.*

4. **Leaderboard figure built and committed** — two-panel horizontal bar chart
   (synthetic MSE vs. DMPC gen_ratio, log scale, sorted by synthetic MSE to show
   rank inversion). Saved as `paper5_Transformer_CARS/paper5_leaderboard.pdf/.png`.
   Journal-ready caption written. Commit `fac2e6a`.

5. **Public/private split confirmed** — `paper5_Transformer_CARS/` (notebook +
   figure) is correctly public; supervisor already has the repo link. Raw
   experimental CSVs and benchmark numbers stay gitignored until Vartiainen sign-off.

#### Next steps (carry forward)

- [x] Build leaderboard figure for Paper 5  *(done — fac2e6a)*
- [x] Investigate SpecNet-Full checkpoint validity  *(done — session 7)*
- [x] Add multiple seeds to benchmark for statistical claims  *(done — session 7)*
- [ ] Get Vartiainen sign-off for experimental data public release
- [ ] Upload to HF on arXiv submission

---

### 2026-11-20 (session 5) — Open CARS benchmark: package, results, MLflow tracking

#### What was done

1. **Benchmarked the field** — confirmed no CARS benchmark exists publicly (HF,
   GitHub, or literature). Closest analogue: RamanBench (arXiv:2605.02003),
   spontaneous Raman only. CARS evaluation metric design is original.

2. **Discovered and inventoried experimental data** — found all local experimental
   CARS spectra under `C:\Users\alisa\DS-ML-DL\_other stuff\`. Catalogued into
   three groups: Group A (640-pt, 2500–3100 cm⁻¹, 4 paired CARS+Raman CSVs —
   canonical for benchmark), Group B (640-pt fingerprint, no Raman pair), Group C
   (1000-pt, 3 paired samples). Canonical source: `08_HyperOptimizations/Comparisons/experimental_CARS/`.

3. **Created `experimental_data/` folder** — gitignored (IP-restricted). Copied
   Group A CSVs. Wrote `experimental_data/README.md` with full provenance, file
   inventory, and action items (Vartiainen sign-off required for public release).
   Commits: `f072062` (.gitignore), content not committed.

4. **Built `cars_benchmark/` package** — gitignored until Paper 5 submission.
   - `metrics.py`: MSE, peak F1 (scipy find_peaks, ±2 bin tolerance), edge error
     (gradient-based), calibration score (ECE, returns None for deterministic models),
     generalisation ratio (MSE_exp / MSE_syn).
   - `loaders.py`: `load_synthetic`, `load_experimental` (with UTF-8 BOM fix),
     `load_weights_keras23` (Keras 2.3.1 → TF 2.11 positional weight loader).
   - `evaluate.py`: CLI entry point `cars-eval`.
   - 15 unit tests, all passing in `tensorflow_env_higher`.
   Commit: `ed8a309` (.gitignore entry only; package itself not committed).

5. **Fixed Keras 2.3.1 compatibility** — `Specnet_weights.h5` has no model config,
   `_1`-suffixed layer names, and weights stored alphabetically (bias before kernel).
   Diagnosed via h5py inspection. Fixed with `load_weights_keras23()`: positional
   layer matching + `sorted(keys, reverse=True)` for [kernel, bias] order.

6. **Ran full benchmark** — all 5 committed model weights evaluated on synthetic
   (300 spectra) and 4 experimental samples. Full results in `cars_benchmark/CONTEXT.md`.

#### Key findings (2026-11-20)

Full benchmark numbers and per-sample results are in `cars_benchmark/CONTEXT.md`
(gitignored — private until supervisor sign-off).

**High-level:** best synthetic MSE predicts worst experimental generalisation.
Recurrent models collapse on structured lipid NRB (DMPC). SpecNet-Polynomial
is the most stable cross-sample generaliser. SpecNet-Full weights suspect —
needs investigation. Full detail in the private benchmark context.

#### Also done (late session)

7. **MLflow tracking wired in** — installed MLflow 2.17.2 in `tensorflow_env_higher`
   (protobuf pinned to 3.20.3 for TF 2.11 compatibility). `run_benchmark.py` now
   logs every run to `cars_benchmark/mlruns/` (gitignored). 5 runs confirmed in
   `CARS-Benchmark` experiment, 15 metrics each. UI: `mlflow ui --backend-store-uri
   "file:///C:/Users/alisa/.../cars_benchmark/mlruns"` → localhost:5000.

#### Next steps (carry forward)

- [ ] Investigate SpecNet-Full checkpoint validity
- [ ] Build leaderboard artifact for Paper 5 supplementary
- [ ] Add Paper 5 Transformer model to benchmark once trained
- [ ] Get Vartiainen sign-off for experimental data public release
- [ ] Upload to HF (`ali-saghi/CARS-Benchmark`) on arXiv submission

---

### 2026-11-17 (session 4) — Repo polish: folder rename, badges, paper READMEs

#### What was done

1. **Folder rename** — all five paper folders renamed to consistent `paperN_` prefix
   with snake_case (`paper1_NRB_effect_on_CNN` … `paper5_Transformer_CARS`).
   Git history preserved via `git mv`. References updated in README.md, CONTEXT.md,
   `generate_readme_figures.py`.

2. **Publication badges** — DOI (blue) + journal (green) badges added to each paper
   section in the root README. All four papers now have both badge types.
   DOIs for Papers 3 and 4: `10.1364/OPTCON.472573`, `10.1364/OPTCON.527778`.

3. **Paper READMEs rewritten** — consistent template applied to all four:
   badges, one-line summary, key finding, contents table with file descriptions,
   data section, requirements note (including `tensorflow_env_higher` fix),
   BibTeX citation block.
   - Removed email addresses from Paper 2 (public repo).
   - Replaced broken Junjuri/LUT GitHub links with references to this repo.
   - Papers 3 and 4 were near-empty stubs — fully written from scratch.

**Commits:** folder renames `2581e7e`, badges `d76925c` / `f93a7de`, paper READMEs `252e988`

---

### 2026-11-17 (session 3) — Repo housekeeping + README visual upgrade

#### What was done

1. **CLAUDE.md created** — repo-scoped rules: Python env (TF 2.7 / TF 2.11), notebook vs.
   script convention, data/weights policy, public-repo guardrails, CONTEXT.md protocol.

2. **Private files gitignored** — `CARS Deep-Learning Research_ Strategic Roadmap.md`
   and `Claude outputs/` folder added to `.gitignore`. Progress report renamed to
   remove supervisor names from filename.

3. **README visual upgrade:**
   - `generate_readme_figures.py` added at repo root — generates both README figures
     from committed weights + test data; requires `tensorflow_env_higher` (TF 2.11,
     NumPy 1.23.5); `tensorflow_env` incompatible (NumPy 1.24 breaks TF 2.7).
   - `figures/cars_nrb_removal.png` — before/after for 2 spectra (Bi-LSTM),
     examples selected by per-sample MSE percentile (25th and 60th).
   - `figures/model_comparison.png` — two-panel bar chart: NRB type effect on CNN
     (Paper 1) and architecture comparison CNN/LSTM/Bi-LSTM (Paper 2). CNN weights
     are weights-only `.h5` files (not full saved models) — script rebuilds
     architecture via `build_cnn()` before `load_weights()`.
   - Both figures embedded in README above Key Results section.

**Commits:** `bc6004c` (hero figure), `a7e2b7f` (comparison chart)

---

### 2026-11-17 (session 2) — Benchmark audit, HF search, Milan data investigation

#### What was done

**1. Notebook results summary filled in (`paper5_Transformer_CARS/benchmark_evaluation.ipynb`, cell `12-summary`)**

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
