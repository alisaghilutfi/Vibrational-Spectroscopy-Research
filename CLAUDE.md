# Vibrational-Spectroscopy-Research — repo rules

Global guardrails: `~/.claude/CLAUDE.md`.

## Repo purpose

Python research code for deep learning on CARS spectroscopy (NRB removal /
Raman signal extraction). One folder per published paper or active study.
No Fabric, no Power BI, no MCP, no TMDL — none of that tooling applies here.

## Structure convention

```
<StudyName>/          one folder per paper or study, named after the topic
  *.py                training / prediction scripts
  *.ipynb             exploratory notebooks, hyperparameter search, benchmarks
  *_weights.h5        trained weights (published papers only — see weights policy)
  README.md           paper reference, key result, dataset note
CONTEXT.md            living research log — findings, blockers, next steps
requirements.txt      pinned dependencies
```

Do not create subfolders inside a study folder for intermediate outputs. Scratch
outputs go in the session scratchpad, not committed.

## Python environment

- Python 3, TensorFlow 2.7, `tensorflow.keras`
- Hyperparameter search: Optuna (Bayesian optimisation)
- Data: NumPy arrays (`.npy`), loaded via `np.load`
- Do not upgrade framework versions without verifying weight compatibility across
  all `.h5` files in the repo.

## Notebook vs. script convention

- `.py` for training runs (reproducible, scriptable, CI-friendly)
- `.ipynb` for exploration, hyperparameter search, benchmark evaluation, and
  result visualisation
- Keep notebooks clean before committing: clear outputs on cells that produce
  large arrays or plots, unless the output is the documented result

## Data rules

**Committed:**
- Synthetic test spectra (`.npy`) when small enough for Git LFS or direct commit
- Trained weights (`.h5`) for published papers

**Never committed:**
- Experimental CARS spectra (IP/data-sharing constraints — available on request
  from the original lab)
- Large training datasets — host on Hugging Face and reference by URL
- Any file containing lab names, supervisor contacts, or institution-internal paths

**Weights policy:**
- Published paper weights: commit to repo under the study folder
- New experiment weights: gitignore until the paper is accepted; then push to
  Hugging Face model hub and add a model card

## Git

- Follow global conventional commits (`feat:`, `fix:`, `data:`, `exp:` for experiments)
- Branch: work on `dev`, merge to `main` via PR — never push directly to `main`
- Commit message subject <= 50 chars
- Do not commit notebooks with uncleared large outputs (arrays, model summaries
  over ~50 lines)

## Public repo — what to keep out

This repo is public. Never include:
- Personal contact details, supervisor names, or institution-internal identifiers
- Postdoc plans, funding strategy, or career notes (keep those private)
- Experimental spectra or data under a data-sharing agreement
- API keys, tokens, or any credentials

## CONTEXT.md

Read `CONTEXT.md` at session start. Update the session log section at session end.
Do not rewrite the Instructions layer mid-session — that is a ProjectPlanner task.
