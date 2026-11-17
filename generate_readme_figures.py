"""
Generate README hero figure: CARS input vs. Bi-LSTM Raman extraction.

Outputs
-------
figures/cars_nrb_removal.png   -- before/after for 2 representative spectra

Usage (from repo root)
----------------------
    python generate_readme_figures.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent
DATA_DIR    = REPO / "NRBs_on_extraction_of_Raman_signals_from_CARS_spectra"
WEIGHTS_DIR = REPO / "Four_DL_models_comparison"
OUT_DIR     = REPO / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load model and test data
# ---------------------------------------------------------------------------
print("Loading Bi-LSTM model...")
model = load_model(str(WEIGHTS_DIR / "Bi_LSTM_model_weights.h5"))

print("Loading test data...")
x_test = np.load(str(DATA_DIR / "x_test_300_merge_spectra3.npy"))  # (300, 640)
y_test = np.load(str(DATA_DIR / "y_test_300_merge_spectra3.npy"))  # (300, 640)

n_points = x_test.shape[1]
nu       = np.linspace(0, 1, n_points)

# ---------------------------------------------------------------------------
# Predict (batch)
# ---------------------------------------------------------------------------
print("Running predictions...")
x_input = x_test[:, :, np.newaxis]              # (300, 640, 1)
y_pred  = model.predict(x_input, verbose=0)     # (300, 640, 1)
y_pred  = y_pred.squeeze(-1)                    # (300, 640)

# ---------------------------------------------------------------------------
# Pick two representative spectra by per-sample MSE
#   idx_lo:  MSE near 25th percentile  → clean fit, easy to read
#   idx_mid: MSE near 60th percentile  → shows some challenge
# ---------------------------------------------------------------------------
per_sample_mse = np.mean((y_pred - y_test) ** 2, axis=1)
mse_sorted_idx = np.argsort(per_sample_mse)
idx_lo  = mse_sorted_idx[int(0.25 * len(mse_sorted_idx))]
idx_mid = mse_sorted_idx[int(0.60 * len(mse_sorted_idx))]
indices = [idx_lo, idx_mid]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
C_CARS = "#7B93B4"   # muted steel blue  -- raw CARS input
C_TRUE = "#27AE60"   # green             -- ground truth Raman
C_PRED = "#E67E22"   # warm orange       -- Bi-LSTM prediction

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
fig.patch.set_facecolor("white")

for col, idx in enumerate(indices):
    ax = axes[col]
    cars_norm = x_test[idx] / x_test[idx].max()   # normalize for overlay

    ax.plot(nu, cars_norm,    color=C_CARS, lw=1.4, alpha=0.65, label="CARS spectrum (input)")
    ax.plot(nu, y_test[idx],  color=C_TRUE, lw=1.8,              label="Ground truth  Im[χ⁽³⁾]")
    ax.plot(nu, y_pred[idx],  color=C_PRED, lw=1.5, ls="--",    label="Bi-LSTM prediction")

    ax.set_xlabel("Normalized wavenumber", fontsize=11)
    ax.set_ylabel("Intensity (a.u.)",       fontsize=11)
    ax.set_title(f"Example {col + 1}", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    sample_mse = per_sample_mse[idx]
    ax.text(0.98, 0.97, f"MSE = {sample_mse:.2e}",
            transform=ax.transAxes, fontsize=8,
            ha="right", va="top", color="#555555")

axes[0].legend(fontsize=9, framealpha=0.35, loc="upper left")

fig.suptitle("NRB removal from CARS spectra via Bi-LSTM",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()

out_path = OUT_DIR / "cars_nrb_removal.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out_path}")
plt.show()
