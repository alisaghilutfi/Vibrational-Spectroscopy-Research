"""
Generate README figures.

Outputs
-------
figures/cars_nrb_removal.png   -- before/after for 2 representative spectra
figures/model_comparison.png   -- MSE bar charts: NRB type effect + architecture comparison

Usage (from repo root)
----------------------
    python generate_readme_figures.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    BatchNormalization, Activation, Conv1D, Dense, Flatten, Dropout
)
from tensorflow.keras import regularizers

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent
DATA_DIR    = REPO / "paper1_NRB_effect_on_CNN"
WEIGHTS_DIR = REPO / "paper2_four_architectures"
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

# ===========================================================================
# Figure 2 — Model comparison bar chart
# Left panel:  NRB type effect on CNN         (Paper 1)
# Right panel: Architecture comparison        (Paper 2)
# ===========================================================================

def mean_mse(model, x, y_true):
    """Return mean MSE across all test spectra."""
    pred = model.predict(x, verbose=0)
    pred = pred.squeeze(-1) if pred.ndim == 3 else pred
    return float(np.mean((pred - y_true) ** 2))

print("\nLoading models for comparison chart...")

PAPER1_DIR = REPO / "paper1_NRB_effect_on_CNN"
PAPER2_DIR = REPO / "paper2_four_architectures"

def build_cnn(n_points=640):
    """Reconstruct SpecNet CNN architecture (identical across all NRB variants)."""
    m = Sequential([
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,
                           input_shape=(n_points, 1)),
        Activation("relu"),
        Conv1D(128, kernel_size=32, activation="relu"),
        Conv1D(64,  kernel_size=16, activation="relu"),
        Conv1D(16,  kernel_size=8,  activation="relu"),
        Conv1D(16,  kernel_size=8,  activation="relu"),
        Conv1D(16,  kernel_size=8,  activation="relu"),
        Dense(32, activation="relu",
              kernel_regularizer=regularizers.l1_l2(l1=0, l2=0.1)),
        Dense(16, activation="relu",
              kernel_regularizer=regularizers.l1_l2(l1=0, l2=0.1)),
        Flatten(),
        Dropout(0.25),
        Dense(n_points, activation="relu"),
    ])
    m.compile(loss="mse", optimizer="Adam")
    return m

def load_cnn(weights_path):
    m = build_cnn()
    m.load_weights(str(weights_path))
    return m

nrb_models = {
    "Two-sigmoid\n(SpecNet)": load_cnn(PAPER1_DIR / "Specnet_weights.h5"),
    "One-sigmoid":            load_cnn(PAPER1_DIR / "One_sigmoid_NRB_model_weights.h5"),
    "Polynomial\n(best)":     load_cnn(PAPER1_DIR / "Polynomial_NRB_model_weights.h5"),
}

arch_models = {
    "CNN\n(Polynomial)": load_cnn(PAPER1_DIR / "Polynomial_NRB_model_weights.h5"),
    "LSTM":              load_model(str(PAPER2_DIR / "LSTM_model_weights.h5")),
    "Bi-LSTM\n(best)":   load_model(str(PAPER2_DIR / "Bi_LSTM_model_weights.h5")),
}

print("Computing MSE for each model...")
nrb_mse   = {k: mean_mse(m, x_input, y_test) for k, m in nrb_models.items()}
arch_mse  = {k: mean_mse(m, x_input, y_test) for k, m in arch_models.items()}

# --- palette ---
C_DEFAULT  = "#A8C0D6"   # muted blue for standard bars
C_BEST     = "#E67E22"   # warm orange for the winner
C_GRID     = "#E0E0E0"

def bar_colors(labels, best_label_fragment):
    return [C_BEST if best_label_fragment in lbl else C_DEFAULT for lbl in labels]

fig2, (ax_nrb, ax_arch) = plt.subplots(1, 2, figsize=(11, 4))
fig2.patch.set_facecolor("white")

for ax, mse_dict, best_frag, title, paper_tag in [
    (ax_nrb,  nrb_mse,  "Polynomial", "Effect of NRB type on CNN",        "Paper 1"),
    (ax_arch, arch_mse, "Bi-LSTM",    "Architecture comparison",           "Paper 2"),
]:
    labels = list(mse_dict.keys())
    values = list(mse_dict.values())
    colors = bar_colors(labels, best_frag)

    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.015,
                f"{val:.2e}", ha="center", va="bottom", fontsize=9, color="#444444")

    ax.set_ylabel("Mean MSE (300 test spectra)", fontsize=10)
    ax.set_title(f"{title}\n({paper_tag})", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(values) * 1.22)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8)

fig2.suptitle("Deep learning for CARS NRB removal — model benchmarks",
              fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()

out_path2 = OUT_DIR / "model_comparison.png"
fig2.savefig(out_path2, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved -> {out_path2}")
plt.show()
