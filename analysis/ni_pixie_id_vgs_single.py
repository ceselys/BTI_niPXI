import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the CSV
path = Path(r"C:\Users\RSG\Documents\CMU_BTI_Testing\data\20251022-175750_cnfet_scaling_w2_x_2_y_1_lc_400_lch_160_col1_23_dc\initial_iv.csv")  # update with your actual path
df = pd.read_csv(path)

# Compute Vgs = Vwl - Vsl
df["Vgs"] = df["v_wl"] - df["v_sl"]

# Define Id = i_bl
df["Id"] = abs(df["i_bl"])

# Sort by Vgs
df = df.sort_values("Vgs")

# Plot Id vs Vgs on a log scale
plt.figure(figsize=(7,5))
plt.semilogy(df["Vgs"], df["Id"], "o-", label="Id–Vgs")

plt.xlabel("Vgs (V)")
plt.ylabel("|Id| (A)")
plt.title("Id–Vgs")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.tight_layout()
plt.show()
