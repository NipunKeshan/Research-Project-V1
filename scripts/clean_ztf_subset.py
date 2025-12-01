import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

infile = 'ztf_image_catalog_subset.csv'
outfile = 'ztf_image_catalog_subset_cleaned.csv'

print(f"Loading {infile}...")
try:
    df = pd.read_csv(infile)
except FileNotFoundError:
    raise

print('Initial shape:', df.shape)

# Drop exact duplicates
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
after = len(df)
print(f'Dropped {before - after} duplicate rows')

# Try to coerce obvious numeric columns
numeric_cols = []
for c in df.columns:
    try:
        df[c] = pd.to_numeric(df[c], errors='ignore')
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
    except Exception:
        pass

print('Numeric columns detected:', numeric_cols)

# Save cleaned CSV
df.to_csv(outfile, index=False)
print(f"Saved cleaned file to {outfile} (shape: {df.shape})")

# Also produce a small correlation heatmap if possible
num_for_corr = [c for c in numeric_cols if df[c].nunique() > 1]
if len(num_for_corr) >= 2:
    corr = df[num_for_corr].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation matrix (numeric columns)')
    plt.tight_layout()
    plt.savefig('numeric_correlation_heatmap.png')
    print('Saved numeric correlation heatmap to numeric_correlation_heatmap.png')
else:
    print('Not enough numeric columns for correlation heatmap.')
