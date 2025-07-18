import pandas as pd
import cirpy
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.colab import drive
import warnings
warnings.filterwarnings("ignore")

drive.mount('/content/drive')

# Define Drive path
drive_path = '/content/drive/MyDrive/Work'
os.makedirs(drive_path, exist_ok=True)
csv_file = os.path.join(drive_path, 'eco.csv')
df = pd.read_csv(csv_file)

# Now define the column name to clean
cas_column = 'CAS'

# Function to clean CAS numbers
def clean_cas(cas):
    if pd.isna(cas):
        return cas
    try:
        parts = cas.split('-')
        parts[0] = str(int(parts[0]))
        return '-'.join(parts)
    except Exception:
        return cas

df['CAS_cleaned'] = df[cas_column].apply(clean_cas)
cas_column = 'CAS_cleaned'

# Filter valid CAS entries
valid_cas_df = df[
    df[cas_column].notna() &
    (~df[cas_column].astype(str).str.lower().isin(['undefined', 'no result']))
]
cas_numbers = valid_cas_df[cas_column].astype(str).unique()

# --- Parallel SMILES query ---
def get_smiles(cas):
    try:
        return {'CAS_cleaned': cas, 'SMILES': cirpy.resolve(cas, 'smiles')}
    except Exception:
        return {'CAS_cleaned': cas, 'SMILES': None}

results = []
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(get_smiles, cas): cas for cas in cas_numbers}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Resolving SMILES"):
        results.append(future.result())

# Merge and save
smiles_df = pd.DataFrame(results)
merged_df = df.merge(smiles_df, on='CAS_cleaned', how='left')

output_file = os.path.join(drive_path, 'eco-smiles.csv')
merged_df.to_csv(output_file, index=False)

# Summary
valid = smiles_df['SMILES'].notna().sum()
invalid = smiles_df['SMILES'].isna().sum()
print(f"\nValid SMILES retrieved: {valid}")
print(f"Unresolved CAS numbers: {invalid}")
print(f"Saved to: {output_file}")
