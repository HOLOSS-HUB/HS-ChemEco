import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from tqdm import tqdm
from google.colab import drive
drive.mount('/content/drive')
RDLogger.DisableLog('rdApp.*')

input_csv_path = '/content/drive/MyDrive/Work/eco-smiles.csv'
output_csv_path = '/content/drive/MyDrive/Work/eco-smiles-desc.csv'
smiles_column = 'SMILES'

df = pd.read_csv(input_csv_path)

# --- Canonicalize and validate SMILES ---
def canonicalize_and_validate(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        print(f"SMILES error: {smiles} → {e}")
    return None

df['canonical_SMILES'] = df[smiles_column].astype(str).apply(canonicalize_and_validate)

# --- Filter only valid SMILES ---
df_valid = df[df['canonical_SMILES'].notnull()].copy()
print(f"Valid SMILES found: {len(df_valid)} / {len(df)}")

# --- Define safe scalar descriptor list ---
descriptor_columns = [
    'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v',
    'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11',
    'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7',
    'EState_VSA8', 'EState_VSA9', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt', 'Kappa1',
    'Kappa2', 'Kappa3', 'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex',
    'MaxPartialCharge', 'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex',
    'MinPartialCharge', 'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount',
    'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings',
    'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors',
    'NumHDonors', 'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds',
    'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings',
    'NumValenceElectrons', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13',
    'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7',
    'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4',
    'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10',
    'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5',
    'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 'VSA_EState10',
    'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7',
    'VSA_EState8', 'VSA_EState9'
]

# --- Function to compute descriptors ---
def compute_descriptors_debug(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid molecule for descriptor calculation: {smiles}")
        return [np.nan] * len(descriptor_columns)

    descriptor_values = []
    for desc in descriptor_columns:
        try:
            val = getattr(Descriptors, desc)(mol)
            descriptor_values.append(val)
        except Exception as e:
            print(f"Error computing {desc} for {smiles}: {e}")
            descriptor_values.append(np.nan)
    return descriptor_values

# --- Compute descriptors ---
descriptor_data = []
for i, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="Computing descriptors"):
    smiles = row['canonical_SMILES']
    descriptor_data.append(compute_descriptors_debug(smiles))

# --- Combine into final DataFrame ---
desc_df = pd.DataFrame(descriptor_data, columns=descriptor_columns, index=df_valid.index)
final_df = pd.concat([df_valid.reset_index(drop=True), desc_df.reset_index(drop=True)], axis=1)


# --- Save final output ---
final_df.to_csv(output_csv_path, index=False)
print(f"\nDescriptor computation completed.")
print(f"Results saved to: {output_csv_path}")
