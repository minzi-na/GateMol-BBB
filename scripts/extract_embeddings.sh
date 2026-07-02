#!/bin/bash
# ============================================================
# Extract the three pretrained-encoder embeddings in one pass:
#   scage1 (graph-level CLS), scage2 (atom-level mean-pool), mole.
#
# Ported from the study's data_process/generate_all_embeddings.sh.
# The RDKit modalities (ecfp/maccs/avalon/tt/rdkit) are NOT produced here;
# they are computed on the fly by gatemol_bbb.features.assemble_features.
#
# Upstream encoders are third-party and NOT vendored in this repo:
#   SCAGE  https://github.com/KazeDog/scage      (MIT)
#   MolE   https://github.com/recursionpharma/mole_public  (CC BY-NC 4.0)
# Install them separately and set the paths below.
#
# Usage:
#   SCAGE_DIR=... MOLE_DIR=... MOLE_PYTHON=... MOLE_CKPT=... \
#     bash extract_embeddings.sh <input_csv> <output_dir>
#
# Output CSVs (one row per SMILES, 'smiles' column + embedding columns):
#   <basename>_scage1.csv       (512-d, graph CLS)
#   <basename>_scage2_atom.csv  (512-d, atom mean-pool)
#   <basename>_mole.csv         (768-d)
# ============================================================
set -euo pipefail

INPUT_CSV="${1:?usage: extract_embeddings.sh <input_csv> <output_dir>}"
OUTPUT_DIR="${2:?usage: extract_embeddings.sh <input_csv> <output_dir>}"

# --- Locations of the third-party encoder checkouts (override via env) ------
SCAGE_DIR="${SCAGE_DIR:?set SCAGE_DIR to your SCAGE checkout}"
MOLE_DIR="${MOLE_DIR:?set MOLE_DIR to your mole_public checkout}"
MOLE_PYTHON="${MOLE_PYTHON:-python}"            # env with the MolE package
SCAGE_CONDA_ENV="${SCAGE_CONDA_ENV:-scage_new}" # conda env with the SCAGE package
SCAGE_WEIGHT="${SCAGE_WEIGHT:-weights/pretrain/pretrain.pth}"
MOLE_CKPT="${MOLE_CKPT:?set MOLE_CKPT to the MolE checkpoint (.ckpt)}"

BASENAME="$(basename "$INPUT_CSV" .csv)"
TXT_PATH="$OUTPUT_DIR/${BASENAME}.txt"
PKL_PATH="$OUTPUT_DIR/${BASENAME}.pkl"
LMDB_PATH="$OUTPUT_DIR/${BASENAME}.lmdb"          # prepare_data.py rewrites .lmdb -> .pkl
SCAGE1_OUT="$OUTPUT_DIR/${BASENAME}_scage1.csv"
SCAGE2_OUT="$OUTPUT_DIR/${BASENAME}_scage2_atom.csv"
MOLE_OUT="$OUTPUT_DIR/${BASENAME}_mole.csv"

mkdir -p "$OUTPUT_DIR"

# Step 1: CSV -> TXT (one SMILES per line; input CSV needs a 'smiles' column)
echo "[1/5] CSV -> TXT"
python3 -c "
import pandas as pd, sys
df = pd.read_csv('$INPUT_CSV')
open('$TXT_PATH','w').write('\n'.join(df['smiles'].astype(str).tolist()))
"

# Step 2: TXT -> PKL (SCAGE molecular-feature preprocessing)
echo "[2/5] TXT -> PKL (SCAGE preprocess)"
( cd "$SCAGE_DIR" && conda run -n "$SCAGE_CONDA_ENV" python prepare_data.py \
    --taskname pretrain --dataroot "$TXT_PATH" --datatarget "$LMDB_PATH" )

# Step 3: PKL -> scage1 (graph-level CLS embedding)
echo "[3/5] PKL -> scage1 (graph_embed.py)"
( cd "$SCAGE_DIR" && conda run -n "$SCAGE_CONDA_ENV" python graph_embed.py \
    --input_pkl "$PKL_PATH" --weight_path "$SCAGE_WEIGHT" --output_csv "$SCAGE1_OUT" )

# Step 4: PKL -> scage2 (atom-level mean-pooled embedding)
echo "[4/5] PKL -> scage2 (pool_atom_embed.py)"
( cd "$SCAGE_DIR" && conda run -n "$SCAGE_CONDA_ENV" python pool_atom_embed.py \
    --input_pkl "$PKL_PATH" --weight_path "$SCAGE_WEIGHT" --output_csv "$SCAGE2_OUT" )

# Step 5: CSV -> mole (MolE pretrained transformer embedding)
echo "[5/5] CSV -> mole (generate_mole_embed.py)"
( cd "$MOLE_DIR" && "$MOLE_PYTHON" generate_mole_embed.py \
    --input "$INPUT_CSV" --output "$MOLE_OUT" --ckpt "$MOLE_CKPT" )

echo "Done:"
echo "  scage1: $SCAGE1_OUT"
echo "  scage2: $SCAGE2_OUT"
echo "  mole  : $MOLE_OUT"
