#!/usr/bin/env bash
# Fast-track LM queue (seed 42): main methods, matched effective L0, ablations, then evaluation.
# Run from the project root after the BatchTopK and Matryoshka baselines exist.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY=python
export HF_HOME="$PWD/.hf_cache" NLTK_DATA="$PWD/.hf_cache/nltk_data" PYTHONUNBUFFERED=1
mkdir -p data/logs

chain_a() {
  $PY scripts/train_lm_saes.py --methods "DR-SAE" "Tree prior (m=1)" "Post-hoc DR" "AG-SAE-lite" --seeds 42
  # Matched effective L0: lower native k so DR-SAE's effective L0 equals the baselines' k.
  KM=$($PY -c "import json,glob; r=json.load(open(glob.glob('results/lm/L3_w2048_k32_dr-sae_s42.json')[0])); print(max(1, round(32 * r['native_l0'] / r['eff_l0'])))")
  echo "matched k = $KM"
  $PY scripts/train_lm_saes.py --methods "DR-SAE (matched eff. L0)" --seeds 42 --k "$KM"
}

chain_b() {
  $PY scripts/train_lm_saes.py --methods BatchTopK --seeds 42
  $PY scripts/train_lm_saes.py --methods "DR-SAE" --seeds 42 --set lambda_ortho=0.0 --tag __noortho
  $PY scripts/train_lm_saes.py --methods "DR-SAE" --seeds 42 --set use_frequency_order=False --tag __bankorder
  $PY scripts/train_lm_saes.py --methods "DR-SAE" --seeds 42 --set hop_depth=1 --tag __H1
}

chain_a > data/logs/phase3_chain_a.log 2>&1 &
A=$!
chain_b > data/logs/phase3_chain_b.log 2>&1 &
B=$!
wait $A
wait $B
$PY scripts/run_lm_experiments.py > data/logs/phase3_eval.log 2>&1
echo "queue done"
