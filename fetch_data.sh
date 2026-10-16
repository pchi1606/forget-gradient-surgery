#!/usr/bin/env bash
# Fetch the Planetoid datasets (Cora / CiteSeer / PubMed) from the DGL
# mirror. Skipped for any dataset already present (they are bundled in
# data/ in this package; this script is a fallback / refresh path).
set -euo pipefail
cd "$(dirname "$0")"
for ds in cora citeseer pubmed; do
  if [ -f "data/$ds/ind.$ds.graph" ]; then
    echo "$ds: already present, skipping download"
  else
    echo "$ds: downloading..."
    mkdir -p data
    curl -L -o /tmp/$ds.zip "https://data.dgl.ai/dataset/$ds.zip"
    unzip -o -q /tmp/$ds.zip -d data/
    rm -f /tmp/$ds.zip
  fi
done
echo "data ready:"
ls data
