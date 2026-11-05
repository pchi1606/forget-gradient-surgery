#!/usr/bin/env python3
"""Fetch the Planetoid raw files the harness loader (code/data.py) expects.

The package does not redistribute the Planetoid datasets (see README,
"Data provenance, licenses, and split hygiene"). This script downloads
the public DGL mirror archives -- the same source the original
experiment tree used (code/data.py documents why) -- and unpacks them
into code/data/<dataset>/, the layout code/data.py reads.

    python3 scripts/fetch_data.py            # fetch all three datasets
    python3 scripts/fetch_data.py cora       # fetch one
"""
import io
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "code", "data")
MIRROR = "https://data.dgl.ai/dataset/{}.zip"


def fetch(name):
    dest = os.path.join(DATA_DIR, name)
    marker = os.path.join(dest, f"ind.{name}.graph")
    if os.path.exists(marker):
        print(f"{name}: already present, skipping")
        return
    url = MIRROR.format(name)
    print(f"{name}: downloading {url}")
    with urllib.request.urlopen(url) as r:
        blob = r.read()
    zf = zipfile.ZipFile(io.BytesIO(blob))
    os.makedirs(dest, exist_ok=True)
    zf.extractall(dest)  # the mirror archives are flat (ind.<name>.* files)
    if not os.path.exists(marker):
        raise RuntimeError(f"{name}: unpacked archive lacks {marker}")
    print(f"{name}: unpacked to {dest}")


if __name__ == "__main__":
    names = sys.argv[1:] or ["cora", "citeseer", "pubmed"]
    for nm in names:
        fetch(nm)
