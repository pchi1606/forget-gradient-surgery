"""Fixed-step iterate runner (stream). Runs run_config for a list of configs
and saves the recs list to a JSON file for later assembly.

Usage: python3 run_fixedstep_stream.py <out_recs.json> <ds1:seed1> <ds2:seed2> ...
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiment_fixedstep as EF


def main():
    out_path = sys.argv[1]
    configs = []
    for tok in sys.argv[2:]:
        ds, seed = tok.split(":")
        configs.append((ds, int(seed)))
    recs = []
    for (ds, seed) in configs:
        try:
            recs.append(EF.run_config(ds, seed))
        except Exception as e:
            print(f"[fixedstep-stream] FAILED {ds} seed={seed}: {e!r}",
                  flush=True)
            raise
    with open(out_path, "w") as f:
        json.dump(recs, f, indent=1)
    print(f"[fixedstep-stream] wrote {out_path} with {len(recs)} recs")


if __name__ == "__main__":
    main()
