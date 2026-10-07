"""Recompute the encoder-sweep rank statistics (C0 vs LAP-L gain) from the per-seed JSON.

    python scripts/verify_sweep_statistics.py [--output stats.json]

The primary set is the eight records with anchor=false, tested by enumerating all 8!
permutations; the combined comparison includes all 13 records and uses a seeded PCG64
Monte Carlo permutation test, reset to the same seed at each endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {"100k": ROOT / "results/sweep_100k.json",
          "400k": ROOT / "results/sweep_400k.json"}


def ranks(values):
    """One-based average ranks, including ties."""
    values = np.asarray(values, dtype=float)
    unique, inverse, counts = np.unique(values, return_inverse=True,
                                        return_counts=True)
    del unique
    ends = np.cumsum(counts)
    return ((ends - counts + 1 + ends) / 2.0)[inverse]


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def partial(a, b, control):
    ab, ac, bc = corr(a, b), corr(a, control), corr(b, control)
    return float((ab - ac * bc) / np.sqrt((1 - ac * ac) * (1 - bc * bc)))


def summarize(records):
    c0 = np.array([row["c0_raw"] for row in records])
    delta = c0 - np.array([row["c0_lapl"] for row in records])
    raw = np.array([np.mean(row["raw"]) for row in records])
    gains = np.array([np.mean(np.array(row["raw"]) - row["lapl"])
                      for row in records])
    rc, rd, rr, rg = map(ranks, (c0, delta, raw, gains))
    return {
        "n": len(records),
        "names": [row["name"] for row in records],
        "gains": dict(zip((row["name"] for row in records), gains.tolist())),
        "spearman_gain_c0": corr(rg, rc),
        "spearman_gain_delta_c0": corr(rg, rd),
        "spearman_gain_raw_fid": corr(rg, rr),
        "spearman_c0_raw_fid": corr(rc, rr),
        "partial_rank_gain_c0_given_raw_fid": partial(rg, rc, rr),
        "partial_rank_gain_raw_fid_given_c0": partial(rg, rr, rc),
    }, rc, rg


def exact_permutation_test(c0_ranks, gain_ranks):
    """Enumerate label permutations, counting ties in both tails."""
    x = c0_ranks - np.mean(c0_ranks)
    y = gain_ranks - np.mean(gain_ranks)
    observed = float(np.dot(x, y))
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    draws = np.array(list(itertools.permutations(y)))
    dots = np.sum(draws * x, axis=1)
    total = math.factorial(len(y))
    one_sided = int(np.count_nonzero(dots >= observed - 1e-12))
    two_sided = int(np.count_nonzero(np.abs(dots) >= abs(observed) - 1e-12))
    return {
        "method": "exact enumeration",
        "permutations": total,
        "null": "uniformly permute gain ranks across encoder labels; fix C0 ranks",
        "positive_alternative": "Spearman rho > 0",
        "observed_rho": observed / denominator,
        "one_sided_exceedances": one_sided,
        "one_sided_p": one_sided / total,
        "two_sided_exceedances": two_sided,
        "two_sided_p": two_sided / total,
        "p_formula": "exceedances / n!; absolute values for two-sided",
    }


def permutation_test(c0_ranks, gain_ranks, permutations, seed):
    x = c0_ranks - np.mean(c0_ranks)
    y = gain_ranks - np.mean(gain_ranks)
    observed = float(np.dot(x, y))
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    rng = np.random.Generator(np.random.PCG64(seed))
    one_sided = two_sided = 0
    for start in range(0, permutations, 10000):
        count = min(10000, permutations - start)
        draws = rng.permuted(np.broadcast_to(y, (count, len(y))), axis=1)
        dots = np.sum(draws * x, axis=1)
        one_sided += int(np.count_nonzero(dots >= observed - 1e-12))
        two_sided += int(np.count_nonzero(np.abs(dots) >= abs(observed) - 1e-12))
    return {
        "seed": seed,
        "bit_generator": "PCG64",
        "permutations": permutations,
        "null": "uniformly permute gain ranks across encoder labels; fix C0 ranks",
        "positive_alternative": "Spearman rho > 0",
        "observed_rho": observed / denominator,
        "one_sided_exceedances": one_sided,
        "one_sided_p": (one_sided + 1) / (permutations + 1),
        "two_sided_exceedances": two_sided,
        "two_sided_p": (two_sided + 1) / (permutations + 1),
        "p_formula": "(number of permuted statistics >= observed + 1) / (B + 1); absolute values for two-sided",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--output", type=Path, help="optional JSON file for the full result")
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error("--permutations must be positive")
    result = {"numpy_version": np.__version__, "endpoints": {}}
    combined_gains = {}
    for endpoint, path in INPUTS.items():
        records = json.loads(path.read_text())["encoders"]
        assert len(records) == 13 and sum(not row["anchor"] for row in records) == 8
        combined, rc, rg = summarize(records)
        added, added_rc, added_rg = summarize([row for row in records if not row["anchor"]])
        result["endpoints"][endpoint] = {
            "input": str(path.relative_to(ROOT)),
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "primary_eight_added": added,
            "primary_permutation_test": exact_permutation_test(added_rc, added_rg),
            "amended_thirteen_combined": combined,
            "combined_permutation_test": permutation_test(rc, rg, args.permutations, args.seed),
        }
        combined_gains[endpoint] = combined["gains"]
    names = sorted(combined_gains["100k"])
    assert names == sorted(combined_gains["400k"])
    result["gain_rank_agreement_100k_400k"] = corr(
        ranks([combined_gains["100k"][name] for name in names]),
        ranks([combined_gains["400k"][name] for name in names]))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(args.output)
    for endpoint, row in result["endpoints"].items():
        for key in ("primary_eight_added", "amended_thirteen_combined"):
            stats = row[key]
            print(f"{endpoint} {key}: rho={stats['spearman_gain_c0']:.9f}, "
                  f"partial={stats['partial_rank_gain_c0_given_raw_fid']:.9f}")
        for key in ("primary_permutation_test", "combined_permutation_test"):
            test = row[key]
            print(f"{endpoint} {key}: one-sided p={test['one_sided_p']:.9g}, "
                  f"two-sided p={test['two_sided_p']:.9g}")


if __name__ == "__main__":
    main()
