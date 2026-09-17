"""
ceho_core.py
============
Core engine for CEHO-MSA (Clustering-based Elephant Herding Optimization
for Multiple Sequence Alignment).

The method combines two ideas from the literature:

  1. Graph/threshold-based sequence clustering with progressive profile
     merging, inspired by CPA-FL (Hajieghrari, 2026, Biochem. Biophys.
     Reports 45:102523).
  2. A population-based metaheuristic refinement stage inspired by the
     Elephant Herding Optimization (EHO) algorithm applied to MSA
     (Rios-Willars et al., 2026, Evolutionary Intelligence 19:81), using
     its Clan Updating Operator (CUO) and Separating Operator (SO).

Pipeline
--------
1.  Parse un-aligned FASTA sequences.
2.  Estimate pairwise distances with a fast, alignment-free k-mer
    profile (cosine distance) -- used only to *guide* clustering, not
    for the final alignment itself.
3.  Build a distance-thresholded graph (theta = mean - 0.5*std of all
    pairwise distances). Connected components become the initial
    clusters. If the threshold degenerates (one giant cluster or all
    singletons), fall back to an average-linkage agglomerative cut into
    k=3 clusters -- exactly the fallback used in CPA-FL.
4.  Align each cluster internally with a nearest-neighbour-ordered
    progressive alignment (profile-profile Needleman-Wunsch).
5.  Progressively merge the cluster profiles (closest-representative
    first) into one global alignment -- this is the "matriarch" seed
    solution.
6.  Refine that seed with an Elephant Herding Optimization loop:
    the population is a set of alignments (gap-column perturbations of
    the seed), grouped into clans. The Clan Updating Operator performs
    column-wise crossover of each clan member toward its clan's fittest
    member (matriarch); the Separating Operator replaces each clan's
    worst member with a freshly perturbed copy of the matriarch. Fitness
    is the Sum-of-Pairs (SP) score. Elitism guarantees the best
    alignment found is never lost.
7.  All-gap columns introduced during refinement are stripped from the
    final alignment.

Only the Python standard library is used so the script runs anywhere
without extra installs.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Scoring scheme (shared by the aligner AND the evaluation metrics, per the
# assignment's formulas 46-48)
# ---------------------------------------------------------------------------
MATCH = 2
MISMATCH = -1
GAP = -2


# ---------------------------------------------------------------------------
# FASTA I/O
# ---------------------------------------------------------------------------
def read_fasta(path: str) -> "dict[str, str]":
    """Read a (possibly multi-line) FASTA file into an ordered dict."""
    seqs: "dict[str, str]" = {}
    name = None
    buf: List[str] = []
    with open(path, "r") as fh:
        for raw in fh:
            line = raw.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line.strip())
        if name is not None:
            seqs[name] = "".join(buf)
    return seqs


def write_fasta(seqs: "dict[str, str]", path: str, width: int = 60) -> None:
    with open(path, "w") as fh:
        for name, seq in seqs.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), width):
                fh.write(seq[i:i + width] + "\n")


def is_dna(seqs: "dict[str, str]") -> bool:
    letters = set("".join(seqs.values()).upper())
    return letters.issubset(set("ACGTNU-"))


# ---------------------------------------------------------------------------
# k-mer profiles (alignment-free, used only to steer clustering / guide-tree)
# ---------------------------------------------------------------------------
def kmer_profile(seq: str, k: int = 3) -> "dict[str, int]":
    seq = seq.upper()
    prof: "dict[str, int]" = defaultdict(int)
    if len(seq) < k:
        prof[seq] += 1
        return prof
    for i in range(len(seq) - k + 1):
        prof[seq[i:i + k]] += 1
    return prof


def kmer_distance(p1: "dict[str, int]", p2: "dict[str, int]") -> float:
    keys = set(p1) | set(p2)
    dot = sum(p1.get(x, 0) * p2.get(x, 0) for x in keys)
    n1 = math.sqrt(sum(v * v for v in p1.values()))
    n2 = math.sqrt(sum(v * v for v in p2.values()))
    if n1 == 0 or n2 == 0:
        return 1.0
    cos_sim = dot / (n1 * n2)
    return max(0.0, 1.0 - cos_sim)


# ---------------------------------------------------------------------------
# Profile-profile Needleman-Wunsch alignment
# ---------------------------------------------------------------------------
def _freq(col: Sequence[str]) -> "dict[str, int]":
    """Plain-dict character frequency count (faster than collections.Counter
    for this hot path -- no ABC/instance-check overhead)."""
    d: "dict[str, int]" = {}
    for ch in col:
        d[ch] = d.get(ch, 0) + 1
    return d


def _col_score_freq(fa: "dict[str, int]", fb: "dict[str, int]") -> int:
    """Score of aligning a column with character-frequency dict fa against
    one with frequency dict fb: sum over every cross pair."""
    total = 0
    for a, na in fa.items():
        for b, nb in fb.items():
            if a == "-" or b == "-":
                total += GAP * na * nb
            elif a == b:
                total += MATCH * na * nb
            else:
                total += MISMATCH * na * nb
    return total


def profile_nw(profileA: List[str], profileB: List[str]) -> List[str]:
    """Global (Needleman-Wunsch) alignment of two profiles (lists of
    equal-length aligned strings). Returns the merged list of aligned
    strings, profileA rows first, then profileB rows, in original order."""
    nA, nB = len(profileA[0]), len(profileB[0])
    dA, dB = len(profileA), len(profileB)
    colsA = [[s[i] for s in profileA] for i in range(nA)]
    colsB = [[s[j] for s in profileB] for j in range(nB)]
    # Precompute each column's character-frequency dict once -- this is the
    # dominant cost driver, so caching it here (instead of rebuilding a
    # Counter for every DP cell) is what makes the DP tractable for
    # realistic protein-family sizes.
    freqsA = [_freq(c) for c in colsA]
    freqsB = [_freq(c) for c in colsB]

    gap_cost = GAP * dA * dB  # constant: aligning a whole column against gaps

    dp = [[0] * (nB + 1) for _ in range(nA + 1)]
    tb = [[0] * (nB + 1) for _ in range(nA + 1)]  # 1=diag 2=up(consume A) 3=left(consume B)

    for i in range(1, nA + 1):
        dp[i][0] = dp[i - 1][0] + gap_cost
        tb[i][0] = 2
    for j in range(1, nB + 1):
        dp[0][j] = dp[0][j - 1] + gap_cost
        tb[0][j] = 3

    for i in range(1, nA + 1):
        dp_i = dp[i]
        dp_im1 = dp[i - 1]
        fa = freqsA[i - 1]
        for j in range(1, nB + 1):
            diag = dp_im1[j - 1] + _col_score_freq(fa, freqsB[j - 1])
            up = dp_im1[j] + gap_cost
            left = dp_i[j - 1] + gap_cost
            best = diag
            move = 1
            if up > best:
                best, move = up, 2
            if left > best:
                best, move = left, 3
            dp_i[j] = best
            tb[i][j] = move

    # traceback
    i, j = nA, nB
    newA = [[] for _ in range(dA)]
    newB = [[] for _ in range(dB)]
    while i > 0 or j > 0:
        move = tb[i][j]
        if move == 1:
            for r in range(dA):
                newA[r].append(colsA[i - 1][r])
            for r in range(dB):
                newB[r].append(colsB[j - 1][r])
            i -= 1
            j -= 1
        elif move == 2:
            for r in range(dA):
                newA[r].append(colsA[i - 1][r])
            for r in range(dB):
                newB[r].append("-")
            i -= 1
        else:
            for r in range(dA):
                newA[r].append("-")
            for r in range(dB):
                newB[r].append(colsB[j - 1][r])
            j -= 1

    for r in range(dA):
        newA[r].reverse()
    for r in range(dB):
        newB[r].reverse()

    return ["".join(r) for r in newA] + ["".join(r) for r in newB]


# ---------------------------------------------------------------------------
# Clustering (graph-threshold, with agglomerative fallback -- CPA-FL style)
# ---------------------------------------------------------------------------
def _agglomerative_k(indices: List[int], dist: List[List[float]], k: int) -> List[List[int]]:
    clusters = [[idx] for idx in indices]

    def cluster_dist(c1: List[int], c2: List[int]) -> float:
        return sum(dist[a][b] for a in c1 for b in c2) / (len(c1) * len(c2))

    while len(clusters) > k:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = cluster_dist(clusters[a], clusters[b])
                if best is None or d < best[0]:
                    best = (d, a, b)
        _, a, b = best
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    return clusters


def cluster_sequences(n: int, dist: List[List[float]]) -> List[List[int]]:
    if n <= 2:
        return [[i] for i in range(n)]

    pair_dists = [dist[i][j] for i in range(n) for j in range(i + 1, n)]
    mu = sum(pair_dists) / len(pair_dists)
    var = sum((d - mu) ** 2 for d in pair_dists) / len(pair_dists)
    sigma = math.sqrt(var)
    theta = max(0.0, mu - 0.5 * sigma)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if dist[i][j] <= theta:
                union(i, j)

    groups: "dict[int, List[int]]" = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    clusters = list(groups.values())

    # Fallback (mirrors CPA-FL's k=3 fallback for degenerate thresholding)
    if len(clusters) == 1 or len(clusters) == n:
        k = max(2, min(3, n))
        clusters = _agglomerative_k(list(range(n)), dist, k)

    return clusters


def _nn_chain_order(indices: List[int], dist: List[List[float]]) -> List[int]:
    remaining = list(indices)
    order = [remaining.pop(0)]
    while remaining:
        last = order[-1]
        nxt = min(remaining, key=lambda x: dist[last][x])
        order.append(nxt)
        remaining.remove(nxt)
    return order


def progressive_align(seq_dict: "dict[str, str]", order: List[str]) -> "dict[str, str]":
    if len(order) == 1:
        return {order[0]: seq_dict[order[0]]}
    profile_names = [order[0]]
    profile_seqs = [seq_dict[order[0]]]
    for name in order[1:]:
        merged = profile_nw(profile_seqs, [seq_dict[name]])
        profile_seqs = merged
        profile_names.append(name)
    return dict(zip(profile_names, profile_seqs))


def merge_cluster_profiles(cluster_profiles: List["dict[str, str]"],
                            seq_dict: "dict[str, str]", k: int = 3) -> "dict[str, str]":
    profiles = list(cluster_profiles)
    reps = [kmer_profile(seq_dict[next(iter(cp))], k) for cp in profiles]

    while len(profiles) > 1:
        best = None
        for a in range(len(profiles)):
            for b in range(a + 1, len(profiles)):
                d = kmer_distance(reps[a], reps[b])
                if best is None or d < best[0]:
                    best = (d, a, b)
        _, a, b = best
        namesA, namesB = list(profiles[a].keys()), list(profiles[b].keys())
        seqsA = [profiles[a][nm] for nm in namesA]
        seqsB = [profiles[b][nm] for nm in namesB]
        merged_seqs = profile_nw(seqsA, seqsB)
        profiles[a] = dict(zip(namesA + namesB, merged_seqs))
        del profiles[b]
        del reps[b]
    return profiles[0]


# ---------------------------------------------------------------------------
# Elephant Herding Optimization refinement stage
# ---------------------------------------------------------------------------
def _sp_score(seqs: List[str]) -> int:
    """Sum-of-Pairs score, per formula (46): sum over all columns and all
    unordered sequence pairs of the pairwise character score."""
    L = len(seqs[0])
    total = 0
    for col in range(L):
        cnt = _freq([s[col] for s in seqs])
        items = list(cnt.items())
        for idx1 in range(len(items)):
            a, na = items[idx1]
            for idx2 in range(idx1, len(items)):
                b, nb = items[idx2]
                pairs = na * (na - 1) // 2 if idx1 == idx2 else na * nb
                if pairs == 0:
                    continue
                if a == "-" or b == "-":
                    total += GAP * pairs
                elif a == b:
                    total += MATCH * pairs
                else:
                    total += MISMATCH * pairs
    return total


def _remove_allgap_columns(seqs: List[str]) -> List[str]:
    L = len(seqs[0])
    keep = [c for c in range(L) if any(s[c] != "-" for s in seqs)]
    return ["".join(s[c] for c in keep) for s in seqs]


# ---- Content-safe "elephant" representation -------------------------------
# Each elephant in the population is NOT a raw character matrix (crossing
# raw characters between two differently-gapped alignments can silently
# duplicate or drop residues -- verified empirically during testing). Instead
# every elephant is an *insertion vector* over the fixed core alignment
# produced by the clustering/progressive stage: insert_counts[i] is the
# number of extra all-gap columns spliced in immediately before core column
# i (insert_counts[L0] = extra gap columns appended after the last core
# column). Because the core columns themselves are never touched, every
# elephant -- after crossover or mutation -- reconstructs *exactly* the
# original input sequences with only gap characters added. This keeps the
# metaheuristic search valid by construction while still letting it explore
# alternative gap placements to improve the SP score.
def _materialize(core: List[str], insert_counts: List[int]) -> List[str]:
    depth = len(core)
    L0 = len(core[0])
    cols: List[List[str]] = []
    for i in range(L0):
        for _ in range(insert_counts[i]):
            cols.append(["-"] * depth)
        cols.append([core[r][i] for r in range(depth)])
    for _ in range(insert_counts[L0]):
        cols.append(["-"] * depth)
    return ["".join(cols[c][r] for c in range(len(cols))) for r in range(depth)]


def _random_insert_vector(L0: int, p: float, rng: random.Random) -> List[int]:
    return [1 if rng.random() < p else 0 for _ in range(L0 + 1)]


def elephant_herding_refine(seed_alignment: "dict[str, str]",
                             iterations: int = 30,
                             population_size: int = 12,
                             num_clans: int = 3,
                             crossover_p: float = 0.6,
                             gap_insert_p: float = 0.08,
                             seed: int = 42) -> "dict[str, str]":
    rng = random.Random(seed)
    names = list(seed_alignment.keys())
    core = [seed_alignment[n] for n in names]
    L0 = len(core[0])

    def fitness(ic: List[int]) -> int:
        return _sp_score(_materialize(core, ic))

    # Population = the unperturbed seed (all-zero insertion vector) plus
    # randomly perturbed variants.
    population: List[List[int]] = [[0] * (L0 + 1)] + [
        _random_insert_vector(L0, gap_insert_p, rng) for _ in range(population_size - 1)
    ]
    fits = [fitness(ic) for ic in population]

    best_idx = max(range(len(population)), key=lambda i: fits[i])
    best_ic = list(population[best_idx])
    best_fit = fits[best_idx]

    for _ in range(iterations):
        # Rank population and split round-robin into clans.
        order = sorted(range(len(population)), key=lambda i: fits[i], reverse=True)
        clans = [order[c::num_clans] for c in range(num_clans)]

        for clan in clans:
            if not clan:
                continue
            matriarch_idx = max(clan, key=lambda i: fits[i])
            matriarch_ic = population[matriarch_idx]

            # Clan Updating Operator (CUO): crossover each non-matriarch
            # member's insertion vector with the matriarch's at a random
            # cut point. Always content-safe: both vectors describe gap
            # placement only, over the same fixed core columns.
            for idx in clan:
                if idx == matriarch_idx:
                    continue
                c = rng.randint(0, L0)
                child_ic = population[idx][:c] + matriarch_ic[c:]
                child_fit = fitness(child_ic)
                if rng.random() < crossover_p or child_fit > fits[idx]:
                    population[idx] = child_ic
                    fits[idx] = child_fit

            # Separating Operator (SO): the clan's worst member is replaced
            # by a freshly perturbed variant of the matriarch's pattern
            # (bit-flip mutation with probability gap_insert_p per slot).
            worst_idx = min(clan, key=lambda i: fits[i])
            new_ic = [1 - v if rng.random() < gap_insert_p else v for v in matriarch_ic]
            new_fit = fitness(new_ic)
            population[worst_idx] = new_ic
            fits[worst_idx] = new_fit

        # Elitism: never lose the best alignment found so far.
        cur_best_idx = max(range(len(population)), key=lambda i: fits[i])
        if fits[cur_best_idx] > best_fit:
            best_fit = fits[cur_best_idx]
            best_ic = list(population[cur_best_idx])

    final_seqs = _remove_allgap_columns(_materialize(core, best_ic))
    return dict(zip(names, final_seqs))


# ---------------------------------------------------------------------------
# Full CEHO-MSA pipeline
# ---------------------------------------------------------------------------
def ceho_align(seq_dict: "dict[str, str]",
               k: int = 3,
               eho_iterations: int = 30,
               eho_population: int = 12,
               num_clans: int = 3,
               seed: int = 42) -> "dict[str, str]":
    names = list(seq_dict.keys())
    n = len(names)

    if n == 1:
        return dict(seq_dict)
    if n == 2:
        aligned = profile_nw([seq_dict[names[0]]], [seq_dict[names[1]]])
        seed_aln = dict(zip(names, aligned))
        return elephant_herding_refine(seed_aln, iterations=eho_iterations,
                                        population_size=eho_population,
                                        num_clans=min(num_clans, 2), seed=seed)

    profiles = [kmer_profile(seq_dict[nm], k) for nm in names]
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = kmer_distance(profiles[i], profiles[j])
            dist[i][j] = dist[j][i] = d

    clusters_idx = cluster_sequences(n, dist)

    cluster_profiles: List["dict[str, str]"] = []
    for cl in clusters_idx:
        if len(cl) == 1:
            nm = names[cl[0]]
            cluster_profiles.append({nm: seq_dict[nm]})
            continue
        order_idx = _nn_chain_order(cl, dist)
        order_names = [names[i] for i in order_idx]
        cluster_profiles.append(progressive_align(seq_dict, order_names))

    seed_alignment = merge_cluster_profiles(cluster_profiles, seq_dict, k=k)

    clans = max(1, min(num_clans, len(clusters_idx), eho_population))
    refined = elephant_herding_refine(seed_alignment,
                                       iterations=eho_iterations,
                                       population_size=eho_population,
                                       num_clans=clans,
                                       seed=seed)
    return refined


# ---------------------------------------------------------------------------
# Evaluation metrics (formulas 46-48)
# ---------------------------------------------------------------------------
def compute_sp(aligned: "dict[str, str]") -> int:
    return _sp_score(list(aligned.values()))


def compute_cs(aligned: "dict[str, str]") -> float:
    seqs = list(aligned.values())
    L = len(seqs[0])
    if L == 0:
        return 0.0
    conserved = 0
    for col in range(L):
        chars = {s[col] for s in seqs}
        if len(chars) == 1 and "-" not in chars:
            conserved += 1
    return conserved / L


def compute_gap_pct(aligned: "dict[str, str]") -> float:
    seqs = list(aligned.values())
    N = len(seqs)
    L = len(seqs[0]) if seqs else 0
    if N == 0 or L == 0:
        return 0.0
    gaps = sum(s.count("-") for s in seqs)
    return gaps / (N * L) * 100.0
