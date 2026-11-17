from typing import Any, Dict, List


OBJECTIVE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "accuracy": {"accuracy": 0.85, "speed": 0.08, "memory": 0.07},
    "speed": {"accuracy": 0.35, "speed": 0.45, "memory": 0.20},
    "balanced": {"accuracy": 0.55, "speed": 0.25, "memory": 0.20},
}


def _dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """
    Returns True if architecture `a` Pareto-dominates `b` on:
      - Maximize accuracy
      - Minimize inference_latency_ms
      - Minimize model_size_kb (proxy for memory/storage footprint)
    """
    acc_a = float(a.get("accuracy") or 0.0)
    acc_b = float(b.get("accuracy") or 0.0)
    lat_a = float(a.get("inference_latency_ms") or 1e6)
    lat_b = float(b.get("inference_latency_ms") or 1e6)
    mem_a = float(a.get("model_size_kb") or 1e6)
    mem_b = float(b.get("model_size_kb") or 1e6)

    not_worse = (acc_a >= acc_b) and (lat_a <= lat_b) and (mem_a <= mem_b)
    strictly_better = (acc_a > acc_b) or (lat_a < lat_b) or (mem_a < mem_b)
    return not_worse and strictly_better


def fast_non_dominated_sort(candidates: List[Dict[str, Any]]) -> List[List[int]]:
    """NSGA-II fast non-dominated sorting returning lists of indices per Pareto front."""
    n = len(candidates)
    if n == 0:
        return []

    domination_sets: List[List[int]] = [[] for _ in range(n)]
    dominated_counts: List[int] = [0] * n
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(candidates[p], candidates[q]):
                domination_sets[p].append(q)
            elif _dominates(candidates[q], candidates[p]):
                dominated_counts[p] += 1
        if dominated_counts[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in domination_sets[p]:
                dominated_counts[q] -= 1
                if dominated_counts[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1

    return fronts


def compute_crowding_distance(candidates: List[Dict[str, Any]], front: List[int]) -> Dict[int, float]:
    if len(front) <= 2:
        return {idx: 1.0 for idx in front}

    distances: Dict[int, float] = {idx: 0.0 for idx in front}
    objectives = [
        lambda c: float(c.get("accuracy") or 0.0),
        lambda c: float(c.get("inference_latency_ms") or 0.0),
        lambda c: float(c.get("model_size_kb") or 0.0),
    ]

    for obj_fn in objectives:
        sorted_front = sorted(front, key=lambda idx: obj_fn(candidates[idx]))
        min_val = obj_fn(candidates[sorted_front[0]])
        max_val = obj_fn(candidates[sorted_front[-1]])
        span = max_val - min_val
        distances[sorted_front[0]] += 1.0
        distances[sorted_front[-1]] += 1.0
        if span <= 1e-12:
            continue
        for pos in range(1, len(sorted_front) - 1):
            prev_val = obj_fn(candidates[sorted_front[pos - 1]])
            next_val = obj_fn(candidates[sorted_front[pos + 1]])
            distances[sorted_front[pos]] += (next_val - prev_val) / span

    return {k: round(v, 4) for k, v in distances.items()}


def compute_fitness_and_pareto(
    candidates: List[Dict[str, Any]], optimization_mode: str = "balanced"
) -> List[Dict[str, Any]]:
    """
    Computes Pareto rank (1 = Pareto-optimal frontier), crowding distance, and a
    mode-aware multi-objective fitness score (0..100) for all evaluated candidates.
    Mutates and returns the candidate dicts.
    """
    if not candidates:
        return candidates

    weights = OBJECTIVE_WEIGHTS.get(optimization_mode, OBJECTIVE_WEIGHTS["balanced"])
    fronts = fast_non_dominated_sort(candidates)

    for rank_idx, front in enumerate(fronts, start=1):
        crowding = compute_crowding_distance(candidates, front)
        for idx in front:
            candidates[idx]["pareto_rank"] = rank_idx
            candidates[idx]["is_pareto_optimal"] = rank_idx == 1
            candidates[idx]["crowding_distance"] = crowding.get(idx, 0.0)

    # Compute normalized efficiency metrics relative to population bounds with smooth anchors
    latencies = [float(c.get("inference_latency_ms") or 0.1) for c in candidates]
    sizes = [float(c.get("model_size_kb") or 10.0) for c in candidates]

    min_lat, max_lat = min(latencies), max(latencies)
    min_size, max_size = min(sizes), max(sizes)

    for c in candidates:
        acc = max(0.0, min(1.0, float(c.get("accuracy") or 0.0)))
        lat = float(c.get("inference_latency_ms") or 0.1)
        size = float(c.get("model_size_kb") or 10.0)

        # Normalized speed score in [0, 1] (lower latency = higher score)
        if max_lat - min_lat > 1e-6:
            speed_rel = 1.0 - (lat - min_lat) / (max_lat - min_lat)
        else:
            speed_rel = 1.0 / (1.0 + lat)
        # Blend absolute latency anchor so single-candidate scores remain meaningful
        speed_abs = 1.0 / (1.0 + lat / 1.5)
        speed_score = 0.55 * speed_rel + 0.45 * speed_abs

        # Normalized compactness score in [0, 1] (smaller model = higher score)
        if max_size - min_size > 1e-6:
            mem_rel = 1.0 - (size - min_size) / (max_size - min_size)
        else:
            mem_rel = 1.0 / (1.0 + size / 500.0)
        mem_abs = 1.0 / (1.0 + size / 800.0)
        mem_score = 0.55 * mem_rel + 0.45 * mem_abs

        # Small Pareto frontier bonus in balanced mode
        pareto_bonus = 0.025 if (c.get("is_pareto_optimal") and optimization_mode == "balanced") else 0.0

        raw_fitness = (
            weights["accuracy"] * acc
            + weights["speed"] * speed_score
            + weights["memory"] * mem_score
            + pareto_bonus
        )
        c["fitness_score"] = round(min(100.0, max(0.0, raw_fitness * 100.0)), 3)

    return candidates
