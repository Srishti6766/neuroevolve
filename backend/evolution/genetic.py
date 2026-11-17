import random
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from backend.engine.search_space import SearchSpace, architecture_signature, clone_arch_config
from backend.reinforcement_learning.rl_selector import RLArchitectureController


def initialize_population(
    search_space: SearchSpace,
    population_size: int,
    batch_size: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Generate a diverse initial population of candidate architecture specs."""
    candidates = []
    seen = set()
    attempts = 0
    while len(candidates) < population_size and attempts < population_size * 12:
        attempts += 1
        cfg = search_space.sample_architecture(rng, batch_size=batch_size)
        sig = architecture_signature(cfg)
        if sig not in seen or attempts > population_size * 8:
            seen.add(sig)
            candidates.append(
                {
                    "config": cfg,
                    "origin": "random",
                    "parent_ids": [],
                    "operation_details": "Initial search-space sampling",
                }
            )
    return candidates


def tournament_select(
    evaluated_population: List[Dict[str, Any]],
    rng: random.Random,
    k: int = 3,
) -> Dict[str, Any]:
    """Select a parent via tournament selection prioritizing fitness and Pareto rank."""
    pool_size = min(k, len(evaluated_population))
    contenders = rng.sample(evaluated_population, pool_size)
    return max(
        contenders,
        key=lambda c: (
            float(c.get("fitness_score") or 0.0),
            -(int(c.get("pareto_rank") or 99)),
            float(c.get("accuracy") or 0.0),
        ),
    )


def crossover_architectures(
    parent_a: Dict[str, Any],
    parent_b: Dict[str, Any],
    search_space: SearchSpace,
    rng: random.Random,
) -> Tuple[Dict[str, Any], str]:
    """
    Combine layers and hyperparameters from two parent architectures via structural crossover.
    """
    cfg_a = clone_arch_config(parent_a["config"])
    cfg_b = clone_arch_config(parent_b["config"])

    layers_a = cfg_a["hidden_layers"]
    layers_b = cfg_b["hidden_layers"]

    split_a = rng.randint(1, len(layers_a)) if len(layers_a) > 1 else 1
    split_b = rng.randint(0, len(layers_b) - 1) if len(layers_b) > 1 else 0

    child_layers = layers_a[:split_a] + layers_b[split_b:]
    # Enforce search space layer bounds
    if len(child_layers) > search_space.max_layers:
        child_layers = child_layers[: search_space.max_layers]
    while len(child_layers) < search_space.min_layers:
        child_layers.append(search_space.sample_layer(rng))

    opt = cfg_a["optimizer"] if rng.random() < 0.5 else cfg_b["optimizer"]
    lr = cfg_a["learning_rate"] if rng.random() < 0.5 else cfg_b["learning_rate"]

    child_cfg = {
        "hidden_layers": child_layers,
        "optimizer": opt,
        "learning_rate": lr,
        "batch_size": cfg_a.get("batch_size", 64),
    }
    desc = (
        f"Crossover({parent_a['name']}[:{split_a}] + {parent_b['name']}[{split_b}:], "
        f"opt={opt}, lr={lr})"
    )
    return child_cfg, desc


def mutate_architecture(
    base_config: Dict[str, Any],
    search_space: SearchSpace,
    rng: random.Random,
    num_mutations: int = 1,
) -> Tuple[Dict[str, Any], str]:
    """
    Apply 1 or more structural / hyperparameter mutations to a cloned architecture config.
    """
    cfg = clone_arch_config(base_config)
    layers = cfg["hidden_layers"]
    ops_applied: List[str] = []

    for _ in range(max(1, num_mutations)):
        possible_ops = ["units", "activation", "dropout", "batch_norm", "optimizer", "learning_rate"]
        if len(layers) < search_space.max_layers:
            possible_ops.append("add_layer")
        if len(layers) > search_space.min_layers:
            possible_ops.append("remove_layer")

        op = rng.choice(possible_ops)

        if op == "add_layer":
            new_layer = search_space.sample_layer(rng)
            pos = rng.randint(0, len(layers))
            layers.insert(pos, new_layer)
            ops_applied.append(f"+layer@{pos + 1}({new_layer['units']} {new_layer['activation']})")

        elif op == "remove_layer":
            pos = rng.randrange(len(layers))
            removed = layers.pop(pos)
            ops_applied.append(f"-layer@{pos + 1}({removed['units']})")

        elif op == "units":
            idx = rng.randrange(len(layers))
            old_u = layers[idx]["units"]
            alts = [u for u in search_space.neuron_choices if u != old_u] or search_space.neuron_choices
            new_u = rng.choice(alts)
            layers[idx]["units"] = new_u
            ops_applied.append(f"L{idx + 1}.units:{old_u}->{new_u}")

        elif op == "activation":
            idx = rng.randrange(len(layers))
            old_a = layers[idx]["activation"]
            alts = [a for a in search_space.activations if a != old_a] or search_space.activations
            new_a = rng.choice(alts)
            layers[idx]["activation"] = new_a
            ops_applied.append(f"L{idx + 1}.act:{old_a}->{new_a}")

        elif op == "dropout":
            idx = rng.randrange(len(layers))
            old_d = layers[idx]["dropout"]
            alts = [d for d in search_space.dropout_choices if d != old_d] or search_space.dropout_choices
            new_d = rng.choice(alts)
            layers[idx]["dropout"] = new_d
            ops_applied.append(f"L{idx + 1}.drop:{old_d}->{new_d}")

        elif op == "batch_norm":
            idx = rng.randrange(len(layers))
            old_bn = bool(layers[idx].get("batch_norm", False))
            if len(search_space.batch_norm_choices) > 1:
                layers[idx]["batch_norm"] = not old_bn
                ops_applied.append(f"L{idx + 1}.bn:{old_bn}->{not old_bn}")
            else:
                # Fallback to units mutation
                new_u = rng.choice(search_space.neuron_choices)
                layers[idx]["units"] = new_u
                ops_applied.append(f"L{idx + 1}.units->{new_u}")

        elif op == "optimizer":
            old_o = cfg["optimizer"]
            alts = [o for o in search_space.optimizers if o != old_o] or search_space.optimizers
            new_o = rng.choice(alts)
            cfg["optimizer"] = new_o
            ops_applied.append(f"opt:{old_o}->{new_o}")

        elif op == "learning_rate":
            old_lr = cfg["learning_rate"]
            alts = [lr for lr in search_space.learning_rates if lr != old_lr] or search_space.learning_rates
            new_lr = rng.choice(alts)
            cfg["learning_rate"] = new_lr
            ops_applied.append(f"lr:{old_lr}->{new_lr}")

    return cfg, "Mutate(" + ", ".join(ops_applied) + ")"


def create_next_generation(
    previous_generation: List[Dict[str, Any]],
    search_space: SearchSpace,
    population_size: int,
    batch_size: int,
    strategy: str,
    rng: random.Random,
    rl_controller: Optional[RLArchitectureController] = None,
    elitism_count: int = 1,
) -> List[Dict[str, Any]]:
    """
    Create the next generation of candidate architectures using:
      - 'genetic': Elitism + Tournament Selection + Crossover + Mutation
      - 'hybrid_rl': Elitism + Genetic Crossover/Mutation + RL UCB1/Q-Learning proposals
      - 'rl_guided': Elitism + RL Bandit/Q-Learning proposals + light mutation
    """
    sorted_pop = sorted(
        previous_generation,
        key=lambda c: (float(c.get("fitness_score") or 0.0), float(c.get("accuracy") or 0.0)),
        reverse=True,
    )

    next_gen: List[Dict[str, Any]] = []
    seen_sigs = set()

    # 1. Elitism (carry forward top performer(s))
    actual_elitism = min(max(1, elitism_count), len(sorted_pop), max(1, population_size // 3))
    for elite in sorted_pop[:actual_elitism]:
        cfg = clone_arch_config(elite["config"])
        seen_sigs.add(architecture_signature(cfg))
        next_gen.append(
            {
                "config": cfg,
                "origin": "elitism",
                "parent_ids": [elite["id"]],
                "operation_details": f"Elite preserved from {elite['name']} (fitness {elite.get('fitness_score', 0):.2f})",
            }
        )

    # 2. RL-guided proposals (for hybrid_rl and rl_guided strategies)
    rl_slots = 0
    if rl_controller is not None:
        if strategy == "rl_guided":
            rl_slots = max(1, population_size - len(next_gen))
        elif strategy == "hybrid_rl":
            rl_slots = max(1, (population_size - len(next_gen)) // 3)

    best_parent_id = sorted_pop[0]["id"] if sorted_pop else None
    for _ in range(rl_slots):
        if len(next_gen) >= population_size:
            break
        rl_cfg = rl_controller.propose_architecture(rng, batch_size=batch_size)
        sig = architecture_signature(rl_cfg)
        if sig in seen_sigs:
            rl_cfg, mut_desc = mutate_architecture(rl_cfg, search_space, rng, num_mutations=1)
            desc = f"RL UCB1/Q-Learning proposal + {mut_desc}"
        else:
            desc = f"RL UCB1/Q-Learning policy proposal (eps={rl_controller.epsilon:.2f})"
        seen_sigs.add(architecture_signature(rl_cfg))
        next_gen.append(
            {
                "config": rl_cfg,
                "origin": "rl_guided",
                "parent_ids": [best_parent_id] if best_parent_id else [],
                "operation_details": desc,
            }
        )

    # 3. Genetic Crossover & Mutation to fill remaining slots
    attempts = 0
    while len(next_gen) < population_size and attempts < population_size * 15:
        attempts += 1
        use_crossover = len(sorted_pop) >= 2 and rng.random() < 0.55
        if use_crossover:
            p1 = tournament_select(sorted_pop, rng)
            p2 = tournament_select(sorted_pop, rng)
            if p2["id"] == p1["id"] and len(sorted_pop) > 1:
                p2 = rng.choice([c for c in sorted_pop if c["id"] != p1["id"]])
            child_cfg, desc = crossover_architectures(p1, p2, search_space, rng)
            # 40% chance of light post-crossover mutation
            if rng.random() < 0.4:
                child_cfg, m_desc = mutate_architecture(child_cfg, search_space, rng, num_mutations=1)
                desc = f"{desc} + {m_desc}"
            origin = "crossover"
            parent_ids = [p1["id"], p2["id"]]
        else:
            parent = tournament_select(sorted_pop, rng)
            num_m = 2 if rng.random() < 0.3 else 1
            child_cfg, desc = mutate_architecture(parent["config"], search_space, rng, num_mutations=num_m)
            origin = "mutation"
            parent_ids = [parent["id"]]

        sig = architecture_signature(child_cfg)
        if sig not in seen_sigs or attempts > population_size * 10:
            seen_sigs.add(sig)
            next_gen.append(
                {
                    "config": child_cfg,
                    "origin": origin,
                    "parent_ids": parent_ids,
                    "operation_details": desc,
                }
            )

    return next_gen


def build_evolution_graph(architectures: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a NetworkX directed acyclic lineage graph for the Evolution History page,
    computing generation columns, node positions, parent->child edges, and per-generation summaries.
    """
    G = nx.DiGraph()
    by_gen: Dict[int, List[Dict[str, Any]]] = {}
    for arch in architectures:
        by_gen.setdefault(arch["generation"], []).append(arch)
        G.add_node(
            arch["id"],
            name=arch["name"],
            generation=arch["generation"],
            candidate_index=arch["candidate_index"],
            origin=arch["origin"],
            operation_details=arch.get("operation_details", ""),
            accuracy=arch.get("accuracy"),
            fitness_score=arch.get("fitness_score"),
            inference_latency_ms=arch.get("inference_latency_ms"),
            model_size_kb=arch.get("model_size_kb"),
            parameter_count=arch.get("parameter_count"),
            is_pareto_optimal=bool(arch.get("is_pareto_optimal")),
            pareto_rank=arch.get("pareto_rank"),
            status=arch.get("status"),
        )

    id_set = {a["id"] for a in architectures}
    for arch in architectures:
        for pid in arch.get("parent_ids", []):
            if pid in id_set:
                G.add_edge(pid, arch["id"], operation=arch["origin"], details=arch.get("operation_details", ""))

    # Compute structured grid layout coordinates
    nodes = []
    generations_summary = []
    for gen in sorted(by_gen.keys()):
        gen_archs = sorted(by_gen[gen], key=lambda x: x["candidate_index"])
        evaluated = [a for a in gen_archs if a.get("status") == "evaluated"]
        best = max(evaluated, key=lambda x: x.get("fitness_score") or 0.0) if evaluated else None

        generations_summary.append(
            {
                "generation": gen,
                "total": len(gen_archs),
                "evaluated": len(evaluated),
                "best_id": best["id"] if best else None,
                "best_name": best["name"] if best else None,
                "best_accuracy": best.get("accuracy") if best else None,
                "best_fitness": best.get("fitness_score") if best else None,
                "avg_accuracy": (
                    round(sum(a.get("accuracy") or 0.0 for a in evaluated) / len(evaluated), 4)
                    if evaluated
                    else None
                ),
                "avg_fitness": (
                    round(sum(a.get("fitness_score") or 0.0 for a in evaluated) / len(evaluated), 2)
                    if evaluated
                    else None
                ),
                "origins": {
                    op: sum(1 for a in gen_archs if a.get("origin") == op)
                    for op in ["random", "elitism", "crossover", "mutation", "rl_guided"]
                },
            }
        )

        for row_idx, arch in enumerate(gen_archs):
            data = dict(G.nodes[arch["id"]])
            data["id"] = arch["id"]
            data["x"] = gen
            data["y"] = row_idx
            data["is_gen_best"] = bool(best and arch["id"] == best["id"])
            data["in_degree"] = G.in_degree(arch["id"])
            data["out_degree"] = G.out_degree(arch["id"])
            nodes.append(data)

    edges = [
        {
            "source": u,
            "target": v,
            "operation": d.get("operation", "mutation"),
            "details": d.get("details", ""),
        }
        for u, v, d in G.edges(data=True)
    ]

    return {
        "nodes": nodes,
        "edges": edges,
        "generations": generations_summary,
        "stats": {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "total_generations": len(by_gen),
        },
    }
