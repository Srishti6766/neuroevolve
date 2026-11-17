import math
import random
from typing import Any, Dict, List

from backend.engine.search_space import SearchSpace


class RLArchitectureController:
    """
    Lightweight Tabular Q-Learning + UCB1 Contextual Bandit controller for Neural Architecture Search.
    Learns expected multi-objective rewards for discrete architectural choices (layer count, neuron widths,
    activations, dropout rates, batch norm, optimizer, and learning rate) from real PyTorch evaluations.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        alpha: float = 0.35,
        epsilon: float = 0.25,
        epsilon_decay: float = 0.92,
        ucb_c: float = 0.45,
    ):
        self.search_space = search_space
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.ucb_c = ucb_c
        self.total_updates = 0

        self.arms: Dict[str, List[Any]] = {
            "num_layers": list(range(search_space.min_layers, search_space.max_layers + 1)),
            "units": list(search_space.neuron_choices),
            "activation": list(search_space.activations),
            "dropout": list(search_space.dropout_choices),
            "batch_norm": list(search_space.batch_norm_choices),
            "optimizer": list(search_space.optimizers),
            "learning_rate": list(search_space.learning_rates),
        }

        self.q_values: Dict[str, Dict[str, float]] = {
            group: {str(val): 0.5 for val in choices} for group, choices in self.arms.items()
        }
        self.counts: Dict[str, Dict[str, int]] = {
            group: {str(val): 0 for val in choices} for group, choices in self.arms.items()
        }
        self.reward_history: List[Dict[str, Any]] = []

    def _select_arm(self, group: str, rng: random.Random) -> Any:
        choices = self.arms[group]
        if rng.random() < self.epsilon:
            return rng.choice(choices)

        total_n = max(1, sum(self.counts[group].values()))
        best_score = -1e9
        best_choices: List[Any] = []
        for choice in choices:
            key = str(choice)
            q = self.q_values[group].get(key, 0.5)
            n = self.counts[group].get(key, 0)
            bonus = self.ucb_c * math.sqrt(math.log(total_n + 1.0) / (n + 1.0))
            score = q + bonus
            if score > best_score + 1e-9:
                best_score = score
                best_choices = [choice]
            elif abs(score - best_score) <= 1e-9:
                best_choices.append(choice)
        return rng.choice(best_choices)

    def propose_architecture(self, rng: random.Random, batch_size: int = 64) -> Dict[str, Any]:
        num_layers = int(self._select_arm("num_layers", rng))
        hidden_layers = []
        for _ in range(num_layers):
            hidden_layers.append(
                {
                    "units": int(self._select_arm("units", rng)),
                    "activation": str(self._select_arm("activation", rng)),
                    "dropout": float(self._select_arm("dropout", rng)),
                    "batch_norm": bool(self._select_arm("batch_norm", rng)),
                }
            )
        # Mild structural regularization: bias toward non-increasing widths
        if num_layers > 1 and rng.random() < 0.65:
            sorted_u = sorted((l["units"] for l in hidden_layers), reverse=True)
            for l, u in zip(hidden_layers, sorted_u):
                l["units"] = u

        return {
            "hidden_layers": hidden_layers,
            "optimizer": str(self._select_arm("optimizer", rng)),
            "learning_rate": float(self._select_arm("learning_rate", rng)),
            "batch_size": batch_size,
        }

    def update(self, arch_config: Dict[str, Any], fitness_score: float, generation: int) -> float:
        """
        Update Q-values using reward = fitness_score / 100.0 from a genuinely evaluated PyTorch architecture.
        """
        reward = max(0.0, min(1.0, float(fitness_score) / 100.0))
        self.total_updates += 1

        layers = arch_config.get("hidden_layers", [])
        observations = [
            ("num_layers", str(len(layers))),
            ("optimizer", str(arch_config.get("optimizer", "adam"))),
            ("learning_rate", str(arch_config.get("learning_rate", 0.001))),
        ]
        for layer in layers:
            observations.append(("units", str(layer["units"])))
            observations.append(("activation", str(layer["activation"])))
            observations.append(("dropout", str(layer["dropout"])))
            observations.append(("batch_norm", str(bool(layer.get("batch_norm", False)))))

        for group, key in observations:
            if group in self.q_values and key in self.q_values[group]:
                old_q = self.q_values[group][key]
                self.q_values[group][key] = round(old_q + self.alpha * (reward - old_q), 4)
                self.counts[group][key] = self.counts[group].get(key, 0) + 1

        self.epsilon = max(0.08, self.epsilon * self.epsilon_decay)
        self.reward_history.append(
            {
                "step": self.total_updates,
                "generation": generation,
                "reward": round(reward, 4),
                "epsilon": round(self.epsilon, 4),
            }
        )
        return round(reward, 4)

    def get_policy_summary(self) -> Dict[str, Any]:
        top_preferences: Dict[str, Any] = {}
        for group, qmap in self.q_values.items():
            best_key = max(qmap.keys(), key=lambda k: qmap[k])
            top_preferences[group] = {
                "best_choice": best_key,
                "q_value": round(qmap[best_key], 4),
                "trials": self.counts[group].get(best_key, 0),
            }
        return {
            "algorithm": "Tabular Q-Learning + UCB1 Contextual Bandit",
            "total_updates": self.total_updates,
            "current_epsilon": round(self.epsilon, 4),
            "alpha": self.alpha,
            "ucb_c": self.ucb_c,
            "top_preferences": top_preferences,
            "q_table": self.q_values,
            "counts": self.counts,
            "reward_history": self.reward_history[-40:],
        }
