import copy
import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List


DEFAULT_SEARCH_SPACE: Dict[str, Any] = {
    "min_layers": 1,
    "max_layers": 4,
    "neuron_choices": [32, 64, 128, 256, 384],
    "activations": ["relu", "gelu", "tanh", "silu", "leaky_relu"],
    "dropout_choices": [0.0, 0.1, 0.2, 0.3],
    "optimizers": ["adam", "adamw", "sgd"],
    "learning_rates": [0.0005, 0.001, 0.003, 0.005],
    "batch_norm_choices": [False, True],
}


@dataclass
class SearchSpace:
    min_layers: int = 1
    max_layers: int = 4
    neuron_choices: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 384])
    activations: List[str] = field(default_factory=lambda: ["relu", "gelu", "tanh", "silu", "leaky_relu"])
    dropout_choices: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3])
    optimizers: List[str] = field(default_factory=lambda: ["adam", "adamw", "sgd"])
    learning_rates: List[float] = field(default_factory=lambda: [0.0005, 0.001, 0.003, 0.005])
    batch_norm_choices: List[bool] = field(default_factory=lambda: [False, True])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchSpace":
        merged = {**DEFAULT_SEARCH_SPACE, **(data or {})}
        min_l = max(1, min(6, int(merged["min_layers"])))
        max_l = max(min_l, min(6, int(merged["max_layers"])))
        neurons = sorted({max(8, min(1024, int(n))) for n in merged["neuron_choices"] if int(n) > 0}) or [64, 128]
        acts = [a.lower() for a in merged["activations"] if a.lower() in {"relu", "gelu", "tanh", "silu", "leaky_relu", "elu"}] or ["relu", "gelu"]
        dropouts = sorted({max(0.0, min(0.7, float(d))) for d in merged["dropout_choices"]}) or [0.0, 0.2]
        opts = [o.lower() for o in merged["optimizers"] if o.lower() in {"adam", "adamw", "sgd", "rmsprop"}] or ["adam"]
        lrs = sorted({max(1e-5, min(0.1, float(lr))) for lr in merged["learning_rates"]}) or [0.001]
        bns = list({bool(b) for b in merged.get("batch_norm_choices", [False, True])}) or [False, True]
        return cls(
            min_layers=min_l,
            max_layers=max_l,
            neuron_choices=neurons,
            activations=acts,
            dropout_choices=dropouts,
            optimizers=opts,
            learning_rates=lrs,
            batch_norm_choices=bns,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_layers": self.min_layers,
            "max_layers": self.max_layers,
            "neuron_choices": self.neuron_choices,
            "activations": self.activations,
            "dropout_choices": self.dropout_choices,
            "optimizers": self.optimizers,
            "learning_rates": self.learning_rates,
            "batch_norm_choices": self.batch_norm_choices,
        }

    def sample_layer(self, rng: random.Random) -> Dict[str, Any]:
        return {
            "units": rng.choice(self.neuron_choices),
            "activation": rng.choice(self.activations),
            "dropout": rng.choice(self.dropout_choices),
            "batch_norm": rng.choice(self.batch_norm_choices),
        }

    def sample_architecture(self, rng: random.Random, batch_size: int = 64) -> Dict[str, Any]:
        num_layers = rng.randint(self.min_layers, self.max_layers)
        layers = [self.sample_layer(rng) for _ in range(num_layers)]
        # Sort slightly descending with 50% probability to produce realistic pyramid topologies too
        if num_layers > 1 and rng.random() < 0.5:
            units_sorted = sorted((l["units"] for l in layers), reverse=True)
            for l, u in zip(layers, units_sorted):
                l["units"] = u
        return {
            "hidden_layers": layers,
            "optimizer": rng.choice(self.optimizers),
            "learning_rate": rng.choice(self.learning_rates),
            "batch_size": batch_size,
        }


def clone_arch_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(config)


def architecture_signature(config: Dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:10]


def architecture_display_name(config: Dict[str, Any], generation: int, index: int) -> str:
    layers = config.get("hidden_layers", [])
    widths = "-".join(str(l["units"]) for l in layers)
    act = layers[0]["activation"].upper() if layers else "LIN"
    return f"NE-G{generation}C{index + 1} [{widths} {act}]"
