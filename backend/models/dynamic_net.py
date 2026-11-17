from typing import Any, Dict, List

import torch
import torch.nn as nn


def build_activation(name: str) -> nn.Module:
    key = (name or "relu").lower()
    mapping = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "leaky_relu": lambda: nn.LeakyReLU(0.1),
        "elu": nn.ELU,
    }
    factory = mapping.get(key, nn.ReLU)
    return factory()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class DynamicNeuralNet(nn.Module):
    """
    Dynamically constructed PyTorch neural network driven by a NEUROEVOLVE genome.
    Provides exact layer-by-layer introspection for the visual Architecture Explorer.
    """

    def __init__(self, input_dim: int, num_classes: int, arch_config: Dict[str, Any]):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.arch_config = arch_config

        layers: List[nn.Module] = []
        layer_summary: List[Dict[str, Any]] = [
            {
                "index": 0,
                "type": "Input",
                "name": "Input Flatten",
                "in_features": self.input_dim,
                "out_features": self.input_dim,
                "params": 0,
                "details": f"{self.input_dim} features",
            }
        ]

        in_features = self.input_dim
        step_idx = 1

        for i, h in enumerate(arch_config.get("hidden_layers", [])):
            out_features = int(h["units"])
            linear = nn.Linear(in_features, out_features)
            layers.append(linear)
            lin_params = in_features * out_features + out_features
            layer_summary.append(
                {
                    "index": step_idx,
                    "type": "Dense",
                    "name": f"Dense {i + 1}",
                    "in_features": in_features,
                    "out_features": out_features,
                    "params": lin_params,
                    "details": f"Linear({in_features} -> {out_features})",
                }
            )
            step_idx += 1

            if h.get("batch_norm", False):
                bn = nn.BatchNorm1d(out_features)
                layers.append(bn)
                bn_params = 2 * out_features
                layer_summary.append(
                    {
                        "index": step_idx,
                        "type": "BatchNorm",
                        "name": f"BatchNorm {i + 1}",
                        "in_features": out_features,
                        "out_features": out_features,
                        "params": bn_params,
                        "details": f"BatchNorm1d({out_features})",
                    }
                )
                step_idx += 1

            act_name = h.get("activation", "relu").lower()
            layers.append(build_activation(act_name))
            layer_summary.append(
                {
                    "index": step_idx,
                    "type": "Activation",
                    "name": act_name.upper(),
                    "in_features": out_features,
                    "out_features": out_features,
                    "params": 0,
                    "details": f"Activation({act_name.upper()})",
                }
            )
            step_idx += 1

            drop_rate = float(h.get("dropout", 0.0))
            if drop_rate > 0.0:
                layers.append(nn.Dropout(p=drop_rate))
                layer_summary.append(
                    {
                        "index": step_idx,
                        "type": "Dropout",
                        "name": f"Dropout ({int(round(drop_rate * 100))}%)",
                        "in_features": out_features,
                        "out_features": out_features,
                        "params": 0,
                        "details": f"Dropout(p={drop_rate})",
                    }
                )
                step_idx += 1

            in_features = out_features

        out_linear = nn.Linear(in_features, self.num_classes)
        layers.append(out_linear)
        out_params = in_features * self.num_classes + self.num_classes
        layer_summary.append(
            {
                "index": step_idx,
                "type": "Output",
                "name": "Output Classifier",
                "in_features": in_features,
                "out_features": self.num_classes,
                "params": out_params,
                "details": f"Linear({in_features} -> {self.num_classes} classes)",
            }
        )

        self.network = nn.Sequential(*layers)
        self._layer_summary = layer_summary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = torch.flatten(x, 1)
        return self.network(x)

    def describe_layers(self) -> List[Dict[str, Any]]:
        return self._layer_summary
