from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class SearchSpaceSchema(BaseModel):
    min_layers: int = Field(default=1, ge=1, le=5)
    max_layers: int = Field(default=4, ge=1, le=6)
    neuron_choices: List[int] = Field(default=[32, 64, 128, 256, 384])
    activations: List[str] = Field(default=["relu", "gelu", "tanh", "silu", "leaky_relu"])
    dropout_choices: List[float] = Field(default=[0.0, 0.1, 0.2, 0.3])
    optimizers: List[str] = Field(default=["adam", "adamw", "sgd"])
    learning_rates: List[float] = Field(default=[0.0005, 0.001, 0.003, 0.005])
    batch_norm_choices: List[bool] = Field(default=[False, True])

    @field_validator("max_layers")
    @classmethod
    def validate_layer_bounds(cls, v: int, info) -> int:
        min_l = info.data.get("min_layers", 1)
        if v < min_l:
            return min_l
        return v


class CreateExperimentRequest(BaseModel):
    name: str = Field(default="NEUROEVOLVE Search", min_length=2, max_length=120)
    dataset: str = Field(default="mnist")
    generations: int = Field(default=3, ge=1, le=30)
    population_size: int = Field(default=4, ge=2, le=24)
    max_epochs: int = Field(default=3, ge=1, le=30)
    batch_size: int = Field(default=64, ge=16, le=512)
    max_train_samples: int = Field(default=4000, ge=200, le=60000)
    max_val_samples: int = Field(default=1000, ge=100, le=10000)
    optimization_mode: Literal["accuracy", "speed", "balanced"] = Field(default="balanced")
    search_strategy: Literal["genetic", "hybrid_rl", "rl_guided"] = Field(default="hybrid_rl")
    cpu_threads: int = Field(default=4, ge=1, le=16)
    elitism_count: int = Field(default=1, ge=1, le=4)
    seed: Optional[int] = Field(default=None)
    search_space: SearchSpaceSchema = Field(default_factory=SearchSpaceSchema)


class CompareRequest(BaseModel):
    architecture_ids: List[str] = Field(..., min_length=1, max_length=12)
