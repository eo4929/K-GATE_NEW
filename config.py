"""Centralized configuration for K-GATE experiments."""

from typing import Optional

from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    judge_model: str = "gpt-4o"
    mutation_model: str = "gpt-4o"
    decomposition_model: str = "gpt-4o"
    target_model: str = "gpt-4o-mini"

    # MCTS: K complete rollouts, each of at most T trajectory steps.
    mcts_simulations: int = Field(default=20, ge=1)
    max_trajectory_length: int = Field(default=5, ge=1)
    num_knowledge_nodes: int = Field(default=5, ge=2)
    c_param: float = Field(default=1.414, ge=0.0)

    harmfulness_threshold: int = Field(default=7, ge=0, le=10)

    data_dir: str = "data"
    advbench_file: str = "advbench_behaviors.csv"
    harmbench_file: str = "harmbench_behaviors_text_all.csv"

    num_samples: Optional[int] = None
    seed: int = 42

    output_dir: str = "results"
    save_intermediate: bool = True
    verbose: bool = True


class APIConfig(BaseModel):
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3


class DatasetConfig(BaseModel):
    dataset_name: str
    file_path: str
    goal_column: str = "goal"
    category_column: Optional[str] = None
    num_samples: Optional[int] = None
    random_seed: int = 42


DEFAULT_CONFIG = ExperimentConfig()
DEFAULT_API_CONFIG = APIConfig()

__all__ = [
    "ExperimentConfig",
    "APIConfig",
    "DatasetConfig",
    "DEFAULT_CONFIG",
    "DEFAULT_API_CONFIG",
]
