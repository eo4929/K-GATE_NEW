"""
K-GATE: Knowledge-Decomposed Graph-based Adversarial Trajectory Mining
"""

# Only re-export core data models eagerly (no external deps)
from src.core import (
    KnowledgeNode,
    KnowledgeDAG,
    EvaluationResult,
    TrajectoryStep,
    AdversarialTrajectory,
    AttackResult,
)

__all__ = [
    "KnowledgeNode",
    "KnowledgeDAG",
    "EvaluationResult",
    "TrajectoryStep",
    "AdversarialTrajectory",
    "AttackResult",
]
