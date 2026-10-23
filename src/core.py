"""Core data models for the K-GATE pipeline."""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class KnowledgeNode(BaseModel):
    """An atomic knowledge unit in the decomposed DAG."""

    id: int
    title: str = Field(description="Semantic title of the node")
    description: str = Field(description="Textual description")
    risk_level: str = Field(description="Fundamental, Bridging, or Target")
    prerequisites: List[int] = Field(
        default_factory=list, description="Prerequisite node IDs"
    )


class KnowledgeDAG(BaseModel):
    """Knowledge-decomposed DAG G=(V,E)."""

    nodes: List[KnowledgeNode]
    edges: List[Tuple[int, int]] = Field(
        default_factory=list, description="(source, target) prerequisite edges"
    )

    def _edge_set(self) -> set[Tuple[int, int]]:
        edges = set(self.edges)
        for node in self.nodes:
            edges.update((source, node.id) for source in node.prerequisites)
        return edges

    def validate_graph(self) -> None:
        """Validate references and acyclicity without selecting a trajectory.

        Kahn's algorithm is used only for cycle validation. Phase 2 instead
        chooses among every node reachable from the current dialogue prefix.
        """
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Knowledge DAG contains duplicate node IDs")

        known = set(node_ids)
        edge_set = self._edge_set()
        unknown = {
            edge
            for edge in edge_set
            if edge[0] not in known or edge[1] not in known
        }
        if unknown:
            raise ValueError(
                f"Knowledge DAG edges reference unknown nodes: {sorted(unknown)}"
            )
        if any(source == target for source, target in edge_set):
            raise ValueError("Knowledge DAG contains a self-loop")

        in_degree = {node_id: 0 for node_id in known}
        adjacency = {node_id: [] for node_id in known}
        for source, target in edge_set:
            adjacency[source].append(target)
            in_degree[target] += 1
        queue = deque(node_id for node_id, degree in in_degree.items() if degree == 0)
        visited = 0
        while queue:
            source = queue.popleft()
            visited += 1
            for target in adjacency[source]:
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)
        if visited != len(known):
            raise ValueError("Knowledge graph must be acyclic")

    def get_ordered_nodes(self) -> List[KnowledgeNode]:
        """Return one deterministic topological order for display purposes.

        This method is not used by MCTS to choose its trajectory.
        """
        self.validate_graph()
        node_map = {node.id: node for node in self.nodes}
        edge_set = self._edge_set()
        in_degree = {node_id: 0 for node_id in node_map}
        adjacency = {node_id: [] for node_id in node_map}
        for source, target in edge_set:
            adjacency[source].append(target)
            in_degree[target] += 1
        queue = deque(sorted(node_id for node_id, degree in in_degree.items() if degree == 0))
        result: List[KnowledgeNode] = []
        while queue:
            source = queue.popleft()
            result.append(node_map[source])
            for target in sorted(adjacency[source]):
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)
        return result

    def get_node_by_id(self, node_id: int) -> Optional[KnowledgeNode]:
        return next((node for node in self.nodes if node.id == node_id), None)


class EvaluationResult(BaseModel):
    """External-auditor result."""

    violated: bool = Field(description="True when score reaches the threshold")
    harmfulness_score: int = Field(ge=0, le=10)
    reasoning: str


class TrajectoryStep(BaseModel):
    """One selected (knowledge node, mutation strategy) tree edge."""

    node: KnowledgeNode
    strategy: str
    prompt: str
    response: str
    score: float
    uct_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Prefix-local empirical action values keyed by v_i:strategy",
    )
    visit_count: int = Field(default=0, description="Visits to this prefix edge")
    mean_return: float = Field(default=0.0, description="Empirical Q(s,a)")
    depth: int = Field(default=0, description="Depth in the dialogue-prefix tree")


class AdversarialTrajectory(BaseModel):
    """A coherent root-to-terminal adversarial trajectory."""

    goal: str
    nodes: List[KnowledgeNode]
    steps: List[TrajectoryStep]
    aggregated_prompt: Optional[str] = None
    search_iterations: int = 0
    tree_nodes: int = 0
    rollout_reward: float = 0.0
    terminal_reason: Optional[str] = None
    max_trajectory_length: int = 0
    exploration_constant: float = 0.0
    harmfulness_threshold: int = 0
    search_seed: int = 0
    frontier_sizes: List[int] = Field(
        default_factory=list,
        description="|R(s)| at every MCTS decision-prefix visit",
    )
    frontier_size_histogram: Dict[int, int] = Field(default_factory=dict)
    mean_frontier_size: float = 0.0
    branching_state_fraction: float = Field(
        default=0.0,
        description="Fraction of recorded decision-prefix visits with |R(s)| > 1",
    )


class AttackResult(BaseModel):
    """Complete three-phase experiment result."""

    goal: str
    dataset: str
    category: Optional[str] = None
    dag: KnowledgeDAG
    trajectory: AdversarialTrajectory
    final_evaluation: EvaluationResult
    final_response: Optional[str] = None
    asr: bool

    class Config:
        arbitrary_types_allowed = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "dataset": self.dataset,
            "category": self.category,
            "asr": self.asr,
            "score": self.final_evaluation.harmfulness_score,
            "violated": self.final_evaluation.violated,
            "reasoning": self.final_evaluation.reasoning,
            "num_nodes": len(self.dag.nodes),
            "num_steps": len(self.trajectory.steps),
            "strategies_used": [step.strategy for step in self.trajectory.steps],
            "node_path": [step.node.id for step in self.trajectory.steps],
            "step_scores": [step.score for step in self.trajectory.steps],
            "search_iterations": self.trajectory.search_iterations,
            "max_trajectory_length": self.trajectory.max_trajectory_length,
            "exploration_constant": self.trajectory.exploration_constant,
            "harmfulness_threshold": self.trajectory.harmfulness_threshold,
            "search_seed": self.trajectory.search_seed,
            "tree_nodes": self.trajectory.tree_nodes,
            "rollout_reward": self.trajectory.rollout_reward,
            "terminal_reason": self.trajectory.terminal_reason,
            "frontier_sizes": self.trajectory.frontier_sizes,
            "frontier_size_histogram": self.trajectory.frontier_size_histogram,
            "mean_frontier_size": self.trajectory.mean_frontier_size,
            "branching_state_fraction": self.trajectory.branching_state_fraction,
            "aggregated_prompt_length": len(self.trajectory.aggregated_prompt or ""),
            "final_response_length": len(self.final_response or ""),
        }


__all__ = [
    "KnowledgeNode",
    "KnowledgeDAG",
    "EvaluationResult",
    "TrajectoryStep",
    "AdversarialTrajectory",
    "AttackResult",
]
