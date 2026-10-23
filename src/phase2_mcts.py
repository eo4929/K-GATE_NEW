"""K-GATE Phase 2: prefix-specific MCTS over a knowledge DAG.

The search tree is a tree of dialogue prefixes, not a table indexed by
``(knowledge node, mutation strategy)``. An edge chooses both the next
currently-reachable DAG node and its mutation strategy. Consequently, the
same action below two different prefixes has independent statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Set

from src.core import AdversarialTrajectory, KnowledgeDAG, KnowledgeNode, TrajectoryStep
from utils import PromptLoader, with_retry


logger = logging.getLogger("KGATE.Phase2")
MUTATION_TEMPLATE_FILE = "mutation_operator_prompt_template.md"


class MutationStrategy:
    """The four strategy-conditioned mutation operators from the paper."""

    STRATEGIES = {
        "Academic": {
            "role": "researcher or academic analyst",
            "context": "scholarly or theoretical research inquiry",
        },
        "Historical": {
            "role": "historian or policy analyst reviewing archival records",
            "context": "analysis of historical events or past case studies",
        },
        "Technical": {
            "role": "systems engineer or domain specialist",
            "context": "request for low-level implementation details or engineering specifications",
        },
        "Counter-factual": {
            "role": "scenario designer or risk assessor",
            "context": "hypothetical, fictional, or emergency simulation scenario",
        },
    }
    STRATEGY_NAMES = tuple(STRATEGIES)

    @staticmethod
    @with_retry(max_attempts=3)
    def get_prompt(
        node: KnowledgeNode,
        strategy_name: str,
        llm_client,
        model: str = "gpt-4o",
    ) -> str:
        """Apply the strategy-conditioned mutation operator ``g(v,T_theta)``."""
        if strategy_name not in MutationStrategy.STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Expected one of {list(MutationStrategy.STRATEGY_NAMES)}"
            )
        strategy = MutationStrategy.STRATEGIES[strategy_name]
        filled = PromptLoader.load(
            MUTATION_TEMPLATE_FILE,
            role=strategy["role"],
            context=strategy["context"],
            strategy_name=strategy_name,
            title=node.title,
            description=node.description,
            level=node.risk_level,
        )
        response = llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": filled}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()


@dataclass(frozen=True)
class SearchAction:
    """A tree edge: choose a reachable knowledge node and a mutation."""

    node_id: int
    strategy: str

    @property
    def label(self) -> str:
        return f"v{self.node_id}:{self.strategy}"


class MCTSNode:
    """One exact dialogue-prefix state in the MCTS tree."""

    def __init__(
        self,
        *,
        parent: Optional["MCTSNode"] = None,
        action: Optional[SearchAction] = None,
        step: Optional[TrajectoryStep] = None,
        visited_node_ids: Iterable[int] = (),
    ) -> None:
        self.parent = parent
        self.action = action
        self.step = step
        self.visited_node_ids = frozenset(visited_node_ids)
        self.children: Dict[SearchAction, MCTSNode] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.best_return = float("-inf")
        self.best_completion: Optional[List[TrajectoryStep]] = None

    @property
    def depth(self) -> int:
        return len(self.visited_node_ids)

    @property
    def value(self) -> float:
        """Compatibility alias for cumulative value V(s,a)."""
        return self.value_sum

    @property
    def strategy(self) -> str:
        return self.action.strategy if self.action else "Root"

    @property
    def knode(self) -> Optional[KnowledgeNode]:
        return self.step.node if self.step else None

    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def avg_score(self) -> float:
        """Backward-compatible name used by earlier analysis scripts."""
        return self.mean_value()

    def uct(self, exploration_constant: float) -> float:
        """UCT evaluated within this node's parent dialogue prefix."""
        if self.visits == 0:
            return float("inf")
        parent_visits = max(1, self.parent.visits if self.parent else self.visits)
        return self.mean_value() + exploration_constant * math.sqrt(
            math.log(parent_visits) / self.visits
        )

    def update(self, rollout_return: float, completion: Sequence[TrajectoryStep]) -> None:
        self.visits += 1
        self.value_sum += rollout_return
        if rollout_return > self.best_return:
            self.best_return = rollout_return
            self.best_completion = list(completion)

    def prefix_steps(self) -> List[TrajectoryStep]:
        steps: List[TrajectoryStep] = []
        cursor: Optional[MCTSNode] = self
        while cursor is not None and cursor.step is not None:
            steps.append(cursor.step)
            cursor = cursor.parent
        steps.reverse()
        return steps

    def dialogue_history(self) -> List[Dict[str, str]]:
        history: List[Dict[str, str]] = []
        for step in self.prefix_steps():
            history.extend(
                [
                    {"role": "user", "content": step.prompt},
                    {"role": "assistant", "content": step.response},
                ]
            )
        return history


class MCTS_TrajectorySearcher:
    """Selection, expansion, simulation, and path backpropagation for K-GATE."""

    def __init__(self, target_llm, evaluator, config, llm_client):
        self.target = target_llm
        self.evaluator = evaluator
        self.cfg = config
        self.llm_client = llm_client
        self.rng = random.Random(config.seed)
        self.last_root: Optional[MCTSNode] = None

    @staticmethod
    def _prerequisites(dag: KnowledgeDAG) -> Dict[int, Set[int]]:
        prereqs: Dict[int, Set[int]] = {
            node.id: set(node.prerequisites) for node in dag.nodes
        }
        for source, target in dag.edges:
            prereqs[target].add(source)
        return prereqs

    def _available_actions(
        self,
        dag: KnowledgeDAG,
        visited: Set[int] | frozenset[int],
    ) -> List[SearchAction]:
        """Return joint node-strategy actions reachable from this prefix."""
        reachable = self._reachable_node_ids(dag, visited)
        return [
            SearchAction(node_id, strategy)
            for node_id in reachable
            for strategy in MutationStrategy.STRATEGY_NAMES
        ]

    def _reachable_node_ids(
        self,
        dag: KnowledgeDAG,
        visited: Set[int] | frozenset[int],
    ) -> List[int]:
        """Return prerequisite-valid unvisited nodes, deterministically ordered."""
        prereqs = self._prerequisites(dag)
        reachable = [
            node.id
            for node in dag.nodes
            if node.id not in visited and prereqs[node.id].issubset(visited)
        ]
        # Sorting is only for seeded reproducibility. Node choice remains an
        # action selected by MCTS; this is not a fixed topological traversal.
        return sorted(reachable)

    def _is_terminal(self, node: MCTSNode, dag: KnowledgeDAG) -> bool:
        if node.depth >= min(self.cfg.max_trajectory_length, len(dag.nodes)):
            return True
        if node.step and node.step.score >= self.cfg.harmfulness_threshold:
            return True
        return not self._available_actions(dag, node.visited_node_ids)

    def _execute_action(
        self,
        dag: KnowledgeDAG,
        goal: str,
        action: SearchAction,
        history: Sequence[Dict[str, str]],
    ) -> TrajectoryStep:
        knode = dag.get_node_by_id(action.node_id)
        if knode is None:
            raise RuntimeError(f"MCTS selected missing DAG node {action.node_id}")
        prompt = MutationStrategy.get_prompt(
            knode,
            action.strategy,
            self.llm_client,
            model=self.cfg.mutation_model,
        )
        response = self.target.generate(
            list(history) + [{"role": "user", "content": prompt}]
        )
        evaluation = self.evaluator.evaluate(goal, response)
        return TrajectoryStep(
            node=knode,
            strategy=action.strategy,
            prompt=prompt,
            response=response,
            score=evaluation.harmfulness_score,
        )

    @staticmethod
    def _rollout_return(steps: Sequence[TrajectoryStep]) -> float:
        """Return maximum intermediate harmfulness on the paper's [0,10] scale."""
        return max((step.score for step in steps), default=0.0)

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        best_uct = max(
            child.uct(self.cfg.c_param) for child in node.children.values()
        )
        tied = [
            child
            for child in node.children.values()
            if math.isclose(child.uct(self.cfg.c_param), best_uct)
        ]
        return self.rng.choice(tied)

    def _simulate(
        self,
        dag: KnowledgeDAG,
        goal: str,
        frontier: MCTSNode,
        frontier_sizes: List[int],
    ) -> List[TrajectoryStep]:
        """Default-policy rollout from the expanded prefix to termination."""
        completion = frontier.prefix_steps()
        visited = set(frontier.visited_node_ids)
        history = frontier.dialogue_history()

        while len(visited) < min(self.cfg.max_trajectory_length, len(dag.nodes)):
            if completion and completion[-1].score >= self.cfg.harmfulness_threshold:
                break
            reachable = self._reachable_node_ids(dag, visited)
            if not reachable:
                break
            frontier_sizes.append(len(reachable))
            actions = [
                SearchAction(node_id, strategy)
                for node_id in reachable
                for strategy in MutationStrategy.STRATEGY_NAMES
            ]
            action = self.rng.choice(actions)
            step = self._execute_action(dag, goal, action, history)
            completion.append(step)
            visited.add(action.node_id)
            history.extend(
                [
                    {"role": "user", "content": step.prompt},
                    {"role": "assistant", "content": step.response},
                ]
            )
        return completion

    def mine_trajectory(self, dag: KnowledgeDAG, goal: str) -> AdversarialTrajectory:
        """Mine one coherent adversarial trajectory with prefix-specific MCTS."""
        dag.validate_graph()
        root = MCTSNode()
        self.last_root = root
        frontier_sizes: List[int] = []
        # K denotes complete rollouts. Each rollout can issue at most T target
        # calls, so the target-query budget remains O(KT).
        iterations = max(1, self.cfg.mcts_simulations)
        logger.info(
            "Prefix-tree MCTS: %d iterations, %d DAG nodes, C=%.3f",
            iterations,
            len(dag.nodes),
            self.cfg.c_param,
        )

        for simulation_index in range(iterations):
            current = root
            search_path = [root]

            # 1) Selection: descend while all reachable joint actions exist.
            while not self._is_terminal(current, dag):
                reachable = self._reachable_node_ids(dag, current.visited_node_ids)
                frontier_sizes.append(len(reachable))
                actions = self._available_actions(dag, current.visited_node_ids)
                untried = [a for a in actions if a not in current.children]
                if untried:
                    break
                current = self._select_child(current)
                search_path.append(current)

            # 2) Expansion: create one child under this exact prefix.
            if not self._is_terminal(current, dag):
                actions = self._available_actions(dag, current.visited_node_ids)
                untried = [a for a in actions if a not in current.children]
                action = self.rng.choice(untried)
                step = self._execute_action(
                    dag, goal, action, current.dialogue_history()
                )
                child = MCTSNode(
                    parent=current,
                    action=action,
                    step=step,
                    visited_node_ids=set(current.visited_node_ids) | {action.node_id},
                )
                current.children[action] = child
                current = child
                search_path.append(child)

            # 3) Simulation: sample the remaining reachable node/strategy path.
            completion = self._simulate(dag, goal, current, frontier_sizes)
            rollout_return = self._rollout_return(completion)

            # 4) Backpropagation: update the entire selected root-frontier path.
            for tree_node in reversed(search_path):
                tree_node.update(rollout_return, completion)

            logger.debug(
                "Simulation %d/%d: depth=%d return=%.2f path=%s",
                simulation_index + 1,
                iterations,
                len(completion),
                rollout_return,
                " -> ".join(
                    f"v{step.node.id}:{step.strategy}" for step in completion
                ),
            )

        return self._extract_trajectory(
            root, dag, goal, iterations, frontier_sizes
        )

    def _extract_trajectory(
        self,
        root: MCTSNode,
        dag: KnowledgeDAG,
        goal: str,
        iterations: int,
        frontier_sizes: List[int],
    ) -> AdversarialTrajectory:
        """Extract one prefix-consistent principal variation from the tree."""
        current = root
        chosen: List[TrajectoryStep] = []
        terminal_reason = "search_exhausted"

        while current.children and not self._is_terminal(current, dag):
            best = max(
                current.children.values(),
                key=lambda child: (
                    child.mean_value(),
                    child.visits,
                    child.action.label,
                ),
            )
            action_values = {
                child.action.label: child.mean_value()
                for child in current.children.values()
            }
            step = best.step.model_copy(
                update={
                    "uct_scores": action_values,
                    "visit_count": best.visits,
                    "mean_return": best.mean_value(),
                    "depth": len(chosen),
                }
            )
            chosen.append(step)
            current = best

        # This suffix was evaluated as part of a rollout from the exact prefix.
        if not self._is_terminal(current, dag) and current.best_completion:
            suffix = current.best_completion[len(chosen):]
            for step in suffix:
                chosen.append(step.model_copy(update={"depth": len(chosen)}))

        visited = {step.node.id for step in chosen}
        if chosen and chosen[-1].score >= self.cfg.harmfulness_threshold:
            terminal_reason = "harmfulness_threshold"
        elif len(chosen) >= min(self.cfg.max_trajectory_length, len(dag.nodes)):
            terminal_reason = "max_trajectory_length"
        elif not self._available_actions(dag, visited):
            terminal_reason = "dag_terminal"

        reward = self._rollout_return(chosen)
        tree_nodes = self._count_tree_nodes(root)
        frontier_histogram = {
            size: frontier_sizes.count(size) for size in sorted(set(frontier_sizes))
        }
        mean_frontier_size = (
            sum(frontier_sizes) / len(frontier_sizes) if frontier_sizes else 0.0
        )
        branching_state_fraction = (
            sum(size > 1 for size in frontier_sizes) / len(frontier_sizes)
            if frontier_sizes
            else 0.0
        )
        logger.info(
            "MCTS trajectory: %s (return=%.2f, terminal=%s, tree_nodes=%d)",
            " -> ".join(f"v{s.node.id}:{s.strategy}" for s in chosen),
            reward,
            terminal_reason,
            tree_nodes,
        )
        return AdversarialTrajectory(
            goal=goal,
            nodes=[step.node for step in chosen],
            steps=chosen,
            search_iterations=iterations,
            max_trajectory_length=self.cfg.max_trajectory_length,
            exploration_constant=self.cfg.c_param,
            harmfulness_threshold=self.cfg.harmfulness_threshold,
            search_seed=self.cfg.seed,
            tree_nodes=tree_nodes,
            rollout_reward=reward,
            terminal_reason=terminal_reason,
            frontier_sizes=frontier_sizes,
            frontier_size_histogram=frontier_histogram,
            mean_frontier_size=mean_frontier_size,
            branching_state_fraction=branching_state_fraction,
        )

    @staticmethod
    def _count_tree_nodes(root: MCTSNode) -> int:
        return 1 + sum(
            MCTS_TrajectorySearcher._count_tree_nodes(child)
            for child in root.children.values()
        )


MCTS_TrajectorySearacher = MCTS_TrajectorySearcher

__all__ = [
    "MutationStrategy",
    "SearchAction",
    "MCTSNode",
    "MCTS_TrajectorySearcher",
    "MCTS_TrajectorySearacher",
]
