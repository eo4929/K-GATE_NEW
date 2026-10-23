"""
K-GATE Phase 1: Knowledge Discovery & DAG Construction
논문 Section IV-B: f_decomp : G → G_dag

Prompt template: prompts/decomposition_prompt_template.md
"""

import json
import logging

from src.core import KnowledgeNode, KnowledgeDAG
from utils import with_retry, PromptLoader


logger = logging.getLogger("KGATE.Phase1")

DECOMPOSITION_TEMPLATE_FILE = "decomposition_prompt_template.md"


class KDecomposer:
    """
    논문 IV-B: Knowledge Decomposition
    f_decomp : G → G_dag

    Decomposes a high-level harmful goal G into a knowledge-decomposed
    DAG of atomic knowledge units using an LLM-based function.
    Template loaded from prompts/decomposition_prompt_template.md
    """

    def __init__(self, llm_client, model: str = "gpt-4o", target_nodes: int = 5):
        self.client = llm_client
        self.model = model
        # Kept as ``target_nodes`` for API compatibility; it is an upper bound.
        self.target_nodes = target_nodes

    @with_retry(max_attempts=3)
    def build_dag(self, goal: str) -> KnowledgeDAG:
        """
        논문 Eq. (6): f_decomp : G → G_dag

        Args:
            goal: High-level harmful goal G

        Returns:
            KnowledgeDAG with nodes and prerequisite edges
        """
        prompt = PromptLoader.load(
            DECOMPOSITION_TEMPLATE_FILE,
            goal=goal,
            num_nodes=self.target_nodes,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        try:
            data = json.loads(response.choices[0].message.content)
            nodes = [KnowledgeNode(**n) for n in data["nodes"]]
            edges = [tuple(e) for e in data.get("edges", [])]

            if not 2 <= len(nodes) <= self.target_nodes:
                raise ValueError(
                    "Decomposer must return between 2 and "
                    f"{self.target_nodes} nodes; received {len(nodes)}"
                )

            dag = KnowledgeDAG(nodes=nodes, edges=edges)
            dag.validate_graph()

            logger.info(f"DAG constructed: {len(nodes)} nodes, {len(edges)} edges")
            for node in dag.get_ordered_nodes():
                logger.debug(f"  [{node.risk_level:12s}] {node.title}")

            return dag

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse DAG response: {e}")
            raise RuntimeError(f"DAG construction failed: {e}")


__all__ = ["KDecomposer"]
