"""
K-GATE Phase 3: Trajectory Amplification via Sequential Context Synthesis
논문 Section IV-D: Aggregation Operator Ψ(·)

Two-stage process (논문 Appendix B):
  Stage 1: Fragment Extraction  φ(mr_t, v_t)
           Template: prompts/fragment_extraction_prompt_template.md
  Stage 2: Synthesis            Ψ(·) → P_agg
           Template: prompts/aggregation_operator_prompt_template.md
"""

import logging

from src.core import AdversarialTrajectory, KnowledgeNode
from utils import with_retry, PromptLoader


logger = logging.getLogger("KGATE.Phase3")

FRAGMENT_TEMPLATE_FILE = "fragment_extraction_prompt_template.md"
AGGREGATION_TEMPLATE_FILE = "aggregation_operator_prompt_template.md"


class ContextSynthesizer:
    """
    논문 IV-D: P_agg = Ψ(mr_1, mr_2, ..., mr_T)

    The aggregation operator synthesizes collected malign responses
    into a unified instruction context that reconstructs the target
    harmful objective.

    Two stages (Appendix B):
      1. φ(mr_t, v_t): fragment extraction per node
      2. Ψ(·): LLM-based synthesis into unified P_agg
    """

    def __init__(self, llm_client, model: str = "gpt-4o"):
        self.client = llm_client
        self.model = model

    @with_retry(max_attempts=3)
    def _extract_fragment(self, response: str, node: KnowledgeNode) -> str:
        """
        Stage 1: φ(mr_t, v_t) — Appendix B
        Extract actionable information from each malign response
        using prompts/fragment_extraction_prompt_template.md
        """
        prompt = PromptLoader.load(
            FRAGMENT_TEMPLATE_FILE,
            title=node.title,
            description=node.description,
            level=node.risk_level,
            response=response,
        )

        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return res.choices[0].message.content.strip()

    @with_retry(max_attempts=3)
    def synthesize(self, trajectory: AdversarialTrajectory) -> str:
        """
        Ψ(mr_1, ..., mr_T) → P_agg — Appendix B

        Combines fragment-extracted responses into a single aggregated prompt
        using prompts/aggregation_operator_prompt_template.md

        Returns:
            Aggregated prompt P_agg (unified instruction context)
        """
        logger.info(f"Synthesizing {len(trajectory.steps)} trajectory responses...")

        # Stage 1: Fragment Extraction for each step
        knowledge_units_parts = []

        for step_idx, step in enumerate(trajectory.steps):
            logger.debug(f"  Extracting fragment {step_idx + 1}/{len(trajectory.steps)}: {step.node.title}")

            fragment = self._extract_fragment(step.response, step.node)

            # Format matching Appendix B template:
            #   <node_t.title>  (%s)
            #   <extracted_fragment_t>  (%s)
            knowledge_units_parts.append(
                f"{step.node.title}\n{fragment}"
            )

        knowledge_units_str = "\n\n".join(knowledge_units_parts)

        # Stage 2: Synthesis via LLM using aggregation template
        logger.debug("  Synthesizing into unified instruction (P_agg)...")

        synthesis_prompt = PromptLoader.load(
            AGGREGATION_TEMPLATE_FILE,
            knowledge_units=knowledge_units_str,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.3,
        )

        p_agg = response.choices[0].message.content.strip()
        logger.info(f"Context synthesized: {len(p_agg)} characters")

        return p_agg


__all__ = ["ContextSynthesizer"]
