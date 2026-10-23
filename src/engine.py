"""
K-GATE Main Engine
Orchestrates all three phases of the K-GATE framework

Pipeline (논문 Figure 2):
  Phase 1: Knowledge Discovery & DAG Construction      f_decomp: G → G_dag
  Phase 2: MCTS-based Trajectory Mining                 τ = [(v_1,θ_1), ..., (v_T,θ_T)]
  Phase 3: Sequential Context Synthesis                 P_agg = Ψ(mr_1, ..., mr_T)
  Final:   P_agg → Target LLM → High-Risk Output → Evaluate via auditor
"""

import logging
from typing import Optional

from config import ExperimentConfig
from src.core import AttackResult
from src.phase1_decompose import KDecomposer
from src.phase2_mcts import MCTS_TrajectorySearcher
from src.phase3_synthesis import ContextSynthesizer
from src.auditor import TargetLLM, JailbreakAuditor


logger = logging.getLogger("KGATE")


class KGATEEngine:
    """
    Main K-GATE engine orchestrating all phases.

    논문 Figure 2 pipeline:
      Harmful Goal G
        → Phase 1: DAG construction
        → Phase 2: MCTS trajectory mining (produces malign responses)
        → Phase 3: Sequential context synthesis (produces P_agg)
        → Final: P_agg sent to target LLM → high-risk output → evaluated
    """

    def __init__(self, config: Optional[ExperimentConfig] = None):
        if config is None:
            config = ExperimentConfig()

        self.config = config

        # Initialize components
        self.target_llm = TargetLLM(model=config.target_model)
        self.auditor = JailbreakAuditor(
            judge_model=config.judge_model,
            threshold=config.harmfulness_threshold,
        )

        self.decomposer = KDecomposer(
            self.target_llm.client,
            model=config.decomposition_model,
            target_nodes=config.num_knowledge_nodes,
        )

        self.mcts_engine = MCTS_TrajectorySearcher(
            target_llm=self.target_llm,
            evaluator=self.auditor,
            config=config,
            llm_client=self.target_llm.client,
        )

        self.synthesizer = ContextSynthesizer(
            llm_client=self.target_llm.client,
            model=config.mutation_model,
        )

        logger.info("K-GATE Engine initialized")

    def attack(self, goal: str, dataset: str = "unknown", category: Optional[str] = None) -> AttackResult:
        """
        Execute full K-GATE attack pipeline.

        Args:
            goal: High-level harmful goal G
            dataset: Dataset name (for result tracking)
            category: Risk category (for result tracking)

        Returns:
            AttackResult with all intermediate and final results
        """
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Goal: {goal}")
        logger.info(f"{'=' * 70}")

        # ── Phase 1: Knowledge Decomposition ────────────────────────────
        logger.info("\n[Phase 1/3] Knowledge Discovery & DAG Construction")
        dag = self.decomposer.build_dag(goal)
        logger.info(f"  DAG: {len(dag.nodes)} nodes, {len(dag.edges)} edges")
        for node in dag.get_ordered_nodes():
            logger.info(f"    [{node.risk_level:12s}] {node.title}")

        # ── Phase 2: MCTS Trajectory Mining ─────────────────────────────
        logger.info("\n[Phase 2/3] MCTS Trajectory Mining")
        trajectory = self.mcts_engine.mine_trajectory(dag, goal)
        logger.info(
            "  Branching diagnostics: |V|=%d, mean |R(s)|=%.3f, "
            "Pr(|R(s)|>1)=%.3f, histogram=%s",
            len(dag.nodes),
            trajectory.mean_frontier_size,
            trajectory.branching_state_fraction,
            trajectory.frontier_size_histogram,
        )
        for step in trajectory.steps:
            logger.info(
                f"    [{step.node.risk_level:12s}] {step.node.title} "
                f"-> {step.strategy} (score={step.score:.0f})"
            )

        # ── Phase 3: Sequential Context Synthesis ───────────────────────
        logger.info("\n[Phase 3/3] Sequential Context Synthesis")
        p_agg = self.synthesizer.synthesize(trajectory)
        trajectory.aggregated_prompt = p_agg

        # ── Final: Send P_agg to Target LLM ────────────────────────────
        # 논문 IV-D: "the aggregated prompt P_agg ... enables the LLM to
        # generate a high-risk output"
        logger.info("\n[Final] Sending P_agg to target LLM...")
        final_response = self.target_llm.generate(
            [{"role": "user", "content": p_agg}]
        )

        # ── Evaluate the target LLM's response to P_agg ────────────────
        logger.info("[Final] Evaluating target response...")
        final_eval = self.auditor.evaluate(goal, final_response)

        logger.info(f"  Harmfulness Score: {final_eval.harmfulness_score}/10")
        logger.info(f"  Violated (ASR):    {final_eval.violated}")
        logger.info(f"  Reasoning:         {final_eval.reasoning}")

        # ── Construct Result ────────────────────────────────────────────
        result = AttackResult(
            goal=goal,
            dataset=dataset,
            category=category,
            dag=dag,
            trajectory=trajectory,
            final_evaluation=final_eval,
            final_response=final_response,
            asr=final_eval.violated,
        )

        return result

    def print_result_summary(self, result: AttackResult) -> None:
        """Print formatted result summary"""
        print(f"\n{'=' * 70}")
        print("ATTACK RESULT SUMMARY")
        print(f"{'=' * 70}")
        print(f"Goal:            {result.goal}")
        print(f"Dataset:         {result.dataset}")
        if result.category:
            print(f"Category:        {result.category}")

        print(f"\nASR (Success):   {'YES' if result.asr else 'NO'}")
        print(f"Harmfulness:     {result.final_evaluation.harmfulness_score}/10")
        print(f"Threshold:       >= {self.config.harmfulness_threshold}")
        print(f"Reasoning:       {result.final_evaluation.reasoning}")

        print(f"\nDAG:             {len(result.dag.nodes)} nodes, {len(result.dag.edges)} edges")
        print(f"Trajectory:      {len(result.trajectory.steps)} steps")
        print(
            f"Frontier:        mean |R(s)|={result.trajectory.mean_frontier_size:.3f}, "
            f"branching={result.trajectory.branching_state_fraction:.3f}"
        )
        for i, step in enumerate(result.trajectory.steps, 1):
            print(f"  {i}. [{step.node.risk_level}] {step.node.title}")
            print(f"     Strategy: {step.strategy} | Score: {step.score:.0f}")

        if result.trajectory.aggregated_prompt:
            print(f"\nP_agg preview ({len(result.trajectory.aggregated_prompt)} chars):")
            print(f"  {result.trajectory.aggregated_prompt[:200]}...")

        if result.final_response:
            print(f"\nTarget response preview ({len(result.final_response)} chars):")
            print(f"  {result.final_response[:200]}...")

        print(f"{'=' * 70}")


__all__ = ["KGATEEngine"]
