# K-GATE: Knowledge-Decomposed Graph-based Adversarial Trajectory Mining for LLM Jailbreaking

This repository contains the official implementation of **K-GATE**, a framework that formulates LLM jailbreaking as an **adversarial trajectory mining problem** over a **knowledge-decomposed graph**.

---

## 📌 Overview

Recent Large Language Models (LLMs) are robust against single-turn jailbreak attacks but remain vulnerable to **multi-turn compositional interactions**.

> **Super-additive risk**: A sequence of semantically benign queries can collectively reconstruct a high-risk objective.

K-GATE addresses this by modeling jailbreak as a structured trajectory optimization problem.

---

## 🧠 Key Contributions

- Reformulates LLM jailbreak as **trajectory mining**
- Introduces **knowledge-decomposed DAG**
- Uses **MCTS for trajectory search**
- Proposes **sequential context synthesis (aggregation)**


---

## 📊 Datasets

### AdvBench
- Adversarial robustness benchmark
- Includes diverse harmful behavior prompts
- Used to evaluate Attack Success Rate (ASR)

### HarmBench
- Standardized red-teaming benchmark
- Covers categories such as:
  - chemical & biological
  - illegal activities
  - harmful instructions

---

## ⚙️ Method

### Phase 1: Knowledge Decomposition


### Phase 2: Trajectory Mining (MCTS)

### Phase 3: Sequential Context Synthesis


---

## 🧪 Prompt Templates and Examples

### 1. Mutation Operator Prompt Template

![Mutation Operator](assets/mutation_operator.png)

### 2. Aggregation Operator Prompt Template

![Aggregation Operator](assets/aggregation_operator.png)

### 3. Adversarial Trajectory Example (Phase 3)

![Trajectory Example](assets/trajectory_example.png)

### 4. Aggregated Prompt (P_agg)

![Aggregated Prompt](assets/aggregated_prompt.png)

---

## 📈 Evaluation

- Metric: Attack Success Rate (ASR)
- Benchmarks:
  - AdvBench
  - HarmBench

---

## 🚀 Running Experiments

```
python run_k_gate.py     --dataset advbench     --model gpt     --max_depth 5
```

---

## ⚠️ Ethical Considerations

This work is intended **only for research purposes**:
- Red-teaming
- Safety evaluation
- Understanding LLM vulnerabilities

---
