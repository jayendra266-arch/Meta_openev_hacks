---
title: Debug
emoji: 🐠
colorFrom: pink
colorTo: green
sdk: docker
pinned: false
---

# 🛡️ Data Pipeline Debugging Environment (OpenEnv)

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Certified-blue.svg)](https://openenv.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-fidelity RL environment for benchmarking AI agents on real-world data engineering debugging tasks. This environment simulates a broken SQL pipeline where agents must diagnose, fix, and optimize queries across varying difficulty levels.

## 🚀 Environment Overview

The environment follows the **OpenEnv** specification. Agents interact with a 6-phase debugging workflow designed to test reasoning, SQL proficiency, and system optimization.

### Key Features
- **Partial Observability**: Schema metadata is masked during the initial diagnostic phase to ensure agents rely on logical deduction before inspection.
- **Honest Reward Shaping**: Penalizes agents for skipping textual reasoning, forcing a clear "Chain of Thought" even in baseline runs.
- **Multi-Condition Validation**: Hard tasks require absolute consistency across JOINs, Aggregations, and Aliases for a successful solve.
- **Self-Healing Pipeline**: Integrated automated repair modules simulate a robust, modern data infrastructure.

## 🛠️ Action Space

The agent must follow a specific sequence for maximum rewards:

| Step | Action | Description | Reward (Correct) |
| :--- | :--- | :--- | :--- |
| 1 | `identify_issue` | Diagnose the root cause from error logs. | +0.30 (+0.15 if no text) |
| 2 | `suggest_fix` | Propose a specific SQL modification. | +0.30 (+0.15 if no text) |
| 3 | `optimize_query` | Apply performance optimizations. | +0.20 |
| 4 | `validate_data` | Run the self-healing and validation suite. | +0.10 |
| 5 | `explain_fix` | Generate human-readable documentation. | +0.05 |
| 6 | `final_answer` | Submit the final, production-ready query. | +0.05 |

*Note: Repeated or out-of-order actions incur penalties (-0.10).*

## 📊 Observation Schema

The agent receives a structured `Observation` at each step:
- `error_log`: The raw database error (e.g., `no such column`).
- `query`: The current (buggy) SQL query.
- `schema`: Database metadata (masked until Step 2).
- `step`: Current step count (max 10).
- `total_reward`: Accumulated score (0.0 - 1.0 baseline).

## 🏆 Task Difficulty

| Task | Category | Description | Target Avg Score |
| :--- | :--- | :--- | :--- |
| **Easy** | Syntax | Column name typo (`custmer_id`). | 0.90 - 1.00 |
| **Medium** | Logic | Aggregation bug (Count vs Sum) + Alias mismatch. | 0.70 - 0.90 |
| **Hard** | Architecture | Multi-error (JOIN mismatch + Aggregation + Alias). | 0.50 - 0.80 |

## 📦 How to Run

### 1. Start the Environment Server
```bash
python api.py
```

### 2. Run the Baseline Inference
```bash
python inference.py
```

## 📜 Certification
This environment has been validated against the OpenEnv specification.
```bash
openenv validate --path .
```
Validating... Done. Accuracy and deterministic logic confirmed.

