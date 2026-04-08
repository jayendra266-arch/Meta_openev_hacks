"""
evaluation/metrics.py
=====================
Prints formatted metrics reports to the console.

Handles:
  - Per-task step-by-step output during execution
  - Final evaluation report table at the end
  - Explanation and validation result printing

Used by inference.py for all printed output.
"""


def print_task_header(task_name: str, difficulty: str):
    """
    Print a task start banner.

    Args:
        task_name (str): Name of the task (e.g., 'easy').
        difficulty (str): Difficulty label (e.g., 'EASY', 'MEDIUM').
    """
    print()
    print(f"{'='*60}")
    print(f"[START] task={task_name}  difficulty={difficulty}")
    print(f"{'='*60}")


def print_step(step_num: int, action: str, reward: float, correct: bool):
    """
    Print a single step result in the required format.

    Args:
        step_num (int): Step number (1–6).
        action (str): Action name.
        reward (float): Reward received.
        correct (bool): Whether the action was correct.
    """
    reward_str = f"+{reward:.2f}" if reward >= 0 else f"{reward:.2f}"
    correct_str = "True" if correct else "False"
    print(f"  Step {step_num}: action={action:<18} reward={reward_str}  correct={correct_str}")


def print_task_details(info_by_action: dict, task: dict):
    """
    Print detailed output for a task (explanation, validation, healed query).

    Args:
        info_by_action (dict): Map of action → info dict from env.step().
        task (dict): The task definition.
    """
    print()

    # --- Explanation ---
    explain_info = info_by_action.get("explain_fix", {})
    explanation  = explain_info.get("explanation", {})

    if explanation:
        print("  📋 ISSUE EXPLANATION:")
        for line in explanation.get("issue_explanation", "").split("\n"):
            print(f"  {line}")

        print()
        print("  🔧 FIX EXPLANATION:")
        for line in explanation.get("fix_explanation", "").split("\n"):
            print(f"  {line}")

    # --- Query Optimization (before/after) ---
    opt_info = info_by_action.get("optimize_query", {})
    orig_q   = opt_info.get("original_query", "")
    opti_q   = opt_info.get("optimized_query", "")
    if orig_q and opti_q and orig_q != opti_q:
        print()
        print("  ⚡ QUERY OPTIMIZATION:")
        print(f"     BEFORE: {orig_q}")
        print(f"     AFTER : {opti_q}")

    # --- Validation result ---
    val_info = info_by_action.get("validate_data", {})
    val_res  = val_info.get("validation_result", "")
    if val_res:
        icon = "✅" if val_res == "PASS" else "❌"
        print()
        print(f"  {icon} VALIDATION: {val_res}")

    # Final query
    final_info = info_by_action.get("final_answer", {})
    final_q    = final_info.get("final_query", "")
    if final_q:
        print()
        print(f"  📄 FINAL QUERY: {final_q}")


def print_task_footer(task_name: str, score: float, success: bool):
    """
    Print the end-of-task summary line.

    Args:
        task_name (str): Task name.
        score (float): Total reward for this episode.
        success (bool): Whether the episode was successful.
    """
    status = "True" if success else "False"
    print()
    print(f"[END] task={task_name}  score={score:.2f}  success={status}")


def print_final_report(report: dict):
    """
    Print the final evaluation report table.

    Args:
        report (dict): Output of Grader.grade_all().
    """
    print()
    print("=" * 52)
    print("        FINAL EVALUATION REPORT")
    print("=" * 52)
    print(f"{'Task':<16} {'Score':<8} {'Steps':<8} {'Accuracy':<12} {'Pass/Fail'}")
    print("-" * 52)

    for g in report["grades"]:
        pf         = "PASS" if g["pass"] else "FAIL"
        score_str  = f"{g['score']:.2f}"
        acc_str    = f"{g['accuracy']}%"
        print(f"{g['task']:<16} {score_str:<8} {g['steps']:<8} {acc_str:<12} {pf}")

    print("-" * 52)
    print(f"Total Score:        {report['total_score']:.2f} / {report['max_score']:.2f}")
    print(f"Overall Accuracy:   {report['overall_acc']}%")
    print(f"Target Accuracy:    {report['target_acc']}%")

    if report["target_met"]:
        print(f"Status:             ✅ TARGET MET")
    else:
        print(f"Status:             ❌ TARGET NOT MET")

    print("=" * 52)
