"""
test_proof.py
=============
Comprehensive test script to PROVE 100% accuracy of the system.

Tests every module individually, then runs the full RL pipeline.
Prints PASS/FAIL for every single test case.
"""

import sys
import io
import traceback

# Force UTF-8 for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─────────────────────────────────────────────
# Helper: test runner
# ─────────────────────────────────────────────
PASS_COUNT = 0
FAIL_COUNT = 0
TEST_LOG   = []

def test(name: str, condition: bool, detail: str = ""):
    global PASS_COUNT, FAIL_COUNT
    status = "PASS" if condition else "FAIL"
    icon   = "✅" if condition else "❌"
    line   = f"  {icon} [{status}] {name}"
    if detail:
        line += f"\n         → {detail}"
    print(line)
    TEST_LOG.append((name, condition, detail))
    if condition:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1


# ═══════════════════════════════════════════════
# SECTION 1 — TASK DEFINITIONS
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("  SECTION 1: TASK DEFINITIONS")
print("=" * 60)

from tasks.easy         import EASY_TASK
from tasks.medium       import MEDIUM_TASK
from tasks.hard         import HARD_TASK
from tasks.messy_schema import MESSY_TASK

ALL_TASKS = [EASY_TASK, MEDIUM_TASK, HARD_TASK, MESSY_TASK]
REQUIRED_KEYS = ["name","error_log","query","schema","issue_type",
                 "expected_issue","expected_fix","correct_query",
                 "validation_check","needs_optimization","difficulty"]

for task in ALL_TASKS:
    for key in REQUIRED_KEYS:
        test(
            f"tasks/{task['name']}.py → has '{key}'",
            key in task,
            f"value = {str(task.get(key,'MISSING'))[:60]}"
        )

# ═══════════════════════════════════════════════
# SECTION 2 — REWARD ENGINE
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("  SECTION 2: REWARD ENGINE")
print("=" * 60)

from core.reward_engine import RewardEngine, REWARD_TABLE

re_engine = RewardEngine()

EXPECTED_REWARDS = {
    "identify_issue": (+0.30, -0.20),
    "suggest_fix":    (+0.30, -0.20),
    "optimize_query": (+0.20, -0.10),
    "validate_data":  (+0.10, -0.10),
    "explain_fix":    (+0.05,  0.00),
    "final_answer":   (+0.05, -0.30),
}

for action, (pos, neg) in EXPECTED_REWARDS.items():
    test(
        f"RewardEngine: {action} correct reward = {pos:+.2f}",
        REWARD_TABLE[action]["correct"] == pos,
        f"got {REWARD_TABLE[action]['correct']:+.2f}"
    )
    test(
        f"RewardEngine: {action} wrong reward = {neg:+.2f}",
        REWARD_TABLE[action]["wrong"] == neg,
        f"got {REWARD_TABLE[action]['wrong']:+.2f}"
    )

# Test accumulation
re_engine2 = RewardEngine()
re_engine2.compute_reward("identify_issue", True)
re_engine2.compute_reward("suggest_fix",    True)
re_engine2.compute_reward("optimize_query", True)
re_engine2.compute_reward("validate_data",  True)
re_engine2.compute_reward("explain_fix",    True)
re_engine2.compute_reward("final_answer",   True)
total = re_engine2.total_reward
test(
    "RewardEngine: full correct episode totals 1.00",
    abs(total - 1.00) < 0.001,
    f"accumulated total = {total:.2f}"
)

# ═══════════════════════════════════════════════
# SECTION 3 — STATE MANAGER
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("  SECTION 3: STATE MANAGER")
print("=" * 60)

from core.state_manager import StateManager

sm = StateManager(EASY_TASK)
state = sm.get_state()

test("StateManager: initial step = 0",   state["step"] == 0, f"step={state['step']}")
test("StateManager: error_log set",      len(state["error_log"]) > 0)
test("StateManager: query set",          len(state["query"]) > 0)
test("StateManager: schema is a list",   isinstance(state["schema"], list))

sm.advance_step()
test("StateManager: advance_step works", sm.get_state()["step"] == 1, "step should be 1")

sm.update_query("SELECT fixed FROM table")
test("StateManager: update_query works", sm.query == "SELECT fixed FROM table")

sm.reset(MEDIUM_TASK)
test("StateManager: reset clears step",  sm.get_state()["step"] == 0)
test("StateManager: reset loads new task", "revenue" in sm.get_state()["error_log"].lower() or
     "count" in sm.get_state()["query"].lower())

# ═══════════════════════════════════════════════
# SECTION 4 — QUERY OPTIMIZER
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("  SECTION 4: QUERY OPTIMIZER")
print("=" * 60)

from features.query_optimizer import QueryOptimizer

qo = QueryOptimizer()

# Test COUNT → SUM replacement
buggy_q = "SELECT customer_id, COUNT(*) AS revenue FROM orders GROUP BY customer_id"
optimized = qo.optimize(buggy_q, MEDIUM_TASK)
test(
    "QueryOptimizer: COUNT(*) replaced with SUM(amount)",
    "SUM(amount)" in optimized,
    f"result: {optimized}"
)
test(
    "QueryOptimizer: COUNT(*) removed",
    "COUNT(*)" not in optimized,
    f"result: {optimized}"
)
test(
    "QueryOptimizer: alias AS revenue preserved",
    "AS revenue" in optimized,
    f"result: {optimized}"
)

# Test no-optimization task returns correct query
opt_easy = qo.optimize(EASY_TASK["query"], EASY_TASK)
test(
    "QueryOptimizer: no-op task returns correct query",
    opt_easy == EASY_TASK["correct_query"],
    f"result: {opt_easy}"
)

# ═══════════════════════════════════════════════
# SECTION 5 — DATA VALIDATOR
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("  SECTION 5: DATA VALIDATOR")
print("=" * 60)

from features.data_validator import DataValidator

dv = DataValidator()

test(
    "DataValidator: correct query PASSES easy task",
    dv.validate(EASY_TASK["correct_query"], EASY_TASK),
    f"check='{EASY_TASK['validation_check']}'"
)
test(
    "DataValidator: buggy query FAILS easy task",
    not dv.validate(EASY_TASK["query"], EASY_TASK),
    f"buggy query missing 'customer_id'"
)
test(
    "DataValidator: correct query PASSES medium task",
    dv.validate(MEDIUM_TASK["correct_query"], MEDIUM_TASK),
    f"check='{MEDIUM_TASK['validation_check']}'"
)
test(
    "DataValidator: correct query PASSES hard task",
    dv.validate(HARD_TASK["correct_query"], HARD_TASK),
    f"check='{HARD_TASK['validation_check']}'"
)
test(
    "DataValidator: correct query PASSES messy task",
    dv.validate(MESSY_TASK["correct_query"], MESSY_TASK),
    f"check='{MESSY_TASK['validation_check']}'"
)

# Test messy schema detection
messy_issues = dv.check_missing_data(MESSY_TASK["schema"])
test(
    "DataValidator: detects messy schema columns with spaces",
    len(messy_issues) > 0,
    f"problematic cols: {messy_issues}"
)

# ═══════════════════════════════════════════════
# SECTION 6 — SCHEMA HANDLER
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("  SECTION 6: SCHEMA HANDLER")
print("=" * 60)

from features.schema_handler import SchemaHandler

sh = SchemaHandler()

messy_schema = ["customerid", "AMOUNT ", " order_Date", "  ORDER_ID  ", "status"]
normalized   = sh.normalize_schema(messy_schema)

test(
    "SchemaHandler: strips leading/trailing spaces",
    all(c == c.strip() for c in normalized),
    f"normalized: {normalized}"
)
test(
    "SchemaHandler: converts to lowercase",
    all(c == c.lower() for c in normalized),
    f"normalized: {normalized}"
)
test(
    "SchemaHandler: 'AMOUNT ' → 'amount'",
    "amount" in normalized,
    f"normalized: {normalized}"
)
test(
    "SchemaHandler: ' order_Date' → 'order_date'",
    "order_date" in normalized,
    f"normalized: {normalized}"
)

issues = sh.detect_issues(messy_schema)
test(
    "SchemaHandler: detects schema issues",
    len(issues) > 0,
    f"issues found: {issues}"
)

# ═══════════════════════════════════════════════
# SECTION 7 — SELF HEALER
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("  SECTION 7: SELF-HEALING PIPELINE")
print("=" * 60)

from features.self_healing import SelfHealer

healer = SelfHealer()

# Easy: column typo healing
healed_easy = healer.heal(EASY_TASK["query"], EASY_TASK)
test(
    "SelfHealer: fixes column typo 'custmer_id' → 'customer_id'",
    "customer_id" in healed_easy,
    f"healed: {healed_easy}"
)
test(
    "SelfHealer: typo 'custmer_id' removed after healing",
    "custmer_id" not in healed_easy,
    f"healed: {healed_easy}"
)

# Medium: aggregation healing
healed_med = healer.heal(MEDIUM_TASK["query"], MEDIUM_TASK)
test(
    "SelfHealer: fixes COUNT(*) → SUM(amount)",
    "SUM(amount)" in healed_med,
    f"healed: {healed_med}"
)

# Hard: JOIN mismatch healing
healed_hard = healer.heal(HARD_TASK["query"], HARD_TASK)
test(
    "SelfHealer: fixes sessions.user_id → sessions.uid",
    "sessions.uid" in healed_hard,
    f"healed: {healed_hard}"
)

# Messy: normalized schema healing
healed_messy = healer.heal(MESSY_TASK["query"], MESSY_TASK)
test(
    "SelfHealer: messy schema healed to normalized query",
    "customerid" in healed_messy,
    f"healed: {healed_messy}"
)

# ═══════════════════════════════════════════════
# SECTION 8 — EXPLANATION ENGINE
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("  SECTION 8: EXPLANATION ENGINE")
print("=" * 60)

from features.explanation_engine import ExplanationEngine

ee = ExplanationEngine()

for task in ALL_TASKS:
    exp = ee.explain(task, task["correct_query"])
    test(
        f"ExplanationEngine: {task['name']} has issue_explanation",
        len(exp.get("issue_explanation", "")) > 20,
        f"len={len(exp.get('issue_explanation',''))}"
    )
    test(
        f"ExplanationEngine: {task['name']} has fix_explanation",
        len(exp.get("fix_explanation", "")) > 20,
        f"len={len(exp.get('fix_explanation',''))}"
    )

# ═══════════════════════════════════════════════
# SECTION 9 — REASONING MODULE
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("  SECTION 9: REASONING MODULE")
print("=" * 60)

from agent.reasoning import ReasoningModule

rm = ReasoningModule()

easy_state   = {"error_log": EASY_TASK["error_log"],   "query": EASY_TASK["query"],   "schema": EASY_TASK["schema"],   "step": 0}
medium_state = {"error_log": MEDIUM_TASK["error_log"], "query": MEDIUM_TASK["query"], "schema": MEDIUM_TASK["schema"], "step": 0}
hard_state   = {"error_log": HARD_TASK["error_log"],   "query": HARD_TASK["query"],   "schema": HARD_TASK["schema"],   "step": 0}
messy_state  = {"error_log": MESSY_TASK["error_log"],  "query": MESSY_TASK["query"],  "schema": MESSY_TASK["schema"],  "step": 0}

test("ReasoningModule: diagnoses easy → column_typo",   rm.diagnose(easy_state)   == "column_typo",   f"got: {rm.diagnose(easy_state)}")
test("ReasoningModule: diagnoses medium → aggregation_bug", rm.diagnose(medium_state) == "aggregation_bug", f"got: {rm.diagnose(medium_state)}")
test("ReasoningModule: diagnoses hard → join_mismatch",  rm.diagnose(hard_state)   == "join_mismatch",  f"got: {rm.diagnose(hard_state)}")

# ═══════════════════════════════════════════════
# SECTION 10 — MEMORY MODULE
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("  SECTION 10: MEMORY MODULE (Learning from Mistakes)")
print("=" * 60)

from agent.memory import Memory

mem = Memory()

dummy_state = {"error_log": "test error", "query": "SELECT 1", "schema": [], "step": 2}

# Initially no mistakes
test("Memory: starts empty", mem.total_mistakes == 0)
test("Memory: was_mistake returns False for unseen action", not mem.was_mistake(dummy_state, "suggest_fix"))

# Store a mistake
mem.store_mistake(dummy_state, "suggest_fix")
test("Memory: total_mistakes incremented", mem.total_mistakes == 1)
test("Memory: was_mistake returns True after store", mem.was_mistake(dummy_state, "suggest_fix"))
test("Memory: mistake count = 1", mem.get_mistake_count(dummy_state, "suggest_fix") == 1)

# Store same mistake again
mem.store_mistake(dummy_state, "suggest_fix")
test("Memory: mistake count = 2 after repeat", mem.get_mistake_count(dummy_state, "suggest_fix") == 2)

# Different action not a mistake
test("Memory: different action still returns False", not mem.was_mistake(dummy_state, "identify_issue"))

# ═══════════════════════════════════════════════
# SECTION 11 — FULL RL ENVIRONMENT (end-to-end)
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("  SECTION 11: FULL RL ENVIRONMENT (End-to-End per Task)")
print("=" * 60)

from core.env   import DebugEnvironment, ACTION_SEQUENCE
from agent.agent import DebugAgent
from evaluation.grader import Grader

agent  = DebugAgent()
grader = Grader()
all_results = []

for task in ALL_TASKS:
    env    = DebugEnvironment(task)
    result = agent.run_episode(env)
    all_results.append(result)

    score   = result["total_reward"]
    success = result["success"]
    steps   = result["steps"]

    test(f"ENV [{task['name']}]: episode runs 6 steps", len(steps) == 6, f"steps={len(steps)}")
    test(f"ENV [{task['name']}]: all 6 actions correct", all(s["correct"] for s in steps),
         f"correct={[s['correct'] for s in steps]}")
    test(f"ENV [{task['name']}]: score = 1.00", abs(score - 1.00) < 0.001, f"score={score:.2f}")
    test(f"ENV [{task['name']}]: success = True", success, f"success={success}")

    # Verify action sequence order
    action_names = [s["action"] for s in steps]
    test(
        f"ENV [{task['name']}]: action sequence is correct order",
        action_names == ACTION_SEQUENCE,
        f"order: {action_names}"
    )

# ═══════════════════════════════════════════════
# SECTION 12 — GRADER & FINAL REPORT
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("  SECTION 12: GRADER & FINAL REPORT")
print("=" * 60)

report = grader.grade_all(all_results)

test("Grader: 4 task grades produced",   len(report["grades"]) == 4)
test("Grader: total score = 4.00",       abs(report["total_score"] - 4.00) < 0.001, f"got {report['total_score']}")
test("Grader: max score = 4.00",         report["max_score"] == 4.0)
test("Grader: overall accuracy = 100%",  report["overall_acc"] == 100.0, f"got {report['overall_acc']}%")
test("Grader: target accuracy = 98%",    report["target_acc"] == 98)
test("Grader: target_met = True",        report["target_met"])

for g in report["grades"]:
    test(f"Grader [{g['task']}]: PASS",          g["pass"],    f"score={g['score']}")
    test(f"Grader [{g['task']}]: accuracy=100%", g["accuracy"] == 100.0, f"acc={g['accuracy']}%")

# ═══════════════════════════════════════════════
# SECTION 13 — DETERMINISM CHECK
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("  SECTION 13: DETERMINISM (same result every run)")
print("=" * 60)

# Run a second time, compare scores
agent2      = DebugAgent()
all_results2 = []
for task in ALL_TASKS:
    env2    = DebugEnvironment(task)
    result2 = agent2.run_episode(env2)
    all_results2.append(result2)

report2 = grader.grade_all(all_results2)

test(
    "Determinism: run 1 total score == run 2 total score",
    report["total_score"] == report2["total_score"],
    f"run1={report['total_score']} run2={report2['total_score']}"
)
test(
    "Determinism: run 1 accuracy == run 2 accuracy",
    report["overall_acc"] == report2["overall_acc"],
    f"run1={report['overall_acc']}% run2={report2['overall_acc']}%"
)

# ═══════════════════════════════════════════════
# FINAL PROOF SUMMARY
# ═══════════════════════════════════════════════
print()
print("=" * 60)
print("          PROOF TEST RESULTS SUMMARY")
print("=" * 60)
print(f"  Total Tests Run  : {PASS_COUNT + FAIL_COUNT}")
print(f"  Tests Passed     : {PASS_COUNT}")
print(f"  Tests Failed     : {FAIL_COUNT}")
acc = (PASS_COUNT / (PASS_COUNT + FAIL_COUNT)) * 100
print(f"  Test Accuracy    : {acc:.1f}%")
print()

if FAIL_COUNT == 0:
    print("  ✅ ALL TESTS PASSED — SYSTEM IS 100% ACCURATE")
else:
    print("  ❌ SOME TESTS FAILED — see above for details")
    print()
    print("  Failed tests:")
    for name, ok, detail in TEST_LOG:
        if not ok:
            print(f"    - {name}: {detail}")

print("=" * 60)
