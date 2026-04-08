"""
tasks/hard.py
=============
Hard difficulty task for the Data Pipeline Debugging Environment.

Issue introduced:
MULTI-ERROR scenario:
1. Join mismatch (user_id instead of uid).
2. Wrong aggregation (COUNT(*) instead of SUM(payment)).
3. Alias inconsistency (AS active_users vs AS total_spent).
"""

HARD_TASK = {
    "name": "hard",
    "difficulty": "hard",
    "issue_type": "multi_error_bug",

    # Contains 3 errors
    "query": "SELECT u.name, COUNT(*) AS active_users FROM users u JOIN payments p ON u.id = p.user_id GROUP BY u.name",
    
    # Fully fixed query
    "correct_query": "SELECT u.name, SUM(p.payment) AS total_spent FROM users u JOIN payments p ON u.id = p.uid GROUP BY u.name",
    
    # Expected components for string matching evaluation
    "expected_issue_type": ["join", "aggregation", "alias", "uid", "sum"],
    "expected_fix_pattern": ["sum(p.payment)", "p.uid", "total_spent"],
    "expected_query_pattern": ["sum(p.payment)", "p.uid", "total_spent"],

    "schema": [
        "users.id",
        "users.name",
        "payments.uid",
        "payments.payment"
    ],
    
    "error_log": (
        "ExecutionError: JOIN clause failed. Column 'p.user_id' does not exist. "
        "Also, metric validation failed — expected total transaction volume 'total_spent', not counts."
    ),

    "expected_issue": "JOIN mismatch, aggregation error, and alias inconsistency.",
    "expected_fix": "Fix JOIN to uid, change to SUM, and set total_spent alias.",
    "validation_check": "p.uid",
    "needs_optimization": True
}
