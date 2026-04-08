"""
tasks/medium.py
===============
Medium difficulty task for the Data Pipeline Debugging Environment.

Issue introduced:
1. Aggregation issue (COUNT(*) instead of SUM(amount))
2. Alias mismatch (AS amt instead of AS revenue) -> subtle ambiguity
"""

MEDIUM_TASK = {
    "name": "medium",
    "difficulty": "medium",
    "issue_type": "aggregation_alias_bug",
    
    # Needs two fixes: COUNT(*) -> SUM(amount) AND AS amt -> AS revenue
    "query": "SELECT user_id, COUNT(*) AS amt FROM transactions GROUP BY user_id",
    
    # The true required query that fully fixes both
    "correct_query": "SELECT user_id, SUM(amount) AS revenue FROM transactions GROUP BY user_id",
    
    # Expected components for string matching evaluation
    "expected_issue_type": ["aggregation", "alias", "sum", "revenue"],
    "expected_fix_pattern": ["sum", "revenue"],
    "expected_query_pattern": ["sum(amount)", "as revenue"],

    "schema": [
        "transactions.user_id",
        "transactions.amount",
        "transactions.created_at"
    ],
    
    "error_log": (
        "AnalysisError: Metric mismatch in financial report. "
        "Expected column 'revenue' holding total processed dollars, but got generic counts."
    ),

    "expected_issue": "Aggregation bug and alias mismatch.",
    "expected_fix": "Fix SUM and revenue alias.",
    "validation_check": "sum(amount)",
    "needs_optimization": True
}
