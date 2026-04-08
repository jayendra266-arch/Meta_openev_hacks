"""
tasks/easy.py
=============
EASY task: The pipeline has a column typo.
'custmer_id' must be 'customer_id'.

This is a beginner-level RL task to teach the agent
how to spot a simple naming mistake in a SQL query.
"""

# The task dictionary contains everything the environment needs
EASY_TASK = {
    "name": "easy",

    # What went wrong (shown to the agent as an error log)
    "error_log": "OperationalError: no such column: custmer_id",

    # The buggy SQL query that needs fixing
    "query": "SELECT custmer_id, SUM(amount) FROM orders GROUP BY custmer_id",

    # The database schema (list of valid column names)
    "schema": ["customer_id", "order_id", "amount", "order_date"],

    # What type of issue this is (used by the reward engine)
    "issue_type": "column_typo",

    # The correct, expected issue description
    "expected_issue": "Column name typo: 'custmer_id' should be 'customer_id'",

    # What the fix should look like
    "expected_fix": "Replace 'custmer_id' with 'customer_id'",

    # The corrected query
    "correct_query": "SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id",

    # Validation rule: what pattern must appear in the final query
    "validation_check": "customer_id",

    # Whether any aggregation optimization should be attempted
    "needs_optimization": False,

    # difficulty level for logging/display
    "difficulty": "EASY",

    # NEW: Expected outputs for deterministic grading
    "expected_issue_type": ["column_typo"],
    "expected_fix_pattern": ["customer_id"],
    "expected_query_pattern": ["customer_id", "sum(amount)"]
}
