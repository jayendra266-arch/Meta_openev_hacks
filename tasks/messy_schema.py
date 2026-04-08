"""
tasks/messy_schema.py
=====================
MESSY SCHEMA task: Dirty, inconsistent schema names.
Combines uppercase/lowercase mismatch, extra spaces, and naming inconsistency.

This tests the agent's ability to handle real-world messy data engineering.
"""

MESSY_TASK = {
    "name": "messy_schema",

    # The error comes from mismatched casing in column names
    "error_log": (
        "SchemaError: Column '  Customer_ID  ' not found. "
        "Available columns: ['customerid', 'AMOUNT ', ' order_Date']. "
        "Check for casing or whitespace issues."
    ),

    # Query referencing badly-named schema fields
    "query": "SELECT Customer_ID, SUM(AMOUNT) FROM Orders WHERE order_Date > '2024-01-01'",

    # Messy schema: mixed casing, extra spaces, inconsistent naming
    "schema": ["customerid", "AMOUNT ", " order_Date", "  ORDER_ID  ", "status"],

    "issue_type": "messy_schema",

    "expected_issue": (
        "Dirty schema: uppercase/lowercase mismatch, extra spaces in column names. "
        "'Customer_ID' should be 'customerid', 'AMOUNT' has trailing space, 'order_Date' has leading space."
    ),

    "expected_fix": (
        "Normalize schema: strip whitespace, lowercase all column names. "
        "Update query to match normalized column names."
    ),

    # The cleaned-up version of the query after schema normalization
    "correct_query": "SELECT customerid, SUM(amount) FROM orders WHERE order_date > '2024-01-01'",

    # Validation: final query must use normalized (lowercase, no spaces) column names
    "validation_check": "customerid",

    "needs_optimization": False,

    "difficulty": "MESSY"
}
