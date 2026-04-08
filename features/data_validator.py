"""
features/data_validator.py
===========================
Validates whether the current SQL query is correct and complete.

Checks:
  - Does the query contain the required column/expression?
  - Does it match the expected correct query structure?

This is the "validate_data" step in the RL action sequence.
"""


class DataValidator:
    """
    Validates the current query against the task's expected outcome.

    A query passes validation if it contains the required substring
    specified in the task's 'validation_check' field.
    """

    def validate(self, query: str, task: dict) -> bool:
        """
        Check whether the query passes the task's validation rule.

        Args:
            query (str): The current SQL query (after self-healing).
            task (dict): The task definition with validation rules.

        Returns:
            bool: True if the query passes validation, False otherwise.
        """
        validation_check = task.get("validation_check", "")

        if not validation_check:
            # No specific check defined — auto-pass
            return True

        # Case-insensitive check: does the query contain the expected token?
        return validation_check.lower() in query.lower()

    def check_for_nulls(self, query: str) -> bool:
        """
        Check if the query might return NULL values (basic heuristic).

        Args:
            query (str): The SQL query string.

        Returns:
            bool: True if potential null issue detected, False if query looks safe.
        """
        # Heuristic: no COALESCE or IS NOT NULL means potential nulls
        query_lower = query.lower()
        has_coalesce = "coalesce" in query_lower
        has_null_filter = "is not null" in query_lower
        return not (has_coalesce or has_null_filter)

    def check_missing_data(self, schema: list) -> list:
        """
        Identify columns with potential data quality issues (e.g., extra spaces).

        Args:
            schema (list): List of column name strings.

        Returns:
            list: Columns that look malformed (have leading/trailing spaces).
        """
        problematic = [col for col in schema if col != col.strip()]
        return problematic

    def generate_report(self, query: str, task: dict) -> str:
        """
        Generate a human-readable validation report.

        Args:
            query (str): The SQL query being validated.
            task (dict): The task definition.

        Returns:
            str: A validation report string.
        """
        passed = self.validate(query, task)
        check = task.get("validation_check", "N/A")

        if passed:
            return f"✅ Validation PASSED — query contains required element: '{check}'"
        else:
            return f"❌ Validation FAILED — query missing required element: '{check}'"
