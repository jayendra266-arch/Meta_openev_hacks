"""
features/schema_handler.py
===========================
Handles messy, real-world database schemas.

Real-world schemas often have:
  - Uppercase/lowercase inconsistencies (e.g., 'AMOUNT' vs 'amount')
  - Leading/trailing spaces (e.g., ' order_Date' instead of 'order_date')
  - Inconsistent naming (e.g., 'customerid' vs 'customer_id')

This module normalizes column names to a clean, consistent format.
"""


class SchemaHandler:
    """
    Normalizes and analyzes messy schema column names.
    """

    def normalize_schema(self, schema: list) -> list:
        """
        Clean up a list of column names:
          - Strip leading/trailing whitespace
          - Convert to lowercase

        Args:
            schema (list): Raw schema with potentially messy column names.

        Returns:
            list: Cleaned list of column names.
        """
        return [col.strip().lower() for col in schema]

    def detect_issues(self, schema: list) -> list:
        """
        Identify which column names have formatting problems.

        Args:
            schema (list): List of raw column names.

        Returns:
            list: List of problem descriptions.
        """
        issues = []

        for col in schema:
            problems = []

            # Check for leading or trailing spaces
            if col != col.strip():
                problems.append(f"extra whitespace in '{col}'")

            # Check for uppercase characters (everything should be lowercase for consistency)
            if col != col.lower() and col.strip() != col.strip().lower():
                problems.append(f"uppercase letters in '{col}'")

            # Check for double spaces inside the name
            if "  " in col:
                problems.append(f"double spaces in '{col}'")

            if problems:
                issues.extend(problems)

        return issues

    def find_closest_match(self, target: str, schema: list) -> str:
        """
        Given a column name, find the closest match in the schema.
        Used to help identify what 'customer_id' maps to in a messy schema.

        Args:
            target (str): The column name to look up.
            schema (list): List of available (possibly messy) column names.

        Returns:
            str: The best matching column name, or empty string if none found.
        """
        # Normalize the target for comparison
        target_clean = target.strip().lower().replace("_", "").replace(" ", "")

        for col in schema:
            col_clean = col.strip().lower().replace("_", "").replace(" ", "")
            if target_clean == col_clean:
                return col     # Return original messy version

        return ""

    def generate_mapping(self, schema: list) -> dict:
        """
        Create a mapping from messy column names to normalized ones.

        Args:
            schema (list): Messy schema column names.

        Returns:
            dict: {original: normalized} mapping.
        """
        return {col: col.strip().lower() for col in schema}

    def describe_schema_issues(self, schema: list) -> str:
        """
        Return a human-readable description of all schema problems.

        Args:
            schema (list): Raw schema column names.

        Returns:
            str: Description of detected schema issues.
        """
        issues = self.detect_issues(schema)
        if not issues:
            return "Schema looks clean — no formatting issues detected."

        lines = ["Schema issues detected:"]
        for issue in issues:
            lines.append(f"  - {issue}")
        return "\n".join(lines)
