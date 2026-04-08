"""
agent/reasoning.py
==================
The reasoning module — determines what action to take next.

Uses rule-based logic to analyze the current state and select the
most appropriate action. This simulates how a human data engineer
would think through a debugging problem.

Rules:
  - Read the error log and query
  - Match known patterns (typos, COUNT, JOIN, schema mess)
  - Return the correct action for the current step
"""


class ReasoningModule:
    """
    Rule-based reasoning engine that analyzes state and selects actions.

    The agent always follows the fixed 6-step sequence, but the reasoning
    module helps it understand WHAT to do at each step.
    """

    # Error-log-based patterns (checked first — most reliable signal)
    # Order matters: more specific patterns must come before general ones
    ERROR_LOG_PATTERNS = [
        ("joinerror",        "join_mismatch"),
        ("not found in table", "join_mismatch"),
        ("schemaerror",      "messy_schema"),
        ("dataqualityerror", "aggregation_bug"),
        ("operationalerror", "column_typo"),
        ("no such column",   "column_typo"),
    ]

    # Query-body fallback patterns (used only when error log doesn't match)
    QUERY_PATTERNS = [
        ("custmer_id",       "column_typo"),
        ("count(*)",         "aggregation_bug"),
        ("join",             "join_mismatch"),
    ]

    def diagnose(self, state: dict) -> str:
        """
        Analyze the state and determine the most likely issue type.

        Priority order:
          1. Error log (most reliable — the error TYPE is definitive)
          2. Query body (fallback heuristic)
          3. Schema inspection (last resort for messy schema detection)

        Args:
            state (dict): Current RL state with error_log, query, schema, step.

        Returns:
            str: Detected issue type (e.g., 'column_typo', 'aggregation_bug').
        """
        error_log = state.get("error_log", "").lower()
        query     = state.get("query", "").lower()
        schema    = state.get("schema", [])

        # --- Step 1: Check error log first (highest priority) ---
        for keyword, issue_type in self.ERROR_LOG_PATTERNS:
            if keyword.lower() in error_log:
                return issue_type

        # --- Step 2: Fall back to query-body analysis ---
        for keyword, issue_type in self.QUERY_PATTERNS:
            if keyword.lower() in query:
                return issue_type

        # --- Step 3: Check schema for messy formatting ---
        for col in schema:
            if col != col.strip() or col != col.lower():
                return "messy_schema"

        return "unknown"

    def identify_issue_description(self, state: dict) -> str:
        """
        Generate a description of the detected issue.

        Args:
            state (dict): Current state.

        Returns:
            str: Human-readable issue description.
        """
        issue_type = self.diagnose(state)
        error_log  = state.get("error_log", "")

        descriptions = {
            "column_typo":     f"Column name typo detected in query. Error: '{error_log}'",
            "aggregation_bug": f"Wrong aggregation function (COUNT vs SUM). Error: '{error_log}'",
            "join_mismatch":   f"JOIN column mismatch between tables. Error: '{error_log}'",
            "messy_schema":    f"Messy schema: casing/whitespace issues detected. Error: '{error_log}'",
            "unknown":         f"Unknown issue detected. Error: '{error_log}'"
        }

        return descriptions.get(issue_type, f"Issue in query: {error_log}")

    def suggest_fix_description(self, state: dict) -> str:
        """
        Suggest what fix should be applied based on the detected issue.

        Args:
            state (dict): Current state.

        Returns:
            str: Suggested fix description.
        """
        issue_type = self.diagnose(state)

        suggestions = {
            "column_typo":     "Fix the typo in the column name to match the schema.",
            "aggregation_bug": "Replace COUNT(*) with SUM(amount) for correct revenue computation.",
            "join_mismatch":   "Update the JOIN condition to use the correct column name from each table.",
            "messy_schema":    "Normalize all column names: strip whitespace, convert to lowercase.",
            "unknown":         "Review the query and schema for discrepancies."
        }

        return suggestions.get(issue_type, "Review and fix the query based on the error.")

    def should_optimize(self, state: dict) -> bool:
        """
        Determine if query optimization is needed.

        Args:
            state (dict): Current state.

        Returns:
            bool: True if optimization is recommended.
        """
        query = state.get("query", "").lower()
        # Optimization needed if COUNT is used where SUM would be more appropriate
        return "count(*)" in query

    def summarize_reasoning(self, state: dict) -> str:
        """
        Return a multi-line summary of the agent's reasoning process.

        Args:
            state (dict): Current state.

        Returns:
            str: Reasoning summary.
        """
        issue_type = self.diagnose(state)
        issue_desc = self.identify_issue_description(state)
        fix_desc   = self.suggest_fix_description(state)
        optimize   = self.should_optimize(state)

        return (
            f"[Reasoning]\n"
            f"  Detected Issue Type : {issue_type}\n"
            f"  Issue Description   : {issue_desc}\n"
            f"  Suggested Fix       : {fix_desc}\n"
            f"  Needs Optimization  : {'Yes' if optimize else 'No'}"
        )
