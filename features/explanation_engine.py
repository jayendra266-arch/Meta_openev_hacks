"""
features/explanation_engine.py
================================
Generates human-readable explanations for:
  - What went wrong in the pipeline
  - What fix was applied
  - How the query was improved

This is the "explain_fix" step in the RL action sequence.
"""


class ExplanationEngine:
    """
    Produces beginner-friendly explanations of data pipeline issues and fixes.

    Instead of showing raw error logs, this module translates them into
    clear, actionable language.
    """

    def explain(self, task: dict, final_query: str) -> dict:
        """
        Generate a full explanation for a task's issue and fix.

        Args:
            task (dict): The task definition (issue_type, expected_issue, expected_fix, etc.)
            final_query (str): The query after all fixes have been applied.

        Returns:
            dict: With 'issue_explanation' and 'fix_explanation' keys.
        """
        issue_type = task.get("issue_type", "unknown")

        issue_explanation = self._explain_issue(issue_type, task)
        fix_explanation   = self._explain_fix(issue_type, task, final_query)

        return {
            "issue_explanation": issue_explanation,
            "fix_explanation":   fix_explanation
        }

    def _explain_issue(self, issue_type: str, task: dict) -> str:
        """
        Return a plain-English explanation of what went wrong.

        Args:
            issue_type (str): The category of the issue.
            task (dict): Full task detail.

        Returns:
            str: Human-readable issue description.
        """
        explanations = {
            "column_typo": (
                "🔴 ISSUE: The SQL query referenced a column that doesn't exist in the schema.\n"
                "   The column name 'custmer_id' is a typo — the correct name is 'customer_id'.\n"
                "   This causes an OperationalError when the database tries to find the column."
            ),
            "aggregation_bug": (
                "🔴 ISSUE: The query used COUNT(*) instead of SUM(amount).\n"
                "   COUNT(*) counts the number of rows (transactions) — it does NOT sum the values.\n"
                "   This means 'revenue' was showing transaction count, not actual money earned.\n"
                "   The pipeline reported wrong business metrics silently — very dangerous!"
            ),
            "join_mismatch": (
                "🔴 ISSUE: A JOIN was written with mismatched column names.\n"
                "   The 'users' table uses 'user_id' but the 'sessions' table uses 'uid'.\n"
                "   Writing 'sessions.user_id' in the JOIN condition causes a failure\n"
                "   because that column doesn't exist in the sessions table."
            ),
            "messy_schema": (
                "🔴 ISSUE: The database schema has inconsistent column name formatting.\n"
                "   Problems detected:\n"
                "     - Mixed case: 'Customer_ID' vs 'customerid'\n"
                "     - Leading/trailing spaces: ' order_Date', 'AMOUNT '\n"
                "     - Inconsistent naming style: some use underscores, some don't\n"
                "   SQL is case-sensitive in many databases, and spaces in names cause failures."
            ),
        }

        return explanations.get(
            issue_type,
            f"Issue Type: {issue_type}\nDetails: {task.get('expected_issue', 'N/A')}"
        )

    def _explain_fix(self, issue_type: str, task: dict, final_query: str) -> str:
        """
        Return a plain-English explanation of how the issue was fixed.

        Args:
            issue_type (str): The category of the issue.
            task (dict): Full task detail.
            final_query (str): The final corrected SQL query.

        Returns:
            str: Human-readable fix description.
        """
        fixes = {
            "column_typo": (
                "✅ FIX: Replace 'custmer_id' with the correct column name 'customer_id'.\n"
                "   This is a simple spelling correction. Always verify column names\n"
                "   against the schema before writing queries!"
            ),
            "aggregation_bug": (
                "✅ FIX: Replace COUNT(*) with SUM(amount).\n"
                "   SUM(amount) adds up the actual dollar values in the 'amount' column,\n"
                "   which is the correct way to compute total revenue per customer.\n"
                "   ⭐ Self-healing applied: query was automatically rewritten."
            ),
            "join_mismatch": (
                "✅ FIX: Change the JOIN condition to use 'sessions.uid' instead of 'sessions.user_id'.\n"
                "   The tables use different names for the same concept (user ID).\n"
                "   Always check that JOIN columns exist in BOTH tables being joined."
            ),
            "messy_schema": (
                "✅ FIX: Normalize all schema column names:\n"
                "     - Strip leading/trailing spaces\n"
                "     - Convert to lowercase\n"
                "   Then rewrite the query using normalized names.\n"
                "   Best practice: enforce a schema naming convention in your data warehouse."
            ),
        }

        fix_text = fixes.get(
            issue_type,
            f"Fix: {task.get('expected_fix', 'N/A')}"
        )

        return f"{fix_text}\n\n   📝 Final corrected query:\n   {final_query}"
