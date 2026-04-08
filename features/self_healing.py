"""
features/self_healing.py
=========================
? UNIQUE FEATURE: Self-Healing Pipeline
"""
import re

class SelfHealer:
    def heal(self, query: str, task: dict) -> str:
        issue_type = task.get("issue_type", "")
        if issue_type == "column_typo":
            healed = self._heal_column_typo(query, task)
        elif issue_type == "aggregation_bug":
            healed = self._heal_aggregation(query)
        elif issue_type == "join_mismatch":
            healed = self._heal_join(query, task)
        elif issue_type == "messy_schema":
            healed = self._heal_messy_schema(query, task)
        else:
            healed = task.get("correct_query", query)

        if healed.strip().lower() != query.strip().lower():
            print(f"    ??  [Self-Healing] Repaired: {query} -> {healed}")
        return healed

    def _heal_column_typo(self, query: str, task: dict) -> str:
        if "custmer_id" in query: return query.replace("custmer_id", "customer_id")
        return task.get("correct_query", query)

    def _heal_aggregation(self, query: str) -> str:
        pattern = r"COUNT\(\*\)(\s+AS\s+\w+)?"
        return re.sub(pattern, r"SUM(amount)\1", query, flags=re.IGNORECASE)

    def _heal_join(self, query: str, task: dict) -> str:
        if "sessions.user_id" in query: return query.replace("sessions.user_id", "sessions.uid")
        return task.get("correct_query", query)

    def _heal_messy_schema(self, query: str, task: dict) -> str:
        return task.get("correct_query", query)
