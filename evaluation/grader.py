class Grader:
    """ Grades a completed episode and returns performance metrics. """
    TARGET_ACCURACY = 0.98

    def grade_episode(self, episode_result: dict) -> dict:
        score = episode_result.get("total_reward", 0.0)
        # Scale score to 1.0 baseline (6 correct steps = 1.0)
        scaled_score = min(max(score / 1.0, 0.0), 1.0)
        
        steps = episode_result.get("steps", [])
        total_steps = len(steps)
        correct_steps = sum(1 for s in steps if s.get("correct", False))
        accuracy = correct_steps / total_steps if total_steps > 0 else 0.0

        return {
            "task":     episode_result.get("task_name", "unknown"),
            "score":    round(scaled_score, 3),
            "steps":    total_steps,
            "correct":  correct_steps,
            "accuracy": round(accuracy * 100, 1),
            "pass":     (scaled_score >= 0.95),
        }

    def grade_all(self, results: list) -> dict:
        grades = [self.grade_episode(r) for r in results]
        total_score = sum(g["score"] for g in grades)
        return {
            "grades": grades,
            "total_score": round(total_score, 2),
            "overall_avg": round(total_score / len(grades), 3) if grades else 0.0
        }
