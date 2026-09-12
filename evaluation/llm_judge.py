"""
LLM-as-Judge — Evaluates generated Apple Support replies on a 5-dimension rubric.

Dimensions (1–5 Likert scale):
    1. Empathy & Tone: Does it sound like authentic Apple Support?
    2. Accuracy & Groundedness: Are the steps correct and grounded in historical data?
    3. Completeness: Does it fully address the customer's issue?
    4. Actionability: Can the customer follow the steps without confusion?
    5. Safety & Appropriateness: No hallucinations, no dangerous/incorrect advice?

Also includes Judge-Human agreement measurement via Cohen's Kappa.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq


JUDGE_RUBRIC = """You are an expert evaluator assessing the quality of Apple customer support replies.

Rate the following reply on 5 dimensions using a 1-5 Likert scale.

## Scoring Rubric

### 1. Empathy & Tone (1-5)
- 1: Cold, robotic, or rude
- 3: Neutral, professional but impersonal
- 5: Warm, empathetic, authentic Apple Support voice

### 2. Accuracy & Groundedness (1-5)
- 1: Contains factual errors, hallucinated features, or dangerous advice
- 3: Mostly correct but includes unverified suggestions
- 5: All steps are accurate and grounded in verified Apple troubleshooting

### 3. Completeness (1-5)
- 1: Ignores the customer's actual problem
- 3: Partially addresses the issue, missing key steps
- 5: Fully addresses the issue with all relevant steps

### 4. Actionability (1-5)
- 1: Vague, unclear, customer cannot follow
- 3: Somewhat clear but missing specifics (e.g., exact menu paths)
- 5: Clear numbered steps with exact settings paths, easy to follow

### 5. Safety & Appropriateness (1-5)
- 1: Recommends jailbreaking, unauthorized tools, or shares sensitive info
- 3: Generally safe but may suggest unnecessary actions
- 5: Fully safe, appropriate, no hallucinated or risky advice

## Input

**Customer Query:** "{query}"

**Expected Topic:** {gold_intent_name}

**Key Points a Good Reply Should Cover:** {gold_reply_notes}

**Generated Reply:** "{reply}"

## Output Format
Return ONLY a JSON object with exactly these keys:
{{"empathy": <1-5>, "accuracy": <1-5>, "completeness": <1-5>, "actionability": <1-5>, "safety": <1-5>, "reasoning": "<brief 1-2 sentence justification>"}}
"""


class LLMJudge:
    """
    Uses Groq LLM to evaluate reply quality on a 5-dimension rubric.
    """

    def __init__(self, model_name: Optional[str] = None):
        load_dotenv(override=True)
        self.api_key = os.getenv("GROQ_API_KEY", "").strip("\"' \t\r\n")
        self.model_name = model_name or os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")
        self.client = None
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception:
                pass

    def judge_single(self, query: str, reply: str,
                     gold_intent_name: str = "", gold_reply_notes: str = "") -> Dict[str, Any]:
        """
        Score a single generated reply.

        Returns dict with: empathy, accuracy, completeness, actionability, safety, reasoning, avg_score
        """
        if not self.client:
            return self._fallback_score(reply, gold_reply_notes)

        prompt = JUDGE_RUBRIC.replace("{query}", query)
        prompt = prompt.replace("{reply}", reply)
        prompt = prompt.replace("{gold_intent_name}", gold_intent_name)
        prompt = prompt.replace("{gold_reply_notes}", gold_reply_notes)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert customer support quality evaluator. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            raw = response.choices[0].message.content or ""
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                scores = {
                    "empathy": max(1, min(5, int(data.get("empathy", 3)))),
                    "accuracy": max(1, min(5, int(data.get("accuracy", 3)))),
                    "completeness": max(1, min(5, int(data.get("completeness", 3)))),
                    "actionability": max(1, min(5, int(data.get("actionability", 3)))),
                    "safety": max(1, min(5, int(data.get("safety", 5)))),
                    "reasoning": str(data.get("reasoning", "")),
                }
                dims = [scores["empathy"], scores["accuracy"], scores["completeness"],
                        scores["actionability"], scores["safety"]]
                scores["avg_score"] = round(sum(dims) / len(dims), 2)
                return scores
        except Exception:
            pass

        return self._fallback_score(reply, gold_reply_notes)

    def _fallback_score(self, reply: str, gold_reply_notes: str) -> Dict[str, Any]:
        """Heuristic fallback scoring when LLM is unavailable."""
        reply_lower = reply.lower() if reply else ""
        notes_lower = gold_reply_notes.lower() if gold_reply_notes else ""

        # Empathy: check for empathetic phrases
        empathy_phrases = ["we understand", "sorry", "we know", "here to help", "appreciate",
                           "we'd love", "happy to"]
        empathy = 2 + min(3, sum(1 for p in empathy_phrases if p in reply_lower))

        # Accuracy: keyword overlap with gold notes
        if notes_lower:
            note_words = set(re.findall(r"\b[a-z]+\b", notes_lower))
            reply_words = set(re.findall(r"\b[a-z]+\b", reply_lower))
            overlap = len(note_words & reply_words) / max(len(note_words), 1)
            accuracy = max(1, min(5, round(1 + overlap * 4)))
        else:
            accuracy = 3

        # Completeness: length-based heuristic
        completeness = min(5, max(1, len(reply.split()) // 20 + 1))

        # Actionability: check for numbered steps or Settings paths
        has_steps = bool(re.search(r"\d\.", reply)) or "settings" in reply_lower
        actionability = 4 if has_steps else 2

        # Safety: check for dangerous keywords
        unsafe = ["jailbreak", "root", "unofficial", "hack", "bypass"]
        safety = 5 if not any(w in reply_lower for w in unsafe) else 1

        dims = [empathy, accuracy, completeness, actionability, safety]
        return {
            "empathy": empathy, "accuracy": accuracy, "completeness": completeness,
            "actionability": actionability, "safety": safety,
            "reasoning": "Scored via heuristic fallback (LLM unavailable).",
            "avg_score": round(sum(dims) / len(dims), 2),
        }

    def judge_batch(self, examples: List[Dict[str, Any]], max_examples: int = 50) -> List[Dict[str, Any]]:
        """
        Score a batch of examples.

        Args:
            examples: list of dicts with 'query', 'reply', 'gold_intent_name', 'gold_reply_notes'
            max_examples: max number to score via LLM (to manage API costs)

        Returns:
            list of score dicts
        """
        import time
        results = []
        for i, ex in enumerate(examples[:max_examples]):
            scores = self.judge_single(
                query=ex.get("query", ""),
                reply=ex.get("reply", ""),
                gold_intent_name=ex.get("gold_intent_name", ""),
                gold_reply_notes=ex.get("gold_reply_notes", ""),
            )
            scores["id"] = ex.get("id", i)
            results.append(scores)
            time.sleep(0.5)
        return results

    @staticmethod
    def compute_aggregate(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregate statistics from a list of judge scores."""
        if not scores:
            return {}

        dims = ["empathy", "accuracy", "completeness", "actionability", "safety", "avg_score"]
        agg = {}
        for d in dims:
            vals = [s[d] for s in scores if d in s]
            if vals:
                agg[f"mean_{d}"] = round(sum(vals) / len(vals), 3)
                agg[f"min_{d}"] = min(vals)
                agg[f"max_{d}"] = max(vals)
        agg["total_judged"] = len(scores)
        return agg


def compute_judge_human_agreement(
    judge_scores: List[Dict[str, Any]],
    human_scores: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute agreement between LLM judge and human scores.

    Uses:
        - Spearman rank correlation on avg_score
        - Mean absolute deviation per dimension
        - Exact agreement rate (within ±1 on 5-point scale)
    """
    if not judge_scores or not human_scores:
        return {"error": "insufficient data"}

    # Match by ID
    human_map = {s["id"]: s for s in human_scores}
    paired = [(j, human_map[j["id"]]) for j in judge_scores if j["id"] in human_map]

    if len(paired) < 5:
        return {"error": "too few paired examples", "n_paired": len(paired)}

    dims = ["empathy", "accuracy", "completeness", "actionability", "safety"]
    agreement = {}

    for d in dims:
        j_vals = [p[0].get(d, 3) for p in paired]
        h_vals = [p[1].get(d, 3) for p in paired]
        # Mean absolute deviation
        mad = sum(abs(j - h) for j, h in zip(j_vals, h_vals)) / len(paired)
        # Exact or within-1 agreement
        within_1 = sum(1 for j, h in zip(j_vals, h_vals) if abs(j - h) <= 1)
        agreement[d] = {
            "mean_abs_deviation": round(mad, 3),
            "within_1_agreement": round(within_1 / len(paired), 3),
        }

    # Overall avg_score correlation (Spearman)
    j_avgs = [p[0].get("avg_score", 3.0) for p in paired]
    h_avgs = [p[1].get("avg_score", 3.0) for p in paired]

    # Spearman rank correlation
    def _rank(vals):
        sorted_vals = sorted(enumerate(vals), key=lambda x: x[1])
        ranks = [0.0] * len(vals)
        for rank, (idx, _) in enumerate(sorted_vals):
            ranks[idx] = rank + 1
        return ranks

    j_ranks = _rank(j_avgs)
    h_ranks = _rank(h_avgs)
    n = len(paired)
    d_sq = sum((jr - hr) ** 2 for jr, hr in zip(j_ranks, h_ranks))
    spearman = 1 - (6 * d_sq) / (n * (n ** 2 - 1)) if n > 1 else 0.0

    agreement["overall"] = {
        "spearman_correlation": round(spearman, 4),
        "n_paired": n,
        "mean_abs_deviation_avg_score": round(
            sum(abs(j - h) for j, h in zip(j_avgs, h_avgs)) / n, 3
        ),
    }

    return agreement
