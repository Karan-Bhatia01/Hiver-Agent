"""
Automated Evaluation Metrics for Apple Support AI Agent.

Computes:
    1. Intent Classification Accuracy (Top-1 and Top-3)
    2. Escalation Decision F1 (Precision, Recall, F1 for is_dm)
    3. Guardrail Accuracy (Injection detection + off-topic deflection)
    4. Reply Quality (ROUGE-L, keyword coverage, groundedness heuristic)
"""

import re
from collections import Counter
from typing import Any, Dict, List


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    if not text:
        return []
    return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9]+\b", text)]


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Longest Common Subsequence length (for ROUGE-L)."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(reference: str, hypothesis: str) -> Dict[str, float]:
    """Compute ROUGE-L precision, recall, F1."""
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens) if hyp_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def keyword_coverage(reply: str, gold_notes: str) -> float:
    """Fraction of gold_reply_notes keywords present in the generated reply."""
    if not gold_notes or not reply:
        return 0.0

    # Extract key terms from gold notes (words > 3 chars, excluding common words)
    stop_words = {"the", "and", "for", "with", "that", "this", "from", "your", "have", "check",
                  "update", "settings", "general", "apple", "support", "device", "mention"}
    gold_terms = set(_tokenize(gold_notes)) - stop_words
    gold_terms = {t for t in gold_terms if len(t) > 3}
    if not gold_terms:
        return 0.0

    reply_tokens = set(_tokenize(reply))
    matches = gold_terms & reply_tokens
    return round(len(matches) / len(gold_terms), 4)


def intent_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute intent classification accuracy.

    Args:
        results: list of dicts with 'gold_intent', 'predicted_intent', optionally 'top3_intents'
    """
    in_domain = [r for r in results if r.get("category", "in_domain") in ("in_domain", "ambiguous")]
    if not in_domain:
        return {"top1_accuracy": 0.0, "top3_accuracy": 0.0, "total": 0}

    top1_correct = sum(1 for r in in_domain if r["gold_intent"] == r.get("predicted_intent"))
    top1_acc = top1_correct / len(in_domain)

    top3_correct = 0
    for r in in_domain:
        top3 = r.get("top3_intents", [r.get("predicted_intent")])
        if r["gold_intent"] in top3:
            top3_correct += 1
    top3_acc = top3_correct / len(in_domain)

    # Per-cluster accuracy
    cluster_acc = {}
    for cid in range(12):
        cluster_examples = [r for r in in_domain if r["gold_intent"] == cid]
        if cluster_examples:
            correct = sum(1 for r in cluster_examples if r["gold_intent"] == r.get("predicted_intent"))
            cluster_acc[cid] = round(correct / len(cluster_examples), 4)

    return {
        "top1_accuracy": round(top1_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "total": len(in_domain),
        "top1_correct": top1_correct,
        "per_cluster_accuracy": cluster_acc,
    }


def escalation_f1(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute precision, recall, F1 for escalation (is_dm) prediction.

    Args:
        results: list of dicts with 'gold_is_dm' and 'predicted_is_dm'
    """
    in_domain = [r for r in results if r.get("category", "in_domain") in ("in_domain", "ambiguous")]
    if not in_domain:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = sum(1 for r in in_domain if r["gold_is_dm"] and r.get("predicted_is_dm"))
    fp = sum(1 for r in in_domain if not r["gold_is_dm"] and r.get("predicted_is_dm"))
    fn = sum(1 for r in in_domain if r["gold_is_dm"] and not r.get("predicted_is_dm"))
    tn = sum(1 for r in in_domain if not r["gold_is_dm"] and not r.get("predicted_is_dm"))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": len(in_domain),
    }


def guardrail_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute guardrail detection accuracy.

    Args:
        results: list of dicts with 'category' and guardrail prediction fields
    """
    injection_examples = [r for r in results if r.get("category") == "injection"]
    off_topic_examples = [r for r in results if r.get("category") == "off_topic"]
    gibberish_examples = [r for r in results if r.get("category") == "gibberish"]

    injection_caught = sum(1 for r in injection_examples if not r.get("guardrail_injection_safe", True))
    off_topic_caught = sum(1 for r in off_topic_examples if not r.get("guardrail_valid_query", True))
    gibberish_caught = sum(1 for r in gibberish_examples if not r.get("guardrail_valid_query", True))

    total_guardrail = len(injection_examples) + len(off_topic_examples) + len(gibberish_examples)
    total_caught = injection_caught + off_topic_caught + gibberish_caught

    return {
        "injection_detection_rate": round(injection_caught / max(len(injection_examples), 1), 4),
        "off_topic_detection_rate": round(off_topic_caught / max(len(off_topic_examples), 1), 4),
        "gibberish_detection_rate": round(gibberish_caught / max(len(gibberish_examples), 1), 4),
        "overall_guardrail_accuracy": round(total_caught / max(total_guardrail, 1), 4),
        "injection_total": len(injection_examples),
        "off_topic_total": len(off_topic_examples),
        "gibberish_total": len(gibberish_examples),
    }


def reply_quality_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate reply quality metrics.

    Args:
        results: list of dicts with 'reply', 'gold_reply_notes'
    """
    valid = [r for r in results if r.get("reply") and r.get("gold_reply_notes")
             and r.get("category") in ("in_domain", "ambiguous")]
    if not valid:
        return {"avg_rouge_l_f1": 0.0, "avg_keyword_coverage": 0.0, "total": 0}

    rouge_scores = []
    kw_scores = []
    for r in valid:
        rl = rouge_l(r["gold_reply_notes"], r["reply"])
        rouge_scores.append(rl["f1"])
        kw = keyword_coverage(r["reply"], r["gold_reply_notes"])
        kw_scores.append(kw)

    return {
        "avg_rouge_l_f1": round(sum(rouge_scores) / len(rouge_scores), 4),
        "avg_keyword_coverage": round(sum(kw_scores) / len(kw_scores), 4),
        "total": len(valid),
    }


def compute_all_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute all automated metrics on evaluation results."""
    return {
        "intent_classification": intent_accuracy(results),
        "escalation_decision": escalation_f1(results),
        "guardrail_performance": guardrail_accuracy(results),
        "reply_quality": reply_quality_metrics(results),
    }
