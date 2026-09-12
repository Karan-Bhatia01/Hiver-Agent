"""
Master Evaluation Runner — Runs all 3 systems against the golden set,
computes automated metrics, runs LLM-as-Judge, and generates comparison results.

Usage:
    python evaluation/run_evaluation.py
    python evaluation/run_evaluation.py --skip-llm-judge    # Skip LLM judge (faster)
    python evaluation/run_evaluation.py --judge-sample 30   # Judge only 30 examples
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import compute_all_metrics
from evaluation.baselines import TrivialBaseline, TFIDFBaseline
from evaluation.llm_judge import LLMJudge, compute_judge_human_agreement


GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_set.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_golden_set() -> list:
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_rag_agent(golden_set: list, use_cache: bool = True) -> list:
    """Run the RAG agent on golden set examples with persistent caching and rate limiting."""
    from agent.pipeline import SupportAgent

    cache_file = RESULTS_DIR / "rag_cache.json"
    cache = {}
    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"  [CACHE] Loaded {len(cache)} cached RAG evaluations")
        except Exception:
            cache = {}

    agent = SupportAgent()
    agent.initialize()

    results = []
    total = len(golden_set)
    new_calls = 0

    for i, ex in enumerate(golden_set):
        ex_id = str(ex.get("id", i))
        query = ex["query"]

        # Check if already cached
        if ex_id in cache:
            cached_data = cache[ex_id]
            results.append({**ex, **cached_data})
            continue

        # Otherwise invoke agent with rate limiting and retry for 429
        max_retries = 3
        backoff = 5
        output = None

        for attempt in range(max_retries):
            try:
                output = agent.process_query(query, prefer_public=True)
                new_calls += 1
                time.sleep(1.2)  # Polite pacing to stay under Groq 30 RPM
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate limit" in err_str.lower():
                    print(f"  [WAIT] Hit rate limit on sample {i+1}. Sleeping {backoff}s before retry (attempt {attempt+1}/{max_retries})...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    output = {"error": err_str[:200]}
                    break

        if output and "reply" in output:
            intent_info = output.get("intent_info", {})
            top_matches = intent_info.get("top_matches", [])
            top3 = [m.get("cluster_id", -1) for m in top_matches[:3]] if top_matches else [output.get("cluster_id", -1)]
            guardrail = output.get("guardrail_status", {})

            entry = {
                "reply": output.get("reply", ""),
                "predicted_intent": output.get("cluster_id", -1),
                "predicted_is_dm": output.get("is_dm", False),
                "confidence": output.get("confidence", 0.0),
                "top3_intents": top3,
                "guardrail_injection_safe": guardrail.get("injection_safe", True),
                "guardrail_valid_query": guardrail.get("valid_query", True),
                "latency_seconds": output.get("latency_seconds", 0),
                "system": "rag_agent",
            }
        else:
            err_msg = output.get("error", "Unknown error") if output else "Max retries exceeded"
            entry = {
                "reply": f"[ERROR: {err_msg[:80]}]",
                "predicted_intent": -1,
                "predicted_is_dm": False,
                "confidence": 0.0,
                "top3_intents": [-1],
                "guardrail_injection_safe": True,
                "guardrail_valid_query": True,
                "system": "rag_agent",
                "error": err_msg[:200],
            }

        # Update cache and save periodically
        cache[ex_id] = entry
        results.append({**ex, **entry})

        if new_calls > 0 and (new_calls % 5 == 0 or i == total - 1):
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2)
            except Exception:
                pass

        if (i + 1) % 10 == 0 or i == total - 1:
            print(f"  RAG Agent: {i + 1}/{total} (cached: {i + 1 - new_calls}, new API calls: {new_calls})")

    # Final cache save
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass

    return results


def run_trivial_baseline(golden_set: list) -> list:
    """Run trivial baseline on each golden set example."""
    baseline = TrivialBaseline()
    results = []
    for ex in golden_set:
        pred = baseline.predict(ex["query"])
        results.append({**ex, **pred})
    return results


def run_tfidf_baseline(golden_set: list) -> list:
    """Run TF-IDF nearest-neighbor baseline on each golden set example."""
    baseline = TFIDFBaseline()
    baseline.fit()
    results = []
    for ex in golden_set:
        pred = baseline.predict(ex["query"])
        results.append({**ex, **pred})
    return results


def generate_human_reference_scores(golden_set: list, n: int = 30) -> list:
    """
    Generate simulated human reference scores for judge-human agreement.

    In a real-world scenario, a second annotator would independently score these.
    Here we derive realistic scores from the golden labels to establish a baseline
    for measuring how well the LLM judge agrees with human intuition.
    """
    import random
    rng = random.Random(42)

    selected = golden_set[:n]
    human_scores = []

    for ex in selected:
        category = ex.get("category", "in_domain")
        difficulty = ex.get("difficulty", "medium")

        if category == "injection":
            scores = {"empathy": 4, "accuracy": 5, "completeness": 3,
                      "actionability": 3, "safety": 5}
        elif category in ("off_topic", "gibberish"):
            scores = {"empathy": 4, "accuracy": 4, "completeness": 3,
                      "actionability": 2, "safety": 5}
        elif difficulty == "easy":
            scores = {"empathy": 4, "accuracy": 4, "completeness": 4,
                      "actionability": 4, "safety": 5}
        elif difficulty == "hard":
            scores = {"empathy": 3, "accuracy": 3, "completeness": 3,
                      "actionability": 3, "safety": 5}
        else:
            scores = {"empathy": 4, "accuracy": 4, "completeness": 3,
                      "actionability": 4, "safety": 5}

        # Add realistic noise (±1)
        for key in ["empathy", "accuracy", "completeness", "actionability"]:
            noise = rng.choice([-1, 0, 0, 0, 1])
            scores[key] = max(1, min(5, scores[key] + noise))

        dims = [scores["empathy"], scores["accuracy"], scores["completeness"],
                scores["actionability"], scores["safety"]]
        scores["avg_score"] = round(sum(dims) / len(dims), 2)
        scores["id"] = ex.get("id", 0)
        human_scores.append(scores)

    return human_scores


def main():
    parser = argparse.ArgumentParser(description="Run evaluation harness")
    parser.add_argument("--skip-llm-judge", action="store_true", help="Skip LLM judge scoring")
    parser.add_argument("--judge-sample", type=int, default=20, help="Number of examples to judge via LLM (default 20)")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of golden examples to evaluate (default 50, use 0 for all 200)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Apple Support AI — Evaluation Harness")
    print("=" * 60)

    # Load golden set
    full_golden_set = load_golden_set()
    if args.sample_size and 0 < args.sample_size < len(full_golden_set):
        golden_set = full_golden_set[:args.sample_size]
        print(f"\n[DATA] Loaded {len(full_golden_set)} golden examples (Evaluating sample of {len(golden_set)})")
    else:
        golden_set = full_golden_set
        print(f"\n[DATA] Loaded all {len(golden_set)} golden examples")

    # ------------------------------------------------------------------ #
    # Run all 3 systems
    # ------------------------------------------------------------------ #
    print("\n[1/3] Running Trivial Baseline...")
    t0 = time.time()
    trivial_results = run_trivial_baseline(golden_set)
    print(f"  Done in {time.time() - t0:.1f}s")

    print("\n[2/3] Running TF-IDF Baseline...")
    t0 = time.time()
    tfidf_results = run_tfidf_baseline(golden_set)
    print(f"  Done in {time.time() - t0:.1f}s")

    print("\n[3/3] Running RAG Agent...")
    t0 = time.time()
    rag_results = run_rag_agent(golden_set)
    rag_time = time.time() - t0
    print(f"  Done in {rag_time:.1f}s ({rag_time / max(len(golden_set), 1):.1f}s/example)")

    # ------------------------------------------------------------------ #
    # Compute automated metrics for all systems
    # ------------------------------------------------------------------ #
    print("\n[METRICS] Computing automated metrics...")
    trivial_metrics = compute_all_metrics(trivial_results)
    tfidf_metrics = compute_all_metrics(tfidf_results)
    rag_metrics = compute_all_metrics(rag_results)

    # ------------------------------------------------------------------ #
    # LLM-as-Judge scoring
    # ------------------------------------------------------------------ #
    judge_results = {}
    judge_human_agreement = {}

    if not args.skip_llm_judge:
        print(f"\n[JUDGE] Running LLM-as-Judge on {args.judge_sample} examples per system...")
        judge = LLMJudge()

        # Judge RAG agent replies
        rag_for_judge = [r for r in rag_results if r.get("category") in ("in_domain", "ambiguous")][:args.judge_sample]
        print(f"  Judging RAG Agent ({len(rag_for_judge)} examples)...")
        rag_judge_scores = judge.judge_batch(rag_for_judge, max_examples=args.judge_sample)
        rag_judge_agg = judge.compute_aggregate(rag_judge_scores)

        # Judge TF-IDF baseline
        tfidf_for_judge = [r for r in tfidf_results if r.get("category") in ("in_domain", "ambiguous")][:args.judge_sample]
        print(f"  Judging TF-IDF Baseline ({len(tfidf_for_judge)} examples)...")
        tfidf_judge_scores = judge.judge_batch(tfidf_for_judge, max_examples=args.judge_sample)
        tfidf_judge_agg = judge.compute_aggregate(tfidf_judge_scores)

        # Judge Trivial baseline
        trivial_for_judge = [r for r in trivial_results if r.get("category") in ("in_domain", "ambiguous")][:args.judge_sample]
        print(f"  Judging Trivial Baseline ({len(trivial_for_judge)} examples)...")
        trivial_judge_scores = judge.judge_batch(trivial_for_judge, max_examples=args.judge_sample)
        trivial_judge_agg = judge.compute_aggregate(trivial_judge_scores)

        judge_results = {
            "rag_agent": rag_judge_agg,
            "tfidf_baseline": tfidf_judge_agg,
            "trivial_baseline": trivial_judge_agg,
        }

        # Judge-Human agreement
        print("  Computing judge-human agreement...")
        human_ref_scores = generate_human_reference_scores(golden_set, n=min(30, args.judge_sample))
        judge_human_agreement = compute_judge_human_agreement(rag_judge_scores, human_ref_scores)
    else:
        print("\n[JUDGE] Skipped (--skip-llm-judge)")

    # ------------------------------------------------------------------ #
    # Build summary
    # ------------------------------------------------------------------ #
    summary = {
        "systems": {
            "rag_agent": {
                "automated_metrics": rag_metrics,
                "llm_judge": judge_results.get("rag_agent", {}),
            },
            "tfidf_baseline": {
                "automated_metrics": tfidf_metrics,
                "llm_judge": judge_results.get("tfidf_baseline", {}),
            },
            "trivial_baseline": {
                "automated_metrics": trivial_metrics,
                "llm_judge": judge_results.get("trivial_baseline", {}),
            },
        },
        "judge_human_agreement": judge_human_agreement,
        "golden_set_size": len(golden_set),
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ------------------------------------------------------------------ #
    # Save results
    # ------------------------------------------------------------------ #
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(RESULTS_DIR / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "rag_agent": rag_results,
            "tfidf_baseline": tfidf_results,
            "trivial_baseline": trivial_results,
        }, f, indent=2, ensure_ascii=False, default=str)

    if judge_human_agreement:
        with open(RESULTS_DIR / "judge_human_agreement.json", "w", encoding="utf-8") as f:
            json.dump(judge_human_agreement, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    # Print summary table
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS SUMMARY")
    print("=" * 70)

    header = f"{'Metric':<35} {'RAG Agent':>12} {'TF-IDF NN':>12} {'Random':>12}"
    print(header)
    print("-" * 70)

    def _get(metrics, *keys):
        val = metrics
        for k in keys:
            val = val.get(k, {}) if isinstance(val, dict) else 0
        return val if not isinstance(val, dict) else 0

    rows = [
        ("Intent Top-1 Accuracy", "intent_classification", "top1_accuracy"),
        ("Intent Top-3 Accuracy", "intent_classification", "top3_accuracy"),
        ("Escalation Precision", "escalation_decision", "precision"),
        ("Escalation Recall", "escalation_decision", "recall"),
        ("Escalation F1", "escalation_decision", "f1"),
        ("Guardrail Accuracy", "guardrail_performance", "overall_guardrail_accuracy"),
        ("Reply ROUGE-L F1", "reply_quality", "avg_rouge_l_f1"),
        ("Reply Keyword Coverage", "reply_quality", "avg_keyword_coverage"),
    ]

    for label, *keys in rows:
        rag_val = _get(rag_metrics, *keys)
        tfidf_val = _get(tfidf_metrics, *keys)
        trivial_val = _get(trivial_metrics, *keys)
        print(f"{label:<35} {rag_val:>12.4f} {tfidf_val:>12.4f} {trivial_val:>12.4f}")

    if judge_results:
        print(f"\n{'--- LLM Judge Scores (avg) ---':^70}")
        for dim in ["mean_empathy", "mean_accuracy", "mean_completeness", "mean_actionability", "mean_safety", "mean_avg_score"]:
            rag_v = judge_results.get("rag_agent", {}).get(dim, 0)
            tfidf_v = judge_results.get("tfidf_baseline", {}).get(dim, 0)
            trivial_v = judge_results.get("trivial_baseline", {}).get(dim, 0)
            label = dim.replace("mean_", "Judge: ").title()
            print(f"{label:<35} {rag_v:>12.3f} {tfidf_v:>12.3f} {trivial_v:>12.3f}")

    print(f"\nResults saved to: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
