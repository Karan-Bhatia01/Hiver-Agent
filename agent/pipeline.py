"""
SupportAgent — End-to-end orchestrator for Apple Support AI.

Flow: Guardrails → Intent Classification → Hybrid Retrieval → Rerank → Generation
"""

import time
from typing import Any, Dict, List, Optional

from agent.generator import ReplyGenerator
from agent.retriever import IntentAwareRetriever
from config import DEFAULT_CANDIDATE_K, DEFAULT_TOP_K, PROMPTS_DIR


class Decision:
    def __init__(self, action: str, reason: str):
        self.action = action
        self.reason = reason

    def __getitem__(self, key):
        return getattr(self, key)


class AgentResponse:
    def __init__(self, intent: str, reply: str, decision: Decision, cluster_id: int = 0, retrieved_documents=None, formatted_context=""):
        self.intent = intent
        self.reply = reply
        self.decision = decision
        self.cluster_id = cluster_id
        self.retrieved_documents = retrieved_documents or []
        self.formatted_context = formatted_context

    def __getitem__(self, key):
        if key == "decision":
            return {"action": self.decision.action, "reason": self.decision.reason}
        return getattr(self, key)


class SupportAgent:
    """Orchestrates guardrails, retrieval, and grounded response generation."""

    def __init__(
        self,
        retriever: Optional[IntentAwareRetriever] = None,
        generator: Optional[ReplyGenerator] = None,
        top_k: int = DEFAULT_TOP_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
    ):
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.retriever = retriever or IntentAwareRetriever()
        self.generator = generator or ReplyGenerator(prompts_dir=str(PROMPTS_DIR))
        self._is_ready = False

    def initialize(self):
        """Warm up retriever (loads embeddings, index, models)."""
        if not self._is_ready:
            self.retriever.load()
            self._is_ready = True
        return self

    def handle(self, message: str, context: str = "") -> AgentResponse:
        """Standardized interface: handle(message, context) -> AgentResponse."""
        res = self.process_query(message)
        action = "escalate" if res.get("is_dm") else "auto_reply"
        reason = res.get("escalation_reason", "")
        dec = Decision(action=action, reason=reason)
        return AgentResponse(
            intent=res.get("intent", "general_troubleshooting"),
            reply=res.get("reply", ""),
            decision=dec,
            cluster_id=res.get("cluster_id", 0),
            retrieved_documents=res.get("retrieved_documents", []),
            formatted_context=res.get("formatted_context", ""),
        )

    def process_query(self, query: str, prefer_public: bool = False, top_k: Optional[int] = None) -> Dict[str, Any]:
        """Execute the complete pipeline for a single user query."""
        start_time = time.time()
        k = top_k or self.top_k

        if not self._is_ready:
            self.initialize()

        # Step 1: Guardrails (injection + validity)
        injection_check = self.generator.check_prompt_injection(query)
        if not injection_check["is_safe"]:
            return self._guardrail_response(query, "security_violation", injection_check["reason"],
                                            injection_safe=False, time_start=start_time)

        validity_check = self.generator.check_query_validity(query)
        if not validity_check["is_valid"]:
            cat = validity_check.get("category", "off_topic")
            reply = (
                "We're here to help! It looks like your message might be incomplete or unclear. "
                "Could you please describe what Apple device and issue you need assistance with?"
                if cat in ["gibberish", "spam"] else
                "Thanks for reaching out! We specialize in technical support for Apple products and services "
                "(iPhone, Mac, iPad, Apple Watch, Apple ID, iCloud, and iOS). "
                "If you have an Apple hardware or software issue, please let us know how we can help!"
            )
            return {
                "query": query, "reply": reply, "intent": cat, "cluster_id": -1,
                "confidence": 1.0, "is_dm": False, "escalation_reason": f"Query flagged as {cat}.",
                "retrieved_documents": [],
                "guardrail_status": {"injection_safe": True, "valid_query": False, "validity_reason": validity_check["reason"]},
                "latency_seconds": round(time.time() - start_time, 3),
            }

        # Step 2 & 3: Intent classification + hybrid retrieval
        retrieval_output = self.retriever.retrieve(query=query, top_k=k, prefer_public=prefer_public, candidate_k=self.candidate_k)
        intent_info = retrieval_output["intent_info"]
        retrieved_docs = retrieval_output["documents"]
        formatted_context = retrieval_output["formatted_context"]

        # Step 4: Generation + escalation
        gen_result = self.generator.generate(
            query=query, retrieved_context=formatted_context,
            intent_info=intent_info, retrieved_docs=retrieved_docs,
            skip_guardrails=True, guardrail_status={"injection_safe": True, "valid_query": True},
        )

        return {
            "query": query,
            "reply": gen_result["reply"],
            "intent": intent_info["intent"],
            "cluster_id": intent_info["cluster_id"],
            "confidence": round(intent_info["confidence"], 4),
            "intent_info": intent_info,
            "is_dm": gen_result["is_dm_recommended"],
            "escalation_reason": gen_result["escalation_reason"],
            "retrieved_documents": retrieved_docs,
            "candidates": retrieval_output.get("candidates", []),
            "formatted_context": formatted_context,
            "guardrail_status": gen_result["guardrail_status"],
            "latency_seconds": round(time.time() - start_time, 3),
        }

    def _guardrail_response(self, query: str, intent: str, reason: str, injection_safe: bool, time_start: float) -> Dict[str, Any]:
        """Build a standardized guardrail-blocked response."""
        return {
            "query": query,
            "reply": (
                "Hello! We are here to assist with genuine Apple product, service, and technical inquiries. "
                "If you are experiencing an issue with your Apple device, software, or account, "
                "please describe the problem and we'll be happy to help!"
            ),
            "intent": intent, "cluster_id": -1, "confidence": 1.0,
            "is_dm": False, "escalation_reason": "Blocked by Prompt Injection Guardrail.",
            "retrieved_documents": [],
            "guardrail_status": {"injection_safe": injection_safe, "injection_reason": reason, "valid_query": True},
            "latency_seconds": round(time.time() - time_start, 3),
        }
