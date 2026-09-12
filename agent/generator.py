"""
ReplyGenerator — Grounded Apple Support response generator with security guardrails.

Features:
    1. Prompt injection & jailbreak detection (heuristic + optional LLM)
    2. Query validity & domain relevance check
    3. Escalation decision engine (auto-reply vs DM)
    4. Grounded LLM response generation via Groq
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from groq import Groq


class ReplyGenerator:
    """Generates grounded Apple Support responses with safety guardrails and escalation logic."""

    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|directives|rules|prompts)",
        r"disregard\s+(all\s+)?(previous|above|prior)",
        r"you\s+are\s+now\s+(a\s+)?(dan|jailbreak|unrestricted|god\s+mode|developer\s+mode)",
        r"act\s+as\s+(a\s+)?(dan|jailbreak|unrestricted|hacker)",
        r"(reveal|show|print|leak|output)\s+(your\s+)?(system\s+prompt|hidden\s+prompt|instructions|api\s+key)",
        r"forget\s+(everything|all\s+rules|your\s+training)",
        r"new\s+rule(s)?:\s*",
        r"system\s*:\s*override",
        r"<script\b[^>]*>",
        r"\bdrop\s+table\b",
    ]

    DM_REQUIRED_KEYWORDS = [
        "password", "passcode", "apple id locked", "account recovery", "two factor",
        "2fa", "verification code", "stolen", "lost iphone", "serial number", "imei",
        "billing", "credit card", "debit card", "unauthorized charge", "refund",
        "receipt", "invoice", "cracked screen", "water damage", "swollen battery",
        "battery bulging", "repair appointment", "genius bar appointment", "warranty claim"
    ]

    def __init__(self, prompts_dir: str = "prompts", model_name: Optional[str] = None, max_tokens: int = 800, temperature: float = 0.2):
        load_dotenv(override=True)

        self.prompts_dir = Path(prompts_dir)
        self.api_key = os.getenv("GROQ_API_KEY", "").strip("\"' \t\r\n")
        self.model_name = model_name or os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.client: Optional[Groq] = None
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception:
                pass

        # Load prompt templates
        self.prompts: Dict[str, str] = {}
        for key, filename in {"injection": "injection_check.md", "validity": "validity_check.md",
                              "generation": "generation.md", "escalation": "escalation.md",
                              "classification": "classification.md"}.items():
            path = self.prompts_dir / filename
            self.prompts[key] = path.read_text(encoding="utf-8") if path.exists() else ""

    # ------------------------------------------------------------------ #
    #  Guardrail 1: Prompt Injection Check
    # ------------------------------------------------------------------ #

    def check_prompt_injection(self, query: str, use_llm: bool = False) -> Dict[str, Any]:
        """Check for prompt injection / jailbreak attempts. Returns {is_safe, confidence, reason}."""
        if not query or not query.strip():
            return {"is_safe": True, "confidence": 1.0, "reason": "Empty query."}

        # Fast heuristic check
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                return {"is_safe": False, "confidence": 0.99, "reason": f"Heuristic pattern match: '{pattern}'"}

        # Optional LLM-based deep check
        if use_llm and self.client and self.prompts.get("injection"):
            result = self._llm_json_check(self.prompts["injection"].replace("{query}", query))
            if result:
                return {
                    "is_safe": bool(result.get("is_safe", True)),
                    "confidence": float(result.get("confidence", 0.9)),
                    "reason": str(result.get("reason", "LLM evaluation passed.")),
                }

        return {"is_safe": True, "confidence": 0.95, "reason": "No injection patterns detected."}

    # ------------------------------------------------------------------ #
    #  Guardrail 2: Query Validity Check
    # ------------------------------------------------------------------ #

    def check_query_validity(self, query: str, use_llm: bool = False) -> Dict[str, Any]:
        """Check if query is a legitimate support inquiry. Returns {is_valid, category, reason}."""
        text = query.strip()

        if len(text) < 3:
            return {"is_valid": False, "category": "gibberish", "reason": "Query is too short."}

        if len(set(text.lower().replace(" ", ""))) <= 2 and len(text) > 4:
            return {"is_valid": False, "category": "gibberish", "reason": "Query contains repetitive characters."}

        if use_llm and self.client and self.prompts.get("validity"):
            result = self._llm_json_check(self.prompts["validity"].replace("{query}", text))
            if result:
                return {
                    "is_valid": bool(result.get("is_valid", True)),
                    "category": str(result.get("category", "general")),
                    "reason": str(result.get("reason", "Legitimate Apple support inquiry.")),
                }

        return {"is_valid": True, "category": "general", "reason": "Passed heuristic validity check."}

    # ------------------------------------------------------------------ #
    #  Escalation Decision
    # ------------------------------------------------------------------ #

    def evaluate_escalation(self, query: str, intent: str, retrieved_docs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Decide auto-reply vs DM escalation based on intent + historical DM ratio."""
        # Rule 1: High-risk intents always require DM
        if intent in {"apple_id_account", "subscription_billing", "security_violation"}:
            return {
                "is_dm_recommended": True,
                "reason": f"Intent '{intent}' involves sensitive data requiring DM escalation.",
                "guidance": "Acknowledge the issue, provide safe preliminary info, and instruct customer to DM.",
            }

        # Rule 2: High historical DM ratio
        if retrieved_docs:
            dm_count = sum(1 for d in retrieved_docs if d.get("is_dm", False))
            total = len(retrieved_docs)
            if total > 0 and dm_count / total >= 0.65:
                return {
                    "is_dm_recommended": True,
                    "reason": f"High escalation rate ({dm_count}/{total} similar cases required DM).",
                    "guidance": "Offer quick troubleshooting, but invite DM if the problem persists.",
                }

        # Rule 3: Default public resolution
        return {
            "is_dm_recommended": False,
            "reason": "Standard inquiry with publicly verifiable troubleshooting steps.",
            "guidance": "Provide clear numbered troubleshooting steps. Invite DM only if steps don't resolve.",
        }

    # ------------------------------------------------------------------ #
    #  Core Response Generation
    # ------------------------------------------------------------------ #

    def generate(self, query: str, retrieved_context: str, intent_info: Optional[Dict[str, Any]] = None,
                 retrieved_docs: Optional[List[Dict[str, Any]]] = None,
                 guardrail_status: Optional[Dict[str, Any]] = None,
                 skip_guardrails: bool = False, use_llm_guardrails: bool = False) -> Dict[str, Any]:
        """Full generation pipeline: Guardrails → Escalation → LLM Generation."""
        intent_info = intent_info or {}
        intent_name = intent_info.get("intent", "general_support")
        cluster_id = intent_info.get("cluster_id", 0)
        confidence = intent_info.get("confidence", 0.0)

        # Guardrails (skipped if already verified upstream)
        if not skip_guardrails and not guardrail_status:
            injection_result = self.check_prompt_injection(query, use_llm=use_llm_guardrails)
            if not injection_result["is_safe"]:
                return {
                    "reply": "Hello! We are here to assist with genuine Apple product, service, and technical inquiries. "
                             "Please describe your problem and we'll be happy to help!",
                    "is_dm_recommended": False, "escalation_reason": "Blocked by Prompt Injection Guardrail.",
                    "guardrail_status": {"injection_safe": False, "injection_reason": injection_result["reason"], "valid_query": True},
                    "intent": "security_violation", "cluster_id": -1,
                }

            validity_result = self.check_query_validity(query, use_llm=use_llm_guardrails)
            if not validity_result["is_valid"]:
                cat = validity_result.get("category", "off_topic")
                reply = ("We're here to help! Could you please describe what Apple device and issue you need assistance with?"
                         if cat in ["gibberish", "spam"] else
                         "Thanks for reaching out! We specialize in Apple products and services. "
                         "If you have an Apple hardware or software issue, please let us know!")
                return {
                    "reply": reply, "is_dm_recommended": False, "escalation_reason": f"Query flagged as {cat}.",
                    "guardrail_status": {"injection_safe": True, "valid_query": False, "validity_reason": validity_result["reason"]},
                    "intent": cat, "cluster_id": -1,
                }

        # Escalation decision
        escalation_result = self.evaluate_escalation(query, intent_name, retrieved_docs)
        is_dm = escalation_result["is_dm_recommended"]

        # Build generation prompt
        template = self.prompts.get("generation", "") or (
            "You are Apple Support (@AppleSupport). Provide helpful, polite, and accurate assistance.\n"
            "Customer Query: {query}\nGrounding Context:\n{retrieved_context}\n"
            "Guidance: {escalation_guidance}\nReply:"
        )
        prompt = (template.replace("{query}", query)
                  .replace("{intent}", intent_name)
                  .replace("{cluster_id}", str(cluster_id))
                  .replace("{confidence}", f"{confidence:.1%}")
                  .replace("{retrieved_context}", retrieved_context)
                  .replace("{escalation_guidance}", escalation_result["guidance"]))

        # LLM call
        reply_text = ""
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are Apple Support. Always output warm, concise, professional customer replies grounded in verified steps."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature, max_tokens=self.max_tokens,
                )
                reply_text = (response.choices[0].message.content or "").strip()
            except Exception:
                pass

        # Fallback if LLM unavailable
        if not reply_text:
            reply_text = (
                "We'd like to take a closer look. Please send us a DM with your device model and iOS version: "
                "https://twitter.com/messages/compose?recipient_id=AppleSupport"
                if is_dm else
                "Thanks for reaching out! Try restarting your device and checking Settings > General > Software Update. "
                "If the issue persists, let us know or DM us your details!"
            )

        return {
            "reply": reply_text, "is_dm_recommended": is_dm,
            "escalation_reason": escalation_result["reason"],
            "intent": intent_name, "cluster_id": cluster_id, "confidence": confidence,
            "guardrail_status": {"injection_safe": True, "valid_query": True},
        }

    # ------------------------------------------------------------------ #
    #  Internal Helper
    # ------------------------------------------------------------------ #

    def _llm_json_check(self, prompt: str) -> Optional[dict]:
        """Call LLM with a prompt and parse the first JSON object from response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=250,
            )
            raw_text = response.choices[0].message.content or ""
            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception:
            pass
        return None
