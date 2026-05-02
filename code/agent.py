"""
Core triage agent — orchestrates retrieval → LLM reasoning → structured output
for every support ticket.  Uses Groq (Llama 3.3 70B) for the LLM.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Optional

from groq import Groq

from config import GROQ_API_KEY, LLM_MODEL, TEMPERATURE, SEED
from retriever import Retriever
from prompts import SYSTEM_PROMPT, format_retrieved_context

@dataclass
class TriageResult:
    """Structured triage output for one ticket."""
    status: str          # replied | escalated
    product_area: str
    response: str
    justification: str
    request_type: str    # product_issue | feature_request | bug | invalid

class TriageAgent:
    """End-to-end support-ticket triage agent using Tool Calling."""

    def __init__(self, rebuild_index: bool = False):
        self._llm = Groq(api_key=GROQ_API_KEY)
        self._retriever = Retriever(rebuild=rebuild_index)

    def triage(self, issue: str, subject: str, company: str) -> TriageResult:
        company_key = self._normalise_company(company)
        
        user_msg = f"Ticket Issue: {issue}\nSubject: {subject}\nCompany: {company}\n\nPlease analyze this ticket. You MUST use the `search_corpus` tool to find relevant documentation before answering. If you have enough information, output your final decision in valid JSON format."
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}
        ]
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_corpus",
                    "description": "Search the internal documentation corpus for relevant support articles based on the ticket issue.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to retrieve relevant articles."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
        
        return self._run_agent_loop(messages, tools, company_key)

    def _run_agent_loop(self, messages: list, tools: list, company_key: str, retries: int = 6) -> TriageResult:
        """Call Groq LLM with retries for rate limits and handle tool execution."""
        max_steps = 3
        
        for step in range(max_steps):
            for attempt in range(retries):
                try:
                    response = self._llm.chat.completions.create(
                        model=LLM_MODEL,
                        temperature=TEMPERATURE,
                        seed=SEED,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        max_tokens=2048,
                    )
                    break # Success, break out of retry loop
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        wait = 30 * (attempt + 1)
                        print(f"  ⏳ Rate limited, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
            else:
                return TriageResult("escalated", "unknown", "This ticket requires human review.", "LLM call failed after retries.", "product_issue")
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            
            if tool_calls:
                # Append assistant message with tool calls
                messages.append(response_message)
                
                for tool_call in tool_calls:
                    if tool_call.function.name == "search_corpus":
                        args = json.loads(tool_call.function.arguments)
                        query = args.get("query", "")
                        print(f"  [Agent] Tool Call: search_corpus('{query}')")
                        
                        hits = self._retriever.retrieve(query, company=company_key)
                        context = format_retrieved_context(hits)
                        
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": "search_corpus",
                            "content": context if context else "No relevant documents found."
                        })
            else:
                # No tool calls means the agent is providing the final JSON response
                return self._parse_result(response_message.content)
                
        # If max steps reached without returning
        return TriageResult("escalated", "unknown", "This ticket requires human review.", "Agent loop exceeded maximum steps.", "product_issue")

    @staticmethod
    def _parse_result(raw: str) -> TriageResult:
        try:
            match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
            if match:
                raw = match.group(1)
            parsed = json.loads(raw)
            return TriageResult(
                status=parsed.get("status", "escalated").lower(),
                product_area=parsed.get("product_area", "unknown").lower(),
                response=parsed.get("response", "This ticket requires human review."),
                justification=parsed.get("justification", "Unable to process JSON correctly."),
                request_type=parsed.get("request_type", "product_issue").lower(),
            )
        except Exception:
            return TriageResult(
                status="escalated",
                product_area="unknown",
                response="This ticket requires human review.",
                justification="LLM produced invalid JSON.",
                request_type="product_issue",
            )

    @staticmethod
    def _normalise_company(company: str) -> Optional[str]:
        if not company or company.strip().lower() in ("none", "", "nan"):
            return None
        key = company.strip().lower()
        from config import COMPANY_CORPUS
        for k, v in COMPANY_CORPUS.items():
            if key in v:
                return k
        return None
