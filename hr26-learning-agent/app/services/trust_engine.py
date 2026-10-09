"""
Trust & Conflict Resolution Engine
Minimum Objective #3: Resolve conflicting or outdated information across
multiple sources and autonomously determine what knowledge should be trusted.
"""

from typing import List, Dict, Any, Tuple
from app.models.resource import Resource, TrustSignal
from app.services.agnes_client import get_agnes_client
import json


TRUST_SYSTEM_PROMPT = """You are an expert educational content auditor and knowledge verifier.
Your job is to evaluate the trustworthiness of educational source materials and detect conflicts.

For each resource you receive, analyze:
1. Currency (is the information likely outdated?)
2. Consistency with other sources
3. Specificity and evidence quality
4. Potential bias or oversimplification
5. Alignment with standard curriculum knowledge (where applicable)

Return a structured JSON assessment.
"""


class TrustEngine:
    def __init__(self):
        self.client = get_agnes_client()

    async def evaluate_resource(
        self,
        resource: Resource,
        other_resources: List[Resource] = None,
        domain_context: str = "",
    ) -> Resource:
        """
        Score a single resource and attach trust signals.
        Uses the long-context model to compare against other sources when available.
        """
        other_resources = other_resources or []

        comparison_text = ""
        if other_resources:
            comparison_text = "\n\n--- OTHER SOURCES FOR COMPARISON ---\n"
            for r in other_resources[:5]:  # keep context reasonable
                comparison_text += f"\n[{r.title}] (current trust={r.trust_score:.2f})\n"
                comparison_text += r.original_text[:3000] + "\n"

        prompt = f"""Evaluate this educational resource for trustworthiness.

Domain context: {domain_context or "General education"}

RESOURCE TITLE: {resource.title}
SOURCE TYPE: {resource.source_type}
CONTENT (excerpt):
{resource.original_text[:12000]}

{comparison_text}

Return ONLY valid JSON with this shape:
{{
  "overall_trust_score": 0.0-1.0,
  "signals": [
    {{"signal": "currency|consistency|specificity|bias|curriculum_alignment", "score": 0.0-1.0, "evidence": "short reason"}}
  ],
  "conflicts_detected": [
    {{"with_title": "...", "description": "what conflicts", "severity": "which side is more reliable and why"}}
  ],
  "summary": "one sentence recommendation for the educator"
}}
"""

        messages = [
            {"role": "system", "content": TRUST_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            result = await self.client.chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=1500,
            )
            content = result["choices"][0]["message"]["content"]
            # Clean possible markdown fences
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content)

            resource.trust_score = float(data.get("overall_trust_score", 0.7))
            resource.trust_signals = [
                TrustSignal(**s) for s in data.get("signals", [])
            ]

            # Record conflicts
            for conflict in data.get("conflicts_detected", []):
                # We only store the description for now; full linking happens later
                resource.metadata.setdefault("conflicts", []).append(conflict)

            resource.metadata["trust_summary"] = data.get("summary", "")
            return resource

        except Exception as e:
            # Fallback: keep default score and note the failure
            resource.add_trust_signal(
                signal="evaluation_error",
                score=0.5,
                evidence=f"Automatic evaluation failed: {str(e)[:200]}",
            )
            return resource

    async def resolve_conflicts(
        self,
        resources: List[Resource],
        question: str = "",
    ) -> Dict[str, Any]:
        """
        Given multiple resources, produce a ranked trusted knowledge set
        and an explanation of how conflicts were resolved.
        """
        if len(resources) <= 1:
            return {
                "trusted_resources": resources,
                "resolution_notes": "Only one source available.",
                "final_knowledge_summary": resources[0].original_text[:2000] if resources else "",
            }

        # Sort by current trust
        ranked = sorted(resources, key=lambda r: r.trust_score, reverse=True)

        sources_text = ""
        for i, r in enumerate(ranked):
            sources_text += f"\n=== SOURCE {i+1}: {r.title} (trust={r.trust_score:.2f}, primary={r.is_primary}) ===\n"
            sources_text += r.original_text[:6000] + "\n"

        prompt = f"""You are resolving knowledge conflicts for an educational system.

The educator asked (or the learning goal is): {question or "General accurate knowledge"}

Here are the ranked sources:
{sources_text}

Produce a clean, trustworthy synthesis that an educator can rely on.
Prefer higher-trust and educator-marked primary sources.
When sources conflict, explain the conflict and justify the chosen version.

Return ONLY valid JSON:
{{
  "trusted_order": ["title1", "title2", ...],
  "resolution_notes": "how conflicts were handled",
  "final_knowledge_summary": "clean synthesis the system should use going forward",
  "discarded_or_downgraded": [{{"title": "...", "reason": "..."}}]
}}
"""

        messages = [
            {"role": "system", "content": TRUST_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        result = await self.client.chat_completion(
            messages=messages,
            temperature=0.2,
            max_tokens=2500,
        )
        content = result["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        data = json.loads(content)

        # Re-order resources according to model preference
        title_to_res = {r.title: r for r in resources}
        ordered = []
        for title in data.get("trusted_order", []):
            if title in title_to_res:
                ordered.append(title_to_res[title])
        # append any missing
        for r in ranked:
            if r not in ordered:
                ordered.append(r)

        return {
            "trusted_resources": ordered,
            "resolution_notes": data.get("resolution_notes", ""),
            "final_knowledge_summary": data.get("final_knowledge_summary", ""),
            "discarded_or_downgraded": data.get("discarded_or_downgraded", []),
        }
