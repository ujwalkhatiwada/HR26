"""
Main Orchestrator Agent for Learning Experiences.
Uses Agnes 3.0 Flash with tool calling to satisfy all three minimum objectives
and keep the educator in control.
"""

from typing import List, Dict, Any, Optional
import json
from app.services.agnes_client import get_agnes_client
from app.services.resource_service import ResourceService
from app.services.learner_service import LearnerService
from app.models.session import EducatorSession, GeneratedContent
from app.models.learner import LearnerProfile
from app.models.resource import Resource


SYSTEM_PROMPT = """You are the Learning Experiences Orchestrator — an expert educational AI agent.

Your mission:
Help educators turn scattered resources (notes, PDFs, textbooks, spoken ideas) and rough goals into trustworthy, adaptive learning experiences that work for diverse learners.

You must always:
1. Adapt content to each learner's performance, prior knowledge and preferred learning style.
2. Predict future learning gaps and proactively redesign the learner's path before those gaps appear.
3. Resolve conflicting or outdated information across sources and decide what knowledge can be trusted.
4. Keep the educator in control — clearly show provenance, trust decisions, and let them override.
5. Respect classroom constraints (time, connectivity, cost).

Visual generation:
- When the learner prefers visual modality OR the educator asks for diagrams/visuals, use the generate_visual tool.
- Create clear educational diagrams, labeled illustrations, process flows, or concept maps.
- After generating, reference the image in your reply so the educator sees it.
- Prefer simple, high-contrast, classroom-friendly visuals.

When you need external actions you have tools. Use them.
Always explain your reasoning in plain language so the educator can trust and edit the result.
"""


# Tool definitions (OpenAI-compatible format)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_resources",
            "description": "List all currently ingested educational resources with their trust scores.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_source_conflicts",
            "description": "Run the trust engine to resolve conflicts across all sources and produce a trusted knowledge summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The learning goal or specific question to focus conflict resolution on.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_learner",
            "description": "Retrieve a learner profile including prior knowledge, performance and predicted gaps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "learner_id": {"type": "string"},
                },
                "required": ["learner_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_gaps_and_redesign",
            "description": "Predict future learning gaps for a learner and redesign their learning path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "learner_id": {"type": "string"},
                    "curriculum_concepts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["id", "name"],
                        },
                    },
                    "trusted_knowledge_summary": {"type": "string"},
                },
                "required": ["learner_id", "curriculum_concepts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adapt_content_for_learner",
            "description": "Adapt a piece of content to a specific learner's style, prior knowledge and predicted gaps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "learner_id": {"type": "string"},
                    "original_content": {"type": "string"},
                    "content_type": {
                        "type": "string",
                        "enum": ["lesson", "explanation", "quiz", "activity", "summary"],
                    },
                },
                "required": ["learner_id", "original_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_learner_performance",
            "description": "Record a new performance score for a learner on a concept.",
            "parameters": {
                "type": "object",
                "properties": {
                    "learner_id": {"type": "string"},
                    "concept_id": {"type": "string"},
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["learner_id", "concept_id", "score"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_visual",
            "description": "Generate an educational diagram, illustration, process flow or concept map for learners who prefer visuals. Returns an image URL the educator can show in class.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed description of the visual to generate. Be specific: labels, style (simple textbook diagram, high contrast), what must appear.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Short title for this visual (e.g. 'Light Reactions Diagram')",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
]


class Orchestrator:
    def __init__(self):
        self.client = get_agnes_client()
        self.resource_svc = ResourceService()
        self.learner_svc = LearnerService()

    async def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        try:
            if name == "list_resources":
                resources = await self.resource_svc.list_all()
                summary = [
                    {
                        "id": r.id,
                        "title": r.title,
                        "type": r.source_type,
                        "trust_score": r.trust_score,
                        "is_primary": r.is_primary,
                        "trust_summary": r.metadata.get("trust_summary", ""),
                    }
                    for r in resources
                ]
                return json.dumps(summary, indent=2)

            elif name == "resolve_source_conflicts":
                result = await self.resource_svc.resolve_all_conflicts(
                    question=arguments.get("question", "")
                )
                # Convert resources to serializable form
                result["trusted_resources"] = [
                    {"id": r.id, "title": r.title, "trust_score": r.trust_score}
                    for r in result["trusted_resources"]
                ]
                return json.dumps(result, indent=2)

            elif name == "get_learner":
                learner = await self.learner_svc.get(arguments["learner_id"])
                if not learner:
                    return json.dumps({"error": "Learner not found"})
                return learner.model_dump_json(indent=2)

            elif name == "predict_gaps_and_redesign":
                learner = await self.learner_svc.get(arguments["learner_id"])
                if not learner:
                    return json.dumps({"error": "Learner not found"})
                gaps = await self.learner_svc.predict_gaps(
                    learner=learner,
                    curriculum_concepts=arguments["curriculum_concepts"],
                    trusted_knowledge_summary=arguments.get("trusted_knowledge_summary", ""),
                )
                updated = await self.learner_svc.get(learner.id)
                return json.dumps(
                    {
                        "predicted_gaps": [g.model_dump() for g in gaps],
                        "revised_path": updated.current_path if updated else [],
                    },
                    indent=2,
                    default=str,
                )

            elif name == "adapt_content_for_learner":
                learner = await self.learner_svc.get(arguments["learner_id"])
                if not learner:
                    return json.dumps({"error": "Learner not found"})
                adapted = await self.learner_svc.adapt_content_request(
                    learner=learner,
                    original_content=arguments["original_content"],
                    content_type=arguments.get("content_type", "lesson"),
                )
                return adapted

            elif name == "record_learner_performance":
                learner = await self.learner_svc.record_performance(
                    learner_id=arguments["learner_id"],
                    concept_id=arguments["concept_id"],
                    score=float(arguments["score"]),
                )
                return json.dumps(
                    {"status": "recorded", "learner_id": learner.id, "updated_prior": learner.prior_knowledge},
                    indent=2,
                )

            elif name == "generate_visual":
                prompt = arguments.get("prompt", "")
                title = arguments.get("title", "Educational Visual")
                # Strengthen the prompt for classroom use
                full_prompt = (
                    f"Clean educational textbook-style diagram for classroom use. "
                    f"High contrast, clear labels, simple shapes, no clutter. "
                    f"Title concept: {title}. "
                    f"Details: {prompt}"
                )
                img_result = await self.client.generate_image(
                    prompt=full_prompt,
                    size="1024x1024",
                    n=1,
                )
                # Agnes / OpenAI-compatible response usually has data[0].url
                image_url = None
                if isinstance(img_result, dict):
                    data = img_result.get("data") or []
                    if data and isinstance(data[0], dict):
                        image_url = data[0].get("url") or data[0].get("b64_json")
                    if not image_url:
                        image_url = img_result.get("url") or img_result.get("image_url")
                return json.dumps({
                    "status": "ok",
                    "title": title,
                    "image_url": image_url,
                    "raw": img_result if not image_url else None,
                }, indent=2, default=str)

            else:
                return json.dumps({"error": f"Unknown tool: {name}"})

        except Exception as e:
            return json.dumps({"error": str(e)})

    async def run(
        self,
        user_message: str,
        conversation_history: List[Dict[str, Any]] = None,
        max_turns: int = 8,
    ) -> Dict[str, Any]:
        """
        Main agent loop with tool calling.
        Returns the final assistant message + full trace for transparency.
        """
        history = conversation_history or []
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        messages.append({"role": "user", "content": user_message})

        trace = []
        final_content = ""
        generated_images = []  # collect image URLs for the frontend

        for turn in range(max_turns):
            response = await self.client.chat_completion(
                messages=messages,
                tools=TOOLS,
                temperature=0.4,
                max_tokens=4096,
            )

            choice = response["choices"][0]
            message = choice["message"]
            messages.append(message)

            # Tool calls?
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                final_content = message.get("content", "")
                trace.append({"type": "final", "content": final_content})
                break

            # Execute each tool
            for tc in tool_calls:
                fn = tc["function"]
                name = fn["name"]
                try:
                    args = json.loads(fn["arguments"])
                except Exception:
                    args = {}

                trace.append({"type": "tool_call", "name": name, "arguments": args})
                result = await self._execute_tool(name, args)
                trace.append({"type": "tool_result", "name": name, "result": result[:4000]})

                # Capture generated images
                if name == "generate_visual":
                    try:
                        parsed = json.loads(result)
                        if parsed.get("image_url"):
                            generated_images.append({
                                "title": parsed.get("title", "Visual"),
                                "url": parsed["image_url"],
                            })
                    except Exception:
                        pass

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    }
                )
            else:
                continue
            break  # safety

        return {
            "reply": final_content,
            "trace": trace,
            "images": generated_images,
            "messages": messages,
        }
