"""
Learner modeling, performance tracking, and gap prediction.
Minimum Objectives #1 and #2.
"""

import json
import aiofiles
from pathlib import Path
from typing import List, Optional, Dict, Any
from app.models.learner import LearnerProfile, PredictedGap, LearningStyle
from app.config import get_settings
from app.services.agnes_client import get_agnes_client


class LearnerService:
    def __init__(self):
        self.settings = get_settings()
        self.learners_dir = self.settings.data_dir / "learners"
        self.learners_dir.mkdir(parents=True, exist_ok=True)
        self.client = get_agnes_client()
        self._cache: dict[str, LearnerProfile] = {}

    def _path_for(self, learner_id: str) -> Path:
        return self.learners_dir / f"{learner_id}.json"

    async def save(self, learner: LearnerProfile) -> LearnerProfile:
        path = self._path_for(learner.id)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(learner.model_dump_json(indent=2))
        self._cache[learner.id] = learner
        return learner

    async def get(self, learner_id: str) -> Optional[LearnerProfile]:
        if learner_id in self._cache:
            return self._cache[learner_id]
        path = self._path_for(learner_id)
        if not path.exists():
            return None
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            data = json.loads(await f.read())
        learner = LearnerProfile(**data)
        self._cache[learner_id] = learner
        return learner

    async def create(
        self,
        name: str,
        prior_knowledge: Dict[str, float] = None,
        learning_style: LearningStyle = None,
        language: str = "en",
    ) -> LearnerProfile:
        style = learning_style or LearningStyle(language=language)
        learner = LearnerProfile(
            name=name,
            prior_knowledge=prior_knowledge or {},
            learning_style=style,
        )
        return await self.save(learner)

    async def record_performance(
        self, learner_id: str, concept_id: str, score: float
    ) -> LearnerProfile:
        learner = await self.get(learner_id)
        if not learner:
            raise ValueError(f"Learner {learner_id} not found")
        learner.update_performance(concept_id, score)
        return await self.save(learner)

    async def predict_gaps(
        self,
        learner: LearnerProfile,
        curriculum_concepts: List[Dict[str, str]],
        trusted_knowledge_summary: str = "",
    ) -> List[PredictedGap]:
        """
        Minimum Objective #2:
        Predict future learning gaps and proactively redesign the path.
        """
        concepts_text = "\n".join(
            f"- {c['id']}: {c['name']}  {c.get('description', '')}"
            for c in curriculum_concepts
        )

        history_text = "\n".join(
            f"- {p.concept_id}: score={p.score:.2f}, mastery={p.mastery_level}, attempts={p.attempts}"
            for p in learner.performance_history
        ) or "No performance history yet."

        prior_text = "\n".join(
            f"- {k}: {v:.2f}" for k, v in learner.prior_knowledge.items()
        ) or "No prior knowledge recorded."

        prompt = f"""You are an expert learning scientist and adaptive curriculum designer.

LEARNER PROFILE
Name: {learner.name}
Learning style: {learner.learning_style.model_dump_json()}
Prior knowledge:
{prior_text}

Performance history:
{history_text}

Current planned path: {learner.current_path}

CURRICULUM CONCEPTS (in approximate teaching order):
{concepts_text}

TRUSTED KNOWLEDGE SUMMARY (from educator sources):
{trusted_knowledge_summary[:8000]}

Task:
1. Predict which concepts the learner is likely to struggle with NEXT (even if not yet attempted).
2. Give a probability (0-1) and a clear reason grounded in their profile + the knowledge.
3. Suggest a concrete intervention (extra explanation, scaffold, alternative modality, practice, etc.).
4. Propose a revised ordered path (list of concept ids) that mitigates the predicted gaps.

Return ONLY valid JSON:
{{
  "predicted_gaps": [
    {{
      "concept_id": "...",
      "concept_name": "...",
      "probability": 0.0-1.0,
      "reason": "...",
      "suggested_intervention": "..."
    }}
  ],
  "revised_path": ["concept_id1", "concept_id2", ...]
}}
"""

        messages = [
            {
                "role": "system",
                "content": "You are a precise educational AI that predicts learning difficulties before they appear and redesigns learning paths proactively.",
            },
            {"role": "user", "content": prompt},
        ]

        result = await self.client.chat_completion(
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
        )
        content = result["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        data = json.loads(content)

        gaps = [PredictedGap(**g) for g in data.get("predicted_gaps", [])]
        learner.predicted_gaps = gaps
        if data.get("revised_path"):
            learner.current_path = data["revised_path"]
        await self.save(learner)
        return gaps

    async def adapt_content_request(
        self,
        learner: LearnerProfile,
        original_content: str,
        content_type: str = "lesson",
    ) -> str:
        """
        Minimum Objective #1:
        Adapt learning content based on learner performance, prior knowledge and preferred learning style.
        """
        prompt = f"""Adapt the following educational content for this specific learner.

LEARNER:
- Name: {learner.name}
- Preferred modality: {learner.learning_style.preferred_modality}
- Pace: {learner.learning_style.pace}
- Language: {learner.learning_style.language}
- Needs scaffolding: {learner.learning_style.needs_scaffolding}
- Prior knowledge snapshot: {json.dumps(learner.prior_knowledge)}
- Recent performance: {[p.model_dump() for p in learner.performance_history[-5:]]}
- Predicted gaps: {[g.model_dump() for g in learner.predicted_gaps]}

ORIGINAL CONTENT ({content_type}):
{original_content}

Instructions:
- Rewrite / restructure so it matches the learner's modality preference.
- Adjust depth and scaffolding according to prior knowledge and predicted gaps.
- Keep the core accurate knowledge intact (do not invent facts).
- If language is not English, produce the adapted version in the learner's language.
- Make it ready for classroom use.

Return only the adapted content, no extra commentary.
"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert instructional designer who personalizes learning materials while preserving factual accuracy and source fidelity.",
            },
            {"role": "user", "content": prompt},
        ]

        result = await self.client.chat_completion(
            messages=messages,
            temperature=0.5,
            max_tokens=3000,
        )
        return result["choices"][0]["message"]["content"].strip()
