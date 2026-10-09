from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from datetime import datetime
from uuid import uuid4


class LearningStyle(BaseModel):
    preferred_modality: Literal["visual", "auditory", "reading", "kinesthetic", "mixed"] = "mixed"
    pace: Literal["slow", "moderate", "fast"] = "moderate"
    language: str = "en"
    needs_scaffolding: bool = True


class PerformanceRecord(BaseModel):
    concept_id: str
    score: float = Field(ge=0.0, le=1.0)
    attempts: int = 1
    last_attempt: datetime = Field(default_factory=datetime.utcnow)
    mastery_level: Literal["none", "emerging", "developing", "proficient", "mastered"] = "none"


class PredictedGap(BaseModel):
    concept_id: str
    concept_name: str
    probability: float = Field(ge=0.0, le=1.0)
    reason: str
    suggested_intervention: str
    predicted_at: datetime = Field(default_factory=datetime.utcnow)


class LearnerProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    prior_knowledge: Dict[str, float] = Field(default_factory=dict)  # concept -> mastery 0-1
    learning_style: LearningStyle = Field(default_factory=LearningStyle)
    performance_history: List[PerformanceRecord] = Field(default_factory=list)
    predicted_gaps: List[PredictedGap] = Field(default_factory=list)
    current_path: List[str] = Field(default_factory=list)  # ordered concept ids
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def update_performance(self, concept_id: str, score: float):
        existing = next((p for p in self.performance_history if p.concept_id == concept_id), None)
        if existing:
            existing.score = (existing.score * existing.attempts + score) / (existing.attempts + 1)
            existing.attempts += 1
            existing.last_attempt = datetime.utcnow()
            if existing.score >= 0.9:
                existing.mastery_level = "mastered"
            elif existing.score >= 0.75:
                existing.mastery_level = "proficient"
            elif existing.score >= 0.5:
                existing.mastery_level = "developing"
            elif existing.score >= 0.25:
                existing.mastery_level = "emerging"
            else:
                existing.mastery_level = "none"
        else:
            mastery = "mastered" if score >= 0.9 else "proficient" if score >= 0.75 else "developing" if score >= 0.5 else "emerging" if score >= 0.25 else "none"
            self.performance_history.append(
                PerformanceRecord(concept_id=concept_id, score=score, mastery_level=mastery)
            )
        self.prior_knowledge[concept_id] = score
        self.updated_at = datetime.utcnow()
