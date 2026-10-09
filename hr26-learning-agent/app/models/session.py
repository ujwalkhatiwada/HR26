from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4


class LearningObjective(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    description: str
    concept_ids: List[str] = Field(default_factory=list)
    success_criteria: str = ""


class GeneratedContent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str  # lesson, quiz, explanation, visual_prompt, activity
    title: str
    content: str
    modality: str = "text"
    source_resource_ids: List[str] = Field(default_factory=list)
    trust_notes: str = ""
    adapted_for: Optional[str] = None  # learner id
    created_at: datetime = Field(default_factory=datetime.utcnow)


class EducatorSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    educator_name: str = "Educator"
    goals: List[str] = Field(default_factory=list)
    objectives: List[LearningObjective] = Field(default_factory=list)
    resource_ids: List[str] = Field(default_factory=list)
    learner_ids: List[str] = Field(default_factory=list)
    generated_contents: List[GeneratedContent] = Field(default_factory=list)
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = "active"  # active, ready, archived
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
