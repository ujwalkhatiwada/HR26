from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4
from enum import Enum


class SourceType(str, Enum):
    PDF = "pdf"
    TEXT = "text"
    NOTES = "notes"
    TEXTBOOK = "textbook"
    PRESENTATION = "presentation"
    LINK = "link"
    LECTURE_NOTES = "lecture_notes"


class TrustSignal(BaseModel):
    signal: str
    score: float = Field(ge=0.0, le=1.0)
    evidence: str


class ResourceChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    page: Optional[int] = None
    section: Optional[str] = None
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Resource(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    source_type: SourceType
    filename: Optional[str] = None
    original_text: str = ""
    chunks: List[ResourceChunk] = Field(default_factory=list)
    trust_score: float = Field(default=0.7, ge=0.0, le=1.0)
    trust_signals: List[TrustSignal] = Field(default_factory=list)
    conflicts_with: List[str] = Field(default_factory=list)  # other resource ids
    is_primary: bool = False  # educator-marked as preferred
    language: str = "en"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_trust_signal(self, signal: str, score: float, evidence: str):
        self.trust_signals.append(TrustSignal(signal=signal, score=score, evidence=evidence))
        if self.trust_signals:
            self.trust_score = sum(s.score for s in self.trust_signals) / len(self.trust_signals)
