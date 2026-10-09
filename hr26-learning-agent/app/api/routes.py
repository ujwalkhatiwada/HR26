from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional, Dict, Any
from pathlib import Path
import aiofiles
import tempfile
from pydantic import BaseModel

from app.services.resource_service import ResourceService
from app.services.learner_service import LearnerService
from app.agents.orchestrator import Orchestrator
from app.models.learner import LearningStyle
from app.models.resource import SourceType

router = APIRouter()

resource_svc = ResourceService()
learner_svc = LearnerService()
orchestrator = Orchestrator()


# ---------- Request / Response models ----------
class TextIngestRequest(BaseModel):
    title: str
    text: str
    source_type: str = "notes"
    language: str = "en"
    is_primary: bool = False
    domain_context: str = ""


class CreateLearnerRequest(BaseModel):
    name: str
    prior_knowledge: Dict[str, float] = {}
    preferred_modality: str = "mixed"
    pace: str = "moderate"
    language: str = "en"
    needs_scaffolding: bool = True


class PerformanceRequest(BaseModel):
    concept_id: str
    score: float


class ChatRequest(BaseModel):
    message: str
    conversation_history: List[Dict[str, Any]] = []


class PredictGapsRequest(BaseModel):
    curriculum_concepts: List[Dict[str, str]]
    trusted_knowledge_summary: str = ""


# ---------- Resource endpoints ----------
@router.post("/resources/text")
async def ingest_text(req: TextIngestRequest):
    try:
        source_type = SourceType(req.source_type)
    except ValueError:
        source_type = SourceType.NOTES

    resource = await resource_svc.ingest_text(
        title=req.title,
        text=req.text,
        source_type=source_type,
        language=req.language,
        is_primary=req.is_primary,
        domain_context=req.domain_context,
    )
    return {
        "id": resource.id,
        "title": resource.title,
        "trust_score": resource.trust_score,
        "trust_summary": resource.metadata.get("trust_summary", ""),
        "signals": [s.model_dump() for s in resource.trust_signals],
    }


@router.post("/resources/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    title: str = Form(...),
    language: str = Form("en"),
    is_primary: bool = Form(False),
    domain_context: str = Form(""),
):
    suffix = Path(file.filename or "upload.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        resource = await resource_svc.ingest_pdf(
            title=title,
            file_path=tmp_path,
            language=language,
            is_primary=is_primary,
            domain_context=domain_context,
        )
        return {
            "id": resource.id,
            "title": resource.title,
            "trust_score": resource.trust_score,
            "trust_summary": resource.metadata.get("trust_summary", ""),
            "page_count_approx": len(resource.chunks),
        }
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/resources")
async def list_resources():
    resources = await resource_svc.list_all()
    return [
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


@router.post("/resources/resolve-conflicts")
async def resolve_conflicts(question: str = ""):
    result = await resource_svc.resolve_all_conflicts(question=question)
    result["trusted_resources"] = [
        {"id": r.id, "title": r.title, "trust_score": r.trust_score}
        for r in result["trusted_resources"]
    ]
    return result


# ---------- Learner endpoints ----------
@router.post("/learners")
async def create_learner(req: CreateLearnerRequest):
    style = LearningStyle(
        preferred_modality=req.preferred_modality,  # type: ignore
        pace=req.pace,  # type: ignore
        language=req.language,
        needs_scaffolding=req.needs_scaffolding,
    )
    learner = await learner_svc.create(
        name=req.name,
        prior_knowledge=req.prior_knowledge,
        learning_style=style,
        language=req.language,
    )
    return learner.model_dump()


@router.get("/learners/{learner_id}")
async def get_learner(learner_id: str):
    learner = await learner_svc.get(learner_id)
    if not learner:
        raise HTTPException(status_code=404, detail="Learner not found")
    return learner.model_dump()


@router.post("/learners/{learner_id}/performance")
async def record_performance(learner_id: str, req: PerformanceRequest):
    try:
        learner = await learner_svc.record_performance(
            learner_id=learner_id,
            concept_id=req.concept_id,
            score=req.score,
        )
        return {"status": "ok", "prior_knowledge": learner.prior_knowledge}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/learners/{learner_id}/predict-gaps")
async def predict_gaps(learner_id: str, req: PredictGapsRequest):
    learner = await learner_svc.get(learner_id)
    if not learner:
        raise HTTPException(status_code=404, detail="Learner not found")
    gaps = await learner_svc.predict_gaps(
        learner=learner,
        curriculum_concepts=req.curriculum_concepts,
        trusted_knowledge_summary=req.trusted_knowledge_summary,
    )
    updated = await learner_svc.get(learner_id)
    return {
        "predicted_gaps": [g.model_dump() for g in gaps],
        "revised_path": updated.current_path if updated else [],
    }


# ---------- Main Agent Chat ----------
@router.post("/agent/chat")
async def agent_chat(req: ChatRequest):
    """
    Primary entry point for the educator.
    The orchestrator agent will use tools to hit all three minimum objectives.
    """
    result = await orchestrator.run(
        user_message=req.message,
        conversation_history=req.conversation_history,
    )
    return {
        "reply": result["reply"],
        "trace": result["trace"],
        "images": result.get("images", []),
    }
