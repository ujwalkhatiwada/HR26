"""
Resource ingestion & storage service.
Handles PDFs, plain text, notes, etc.
"""

import json
import aiofiles
from pathlib import Path
from typing import List, Optional
from uuid import uuid4
import pdfplumber
from app.models.resource import Resource, ResourceChunk, SourceType
from app.config import get_settings
from app.services.trust_engine import TrustEngine


class ResourceService:
    def __init__(self):
        self.settings = get_settings()
        self.resources_dir = self.settings.data_dir / "resources"
        self.resources_dir.mkdir(parents=True, exist_ok=True)
        self.trust_engine = TrustEngine()
        self._cache: dict[str, Resource] = {}

    def _path_for(self, resource_id: str) -> Path:
        return self.resources_dir / f"{resource_id}.json"

    async def save(self, resource: Resource) -> Resource:
        path = self._path_for(resource.id)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(resource.model_dump_json(indent=2))
        self._cache[resource.id] = resource
        return resource

    async def get(self, resource_id: str) -> Optional[Resource]:
        if resource_id in self._cache:
            return self._cache[resource_id]
        path = self._path_for(resource_id)
        if not path.exists():
            return None
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            data = json.loads(await f.read())
        resource = Resource(**data)
        self._cache[resource_id] = resource
        return resource

    async def list_all(self) -> List[Resource]:
        resources = []
        for path in self.resources_dir.glob("*.json"):
            rid = path.stem
            r = await self.get(rid)
            if r:
                resources.append(r)
        return resources

    async def ingest_text(
        self,
        title: str,
        text: str,
        source_type: SourceType = SourceType.NOTES,
        language: str = "en",
        is_primary: bool = False,
        run_trust: bool = True,
        domain_context: str = "",
    ) -> Resource:
        resource = Resource(
            title=title,
            source_type=source_type,
            original_text=text,
            language=language,
            is_primary=is_primary,
        )

        # Simple chunking by paragraphs / size
        chunks = self._chunk_text(text)
        resource.chunks = chunks

        if run_trust:
            other = await self.list_all()
            resource = await self.trust_engine.evaluate_resource(
                resource, other_resources=other, domain_context=domain_context
            )

        return await self.save(resource)

    async def ingest_pdf(
        self,
        title: str,
        file_path: Path,
        language: str = "en",
        is_primary: bool = False,
        run_trust: bool = True,
        domain_context: str = "",
    ) -> Resource:
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(f"--- Page {i+1} ---\n{page_text}")

        full_text = "\n\n".join(text_parts)
        resource = Resource(
            title=title,
            source_type=SourceType.PDF,
            filename=file_path.name,
            original_text=full_text,
            language=language,
            is_primary=is_primary,
        )
        resource.chunks = self._chunk_text(full_text)

        if run_trust:
            other = await self.list_all()
            resource = await self.trust_engine.evaluate_resource(
                resource, other_resources=other, domain_context=domain_context
            )

        return await self.save(resource)

    def _chunk_text(self, text: str, max_chars: int = 1800) -> List[ResourceChunk]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[ResourceChunk] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 > max_chars and current:
                chunks.append(ResourceChunk(content=current.strip()))
                current = para
            else:
                current = (current + "\n\n" + para).strip()
        if current:
            chunks.append(ResourceChunk(content=current.strip()))
        return chunks

    async def resolve_all_conflicts(self, question: str = "") -> dict:
        resources = await self.list_all()
        return await self.trust_engine.resolve_conflicts(resources, question=question)
