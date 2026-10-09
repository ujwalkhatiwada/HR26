"""
Quick demo script that exercises the full pipeline.
Requires a valid AGNES_API_KEY in .env
"""

import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.resource_service import ResourceService
from app.services.learner_service import LearnerService
from app.agents.orchestrator import Orchestrator
from app.models.learner import LearningStyle


async def main():
    print("=== HR26 Learning Experiences – Demo Flow ===\n")

    resource_svc = ResourceService()
    learner_svc = LearnerService()
    agent = Orchestrator()

    # 1. Ingest two slightly conflicting sources
    print("1. Ingesting lecture notes...")
    notes = await resource_svc.ingest_text(
        title="Teacher Lecture Notes – Photosynthesis",
        text="""
Photosynthesis is the process by which green plants convert light energy into chemical energy.
The overall equation is: 6CO2 + 6H2O → C6H12O6 + 6O2
It occurs in chloroplasts. Light-dependent reactions happen in the thylakoid membrane.
The Calvin cycle (light-independent) fixes carbon in the stroma.
Chlorophyll a is the main pigment. Water is split, releasing oxygen.
        """,
        source_type="lecture_notes",
        is_primary=True,
        domain_context="High-school biology – photosynthesis",
    )
    print(f"   → Trust score: {notes.trust_score:.2f}")

    print("\n2. Ingesting an older textbook excerpt (slightly outdated wording)...")
    textbook = await resource_svc.ingest_text(
        title="Old Textbook Chapter 7 (2012)",
        text="""
Photosynthesis: Plants use sunlight to make food. The formula is often written as
CO2 + H2O → sugar + O2 (coefficients sometimes omitted in older books).
It takes place only in the presence of chlorophyll. The dark reaction is also called
the Blackman reaction. Some older texts claim the oxygen comes from carbon dioxide,
but modern understanding is that it comes from water.
        """,
        source_type="textbook",
        domain_context="High-school biology – photosynthesis",
    )
    print(f"   → Trust score: {textbook.trust_score:.2f}")

    # 2. Resolve conflicts
    print("\n3. Resolving source conflicts...")
    resolution = await resource_svc.resolve_all_conflicts(
        question="Accurate explanation of photosynthesis for grade 10"
    )
    print("   Resolution notes:", resolution["resolution_notes"][:300], "...")
    print("   Trusted order:", [r.title for r in resolution["trusted_resources"]])

    # 3. Create a learner
    print("\n4. Creating a visual, moderate-pace learner with weak chemistry...")
    learner = await learner_svc.create(
        name="Aisha",
        prior_knowledge={"basic_plant_structure": 0.7, "atoms_and_molecules": 0.3},
        learning_style=LearningStyle(
            preferred_modality="visual",
            pace="moderate",
            language="en",
            needs_scaffolding=True,
        ),
    )
    print(f"   → Learner id: {learner.id}")

    # 4. Talk to the agent
    print("\n5. Asking the Orchestrator to build an adaptive lesson...")
    result = await agent.run(
        user_message=(
            f"I have ingested notes and a textbook about photosynthesis. "
            f"Create a 30-minute learning experience for learner {learner.id} (Aisha). "
            f"She is a visual learner and weak on chemistry basics. "
            f"Predict any gaps and redesign the path if needed. "
            f"Show me the final lesson content and the trust decisions you made."
        )
    )

    print("\n=== AGENT REPLY ===")
    print(result["reply"])
    print("\n=== TOOL TRACE (transparency) ===")
    for step in result["trace"]:
        print(json.dumps(step, indent=2, default=str)[:500])
        print("---")


if __name__ == "__main__":
    asyncio.run(main())
