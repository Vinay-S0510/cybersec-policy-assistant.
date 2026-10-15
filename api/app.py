from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.retrieval import retrieve
from src.generator import generate_answer

app = FastAPI(
    title="Cybersecurity Policy & Compliance Assistant",
    description="An evidence-grounded RAG system for cybersecurity guidance.",
    version="1.0.0",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static",
)


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Cybersecurity policy or compliance question",
    )


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    try:
        evidence = retrieve(question, top_k=5)
        answer = generate_answer(question, evidence)

        sources = []
        seen = set()

        for item in evidence:
            source = item.get("source", "Unknown source")
            page = item.get("page", "Unknown")
            section = item.get("section", "Unknown section")

            key = (source, page, section)

            if key in seen:
                continue

            seen.add(key)

            sources.append({
                "source": source,
                "page": page,
                "section": section,
            })

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to process the question. "
                "Check the server terminal for the underlying error."
            ),
        )
