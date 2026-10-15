from openai import OpenAI

from src.config import OPENAI_API_KEY, CHAT_MODEL


SYSTEM_PROMPT = """
You are a Cybersecurity Policy and Compliance Assistant.

Answer questions using only the supplied evidence from the
provided cybersecurity documents.

Rules:
1. Do not invent controls, requirements, or citations.
2. If the evidence is insufficient, explicitly say so.
3. Distinguish recommendations stated in the documents from
   your own interpretation.
4. Explain technical concepts clearly.
5. Cite evidence using the source and page metadata provided.
6. Do not claim an organization is compliant based only on
   the retrieved excerpts.
"""


def generate_answer(question: str, evidence: list[dict]) -> str:
    """Generate an evidence-grounded answer."""

    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Check the project .env file."
        )

    if not evidence:
        return (
            "I could not retrieve supporting evidence from the "
            "configured cybersecurity documents."
        )

    context_parts = []

    for item in evidence:
        context_parts.append(
            f"""
Source: {item.get('source', 'Unknown')}
Document: {item.get('document', 'Unknown')}
PDF page: {item.get('page', 'Unknown')}
Section: {item.get('section', 'Unknown')}
Evidence:
{item.get('text', '')}
"""
        )

    context = "\n\n---\n\n".join(context_parts)

    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Retrieved evidence:\n{context}\n\n"
                    "Answer using the evidence above. Cite the "
                    "source and PDF page for factual claims."
                ),
            },
        ],
    )

    return response.choices[0].message.content or (
        "The model returned an empty answer."
    )
