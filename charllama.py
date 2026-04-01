import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"


def _kaggle_context(rows: list[dict]) -> str:
    """Convert KaggleRead retrieval rows into prompt context blocks."""
    blocks = []
    for i, r in enumerate(rows):
        blocks.append(
            "\n".join(
                [
                    f"[row={i}]",
                    f"score: {r.get('score', '')}",
                    f"instruction: {r.get('instruction', '')}",
                    f"intent: {r.get('intent', '')}",
                    f"category: {r.get('category', '')}",
                    f"response: {r.get('response', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def ask_kaggle_llama(question: str, retrieved_rows: list[dict]) -> str:
    """Answer a user question using KaggleRead retrieval output only."""
    context = _kaggle_context(retrieved_rows)

    prompt = f"""You are a customer-support assistant.
Use ONLY the provided retrieved rows to answer.

Rules:
1) Do not use external knowledge.
2) If rows do not contain enough information, say so clearly.
3) Prefer actionable steps in plain English. give specific instructions to the user if possible.
4) Cite row ids for factual claims.

Output format:
Answer: <>
Citations: [row=<id>, row=<id>, ...]

Question:
{question}

Retrieved rows:
{context}
"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096,
        },
    }

    resp = requests.post(OLLAMA_URL, data=json.dumps(payload), timeout=120)
    resp.raise_for_status()
    return resp.json()["response"]


if __name__ == "__main__":
    demo_rows = [
        {
            "score": 0.91,
            "instruction": "I want to cancel my order",
            "intent": "cancel_order",
            "category": "order",
            "response": "You can cancel your order from your account if it has not shipped yet.",
        },
        {
            "score": 0.84,
            "instruction": "How do I request a refund?",
            "intent": "request_refund",
            "category": "refund",
            "response": "Please submit a refund request from the order details page.",
        },
    ]

    print(ask_kaggle_llama("I want to cancel my order", demo_rows))
