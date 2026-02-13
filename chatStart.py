import requests, json

OLLAMA_URL = "http://localhost:11434/api/generate"
#this is send the request into the local host and get the answer back from the model
MODEL = "llama3.1:8b" 
#this is the model we are deal with
def ask_ollama(question: str, evidence_chunks: list[dict]) -> str:
    context = "\n\n".join(
        [f"[chunk_id={c['chunk_id']}]\n{c['text']}" for c in evidence_chunks]
    )
    # will looks like this
    #     [chunk_id=12]
    # We operate our business and report our financial performance using three segments...

    # [chunk_id=155]
    # In August 2024, we announced changes to the composition of our segments...

    prompt = f"""You are a grounded QA assistant. Answer using ONLY the provided Context.

MANDATORY RULES:
1) Do NOT use any outside knowledge. If something is not explicitly in the Context, you must say it is not found.
2) Do NOT ask any follow-up questions.
3) Do NOT invent or guess numbers.
4) You must still provide the best possible answer based on what IS in the Context.
5) Every factual statement must be supported by at least one chunk_id citation.
6) If the exact requested figure is missing, write: "The exact figure is not present in the provided context." Then provide the closest relevant figures that ARE present.

OUTPUT FORMAT (must follow exactly):
Answer: <2-6 sentences, best-effort, readable>
Evidence:
- chunk_id=<id>: "<quote up to 25 words>"
- chunk_id=<id>: "<quote up to 25 words>"
Citations: [chunk_id=<id>, chunk_id=<id>, ...]



Question:
{question}

Context:
{context}

Return format:
Answer: ...
Citations: [chunk_id=..., chunk_id=...]
"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,#more slow but more accurate
            "num_ctx": 4096
        }
    }

    r = requests.post(OLLAMA_URL, data=json.dumps(payload))
    r.raise_for_status()
    return r.json()["response"]
