import argparse
import os
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_kaggle_docs(csv_path: str) -> List[Document]:
    """Load CSV and convert each row to a LangChain Document with clean metadata."""
    loader = CSVLoader(file_path=csv_path, encoding="utf-8")
    raw_docs = loader.load()

    docs: List[Document] = []
    for i, d in enumerate(raw_docs):
        # CSVLoader puts whole row in page_content; keep it and add stable row_id metadata.
        md = dict(d.metadata) if d.metadata else {}
        md["row_id"] = i
        docs.append(Document(page_content=d.page_content, metadata=md))
    return docs


class TfidfRetriever:
    """Simple local retriever with invoke() API compatible with this script."""

    def __init__(self, docs: List[Document], k: int = 5):
        self.docs = docs
        self.k = k
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2)
        self.X = self.vectorizer.fit_transform([d.page_content for d in docs])

    def invoke(self, query: str) -> List[Document]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.X).ravel()
        top_idx = np.argsort(-sims)[: self.k]
        return [self.docs[int(i)] for i in top_idx]


def build_retriever(csv_path: str, embedding_model: str, top_k: int, offline: bool, retriever_type: str):
    docs = load_kaggle_docs(csv_path)
    if retriever_type == "tfidf":
        # Fast keyword retriever, no torch embedding build.
        return TfidfRetriever(docs=docs, k=top_k)

    # FAISS + embedding retriever (slower but semantic)
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    emb = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"local_files_only": offline},
    )
    vs = FAISS.from_documents(docs, emb)
    return vs.as_retriever(search_kwargs={"k": top_k})


def make_chain(csv_path: str, embedding_model: str, llm_model: str, top_k: int, offline: bool, retriever_type: str):
    retriever = build_retriever(csv_path, embedding_model, top_k, offline, retriever_type)
    llm = ChatOllama(model=llm_model, temperature=0.2)

    prompt = ChatPromptTemplate.from_template(
        """You are a customer support assistant.
Use only the retrieved context to answer.
If the answer is not in the context, say you cannot find it in the provided data.

Question:
{question}

Context:
{context}

Answer in 2-5 concise sentences.
"""
    )

    def ask(question: str):
        docs = retriever.invoke(question)
        context = "\n\n".join([f"[doc_{i}] {d.page_content}" for i, d in enumerate(docs)])
        msg = prompt.format_messages(question=question, context=context)
        try:
            response = llm.invoke(msg)
            return response.content, docs
        except Exception as e:
            hint = (
                "Failed to call Ollama. Ensure local service is running:\n"
                "1) ollama serve\n"
                "2) ollama pull llama3.1:8b\n"
            )
            raise RuntimeError(f"{hint}\nOriginal error: {e}") from e

    return ask


def main():
    parser = argparse.ArgumentParser(description="LangChain + KaggleRead demo (TF-IDF/FAISS retriever + Ollama).")
    parser.add_argument(
        "--csv",
        default="document/csv/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv",
    )
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--llm-model", default="llama3.1:8b")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retriever", choices=["tfidf", "faiss"], default="tfidf")
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--online", action="store_true", help="Disable offline mode and allow HF Hub network calls.")
    args = parser.parse_args()

    offline = args.offline and (not args.online)
    print(f"Loading retriever={args.retriever} (offline={offline}) ...")

    ask = make_chain(
        csv_path=args.csv,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        top_k=args.top_k,
        offline=offline,
        retriever_type=args.retriever,
    )

    print("LangChain Kaggle RAG is ready. Type your question (or 'exit').")
    while True:
        q = input("\nQ> ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            break
        try:
            ans, docs = ask(q)
        except Exception as e:
            print(f"\n[ERROR] {e}")
            continue
        print("\nA>", ans)
        print("\nTop sources:")
        for i, d in enumerate(docs, start=1):
            rid = d.metadata.get("row", d.metadata.get("row_id", "?"))
            print(f"{i}. row={rid} | {d.page_content[:140].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
