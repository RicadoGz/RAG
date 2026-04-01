import argparse
import csv
import os
import random
import time
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

from KaggleRead import load_rows_csv, build_dense_index, build_tfidf_index


def dcg_at_k(rels: list[int], k: int) -> float:
    score = 0.0
    for i, r in enumerate(rels[:k], start=1):
        score += r / np.log2(i + 1)
    return float(score)


def ndcg_at_k(rels: list[int], k: int) -> float:
    actual = dcg_at_k(rels, k)
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    return 0.0 if ideal == 0.0 else actual / ideal


def sample_query_indices(rows: list[dict], per_intent: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    by_intent: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_intent[r["intent"]].append(i)

    picked = []
    for idxs in by_intent.values():
        rng.shuffle(idxs)
        picked.extend(idxs[: min(per_intent, len(idxs))])
    rng.shuffle(picked)
    return picked


def rel_metrics(rels: list[int], top_k: int) -> dict:
    hit = 1.0 if any(rels) else 0.0
    p_at_k = float(sum(rels)) / top_k

    rr = 0.0
    for rank, rel in enumerate(rels, start=1):
        if rel == 1:
            rr = 1.0 / rank
            break

    return {
        "hit": hit,
        "precision": p_at_k,
        "rr": rr,
        "ndcg": ndcg_at_k(rels, top_k),
    }


def evaluate_dense_detailed(rows: list[dict], query_indices: list[int], top_k: int, model_name: str):
    model = SentenceTransformer(model_name)
    _, corpus_emb = build_dense_index(model, rows)

    per_query = []
    topk_rows = []

    for q_idx in query_indices:
        q_text = rows[q_idx]["instruction"]
        q_intent = rows[q_idx]["intent"]
        query_id = f"dense_q_{q_idx}"

        t0 = time.perf_counter()
        q_emb = model.encode([q_text], normalize_embeddings=True)
        hits = util.semantic_search(q_emb, corpus_emb, top_k=top_k + 1)[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        ranked = []
        ranked_scores = []
        for h in hits:
            idx = h["corpus_id"]
            if idx == q_idx:
                continue
            ranked.append(idx)
            ranked_scores.append(float(h["score"]))
            if len(ranked) == top_k:
                break

        rels = [1 if rows[i]["intent"] == q_intent else 0 for i in ranked]
        m = rel_metrics(rels, top_k)

        per_query.append(
            {
                "query_id": query_id,
                "method": "dense",
                "query_index": q_idx,
                "query_text": q_text,
                "query_intent": q_intent,
                "top_k": top_k,
                "latency_ms": round(latency_ms, 4),
                "top1_score": ranked_scores[0] if ranked_scores else 0.0,
                "hit_at_k": m["hit"],
                "precision_at_k": m["precision"],
                "rr": m["rr"],
                "ndcg_at_k": m["ndcg"],
            }
        )

        for rank, idx in enumerate(ranked, start=1):
            topk_rows.append(
                {
                    "query_id": query_id,
                    "method": "dense",
                    "rank": rank,
                    "doc_index": idx,
                    "score": ranked_scores[rank - 1],
                    "relevant": rels[rank - 1],
                    "query_intent": q_intent,
                    "doc_intent": rows[idx]["intent"],
                    "doc_category": rows[idx]["category"],
                    "instruction": rows[idx]["instruction"],
                    "response": rows[idx]["response"],
                }
            )

    return per_query, topk_rows


def evaluate_tfidf_detailed(rows: list[dict], query_indices: list[int], top_k: int):
    vectorizer, X = build_tfidf_index(rows)

    per_query = []
    topk_rows = []

    for q_idx in query_indices:
        q_text = rows[q_idx]["instruction"]
        q_intent = rows[q_idx]["intent"]
        query_id = f"tfidf_q_{q_idx}"

        t0 = time.perf_counter()
        q = vectorizer.transform([q_text])
        sims = cosine_similarity(q, X).ravel()
        sims[q_idx] = -1.0
        top_idx = np.argsort(-sims)[:top_k]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        rels = [1 if rows[i]["intent"] == q_intent else 0 for i in top_idx]
        scores = [float(sims[i]) for i in top_idx]
        m = rel_metrics(rels, top_k)

        per_query.append(
            {
                "query_id": query_id,
                "method": "tfidf",
                "query_index": q_idx,
                "query_text": q_text,
                "query_intent": q_intent,
                "top_k": top_k,
                "latency_ms": round(latency_ms, 4),
                "top1_score": scores[0] if scores else 0.0,
                "hit_at_k": m["hit"],
                "precision_at_k": m["precision"],
                "rr": m["rr"],
                "ndcg_at_k": m["ndcg"],
            }
        )

        for rank, idx in enumerate(top_idx, start=1):
            topk_rows.append(
                {
                    "query_id": query_id,
                    "method": "tfidf",
                    "rank": rank,
                    "doc_index": int(idx),
                    "score": scores[rank - 1],
                    "relevant": rels[rank - 1],
                    "query_intent": q_intent,
                    "doc_intent": rows[idx]["intent"],
                    "doc_category": rows[idx]["category"],
                    "instruction": rows[idx]["instruction"],
                    "response": rows[idx]["response"],
                }
            )

    return per_query, topk_rows


def summarize_metrics(per_query_rows: list[dict]) -> dict:
    n = len(per_query_rows)
    return {
        "queries": n,
        "Hit@k": sum(r["hit_at_k"] for r in per_query_rows) / n,
        "Precision@k": sum(r["precision_at_k"] for r in per_query_rows) / n,
        "MRR@k": sum(r["rr"] for r in per_query_rows) / n,
        "nDCG@k": sum(r["ndcg_at_k"] for r in per_query_rows) / n,
    }


def summarize_by_intent(per_query_rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in per_query_rows:
        groups[r["query_intent"]].append(r)

    out = []
    for intent, rows in groups.items():
        n = len(rows)
        out.append(
            {
                "method": rows[0]["method"],
                "intent": intent,
                "queries": n,
                "hit_at_k": sum(x["hit_at_k"] for x in rows) / n,
                "precision_at_k": sum(x["precision_at_k"] for x in rows) / n,
                "mrr_at_k": sum(x["rr"] for x in rows) / n,
                "ndcg_at_k": sum(x["ndcg_at_k"] for x in rows) / n,
                "avg_latency_ms": sum(x["latency_ms"] for x in rows) / n,
            }
        )
    out.sort(key=lambda x: (x["method"], x["intent"]))
    return out


def write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def print_report(name: str, metrics: dict):
    print(f"\n=== {name} ===")
    print(f"queries      : {metrics['queries']}")
    print(f"Hit@k        : {metrics['Hit@k']:.4f}")
    print(f"Precision@k  : {metrics['Precision@k']:.4f}")
    print(f"MRR@k        : {metrics['MRR@k']:.4f}")
    print(f"nDCG@k       : {metrics['nDCG@k']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dense vs TF-IDF retrieval automatically using intent labels.")
    parser.add_argument("--csv", default="document/csv/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--per-intent", type=int, default=20, help="How many queries sampled per intent label.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dense-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--out-dir", default="bi_output")
    args = parser.parse_args()

    rows = load_rows_csv(args.csv)
    query_indices = sample_query_indices(rows, per_intent=args.per_intent, seed=args.seed)

    print(f"Loaded rows: {len(rows)}")
    print(f"Sampled queries: {len(query_indices)} (per_intent={args.per_intent})")
    print(f"k={args.top_k}")

    dense_per_query, dense_topk = evaluate_dense_detailed(rows, query_indices, top_k=args.top_k, model_name=args.dense_model)
    tfidf_per_query, tfidf_topk = evaluate_tfidf_detailed(rows, query_indices, top_k=args.top_k)

    dense_metrics = summarize_metrics(dense_per_query)
    tfidf_metrics = summarize_metrics(tfidf_per_query)

    print_report("Dense", dense_metrics)
    print_report("TF-IDF", tfidf_metrics)

    os.makedirs(args.out_dir, exist_ok=True)

    eval_summary_rows = [
        {
            "method": "dense",
            "queries": dense_metrics["queries"],
            "hit_at_k": dense_metrics["Hit@k"],
            "precision_at_k": dense_metrics["Precision@k"],
            "mrr_at_k": dense_metrics["MRR@k"],
            "ndcg_at_k": dense_metrics["nDCG@k"],
        },
        {
            "method": "tfidf",
            "queries": tfidf_metrics["queries"],
            "hit_at_k": tfidf_metrics["Hit@k"],
            "precision_at_k": tfidf_metrics["Precision@k"],
            "mrr_at_k": tfidf_metrics["MRR@k"],
            "ndcg_at_k": tfidf_metrics["nDCG@k"],
        },
    ]

    eval_by_intent_rows = summarize_by_intent(dense_per_query) + summarize_by_intent(tfidf_per_query)
    query_log_rows = dense_per_query + tfidf_per_query
    retrieval_topk_rows = dense_topk + tfidf_topk

    write_csv(
        os.path.join(args.out_dir, "eval_summary.csv"),
        eval_summary_rows,
        ["method", "queries", "hit_at_k", "precision_at_k", "mrr_at_k", "ndcg_at_k"],
    )
    write_csv(
        os.path.join(args.out_dir, "eval_by_intent.csv"),
        eval_by_intent_rows,
        ["method", "intent", "queries", "hit_at_k", "precision_at_k", "mrr_at_k", "ndcg_at_k", "avg_latency_ms"],
    )
    write_csv(
        os.path.join(args.out_dir, "query_log.csv"),
        query_log_rows,
        [
            "query_id",
            "method",
            "query_index",
            "query_text",
            "query_intent",
            "top_k",
            "latency_ms",
            "top1_score",
            "hit_at_k",
            "precision_at_k",
            "rr",
            "ndcg_at_k",
        ],
    )
    write_csv(
        os.path.join(args.out_dir, "retrieval_topk.csv"),
        retrieval_topk_rows,
        [
            "query_id",
            "method",
            "rank",
            "doc_index",
            "score",
            "relevant",
            "query_intent",
            "doc_intent",
            "doc_category",
            "instruction",
            "response",
        ],
    )

    print(f"\nCSV exported to: {args.out_dir}")
    print("- eval_summary.csv")
    print("- eval_by_intent.csv")
    print("- query_log.csv")
    print("- retrieval_topk.csv")


if __name__ == "__main__":
    main()
