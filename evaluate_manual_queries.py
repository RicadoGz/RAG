import argparse
import csv
from pathlib import Path
import math

from sentence_transformers import SentenceTransformer

from KaggleRead import (
    build_dense_index,
    build_tfidf_index,
    dense_search,
    load_rows_csv,
    tfidf_search,
)


def dcg_at_k(rels: list[int], k: int) -> float:
    score = 0.0
    for i, r in enumerate(rels[:k], start=1):
        score += r / math.log2(i + 1)
    return score


def ndcg_at_k(rels: list[int], k: int) -> float:
    actual = dcg_at_k(rels, k)
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    return 0.0 if ideal == 0.0 else actual / ideal


def load_manual_questions(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "query": r["query"].strip(),
                    "expected_intent": r["expected_intent"].strip(),
                }
            )
    return rows


def run_eval(dataset_rows: list[dict], manual_qs: list[dict], top_k: int, dense_model: str):
    dense_available = True
    model = None
    corpus_emb = None

    try:
        model = SentenceTransformer(dense_model)
        _, corpus_emb = build_dense_index(model, dataset_rows)
    except Exception as e:
        dense_available = False
        print(f"[WARN] Dense unavailable, skip Dense. reason: {e}")

    vectorizer, X = build_tfidf_index(dataset_rows)

    out_rows = []
    dense_hits = 0
    tfidf_hits = 0
    dense_top1_hits = 0
    tfidf_top1_hits = 0
    dense_mrr_sum = 0.0
    tfidf_mrr_sum = 0.0
    dense_ndcg_sum = 0.0
    tfidf_ndcg_sum = 0.0

    print(f"\nRunning {len(manual_qs)} manual queries, top_k={top_k}\n")
    for i, q in enumerate(manual_qs, start=1):
        query = q["query"]
        expected = q["expected_intent"]

        dense_results = []
        if dense_available:
            dense_results = dense_search(model, query, corpus_emb, dataset_rows, top_k=top_k)

        tfidf_results = tfidf_search(vectorizer, X, query, dataset_rows, top_k=top_k)

        dense_hit = any(r["intent"] == expected for r in dense_results) if dense_available else False
        tfidf_hit = any(r["intent"] == expected for r in tfidf_results)

        dense_hits += int(dense_hit) if dense_available else 0
        tfidf_hits += int(tfidf_hit)

        dense_top1_intent = dense_results[0]["intent"] if dense_available else "SKIPPED"
        dense_top1_score = dense_results[0]["score"] if dense_available else 0.0
        tfidf_top1_intent = tfidf_results[0]["intent"]
        tfidf_top1_score = tfidf_results[0]["score"]

        dense_rels = [1 if r["intent"] == expected else 0 for r in dense_results] if dense_available else []
        tfidf_rels = [1 if r["intent"] == expected else 0 for r in tfidf_results]

        dense_top1_hit = (dense_rels[0] == 1) if dense_available and dense_rels else False
        tfidf_top1_hit = tfidf_rels[0] == 1 if tfidf_rels else False
        dense_top1_hits += int(dense_top1_hit) if dense_available else 0
        tfidf_top1_hits += int(tfidf_top1_hit)

        if dense_available:
            dense_rr = 0.0
            for rank, rel in enumerate(dense_rels, start=1):
                if rel == 1:
                    dense_rr = 1.0 / rank
                    break
            dense_mrr_sum += dense_rr
            dense_ndcg_sum += ndcg_at_k(dense_rels, top_k)

        tfidf_rr = 0.0
        for rank, rel in enumerate(tfidf_rels, start=1):
            if rel == 1:
                tfidf_rr = 1.0 / rank
                break
        tfidf_mrr_sum += tfidf_rr
        tfidf_ndcg_sum += ndcg_at_k(tfidf_rels, top_k)

        print(f"[{i}] {query}")
        print(f"    expected         : {expected}")
        if dense_available:
            print(
                f"    Dense top1       : intent={dense_top1_intent} score={dense_top1_score:.4f} | hit@{top_k}={dense_hit}"
            )
        else:
            print("    Dense top1       : SKIPPED (model unavailable)")
        print(
            f"    TF-IDF top1      : intent={tfidf_top1_intent} score={tfidf_top1_score:.4f} | hit@{top_k}={tfidf_hit}"
        )
        print()

        out_rows.append(
            {
                "query": query,
                "expected_intent": expected,
                "dense_top1_intent": dense_top1_intent,
                "dense_top1_score": dense_top1_score,
                "dense_top1_hit": int(dense_top1_hit) if dense_available else "",
                "dense_hit_at_k": int(dense_hit) if dense_available else "",
                "tfidf_top1_intent": tfidf_top1_intent,
                "tfidf_top1_score": tfidf_top1_score,
                "tfidf_top1_hit": int(tfidf_top1_hit),
                "tfidf_hit_at_k": int(tfidf_hit),
            }
        )

    n = len(manual_qs)
    print("=== Manual Query Summary ===")
    if dense_available:
        print(f"Dense hit@{top_k} : {dense_hits}/{n} = {dense_hits / n:.4f}")
        print(f"Dense top1 acc   : {dense_top1_hits}/{n} = {dense_top1_hits / n:.4f}")
        print(f"Dense MRR@{top_k}    : {dense_mrr_sum / n:.4f}")
        print(f"Dense nDCG@{top_k}   : {dense_ndcg_sum / n:.4f}")
    else:
        print(f"Dense hit@{top_k} : SKIPPED")
    print(f"TF-IDF hit@{top_k}: {tfidf_hits}/{n} = {tfidf_hits / n:.4f}")
    print(f"TF-IDF top1 acc : {tfidf_top1_hits}/{n} = {tfidf_top1_hits / n:.4f}")
    print(f"TF-IDF MRR@{top_k}  : {tfidf_mrr_sum / n:.4f}")
    print(f"TF-IDF nDCG@{top_k} : {tfidf_ndcg_sum / n:.4f}")

    return out_rows


def save_results(rows: list[dict], path: str):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query",
                "expected_intent",
                "dense_top1_intent",
                "dense_top1_score",
                "dense_top1_hit",
                "dense_hit_at_k",
                "tfidf_top1_intent",
                "tfidf_top1_score",
                "tfidf_top1_hit",
                "tfidf_hit_at_k",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate manually written paraphrase queries.")
    parser.add_argument("--csv", default="document/csv/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
    parser.add_argument("--questions", default="manual_questions.csv")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--out", default="bi_output/manual_query_eval.csv")
    args = parser.parse_args()

    dataset_rows = load_rows_csv(args.csv)
    manual_qs = load_manual_questions(args.questions)

    results = run_eval(dataset_rows, manual_qs, top_k=args.top_k, dense_model=args.dense_model)
    save_results(results, args.out)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
