import argparse
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from KaggleRead import (
    build_dense_index,
    build_tfidf_index,
    dense_search,
    load_rows_csv,
    tfidf_search,
)


@dataclass
class ZhCase:
    query_zh: str
    expected_intent: str


ZH_CASES = [
    ZhCase("我想取消订单", "cancel_order"),
    ZhCase("我可以修改订单吗", "change_order"),
    ZhCase("我想改收货地址", "change_shipping_address"),
    ZhCase("取消订单会收手续费吗", "check_cancellation_fee"),
    ZhCase("我的发票在哪里下载", "get_invoice"),
    ZhCase("你们支持哪些付款方式", "check_payment_methods"),
    ZhCase("我想申请退款", "get_refund"),
    ZhCase("退款什么时候到账", "track_refund"),
    ZhCase("我的订单物流到哪了", "track_order"),
    ZhCase("送达大概需要几天", "delivery_period"),
    ZhCase("我要联系人工客服", "contact_human_agent"),
    ZhCase("我忘记密码了怎么找回", "recover_password"),
]


def run_cases(rows: list[dict], top_k: int, model_name: str):
    model = None
    corpus_emb = None
    dense_available = True
    try:
        model = SentenceTransformer(model_name)
        _, corpus_emb = build_dense_index(model, rows)
    except Exception as e:
        dense_available = False
        print(f"[WARN] Dense model unavailable, skip Dense. reason: {e}")
    vectorizer, X = build_tfidf_index(rows)

    dense_hit = 0
    tfidf_hit = 0

    print(f"\nRunning {len(ZH_CASES)} Chinese queries, top_k={top_k}\n")
    for i, case in enumerate(ZH_CASES, start=1):
        dense = []
        if dense_available:
            dense = dense_search(model, case.query_zh, corpus_emb, rows, top_k=top_k)
        tfidf = tfidf_search(vectorizer, X, case.query_zh, rows, top_k=top_k)

        dense_hit_flag = any(r["intent"] == case.expected_intent for r in dense) if dense_available else False
        tfidf_hit_flag = any(r["intent"] == case.expected_intent for r in tfidf)

        dense_hit += int(dense_hit_flag) if dense_available else 0
        tfidf_hit += int(tfidf_hit_flag)

        dense_top1 = dense[0] if dense_available else None
        tfidf_top1 = tfidf[0]

        print(f"[{i}] query: {case.query_zh}")
        print(f"    expected intent : {case.expected_intent}")
        if dense_available:
            print(
                f"    Dense top1      : intent={dense_top1['intent']} score={dense_top1['score']:.4f} | hit@{top_k}={dense_hit_flag}"
            )
        else:
            print("    Dense top1      : SKIPPED (model unavailable)")
        print(
            f"    TF-IDF top1     : intent={tfidf_top1['intent']} score={tfidf_top1['score']:.4f} | hit@{top_k}={tfidf_hit_flag}"
        )
        print()

    n = len(ZH_CASES)
    print("=== Chinese Query Summary ===")
    if dense_available:
        print(f"Dense hit@{top_k} : {dense_hit}/{n} = {dense_hit / n:.4f}")
    else:
        print(f"Dense hit@{top_k} : SKIPPED")
    print(f"TF-IDF hit@{top_k}: {tfidf_hit}/{n} = {tfidf_hit / n:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test Chinese queries on current KaggleRead retrieval pipelines.")
    parser.add_argument(
        "--csv",
        default="document/csv/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    rows = load_rows_csv(args.csv)
    run_cases(rows, top_k=args.top_k, model_name=args.dense_model)


if __name__ == "__main__":
    main()
