#!/usr/bin/env python3
"""
Onyx retrieval evaluation CLI.

Queries the /search/send-search-message endpoint for each test case,
computes P@k, R@k, and MRR, and writes results to CSV + JSONL.

Usage
-----
    python retrieval_eval.py \\
        --testset testset.jsonl \\
        --endpoint http://localhost:8080 \\
        --api-key  <ONYX_API_KEY> \\
        --top-k 10 \\
        --output-dir ./eval_out

Testset format (JSONL, one JSON object per line):
    {"qid": "q1", "question": "Wie beantrage ich Urlaub?", "expected_doc_ids": ["doc:123"]}

Output
------
    <output-dir>/retrieval_results.jsonl  – per-query retrieved docs + metrics
    <output-dir>/summary.csv             – aggregate P@k, R@k, MRR
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    qid: str
    question: str
    expected_doc_ids: list[str]


@dataclass
class RetrievedDoc:
    doc_id: str
    semantic_identifier: str | None
    score: float | None
    link: str | None
    source_type: str | None


@dataclass
class QueryResult:
    qid: str
    question: str
    expected_doc_ids: list[str]
    retrieved: list[RetrievedDoc]
    # per-query metrics (filled in after retrieval)
    precision_at_k: float = 0.0
    recall_at_k: float | None = None
    reciprocal_rank: float = 0.0
    first_hit_rank: int | None = None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search(
    question: str,
    *,
    base_url: str,
    api_key: str,
    top_k: int,
) -> list[RetrievedDoc]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "search_query": question,
        "filters": {},
        "num_docs_fed_to_llm_selection": top_k,
        "num_hits": top_k,
        "run_query_expansion": False,
        "stream": False,
    }
    r = requests.post(
        f"{base_url}/search/send-search-message",
        headers=headers,
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    body = r.json()

    docs: list[RetrievedDoc] = []
    for d in body.get("search_docs", [])[:top_k]:
        docs.append(
            RetrievedDoc(
                doc_id=d["document_id"],
                semantic_identifier=d.get("semantic_identifier"),
                score=d.get("score"),
                link=d.get("link"),
                source_type=d.get("source_type"),
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def precision_at_k(retrieved: list[RetrievedDoc], expected: set[str], k: int) -> float:
    top = [d.doc_id for d in retrieved[:k]]
    return sum(1 for d in top if d in expected) / k if k else 0.0


def recall_at_k(
    retrieved: list[RetrievedDoc], expected: set[str], k: int
) -> float | None:
    if not expected:
        return None
    top = [d.doc_id for d in retrieved[:k]]
    return sum(1 for d in top if d in expected) / len(expected)


def reciprocal_rank(retrieved: list[RetrievedDoc], expected: set[str]) -> float:
    for i, d in enumerate(retrieved, 1):
        if d.doc_id in expected:
            return 1.0 / i
    return 0.0


def first_hit(retrieved: list[RetrievedDoc], expected: set[str]) -> int | None:
    for i, d in enumerate(retrieved, 1):
        if d.doc_id in expected:
            return i
    return None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_testset(path: Path) -> list[TestCase]:
    cases: list[TestCase] = []
    with path.open() as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cases.append(
                    TestCase(
                        qid=obj["qid"],
                        question=obj["question"],
                        expected_doc_ids=obj["expected_doc_ids"],
                    )
                )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[WARN] Skipping line {lineno}: {e}", file=sys.stderr)
    return cases


def write_jsonl(results: list[QueryResult], path: Path) -> None:
    with path.open("w") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def write_csv(results: list[QueryResult], k: int, path: Path) -> None:
    p_vals = [r.precision_at_k for r in results]
    r_vals = [r.recall_at_k for r in results if r.recall_at_k is not None]
    rr_vals = [r.reciprocal_rank for r in results]

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        # aggregate summary
        writer.writerow(["metric", "value"])
        writer.writerow([f"P@{k}", f"{mean(p_vals):.4f}"])
        writer.writerow([f"R@{k}", f"{mean(r_vals):.4f}"])
        writer.writerow(["MRR", f"{mean(rr_vals):.4f}"])
        writer.writerow(["n_queries", len(results)])
        writer.writerow(["n_with_expected", len(r_vals)])
        writer.writerow([])
        # per-query breakdown
        writer.writerow(
            ["qid", "question", f"P@{k}", f"R@{k}", "RR", "first_hit_rank", "expected_doc_ids"]
        )
        for r in results:
            writer.writerow(
                [
                    r.qid,
                    r.question,
                    f"{r.precision_at_k:.4f}",
                    f"{r.recall_at_k:.4f}" if r.recall_at_k is not None else "",
                    f"{r.reciprocal_rank:.4f}",
                    r.first_hit_rank if r.first_hit_rank is not None else "",
                    "|".join(r.expected_doc_ids),
                ]
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Onyx retrieval quality (P@k, R@k, MRR).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--testset",
        type=Path,
        required=True,
        help="Path to JSONL test set (fields: qid, question, expected_doc_ids).",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("ONYX_API_URL", "http://localhost:8080"),
        help="Base URL of the Onyx API server. Can also be set via ONYX_API_URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ONYX_API_KEY", ""),
        help="Onyx API key (Bearer token). Can also be set via ONYX_API_KEY.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of documents to retrieve per query (used for P@k and R@k).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_out"),
        help="Directory to write retrieval_results.jsonl and summary.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key:
        sys.exit("ERROR: --api-key or ONYX_API_KEY must be set.")

    cases = load_testset(args.testset)
    if not cases:
        sys.exit("ERROR: No valid test cases found in testset.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[QueryResult] = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.qid}: {case.question[:60]}", end=" ... ", flush=True)
        try:
            retrieved = search(
                case.question,
                base_url=args.endpoint,
                api_key=args.api_key,
                top_k=args.top_k,
            )
        except requests.HTTPError as e:
            print(f"HTTP {e.response.status_code} — skipping")
            continue
        except Exception as e:
            print(f"ERROR: {e} — skipping")
            continue

        expected = set(case.expected_doc_ids)
        k = args.top_k
        qr = QueryResult(
            qid=case.qid,
            question=case.question,
            expected_doc_ids=case.expected_doc_ids,
            retrieved=retrieved,
            precision_at_k=precision_at_k(retrieved, expected, k),
            recall_at_k=recall_at_k(retrieved, expected, k),
            reciprocal_rank=reciprocal_rank(retrieved, expected),
            first_hit_rank=first_hit(retrieved, expected),
        )
        results.append(qr)

        hit = f"rank {qr.first_hit_rank}" if qr.first_hit_rank else "not found"
        print(f"P@{k}={qr.precision_at_k:.2f}  RR={qr.reciprocal_rank:.2f}  [{hit}]")

    if not results:
        sys.exit("No results to write.")

    jsonl_path = args.output_dir / "retrieval_results.jsonl"
    csv_path = args.output_dir / "summary.csv"
    write_jsonl(results, jsonl_path)
    write_csv(results, args.top_k, csv_path)

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    p_vals = [r.precision_at_k for r in results]
    r_vals = [r.recall_at_k for r in results if r.recall_at_k is not None]
    rr_vals = [r.reciprocal_rank for r in results]

    print(f"\n=== Results (k={args.top_k}, n={len(results)}) ===")
    print(f"  P@{args.top_k} : {mean(p_vals):.4f}")
    print(f"  R@{args.top_k} : {mean(r_vals):.4f}")
    print(f"  MRR   : {mean(rr_vals):.4f}")
    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
