Python 3.9.9 (tags/v3.9.9:ccb0e6a, Nov 15 2021, 18:08:50) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> #!/usr/bin/env python3
import argparse
import json
import os
import time
import pickle
import hashlib
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from scipy.sparse.linalg import eigsh
from sklearn.model_selection import train_test_split

from tree_sitter import Language, Parser

warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    p = argparse.ArgumentParser("VDISC Preprocessing: dedup + AST + LPE (+ spans)")
    p.add_argument("--cwe-csv", nargs="+", required=True,
                   help="Pairs: CWE_NAME=path.csv (e.g., CWE_119=/path/vdisc_CWE_119.csv)")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--ts-so", required=True, help="tree-sitter compiled languages .so")
    p.add_argument("--ts-lang", default="c", help="tree-sitter language name (default: c)")
    p.add_argument("--k-eig", type=int, default=16)
    p.add_argument("--shuffle-seed", type=int, default=42)
    p.add_argument("--drop-cols", nargs="*", default=["Unnamed: 0", "testCase_ID", "type", "filename"])
    p.add_argument("--store-code", action="store_true",
                   help="Store raw code in Data objects (needed for training/tokenization).")
    p.add_argument("--export-splits", action="store_true",
                   help="Export deterministic split IDs (train/val/test) as JSON.")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--pst-norm", action="store_true", help="Apply per-dimension z-norm to PE (recommended).")
    return p.parse_args()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def canonicalize_code(code: str) -> str:
    # Exact-match dedup uses the canonical string. Keep it conservative.
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    return code


def clean_df(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
    df = df.drop(columns=drop_cols, errors="ignore")
    if "bug" not in df.columns or "code" not in df.columns:
        raise ValueError("CSV must contain columns: ['code', 'bug']")
    df["bug"] = df["bug"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(df["bug"])
    df = df.dropna(subset=["code", "bug"]).copy()
    df["code"] = df["code"].astype(str).map(canonicalize_code)
    df["bug"] = df["bug"].astype(int)
    return df


def build_parser(ts_so: str, ts_lang: str) -> Parser:
    lang = Language(ts_so, ts_lang)
    p = Parser()
    p.set_language(lang)
    return p


def parse_code_to_ast_and_spans(parser: Parser, code: str) -> Tuple[Optional[List[str]], Optional[List[Tuple[int, int]]], Optional[List[Tuple[int, int]]]]:
    """
    Returns:
      node_types: list[str] length N
      edges: list[(parent, child)] in BFS order indices
      spans: list[(start_byte, end_byte)] length N aligned with node_types order
    """
    try:
        tree = parser.parse(code.encode("utf8", errors="ignore"))
        root = tree.root_node
        node_types: List[str] = []
        edges: List[Tuple[int, int]] = []
        spans: List[Tuple[int, int]] = []

        queue: List[Tuple[object, int]] = [(root, -1)]
        while queue:
            node, parent_idx = queue.pop(0)
            cur_idx = len(node_types)
            node_types.append(node.type)
            spans.append((int(node.start_byte), int(node.end_byte)))
            if parent_idx != -1:
                edges.append((parent_idx, cur_idx))
            for child in node.children:
                queue.append((child, cur_idx))

        return node_types, edges, spans
    except Exception:
        return None, None, None


def laplacian_pe(edge_index: torch.Tensor, num_nodes: int, k: int) -> torch.Tensor:
    if num_nodes <= 1 or edge_index.numel() == 0 or edge_index.shape[1] == 0:
        return torch.zeros((num_nodes, k), dtype=torch.float)

    L_idx, L_val = get_laplacian(edge_index, num_nodes=num_nodes, normalization="sym")
    L_sp = to_scipy_sparse_matrix(L_idx, L_val, num_nodes=num_nodes)

    k_compute = min(k, max(0, num_nodes - 2))
    if k_compute <= 0:
        return torch.zeros((num_nodes, k), dtype=torch.float)

    try:
        _, vecs = eigsh(L_sp, k=k_compute, which="SM", tol=1e-5)
        pe = torch.from_numpy(vecs).float()
        if pe.shape[1] < k:
            pe = torch.cat([pe, torch.zeros(num_nodes, k - pe.shape[1])], dim=1)
        return pe
    except Exception:
        return torch.zeros((num_nodes, k), dtype=torch.float)


def pst_norm(pe: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Per-dimension z-normalization across nodes: stabilize variance across graph sizes
    if pe.numel() == 0:
        return pe
    mu = pe.mean(dim=0, keepdim=True)
    sd = pe.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
    return (pe - mu) / sd


def parse_cwe_pairs(pairs: List[str]) -> Dict[str, str]:
    out = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Bad --cwe-csv entry: {item} (expected CWE_NAME=path.csv)")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            raise ValueError(f"Empty CWE name in: {item}")
        out[k] = v
    return out


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cwe_csvs = parse_cwe_pairs(args.cwe_csv)
    parser = build_parser(args.ts_so, args.ts_lang)

    t0 = time.time()
    dfs = []
    for cwe, path in cwe_csvs.items():
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        dfi = pd.read_csv(path)
        dfi = clean_df(dfi, args.drop_cols)
        dfi["cwe"] = cwe
        dfs.append(dfi)

    df_all = pd.concat(dfs, ignore_index=True)

    # Global exact-match dedup with multi-label aggregation
    df_all = df_all.groupby("code").agg({
        "bug": "max",
        "cwe": lambda x: sorted(list(set(x))),
    }).reset_index()

    df_all = df_all.sample(frac=1, random_state=args.shuffle_seed).reset_index(drop=True)

    node_types_all: List[List[str]] = []
    edges_all: List[List[Tuple[int, int]]] = []
    spans_all: List[List[Tuple[int, int]]] = []

    node_type_vocab = set()

    codes = df_all["code"].tolist()
    for i, code in enumerate(codes, start=1):
        ntypes, edges, spans = parse_code_to_ast_and_spans(parser, code)
        node_types_all.append(ntypes)
        edges_all.append(edges)
        spans_all.append(spans)
        if ntypes is not None:
            for nt in ntypes:
                node_type_vocab.add(nt)
        if i % 10000 == 0:
            print(f"Parsed {i}/{len(codes)}")

    df_all["ast_nodes"] = node_types_all
    df_all["ast_edges"] = edges_all
    df_all["ast_spans"] = spans_all

    before = len(df_all)
    df_all = df_all.dropna(subset=["ast_nodes", "ast_edges", "ast_spans"]).reset_index(drop=True)
    print(f"AST parsed: {len(df_all)}/{before}")

    node_vocab = {t: i for i, t in enumerate(sorted(list(node_type_vocab)))}
    print(f"node_vocab size: {len(node_vocab)}")

    all_graphs: List[Data] = []
    graphs_by_cwe: Dict[str, List[Data]] = {k: [] for k in cwe_csvs.keys()}

    for i, row in df_all.iterrows():
        ntypes: List[str] = row["ast_nodes"]
        edges: List[Tuple[int, int]] = row["ast_edges"] or []
        spans: List[Tuple[int, int]] = row["ast_spans"] or []

        node_idx = [node_vocab[n] for n in ntypes]
        x = torch.tensor(node_idx, dtype=torch.long).unsqueeze(1)

        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        y = torch.tensor([int(row["bug"])], dtype=torch.float)

        pe = laplacian_pe(edge_index, num_nodes=len(node_idx), k=args.k_eig)
        if args.pst_norm:
            pe = pst_norm(pe)

        node_span = torch.tensor(spans, dtype=torch.long)  # [N,2] aligned with x
        if node_span.shape[0] != x.shape[0]:
            # Hard fail: alignment becomes unverifiable
            raise RuntimeError(f"node_span length mismatch at row {i}: spans={node_span.shape[0]} nodes={x.shape[0]}")

        d = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            pos_enc=pe,
            node_span=node_span,
        )

        d.cwes = list(row["cwe"])
        d.sample_id = sha256_text(row["code"])

        if args.store_code:
            d.code = row["code"]

        all_graphs.append(d)

        for tag in d.cwes:
            if tag in graphs_by_cwe:
                graphs_by_cwe[tag].append(d)

        if (i + 1) % 10000 == 0:
            print(f"Built {i + 1}/{len(df_all)} graphs")

    with open(os.path.join(args.out_dir, "graphs_with_pe_and_cwe.pkl"), "wb") as f:
        pickle.dump(all_graphs, f)

    with open(os.path.join(args.out_dir, "node_vocab.pkl"), "wb") as f:
        pickle.dump(node_vocab, f)

    # Optional per-CWE subsets (these contain shared Data objects)
    for cwe, lst in graphs_by_cwe.items():
        with open(os.path.join(args.out_dir, f"graphs_{cwe}.pkl"), "wb") as f:
            pickle.dump(lst, f)

    meta = {
        "out_dir": args.out_dir,
        "total_unique_graphs": len(all_graphs),
        "counts_per_cwe_subset": {k: len(v) for k, v in graphs_by_cwe.items()},
        "node_vocab_size": len(node_vocab),
        "k_eig": args.k_eig,
        "pst_norm": bool(args.pst_norm),
        "store_code": bool(args.store_code),
        "shuffle_seed": args.shuffle_seed,
    }

    if args.export_splits:
        labels = [int(d.y.item()) for d in all_graphs]
        train_val, test, _, _ = train_test_split(
            all_graphs, labels, test_size=0.20, random_state=args.split_seed, stratify=labels
        )
        train, val, _, _ = train_test_split(
            train_val,
            [int(d.y.item()) for d in train_val],
            test_size=0.125,
            random_state=args.split_seed,
            stratify=[int(d.y.item()) for d in train_val],
        )
        meta["split_seed"] = args.split_seed
        meta["splits"] = {
            "train": [d.sample_id for d in train],
            "val": [d.sample_id for d in val],
            "test": [d.sample_id for d in test],
        }
        with open(os.path.join(args.out_dir, "splits.json"), "w") as f:
            json.dump(meta["splits"], f)

    with open(os.path.join(args.out_dir, "preprocess_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved graphs: {len(all_graphs)}")
    print(f"Saved meta: {os.path.join(args.out_dir, 'preprocess_meta.json')}")
    print(f"Time: {(time.time() - t0)/60:.2f} min")


if __name__ == "__main__":
    main()