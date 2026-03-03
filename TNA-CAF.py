Python 3.9.9 (tags/v3.9.9:ccb0e6a, Nov 15 2021, 18:08:50) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import argparse
import json
import math
import os
import pickle
import random
import time
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from transformers import AutoTokenizer, AutoModel

try:
    from tree_sitter import Language, Parser
except Exception:
    Language = None
    Parser = None


def parse_args():
    p = argparse.ArgumentParser("Train/Eval TNA-CAF")
    p.add_argument("--graphs", required=True, help="graphs_with_pe_and_cwe.pkl")
    p.add_argument("--node-vocab", required=True, help="node_vocab.pkl")
    p.add_argument("--codebert", default="microsoft/codebert-base", help="HF name or local path")
    p.add_argument("--ts-so", default=None, help="tree-sitter .so (optional; required if graphs lack node_span)")
    p.add_argument("--ts-lang", default="c")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--batch-stage1", type=int, default=16)
    p.add_argument("--batch-stage2", type=int, default=8)
    p.add_argument("--lr-stage1", type=float, default=1e-4)
    p.add_argument("--lr-stage2", type=float, default=1e-5)
    p.add_argument("--epochs-stage1", type=int, default=3)
    p.add_argument("--epochs-stage2", type=int, default=6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--hidden-fusion", type=int, default=512)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--save-dir", default="./checkpoints")
    p.add_argument("--span-cache-cap", type=int, default=20000)
    p.add_argument("--allow-dense-fallback", action="store_true",
                   help="If node spans mismatch, fall back to dense attention instead of error.")
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int, base_seed: int):
    s = base_seed + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def stable_key(code: str) -> bytes:
    return hashlib.sha256(code.encode("utf-8", errors="ignore")).digest()


class GraphEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.node_embedding = nn.Embedding(vocab_size, emb_dim)
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(emb_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.act = nn.ReLU()
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.node_embedding(x.squeeze(1))
        for conv in self.convs:
            h = self.act(conv(h, edge_index))
        return h


class MaskedCrossAttention(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, d_model: int):
        super().__init__()
        self.Wq = nn.Linear(q_dim, d_model)
        self.Wk = nn.Linear(kv_dim, d_model)
        self.Wv = nn.Linear(kv_dim, d_model)
        self.out = nn.Linear(d_model, q_dim)
        self.scale = math.sqrt(d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        scores = (q @ k.T) / self.scale

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
            row_has_any = mask.any(dim=1)
            if (~row_has_any).any():
                scores[~row_has_any] = 0.0

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        ctx = attn @ v
        return self.out(ctx)


class TokenNodeAlignedFusion(nn.Module):
    def __init__(self, vocab_size: int, code_model: nn.Module, tokenizer: AutoTokenizer,
                 hidden_fusion: int = 512, node_emb_dim: int = 128, gnn_hidden: int = 256, d_model: int = 256):
        super().__init__()
        self.tokenizer = tokenizer
        self.code_model = code_model
        self.graph = GraphEncoder(vocab_size, emb_dim=node_emb_dim, hidden_dim=gnn_hidden)

        code_dim = self.code_model.config.hidden_size
        node_dim = gnn_hidden

        self.t2n = MaskedCrossAttention(q_dim=code_dim, kv_dim=node_dim, d_model=d_model)
        self.n2t = MaskedCrossAttention(q_dim=node_dim, kv_dim=code_dim, d_model=d_model)

        self.ln_tok = nn.LayerNorm(code_dim)
        self.ln_node = nn.LayerNorm(node_dim)

        self.cls_head = nn.Sequential(
            nn.Linear(code_dim + node_dim, hidden_fusion),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_fusion, 1),
        )

    def encode_tokens(self, codes: List[str], device: torch.device, max_len: int):
        enc = self.tokenizer(
            codes, padding=True, truncation=True, max_length=max_len,
            return_tensors="pt", return_offsets_mapping=True
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        offsets = enc["offset_mapping"]  # CPU tensor [B,T,2]
        out = self.code_model(input_ids=input_ids, attention_mask=attn_mask)
        tok = out.last_hidden_state  # [B,T,d]
        return tok, offsets, attn_mask

    def forward_one(self, tok_emb: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor, mask_tn: torch.Tensor):
        node_feat = self.graph(x, edge_index)
        tok_delta = self.t2n(tok_emb, node_feat, node_feat, mask_tn)
        tok_enriched = self.ln_tok(tok_emb + tok_delta)

        node_delta = self.n2t(node_feat, tok_emb, tok_emb, mask_tn.transpose(0, 1))
        node_enriched = self.ln_node(node_feat + node_delta)

        cls_vec = tok_enriched[0]
        node_pool = node_enriched.mean(dim=0)
        fused = torch.cat([cls_vec, node_pool], dim=-1)
        return self.cls_head(fused).squeeze(0)


def build_alignment_mask(token_offsets: torch.Tensor, node_spans: torch.Tensor, device: torch.device) -> torch.Tensor:
    token_offsets = token_offsets.to(device)
    node_spans = node_spans.to(device)

    tok_start = token_offsets[:, 0].unsqueeze(1)
    tok_end = token_offsets[:, 1].unsqueeze(1)
    node_start = node_spans[:, 0].unsqueeze(0)
    node_end = node_spans[:, 1].unsqueeze(0)

    no_overlap = (tok_end <= node_start) | (node_end <= tok_start)
    mask = ~no_overlap

    row_has_any = mask.any(dim=1)
    if (~row_has_any).any():
        mask[~row_has_any] = True
    return mask


class SpanProvider:
    def __init__(self, ts_so: Optional[str], ts_lang: str, cache_cap: int):
        self.cache_cap = cache_cap
        self.cache: "OrderedDict[bytes, torch.Tensor]" = OrderedDict()
        self.parser = None
        if ts_so is not None:
            if Language is None or Parser is None:
                raise RuntimeError("tree_sitter is not installed but --ts-so was provided.")
            lang = Language(ts_so, ts_lang)
            p = Parser()
            p.set_language(lang)
            self.parser = p

    def spans_for_code(self, code: str) -> torch.Tensor:
        if self.parser is None:
            return torch.empty((0, 2), dtype=torch.long)

        key = stable_key(code)
        if key in self.cache:
            v = self.cache.pop(key)
            self.cache[key] = v
            return v

        tree = self.parser.parse(code.encode("utf8", errors="ignore"))
        root = tree.root_node
        spans: List[Tuple[int, int]] = []
        q = [root]
        while q:
            n = q.pop(0)
            spans.append((n.start_byte, n.end_byte))
            q.extend(n.children)

        t = torch.tensor(spans, dtype=torch.long)
        self.cache[key] = t
        if len(self.cache) > self.cache_cap:
            self.cache.popitem(last=False)
        return t


def safe_auc_roc(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float("nan") if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, probs))


def safe_auc_pr(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float("nan") if len(np.unique(y_true)) < 2 else float(average_precision_score(y_true, probs))


def metrics_at(y_true: np.ndarray, probs: np.ndarray, t: float):
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    pred = (probs >= t).astype(int)
    acc = float(accuracy_score(y_true, pred))
    prec = float(precision_score(y_true, pred, zero_division=0))
    rec = float(recall_score(y_true, pred, zero_division=0))
    f1v = float(f1_score(y_true, pred, zero_division=0))
    roc = safe_auc_roc(y_true, probs)
    pr = safe_auc_pr(y_true, probs)
    return acc, prec, rec, f1v, roc, pr


def best_threshold_by_f1(y_true: np.ndarray, probs: np.ndarray, grid: np.ndarray):
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    best_t, best_f1v = 0.5, -1.0
    for t in grid:
        pred = (probs >= t).astype(int)
        f1v = f1_score(y_true, pred, zero_division=0)
        if f1v > best_f1v:
            best_f1v = float(f1v)
            best_t = float(t)
    return best_t, best_f1v


@torch.no_grad()
def evaluate_probs(model: TokenNodeAlignedFusion, loader: DataLoader, device: torch.device, max_len: int,
                   span_provider: SpanProvider, allow_dense_fallback: bool):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        data_list = batch.to_data_list()
        codes = [getattr(d, "code", None) for d in data_list]
        if any(c is None for c in codes):
            raise ValueError("Each Data must contain .code for tokenization.")

        tok_all, offsets_all, attn_mask_all = model.encode_tokens(codes, device, max_len)

        for i, d in enumerate(data_list):
            Ti = int(attn_mask_all[i].sum().item())
            Ti = max(Ti, 1)
            tok = tok_all[i, :Ti, :]
            offsets = offsets_all[i, :Ti, :]

            x = d.x.to(device)
            edge_index = d.edge_index.to(device)

            N = int(x.size(0))
            if N == 0:
                x = torch.zeros((1, 1), dtype=torch.long, device=device)
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                N = 1

            if hasattr(d, "node_span") and d.node_span is not None:
                node_spans = d.node_span
            else:
                node_spans = span_provider.spans_for_code(d.code)

            if node_spans.numel() == 0:
                if allow_dense_fallback:
                    mask = torch.ones((Ti, N), dtype=torch.bool, device=device)
                else:
                    raise RuntimeError("Missing node spans. Provide Data.node_span or pass --ts-so.")
            elif node_spans.shape[0] != N:
                if allow_dense_fallback:
                    mask = torch.ones((Ti, N), dtype=torch.bool, device=device)
                else:
                    raise RuntimeError(f"Span/node mismatch: spans={node_spans.shape[0]} vs N={N}.")
            else:
                mask = build_alignment_mask(offsets, node_spans, device)

            logit = model.forward_one(tok, x, edge_index, mask)
            ys.append(int(d.y.item()))
            ps.append(float(torch.sigmoid(logit).item()))

    return np.asarray(ys, dtype=int), np.asarray(ps, dtype=float)


def train_one_epoch(model: TokenNodeAlignedFusion, loader: DataLoader, device: torch.device, max_len: int,
                    optimizer: torch.optim.Optimizer, criterion: nn.Module,
                    span_provider: SpanProvider, allow_dense_fallback: bool) -> float:
    model.train()
    total, steps = 0.0, 0
    for batch in loader:
        data_list = batch.to_data_list()
        codes = [getattr(d, "code", None) for d in data_list]
        if any(c is None for c in codes):
            raise ValueError("Each Data must contain .code for tokenization.")

        tok_all, offsets_all, attn_mask_all = model.encode_tokens(codes, device, max_len)

        logits, targets = [], []
        for i, d in enumerate(data_list):
            Ti = int(attn_mask_all[i].sum().item())
            Ti = max(Ti, 1)
            tok = tok_all[i, :Ti, :]
            offsets = offsets_all[i, :Ti, :]

            x = d.x.to(device)
            edge_index = d.edge_index.to(device)

            N = int(x.size(0))
            if N == 0:
                x = torch.zeros((1, 1), dtype=torch.long, device=device)
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
                N = 1

            if hasattr(d, "node_span") and d.node_span is not None:
                node_spans = d.node_span
            else:
                node_spans = span_provider.spans_for_code(d.code)

            if node_spans.numel() == 0:
                if allow_dense_fallback:
                    mask = torch.ones((Ti, N), dtype=torch.bool, device=device)
                else:
                    raise RuntimeError("Missing node spans. Provide Data.node_span or pass --ts-so.")
            elif node_spans.shape[0] != N:
                if allow_dense_fallback:
                    mask = torch.ones((Ti, N), dtype=torch.bool, device=device)
                else:
                    raise RuntimeError(f"Span/node mismatch: spans={node_spans.shape[0]} vs N={N}.")
            else:
                mask = build_alignment_mask(offsets, node_spans, device)

            logits.append(model.forward_one(tok, x, edge_index, mask))
            targets.append(float(d.y.item()))

        logits_t = torch.stack(logits, dim=0).to(device)
        targets_t = torch.tensor(targets, dtype=torch.float, device=device)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(logits_t, targets_t)
        if torch.isnan(loss):
            continue
        loss.backward()
        optimizer.step()

        total += float(loss.item())
        steps += 1

    return total / max(1, steps)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything(args.train_seed)

    with open(args.graphs, "rb") as f:
        full_dataset = pickle.load(f)
    with open(args.node_vocab, "rb") as f:
        node_vocab = pickle.load(f)

    labels_all = [int(d.y.item()) for d in full_dataset]
    train_val, test_set, _, _ = train_test_split(
        full_dataset, labels_all, test_size=0.20, random_state=args.split_seed, stratify=labels_all
    )
    train_set, val_set, _, _ = train_test_split(
        train_val,
        [int(d.y.item()) for d in train_val],
        test_size=0.125,
        random_state=args.split_seed,
        stratify=[int(d.y.item()) for d in train_val],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.codebert, use_fast=True)
    codebert = AutoModel.from_pretrained(args.codebert).to(device)

    model = TokenNodeAlignedFusion(
        vocab_size=len(node_vocab),
        code_model=codebert,
        tokenizer=tokenizer,
        hidden_fusion=args.hidden_fusion,
    ).to(device)

    span_provider = SpanProvider(args.ts_so, args.ts_lang, args.span_cache_cap)

    g = torch.Generator()
    g.manual_seed(args.train_seed)

    def _seed_worker(wid: int):
        seed_worker(wid, args.train_seed)

    train_loader_s1 = DataLoader(
        train_set, batch_size=args.batch_stage1, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=_seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_stage1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=_seed_worker, generator=g
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_stage1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=_seed_worker, generator=g
    )

    criterion = nn.BCEWithLogitsLoss()
    grid = np.linspace(0.01, 0.99, 99)

    for p in model.code_model.parameters():
        p.requires_grad = False
    for p in model.graph.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_stage1, weight_decay=args.weight_decay
    )

    best_val_f1 = -1.0
    best_state = None
    best_t = 0.5

    for ep in range(1, args.epochs_stage1 + 1):
        tr_loss = train_one_epoch(
            model, train_loader_s1, device, args.max_len, optimizer, criterion,
            span_provider, args.allow_dense_fallback
        )
        yv, pv = evaluate_probs(model, val_loader, device, args.max_len, span_provider, args.allow_dense_fallback)
        t_star, _ = best_threshold_by_f1(yv, pv, grid)
        _, _, _, f1v, _, _ = metrics_at(yv, pv, t_star)
        print(f"S1 {ep:02d} loss={tr_loss:.4f} val_f1={f1v:.4f} t*={t_star:.2f}")
        if f1v > best_val_f1:
            best_val_f1 = f1v
            best_t = t_star
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    for p in model.code_model.parameters():
        p.requires_grad = True
    for p in model.graph.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_stage2, weight_decay=args.weight_decay)

    for ep in range(1, args.epochs_stage2 + 1):
        train_loader_s2 = DataLoader(
            train_set, batch_size=args.batch_stage2, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            worker_init_fn=_seed_worker, generator=g
        )
        tr_loss = train_one_epoch(
            model, train_loader_s2, device, args.max_len, optimizer, criterion,
            span_provider, args.allow_dense_fallback
        )
        yv, pv = evaluate_probs(model, val_loader, device, args.max_len, span_provider, args.allow_dense_fallback)
        t_star, _ = best_threshold_by_f1(yv, pv, grid)
        _, _, _, f1v, _, _ = metrics_at(yv, pv, t_star)
        print(f"S2 {ep:02d} loss={tr_loss:.4f} val_f1={f1v:.4f} t*={t_star:.2f}")
        if f1v > best_val_f1:
            best_val_f1 = f1v
            best_t = t_star
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    yt, pt = evaluate_probs(model, test_loader, device, args.max_len, span_provider, args.allow_dense_fallback)
    acc, prec, rec, f1v, roc, pr = metrics_at(yt, pt, best_t)
    print(f"TEST t*={best_t:.2f} Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1v:.4f} ROC={roc:.4f} PR={pr:.4f}")

    per_cwe = {}
    if len(test_set) > 0 and hasattr(test_set[0], "cwes"):
        for cwe in ["CWE_119", "CWE_120", "CWE_469", "CWE_476", "CWE_OTHERS"]:
            subset = [d for d in test_set if cwe in getattr(d, "cwes", [])]
            if not subset:
                per_cwe[cwe] = {"n": 0}
                continue
            loader = DataLoader(
                subset, batch_size=args.batch_stage1, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
                worker_init_fn=_seed_worker, generator=g
            )
            y, p = evaluate_probs(model, loader, device, args.max_len, span_provider, args.allow_dense_fallback)
            a, prc, rc, f, rauc, pauc = metrics_at(y, p, best_t)
            per_cwe[cwe] = {"n": int(len(subset)), "acc": a, "prec": prc, "rec": rc, "f1": f, "roc_auc": rauc, "pr_auc": pauc}

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ckpt_path = save_dir / f"tna_caf_split{args.split_seed}_seed{args.train_seed}_{stamp}.pth"
    meta_path = save_dir / f"tna_caf_split{args.split_seed}_seed{args.train_seed}_{stamp}.json"

    payload = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "meta": {
            "arch": "TokenNodeAlignedFusion",
            "codebert": args.codebert,
            "split_seed": args.split_seed,
            "train_seed": args.train_seed,
            "best_val_f1": float(best_val_f1),
            "t_star": float(best_t),
            "test": {"acc": acc, "prec": prec, "rec": rec, "f1": f1v, "roc_auc": roc, "pr_auc": pr},
            "per_cwe_test": per_cwe,
        },
    }

    torch.save(payload, ckpt_path)
    with open(meta_path, "w") as f:
        json.dump(payload["meta"], f, indent=2)

    print(f"Saved {ckpt_path}")
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()