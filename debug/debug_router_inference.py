# debug/debug_router_inference.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Union

import pandas as pd
import torch

from src.dataset import RouterBenchSAEDataset
from src.router_model import RouterQNetwork
from src.sae_model import SparseAutoencoder
from src.sbert_encoder import SBERTEncoder
from src.train_router_topk import topk_mask_state

DEFAULT_BEST_CKPT = "ckpts_semantic_search/router_qnet_semantic_best_overall.pt"
LEGACY_CKPT = "ckpts/router_qnet_checkpoint.pt"
FEATURE_CSV = "sae_feature_task_stats.csv"

PromptType = Union[str, List[str]]


@dataclass(frozen=True)
class FeatureSemantic:
    task: str
    ratio: float
    total: int


@dataclass(frozen=True)
class LoadedModels:
    sae: SparseAutoencoder
    sbert: SBERTEncoder
    router: RouterQNetwork
    models: Sequence[str]
    k_top: int | None


def load_feature_semantics(csv_path: str = FEATURE_CSV) -> Dict[int, FeatureSemantic]:
    """Map feature -> (top_task, top_ratio, total) from sae_feature_task_stats.csv."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[load_feature_semantics] {csv_path} not found. Semantics disabled.")
        return {}

    feat_info: Dict[int, FeatureSemantic] = {}
    for _, row in df.iterrows():
        feat_idx = int(row["feature"])
        feat_info[feat_idx] = FeatureSemantic(
            task=str(row["top_task"]),
            ratio=float(row["top_ratio"]),
            total=int(row["total"]),
        )
    print(f"[load_feature_semantics] Loaded semantics for {len(feat_info)} features.")
    return feat_info


def load_models(
    device: str = "cpu",
    ckpt_path: str = DEFAULT_BEST_CKPT,
) -> LoadedModels:
    """Load SAE, SBERT, router Q-net, model list, and k_top."""
    sae = SparseAutoencoder().to(device)
    sae.load_state_dict(torch.load("sae_model.pt", map_location=device))
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    sbert = SBERTEncoder(device=device)

    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except FileNotFoundError:
        print(f"[load_models] {ckpt_path} missing; trying legacy {LEGACY_CKPT}")
        ckpt = torch.load(LEGACY_CKPT, map_location=device)

    latent_dim = ckpt.get("latent_dim", 4096)
    models = ckpt["models"]
    n_models = ckpt.get("n_models", len(models))

    router = RouterQNetwork(latent_dim=latent_dim, n_models=n_models).to(device)
    router.load_state_dict(ckpt["state_dict"])
    router.eval()

    print("[load_models] ==============================")
    print(f"  exp_name          : {ckpt.get('exp_name', 'unknown_exp')}")
    print(f"  latent_dim        : {latent_dim}")
    print(f"  n_models          : {n_models}")
    print(f"  k_top (semantic)  : {ckpt.get('k_top')}")
    print(f"  lambda_cost       : {ckpt.get('lambda_cost')}")
    print(f"  best_epoch        : {ckpt.get('best_epoch')}")
    print(f"  best_val_reward   : {ckpt.get('best_val_reward_router')}")
    print("===========================================")

    return LoadedModels(
        sae=sae,
        sbert=sbert,
        router=router,
        models=models,
        k_top=ckpt.get("k_top"),
    )


def _mask_topk(h: torch.Tensor, k_top: int | None) -> torch.Tensor:
    return topk_mask_state(h, k=k_top) if k_top else h


def _pretty_prompt(p: PromptType) -> str:
    return " / ".join(str(x) for x in p) if isinstance(p, list) else str(p)


def _print_section(title: str, lines: Iterable[str]) -> None:
    print(title)
    for line in lines:
        print(line)
    print()


def _top_q_view(
    q: torch.Tensor,
    models: Sequence[str],
    costs: torch.Tensor | None = None,
    top_k: int = 5,
    router_idx: int | None = None,
    oracle_idx: int | None = None,
) -> List[str]:
    vals, idxs = torch.topk(q, k=min(len(models), top_k))
    vals = vals.cpu().numpy()
    idxs = idxs.cpu().numpy()

    lines: List[str] = []
    for rank, (mid, val) in enumerate(zip(idxs, vals), start=1):
        mark = []
        if router_idx is not None and mid == router_idx:
            mark.append("router")
        if oracle_idx is not None and mid == oracle_idx:
            mark.append("oracle")
        suffix = f"  <-- {', '.join(mark)}" if mark else ""
        cost_str = f", cost={costs[mid].item():.7f}" if costs is not None else ""
        lines.append(
            f"  [{rank}] [{mid}] {models[mid]} : Q={val:.4f}{cost_str}{suffix}"
        )
    return lines


def _top_features_view(
    h: torch.Tensor,
    feat_sem: Dict[int, FeatureSemantic],
    k_show: int,
) -> List[str]:
    vals, idxs = torch.topk(h, k=min(k_show, h.size(0)))
    vals = vals.cpu().numpy()
    idxs = idxs.cpu().numpy()

    lines: List[str] = []
    for i, v in zip(idxs, vals):
        if v <= 0:
            continue
        if i in feat_sem:
            sem = feat_sem[i]
            lines.append(
                f"  feature {i:<4d}: {v:.4f}  | {sem.task} ({sem.ratio:.2f}, n={sem.total})"
            )
        else:
            lines.append(f"  feature {i:<4d}: {v:.4f}")
    return lines


def run_example_prompt(
    prompt: str,
    device: str = "cpu",
    k_show: int = 10,
    ckpt_path: str = DEFAULT_BEST_CKPT,
) -> None:
    artifacts = load_models(device=device, ckpt_path=ckpt_path)
    feat_sem = load_feature_semantics()

    emb = artifacts.sbert.encode([prompt]).to(device)

    with torch.no_grad():
        h = artifacts.sae.encode(emb)  # (1, D)
        h_sparse = _mask_topk(h, artifacts.k_top)
        q = artifacts.router(h_sparse).squeeze(0)

    router_idx = int(torch.argmax(q).item())

    print("\n======================================")
    print("        DEBUG: FREE PROMPT MODE")
    print("======================================\n")

    _print_section("Prompt:", [_pretty_prompt(prompt)])
    _print_section(
        "Router Selected:",
        [f"  [{router_idx}] {artifacts.models[router_idx]}"],
    )
    _print_section(
        "Top-5 Q-value Models:",
        _top_q_view(q, artifacts.models, router_idx=router_idx, top_k=5),
    )
    _print_section(
        f"Top SAE Features (k_show={k_show}):",
        _top_features_view(h.squeeze(0), feat_sem, k_show),
    )
    print("======================================\n")


def run_example_from_dataset(
    idx: int,
    split: str = "val",
    device: str = "cpu",
    k_show: int = 10,
    ckpt_path: str = DEFAULT_BEST_CKPT,
) -> None:
    artifacts = load_models(device=device, ckpt_path=ckpt_path)
    feat_sem = load_feature_semantics()

    ds = RouterBenchSAEDataset(
        "data/routerbench_0shot.pkl",
        artifacts.models,
        sae=artifacts.sae,
        sbert=artifacts.sbert,
        split=split,
        device=device,
        return_meta=True,
    )

    if idx < 0 or idx >= len(ds):
        raise IndexError(f"idx={idx} is out of range (0 ~ {len(ds)-1})")

    h, label, costs, meta = ds[idx]
    prompt = meta.get("prompt", "<prompt not in meta>")
    model_outputs = meta.get("model_outputs", None)

    oracle_idx = int(label.item())
    router_input = h.unsqueeze(0).to(device)
    with torch.no_grad():
        h_sparse = _mask_topk(router_input, artifacts.k_top)
        q = artifacts.router(h_sparse).squeeze(0)

    router_idx = int(torch.argmax(q).item())

    print("\n======================================")
    print(f"  DEBUG: DATASET MODE (split={split}, idx={idx})")
    print("======================================\n")

    _print_section("Prompt:", [_pretty_prompt(prompt)])

    meta_lines = []
    for k, v in meta.items():
        if k not in {"prompt", "model_outputs"}:
            meta_lines.append(f"  {k}: {v}")
    _print_section("Meta info (except long texts):", meta_lines or ["  <empty>"])

    _print_section(
        "Router Selected:",
        [
            f"  [{router_idx}] {artifacts.models[router_idx]}",
            f"  Q(router)  = {q[router_idx].item():.4f}",
            f"  cost(router) = {costs[router_idx].item():.7f}",
        ],
    )

    top_q_lines: List[str] = []
    if isinstance(model_outputs, dict):
        for line in _top_q_view(
            q,
            artifacts.models,
            costs=costs,
            top_k=3,
            router_idx=router_idx,
            oracle_idx=oracle_idx,
        ):
            mid = int(line.split("[")[2].split("]")[0])
            model_name = artifacts.models[mid]
            top_q_lines.append(line)
            if model_name in model_outputs:
                snippet = str(model_outputs[model_name]).replace("\n", " ")
                top_q_lines.append(f"     answer snippet: {snippet}")
            else:
                top_q_lines.append("     (answer text not available)")
    else:
        top_q_lines.append("  (model_outputs is not a dict; cannot show snippets)")
    _print_section("Top-3 models by Q-value (with answer snippets):", top_q_lines)

    _print_section(
        "Top-5 models by Q-value (index view):",
        _top_q_view(q, artifacts.models, router_idx=router_idx, top_k=5),
    )
    _print_section(
        f"Top SAE Features (k_show={k_show}):",
        _top_features_view(h, feat_sem, k_show),
    )
    print("======================================\n")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prompt", "dataset"], default="prompt")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Solve this geometry problem about triangles and angles.",
    )
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--k_show", type=int, default=20)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_BEST_CKPT)
    args = parser.parse_args()

    if args.mode == "prompt":
        run_example_prompt(
            prompt=args.prompt,
            device=device,
            k_show=args.k_show,
            ckpt_path=args.ckpt,
        )
    else:
        run_example_from_dataset(
            idx=args.idx,
            split=args.split,
            device=device,
            k_show=args.k_show,
            ckpt_path=args.ckpt,
        )


if __name__ == "__main__":
    main()
