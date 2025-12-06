# src/eval_router_checkpoint.py

import argparse
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader

from src.sae_model import SparseAutoencoder
from src.sbert_encoder import SBERTEncoder
from src.router_model import RouterQNetwork
from src.train_router_topk import topk_mask_state
from src.dataset import RouterBenchSAEDataset


# ---------------------------------------------------------
# 0. meta → scores 텐서 변환 (correctness)
# ---------------------------------------------------------
def build_scores_tensor_from_metas(
    metas: List[Dict[str, Any]],
    models: List[str],
    device: str = "cpu",
) -> torch.Tensor:
    """
    meta["scores_by_model"][model_name]에서 (B, M) tensor 생성.

    - scores_by_model: 각 모델의 correctness/score (0.0 ~ 1.0)
      예: 1.0 = 정답, 0.0 = 오답, 중간 값도 허용
    """
    batch_scores: List[List[float]] = []
    for meta in metas:
        score_dict = meta.get("scores_by_model", {})
        row = []
        for m in models:
            s = score_dict.get(m, 0.0)
            try:
                s = float(s)
            except Exception:
                s = 0.0
            row.append(s)
        batch_scores.append(row)

    scores = torch.tensor(batch_scores, dtype=torch.float32, device=device)  # (B, M)
    return scores


# ---------------------------------------------------------
# 1. SAE / SBERT 로딩
# ---------------------------------------------------------
def load_sae_sbert(device: str):
    sae = SparseAutoencoder().to(device)
    sae.load_state_dict(torch.load("sae_model.pt", map_location=device))
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    sbert = SBERTEncoder(device=device)
    return sae, sbert


# ---------------------------------------------------------
# 2. 평가용 DataLoader (scores 포함)
# ---------------------------------------------------------
def make_eval_loader(
    models,
    sae,
    sbert,
    device: str,
    batch_size: int,
    split: str = "val",
):
    """
    RouterBenchSAEDataset에서 지정한 split(val/test 등)을 로딩.
    여기서는 (h, label, costs, scores, metas)를 받는다.
    - label (oracle index)는 이번 평가지표에서는 사용하지 않고,
      scores_by_model 기반 correctness만 사용.
    """

    def collate_fn(batch):
        # batch: list of (h, label, costs, meta)
        hs, labels, costs, metas = zip(*batch)
        hs = torch.stack(hs, dim=0)          # (B, H)
        labels = torch.stack(labels, dim=0)  # (B,)
        costs = torch.stack(costs, dim=0)    # (B, M)

        scores = build_scores_tensor_from_metas(
            list(metas), models, device="cpu"
        )  # (B, M)

        return hs, labels, costs, scores, list(metas)

    ds = RouterBenchSAEDataset(
        "data/routerbench_0shot.pkl",
        models,
        sae=sae,
        sbert=sbert,
        split=split,
        device=device,
        return_meta=True,   # 🔥 meta 필요
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return loader


# ---------------------------------------------------------
# 3. 체크포인트 로딩
# ---------------------------------------------------------
def load_router_from_ckpt(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    latent_dim = ckpt["latent_dim"]
    n_models = ckpt["n_models"]
    models = ckpt["models"]
    k_top = ckpt.get("k_top", None)
    lambda_cost = ckpt.get("lambda_cost", 0.0)
    exp_name = ckpt.get("exp_name", "unknown_exp")
    best_epoch = ckpt.get("best_epoch", None)
    best_val_reward = ckpt.get("best_val_reward_router", None)

    router = RouterQNetwork(
        latent_dim=latent_dim,
        n_models=n_models,
    ).to(device)
    router.load_state_dict(ckpt["state_dict"])
    router.eval()

    print("[load_router_from_ckpt]")
    print(f"  ckpt_path       : {ckpt_path}")
    print(f"  exp_name        : {exp_name}")
    print(f"  latent_dim      : {latent_dim}")
    print(f"  n_models        : {n_models}")
    print(f"  k_top           : {k_top}")
    print(f"  lambda_cost     : {lambda_cost}")
    print(f"  best_epoch      : {best_epoch}")
    print(f"  best_val_reward : {best_val_reward}")
    print()

    return router, models, k_top


# ---------------------------------------------------------
# 4. 평가 루프 (정답 correctness 기반)
# ---------------------------------------------------------
def evaluate_checkpoint(
    router: RouterQNetwork,
    eval_loader: DataLoader,
    models,
    k_top,
    device: str,
):
    """
    oracle label이 아니라 scores_by_model(correctness)로 평가.

    - router_top1_acc: 라우터가 고른 모델의 correctness 평균
    - router_top3_acc: 라우터 top-3 중 하나라도 맞춘 비율 (scores 기준)
    - router_top5_acc: top-5 버전
    - best_upper_acc: 각 샘플에서 correctness가 가장 높은 모델의 평균 (upper bound)

    비용/ Q값도 "best_correct" vs "router top1" 비교로 변경.
    """
    router.eval()

    total_samples = 0

    # correctness 기반 통계
    sum_router_top1_correct = 0.0
    sum_router_top3_correct = 0.0
    sum_router_top5_correct = 0.0
    sum_best_correct = 0.0

    # 비용 / Q값 통계 (best vs router)
    sum_cost_best = 0.0
    sum_cost_router = 0.0
    sum_q_best = 0.0
    sum_q_router = 0.0

    with torch.no_grad():
        for h, labels, costs, scores_cpu, metas in eval_loader:
            # labels는 더 이상 사용하지 않음 (oracle index)
            h = h.to(device)                  # (B, H)
            costs = costs.to(device)          # (B, M)
            scores = scores_cpu.to(device)    # (B, M)

            B, M = costs.shape
            batch_idx = torch.arange(B, device=device)

            # state: SAE latent에 top-k 마스킹 적용
            if k_top is not None and k_top > 0:
                h_sparse = topk_mask_state(h, k=k_top)  # (B, H)
            else:
                h_sparse = h

            # Q 예측
            q_pred = router(h_sparse)  # (B, M)

            # Q값 내림차순 정렬 → 라우터가 생각하는 모델 순위
            router_rank = torch.argsort(q_pred, dim=1, descending=True)  # (B, M)

            # Top-1 / Top-3 / Top-5 인덱스
            top1 = router_rank[:, 0]           # (B,)
            top3 = router_rank[:, :3]          # (B, 3)
            top5 = router_rank[:, :5]          # (B, 5)

            # ----- correctness 기반 지표 -----

            # 1) Router top-1 correctness
            router_top1_correct = scores[batch_idx, top1]  # (B,)
            sum_router_top1_correct += router_top1_correct.sum().item()

            # 2) Router top-3 / top-5: top-k 안에 하나라도 맞추면 1
            scores_top3 = torch.gather(scores, 1, top3)  # (B, 3)
            scores_top5 = torch.gather(scores, 1, top5)  # (B, 5)

            # binary correctness라면 max가 1이면 "하나 이상 맞춤"
            router_top3_any = scores_top3.max(dim=1).values  # (B,)
            router_top5_any = scores_top5.max(dim=1).values  # (B,)

            sum_router_top3_correct += router_top3_any.sum().item()
            sum_router_top5_correct += router_top5_any.sum().item()

            # 3) Upper bound: 각 샘플에서 가장 correctness 높은 모델
            best_correct_vals, best_idx = scores.max(dim=1)  # (B,), (B,)
            sum_best_correct += best_correct_vals.sum().item()

            # ----- 비용 / Q값: best_correct vs router top1 -----

            cost_best = costs[batch_idx, best_idx]
            cost_router = costs[batch_idx, top1]

            sum_cost_best += cost_best.sum().item()
            sum_cost_router += cost_router.sum().item()

            q_best = q_pred[batch_idx, best_idx]
            q_router = q_pred[batch_idx, top1]

            sum_q_best += q_best.sum().item()
            sum_q_router += q_router.sum().item()

            total_samples += B

    # 최종 통계
    router_top1_acc = sum_router_top1_correct / total_samples
    router_top3_acc = sum_router_top3_correct / total_samples
    router_top5_acc = sum_router_top5_correct / total_samples
    best_upper_acc = sum_best_correct / total_samples

    avg_cost_best = sum_cost_best / total_samples
    avg_cost_router = sum_cost_router / total_samples

    avg_q_best = sum_q_best / total_samples
    avg_q_router = sum_q_router / total_samples

    stats = {
        "total_samples": total_samples,
        "router_top1_acc": router_top1_acc,
        "router_top3_acc": router_top3_acc,
        "router_top5_acc": router_top5_acc,
        "best_upper_acc": best_upper_acc,
        "avg_cost_best": avg_cost_best,
        "avg_cost_router": avg_cost_router,
        "avg_q_best": avg_q_best,
        "avg_q_router": avg_q_router,
    }
    return stats


# ---------------------------------------------------------
# 5. 여러 config(0~3) 한 번에 평가
# ---------------------------------------------------------
def eval_multiple_configs(
    ckpt_pattern: str,
    cfg_indices: List[int],
    split: str,
    batch_size: int,
    device: str,
):
    print(f"[eval_multiple_configs] device = {device}")
    print(f"[eval_multiple_configs] split  = {split}")
    print()

    results = []

    for cfg_idx in cfg_indices:
        ckpt_path = ckpt_pattern.format(cfg_idx)

        print("========================================")
        print(f"[Config {cfg_idx}] ckpt = {ckpt_path}")
        print("========================================")

        # 1) 체크포인트 로딩
        router, models, k_top = load_router_from_ckpt(ckpt_path, device)

        # 2) SAE / SBERT + DataLoader
        sae, sbert = load_sae_sbert(device)
        eval_loader = make_eval_loader(
            models=models,
            sae=sae,
            sbert=sbert,
            device=device,
            batch_size=batch_size,
            split=split,
        )

        # 3) 평가
        stats = evaluate_checkpoint(
            router=router,
            eval_loader=eval_loader,
            models=models,
            k_top=k_top,
            device=device,
        )

        print("===== Evaluation Result (Config {}) =====".format(cfg_idx))
        print(f"Total samples                 : {stats['total_samples']}")
        print()
        print(f"Router Top-1 correctness      : {stats['router_top1_acc']:.4f}")
        print(f"Router Top-3 any-correct      : {stats['router_top3_acc']:.4f}")
        print(f"Router Top-5 any-correct      : {stats['router_top5_acc']:.4f}")
        print(f"Upper bound (best per sample) : {stats['best_upper_acc']:.4f}")
        print()
        print(f"Avg cost (best correct)       : {stats['avg_cost_best']:.6f}")
        print(f"Avg cost (router top1)        : {stats['avg_cost_router']:.6f}")
        print()
        print(f"Avg Q (best correct)          : {stats['avg_q_best']:.4f}")
        print(f"Avg Q (router top1)           : {stats['avg_q_router']:.4f}")
        print("========================================\n")

        results.append((cfg_idx, stats))

    # 한 줄 요약 표 형태로 다시 한 번 출력
    print("\n========== Summary over configs ==========")
    print("cfg | router_top1 | router_top3 | router_top5 | upper_bound | avg_cost_router")
    for cfg_idx, s in results:
        print(
            f"{cfg_idx:3d} | "
            f"{s['router_top1_acc']:.4f} | "
            f"{s['router_top3_acc']:.4f} | "
            f"{s['router_top5_acc']:.4f} | "
            f"{s['best_upper_acc']:.4f} | "
            f"{s['avg_cost_router']:.6f}"
        )
    print("==========================================\n")


# ---------------------------------------------------------
# 6. main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    # 여러 config 평가용 패턴 (기본: semantic_search에서 cfg{0..3})
    parser.add_argument(
        "--ckpt_pattern",
        type=str,
        default="ckpts_semantic_search/router_semantic_cfg{}_best.pt",
        help="pattern for checkpoint paths, use '{}' as cfg index placeholder",
    )
    parser.add_argument(
        "--cfg_start",
        type=int,
        default=0,
        help="start config index (inclusive)",
    )
    parser.add_argument(
        "--cfg_end",
        type=int,
        default=3,
        help="end config index (inclusive)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="which split to evaluate: train / val / test",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="evaluation batch size",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # cfg_start ~ cfg_end 까지 평가 (기본: 0~3)
    cfg_indices = list(range(args.cfg_start, args.cfg_end + 1))
    eval_multiple_configs(
        ckpt_pattern=args.ckpt_pattern,
        cfg_indices=cfg_indices,
        split=args.split,
        batch_size=args.batch_size,
        device=device,
    )


if __name__ == "__main__":
    main()
