# src/train_router_semantic_reward.py

import argparse
import os
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.sae_model import SparseAutoencoder
from src.sbert_encoder import SBERTEncoder
from src.router_model import RouterQNetwork
from src.train_router_topk import topk_mask_state
from src.dataset import RouterBenchSAEDataset


# ============================================================
# 1. meta → scores 텐서로 변환 (정답 여부 1.0/0.0 등)
# ============================================================

def build_scores_tensor_from_metas(
    metas: List[Dict[str, Any]],
    models: List[str],
    device: str = "cpu",
) -> torch.Tensor:
    """
    meta["scores_by_model"][model_name] 에서 (B, M) tensor 생성.

    - scores_by_model: 각 모델의 score/correctness (0.0 ~ 1.0)
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


# ============================================================
# 2. 배치 단위 리워드 행렬 계산
#    👉 correctness만 사용 (cost는 학습에 쓰지 않음)
# ============================================================

def compute_reward_matrix_for_batch(
    scores: torch.Tensor,   # (B, M) correctness/score (0~1) on device
    w_corr: float,
) -> torch.Tensor:
    """
    각 배치에서 (B, M) 리워드 행렬:

        reward[b, m] = w_corr * correctness[b, m]

    - correctness[b, m]: scores[b, m] (0~1)
    - cost는 reward에 전혀 사용하지 않고, evaluation 로그에만 사용.
    """
    rewards = w_corr * scores
    return rewards


# ============================================================
# 3. SAE + SBERT 로딩 & DataLoader 생성
# ============================================================

def load_sae_sbert(device: str):
    sae = SparseAutoencoder().to(device)
    sae.load_state_dict(torch.load("sae_model.pt", map_location=device))
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    sbert = SBERTEncoder(device=device)
    return sae, sbert


def make_dataloaders(
    models: List[str],
    sae,
    sbert,
    device: str,
    batch_size: int,
):
    """
    RouterBenchSAEDataset을 train/val 두 개로 로더 생성.
    여기서는 return_meta=True 로 해서 model_outputs + scores_by_model를 같이 가져온다.

    + oracle 없는 샘플 필터링:
      - meta["scores_by_model"]가 없거나
      - oracle 모델의 score <= 0.0 이면 해당 샘플은 버림
    """

    def collate_fn(batch):
        # batch: list of (h, label, costs, meta)
        filtered = []
        for (h, label, costs, meta) in batch:
            scores_dict = meta.get("scores_by_model", None)
            if not isinstance(scores_dict, dict) or len(scores_dict) == 0:
                # scores 정보가 없으면 버림
                continue

            # oracle 모델 이름
            oracle_idx = int(label.item())
            if oracle_idx < 0 or oracle_idx >= len(models):
                continue
            oracle_model_name = models[oracle_idx]
            oracle_score = scores_dict.get(oracle_model_name, 0.0)
            try:
                oracle_score = float(oracle_score)
            except Exception:
                oracle_score = 0.0

            # oracle 모델이 "정답"을 전혀 못 맞춘 샘플이면 (0.0 이하) 버림
            if oracle_score <= 0.0:
                continue

            filtered.append((h, label, costs, meta))

        # 이 배치에서 oracle 있는 샘플이 하나도 없으면, 빈 배치 리턴
        if len(filtered) == 0:
            # shape 정보를 위해 원래 배치에서 하나만 참고
            h0, label0, costs0, meta0 = batch[0]
            H = h0.shape[0]
            M = costs0.shape[0]

            hs = torch.empty((0, H), dtype=h0.dtype)
            labels = torch.empty((0,), dtype=label0.dtype)
            costs = torch.empty((0, M), dtype=costs0.dtype)
            scores = torch.empty((0, M), dtype=torch.float32)
            metas_out: List[Dict[str, Any]] = []
            return hs, labels, costs, scores, metas_out

        hs, labels, costs, metas_out = zip(*filtered)
        hs = torch.stack(hs, dim=0)          # (B, H)
        labels = torch.stack(labels, dim=0)  # (B,)
        costs = torch.stack(costs, dim=0)    # (B, M)

        # 정답 여부 / score 행렬 (B, M) - CPU에서 만들고 나중에 device로 옮김
        scores = build_scores_tensor_from_metas(list(metas_out), models, device="cpu")

        return hs, labels, costs, scores, list(metas_out)

    train_ds = RouterBenchSAEDataset(
        "data/routerbench_0shot.pkl",
        models,
        sae=sae,
        sbert=sbert,
        split="train",
        device=device,
        return_meta=True,
    )

    val_ds = RouterBenchSAEDataset(
        "data/routerbench_0shot.pkl",
        models,
        sae=sae,
        sbert=sbert,
        split="val",
        device=device,
        return_meta=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


# ============================================================
# 4. 평가 루프
#    - reward (정답 기반)
#    - cost (로그용)
#    - oracle alignment (top1/top3/top5)
#    - 정답 기준 accuracy (acc_oracle / acc_router)
# ============================================================

def evaluate_router(
    model: RouterQNetwork,
    data_loader: DataLoader,
    models: List[str],   # 현재는 사용하지 않지만, 인터페이스 유지용
    w_corr: float,
    k_top: int,
    device: str,
):
    model.eval()

    total_samples = 0
    sum_reward_oracle = 0.0
    sum_reward_router = 0.0
    sum_cost_oracle = 0.0
    sum_cost_router = 0.0

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0

    sum_corr_oracle = 0.0
    sum_corr_router = 0.0

    with torch.no_grad():
        for h, labels, costs, scores_cpu, metas in data_loader:
            # collate_fn이 빈 배치를 줄 수 있으므로 방어
            if h.size(0) == 0:
                continue

            h = h.to(device)
            labels = labels.to(device)
            costs = costs.to(device)
            scores = scores_cpu.to(device)   # (B, M)

            B, M = costs.shape

            # correctness 기반 reward
            rewards = compute_reward_matrix_for_batch(
                scores=scores,
                w_corr=w_corr,
            )  # (B, M) on device

            # state top-k 마스킹
            h_sparse = topk_mask_state(h, k=k_top)   # (B, H)
            q_pred = model(h_sparse)                 # (B, M)

            batch_idx = torch.arange(B, device=device)
            oracle_idx = labels                      # (B,)
            router_idx = torch.argmax(q_pred, dim=1) # (B,)

            # reward (oracle vs router)
            reward_oracle = rewards[batch_idx, oracle_idx]
            reward_router = rewards[batch_idx, router_idx]
            cost_oracle = costs[batch_idx, oracle_idx]
            cost_router = costs[batch_idx, router_idx]

            sum_reward_oracle += reward_oracle.sum().item()
            sum_reward_router += reward_router.sum().item()
            sum_cost_oracle += cost_oracle.sum().item()
            sum_cost_router += cost_router.sum().item()

            # oracle alignment top-k (oracle 모델과의 일치)
            top1 = router_idx
            top3 = torch.topk(q_pred, k=min(3, M), dim=1).indices
            top5 = torch.topk(q_pred, k=min(5, M), dim=1).indices

            top1_correct += (top1 == oracle_idx).sum().item()
            top3_correct += sum(
                oracle_idx[i].item() in top3[i].tolist()
                for i in range(B)
            )
            top5_correct += sum(
                oracle_idx[i].item() in top5[i].tolist()
                for i in range(B)
            )

            # 정답 기준 correctness
            corr_oracle = scores[batch_idx, oracle_idx]   # (B,)
            corr_router = scores[batch_idx, router_idx]   # (B,)

            sum_corr_oracle += corr_oracle.sum().item()
            sum_corr_router += corr_router.sum().item()

            total_samples += B

    if total_samples == 0:
        # 필터링이 너무 세서 샘플이 하나도 없을 경우 대비
        return {
            "avg_reward_oracle": 0.0,
            "avg_reward_router": 0.0,
            "avg_cost_oracle": 0.0,
            "avg_cost_router": 0.0,
            "top1_acc_oracle_match": 0.0,
            "top3_acc_oracle_match": 0.0,
            "top5_acc_oracle_match": 0.0,
            "acc_oracle": 0.0,
            "acc_router": 0.0,
            "total_samples": 0,
        }

    return {
        "avg_reward_oracle": sum_reward_oracle / total_samples,
        "avg_reward_router": sum_reward_router / total_samples,
        "avg_cost_oracle": sum_cost_oracle / total_samples,
        "avg_cost_router": sum_cost_router / total_samples,
        "top1_acc_oracle_match": top1_correct / total_samples,
        "top3_acc_oracle_match": top3_correct / total_samples,
        "top5_acc_oracle_match": top5_correct / total_samples,
        "acc_oracle": sum_corr_oracle / total_samples,
        "acc_router": sum_corr_router / total_samples,
        "total_samples": total_samples,
    }


# ============================================================
# 5. 한 개 config 훈련 루프
# ============================================================

def train_one_config(
    cfg_idx: int,
    config: Dict[str, Any],
    models: List[str],
    latent_dim: int,
    n_models: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    ckpt_dir: str,
):
    w_corr = config["w_corr"]
    k_top = config["k_top"]
    lr = config["lr"]
    epochs = config["epochs"]
    l1_lambda = config.get("l1_lambda", 0.0)  # SAE feature sparsity용 L1(optional)

    print("\n====================================")
    print(
        f"[Config {cfg_idx}] "
        f"w_corr={w_corr}, k_top={k_top}, "
        f"lr={lr}, epochs={epochs}, l1_lambda={l1_lambda}"
    )
    print("====================================")

    router = RouterQNetwork(
        latent_dim=latent_dim,
        n_models=n_models,
        use_mlp_head=False,   # 해석력 위해 기본은 linear-only
    ).to(device)

    optimizer = torch.optim.Adam(router.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_reward = -1e9
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        router.train()
        epoch_loss = 0.0
        num_batches = 0

        for h, labels, costs, scores_cpu, metas in train_loader:
            if h.size(0) == 0:
                continue

            h = h.to(device)
            labels = labels.to(device)
            costs = costs.to(device)
            scores = scores_cpu.to(device)

            # correctness 기반 reward
            rewards = compute_reward_matrix_for_batch(
                scores=scores,
                w_corr=w_corr,
            )  # (B, M) on device

            h_sparse = topk_mask_state(h, k=k_top)
            q_pred = router(h_sparse)

            loss = criterion(q_pred, rewards)

            # L1 regularization on W (optional)
            if l1_lambda > 0.0:
                l1_reg = l1_lambda * router.W.abs().mean()
                loss = loss + l1_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(1, num_batches)

        # ---------- validation ----------
        val_stats = evaluate_router(
            router,
            val_loader,
            models,
            w_corr=w_corr,
            k_top=k_top,
            device=device,
        )

        print(f"[Config {cfg_idx} | Epoch {epoch}] train MSE={avg_train_loss:.4f}")
        print(
            f"  [Val] acc_oracle={val_stats['acc_oracle']:.3f}, "
            f"acc_router={val_stats['acc_router']:.3f}"
        )
        print(
            f"  [Val] avg_reward_oracle={val_stats['avg_reward_oracle']:.4f}, "
            f"avg_reward_router={val_stats['avg_reward_router']:.4f}"
        )
        print(
            f"  [Val] top1_oracle_match={val_stats['top1_acc_oracle_match']:.3f}, "
            f"top3_oracle_match={val_stats['top3_acc_oracle_match']:.3f}, "
            f"top5_oracle_match={val_stats['top5_acc_oracle_match']:.3f}"
        )
        print(
            f"  [Val] avg_cost_oracle={val_stats['avg_cost_oracle']:.6f}, "
            f"avg_cost_router={val_stats['avg_cost_router']:.6f}, "
            f"total_samples={val_stats['total_samples']}"
        )

        # best 기준: avg_reward_router
        if val_stats["avg_reward_router"] > best_val_reward:
            best_val_reward = val_stats["avg_reward_router"]
            best_state_dict = router.state_dict()

            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(
                ckpt_dir, f"router_semantic_cfg{cfg_idx}_best.pt"
            )

            torch.save(
                {
                    "state_dict": best_state_dict,
                    "latent_dim": latent_dim,
                    "n_models": n_models,
                    "models": models,
                    "k_top": k_top,
                    "w_corr": w_corr,
                    "l1_lambda": l1_lambda,
                    "best_epoch": epoch,
                    "best_val_reward_router": best_val_reward,
                    "exp_name": (
                        f"semantic_wCorr{w_corr}_k{k_top}_l1{l1_lambda}"
                    ),
                },
                ckpt_path,
            )
            print(
                f"  --> [BEST UPDATE] avg_reward_router={best_val_reward:.4f}, "
                f"checkpoint saved to {ckpt_path}"
            )

    return best_val_reward, best_state_dict


# ============================================================
# 6. main: 하이퍼파라미터 서치
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir", type=str, default="ckpts_semantic_search",
        help="directory to save per-config checkpoints"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="batch size"
    )
    parser.add_argument(
        "--base_epochs", type=int, default=5,
        help="epochs per config in search"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    MODELS = [
        "gpt-3.5-turbo-1106",
        "claude-instant-v1",
        "claude-v1",
        "claude-v2",
        "gpt-4-1106-preview",
        "meta/llama-2-70b-chat",
        "mistralai/mixtral-8x7b-chat",
        "zero-one-ai/Yi-34B-Chat",
        "WizardLM/WizardLM-13B-V1.2",
        "meta/code-llama-instruct-34b-chat",
        "mistralai/mistral-7b-chat",
    ]

    # 하이퍼파라미터 config 목록
    # 👉 이제 lambda_cost는 아예 쓰지 않고,
    #    correctness 기반 reward만 사용한다.
    configs: List[Dict[str, Any]] = [
        # Config 0: w_corr=1.0, k_top=32
        {
            "w_corr": 1.0,
            "k_top": 32,
            "lr": 1e-3,
            "epochs": args.base_epochs,
            "l1_lambda": 0.0,
        },

        # Config 1: w_corr=1.0, k_top=64
        {
            "w_corr": 1.0,
            "k_top": 64,
            "lr": 1e-3,
            "epochs": args.base_epochs,
            "l1_lambda": 0.0,
        },

        # Config 2: correctness weight 조금 키움, k_top=32
        {
            "w_corr": 1.5,
            "k_top": 32,
            "lr": 1e-3,
            "epochs": args.base_epochs,
            "l1_lambda": 0.0,
        },

        # Config 3: correctness weight 조금 키움, k_top=64
        {
            "w_corr": 1.5,
            "k_top": 64,
            "lr": 1e-3,
            "epochs": args.base_epochs,
            "l1_lambda": 0.0,
        },
    ]

    print(f"[RouterBenchSAEDataset] device={device}")
    sae, sbert = load_sae_sbert(device)
    train_loader, val_loader = make_dataloaders(
        MODELS, sae, sbert, device, batch_size=args.batch_size
    )

    # latent_dim, n_models 추출
    example_batch = next(iter(train_loader))
    h_ex, _, costs_ex, _, _ = example_batch
    latent_dim = h_ex.shape[1]
    n_models = costs_ex.shape[1]

    print("[Hyperparam Search] latent_dim=", latent_dim, ", n_models=", n_models)
    print("[Hyperparam Search] device=", device)

    best_overall_reward = -1e9
    best_overall_state = None
    best_overall_cfg = None

    for idx, cfg in enumerate(configs):
        val_reward, state_dict = train_one_config(
            cfg_idx=idx,
            config=cfg,
            models=MODELS,
            latent_dim=latent_dim,
            n_models=n_models,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            ckpt_dir=args.ckpt_dir,
        )

        if val_reward > best_overall_reward:
            best_overall_reward = val_reward
            best_overall_state = state_dict
            best_overall_cfg = cfg

    if best_overall_state is not None:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        best_path = os.path.join(
            args.ckpt_dir, "router_qnet_semantic_best_overall.pt"
        )
        torch.save(
            {
                "state_dict": best_overall_state,
                "latent_dim": latent_dim,
                "n_models": n_models,
                "models": MODELS,
                "best_val_reward_router": best_overall_reward,
                "best_config": best_overall_cfg,
                "exp_name": "semantic_reward_best_overall_no_cost",
            },
            best_path,
        )
        print("\n====================================")
        print(
            f"[BEST OVERALL] avg_reward_router={best_overall_reward:.4f}, "
            f"checkpoint saved to {best_path}"
        )
        print("  best_config:", best_overall_cfg)
        print("====================================")
    else:
        print("[WARN] No best_overall_state found (something went wrong?)")


if __name__ == "__main__":
    main()
