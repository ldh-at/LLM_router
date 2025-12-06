# src/train_router.py

import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import csv

from src.dataset import RouterBenchSAEDataset
from src.sae_model import SparseAutoencoder
from src.sbert_encoder import SBERTEncoder
from src.router_model import RouterQNetwork

########################################
# 공통 설정
########################################
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

LATENT_DIM = 4096
N_EPOCHS = 10
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_VAL = 128
LR = 1e-3

# 여러 실험: lambda_cost, k_top 조합
EXPERIMENTS = [
    {"name": "lambda0.3_k64", "lambda_cost": 0.3, "k_top": 64},
    {"name": "lambda1.0_k64", "lambda_cost": 1.0, "k_top": 64},
    {"name": "lambda3.0_k64", "lambda_cost": 3.0, "k_top": 64},
    {"name": "lambda5.0_k32", "lambda_cost": 5.0, "k_top": 32},
    # 필요하면 더 추가해도 됨
]


########################################
# 보상 함수: quality - λ * normalized_cost
########################################
def compute_reward_matrix(labels: torch.Tensor,
                          costs: torch.Tensor,
                          lambda_cost: float) -> torch.Tensor:
    """
    labels: (B,)  oracle model index
    costs : (B, M)  각 모델별 total_cost
    return: rewards (B, M)

    quality:
      - oracle model: 1.0
      - others      : 0.0

    cost는 각 샘플(row) 기준 z-score 정규화 후 penalty로 사용.
    reward = quality - lambda_cost * norm_cost
    """
    B, M = costs.shape

    # 1) quality
    quality = torch.zeros_like(costs)          # (B, M)
    quality[torch.arange(B), labels] = 1.0

    # 2) row-wise z-score
    mean = costs.mean(dim=1, keepdim=True)
    std = costs.std(dim=1, keepdim=True) + 1e-8
    norm_costs = (costs - mean) / std

    # 3) 최종 reward
    rewards = quality - lambda_cost * norm_costs
    return rewards


########################################
# SAE latent를 top-k feature만 남기는 state 변환
########################################
def topk_mask_state(h: torch.Tensor, k: Optional[int]) -> torch.Tensor:
    """
    h: (B, D) SAE latent
    k: 남길 feature 개수 (None/0이면 그대로 반환)

    가장 activation이 큰 k개만 남기고 나머지는 0으로 마스킹.
    """
    if k is None or k <= 0 or k >= h.size(1):
        return h

    # values: (B, k), indices: (B, k)
    values, indices = torch.topk(h, k=k, dim=1)

    mask = torch.zeros_like(h)
    mask.scatter_(1, indices, torch.ones_like(values))

    h_sparse = h * mask
    return h_sparse


########################################
# 하나의 설정(exp)으로 학습 + 검증
########################################
def run_single_experiment(
    device: str,
    train_dl: DataLoader,
    val_dl: DataLoader,
    lambda_cost: float,
    k_top: int,
    exp_name: str,
) -> Dict[str, Any]:
    """
    한 실험 설정(lambda_cost, k_top)에 대해
    N_EPOCHS 동안 학습하고, 각 epoch의 val avg_reward_router 중
    최고값과 그 때의 state_dict를 반환.
    """
    print(f"\n[EXP] ===== {exp_name} (lambda_cost={lambda_cost}, k_top={k_top}) =====")

    router = RouterQNetwork(latent_dim=LATENT_DIM,
                            n_models=len(MODELS)).to(device)
    opt = torch.optim.Adam(router.parameters(), lr=LR)
    mse = torch.nn.MSELoss()

    # CSV 설정
    csv_path = f"router_training_log_{exp_name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "epoch",
            "avg_reward_oracle",
            "avg_reward_router",
            "match_acc",
            "avg_cost_oracle",
            "avg_cost_router",
            "total_samples",
        ]
        header += [f"action_count_{m}" for m in MODELS]
        writer.writerow(header)

    best_val_reward_router = float("-inf")
    best_state_dict = None
    best_epoch = -1

    for epoch in range(1, N_EPOCHS + 1):
        ####################################
        # Train
        ####################################
        router.train()
        total_loss = 0.0

        for h, labels, costs in train_dl:
            h = h.to(device)            # (B, D)
            labels = labels.to(device)  # (B,)
            costs = costs.to(device)    # (B, M)

            # semantic state: top-k feature만 사용
            h_sparse = topk_mask_state(h, k=k_top)

            # target reward
            rewards = compute_reward_matrix(labels, costs,
                                            lambda_cost=lambda_cost).to(device)

            # Q-network 예측
            q = router(h_sparse)        # (B, M)

            loss = mse(q, rewards)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_dl))
        print(f"[{exp_name}][Epoch {epoch}] train MSE={avg_train_loss:.4f}")

        ####################################
        # Validation
        ####################################
        router.eval()
        total_reward_oracle = 0.0
        total_reward_router = 0.0
        total_samples = 0

        total_match = 0
        total_cost_oracle = 0.0
        total_cost_router = 0.0

        action_counts = torch.zeros(len(MODELS), dtype=torch.long)

        with torch.no_grad():
            for h, labels, costs in val_dl:
                h = h.to(device)
                labels = labels.to(device)
                costs = costs.to(device)

                h_sparse = topk_mask_state(h, k=k_top)

                rewards = compute_reward_matrix(labels, costs,
                                                lambda_cost=lambda_cost).to(device)
                q = router(h_sparse)  # (B, M)

                a_router = q.argmax(dim=1)          # (B,)

                r_router = rewards[torch.arange(len(labels)), a_router]
                r_oracle = rewards[torch.arange(len(labels)), labels]

                total_reward_oracle += r_oracle.sum().item()
                total_reward_router += r_router.sum().item()
                total_samples += len(labels)

                total_match += (a_router == labels).sum().item()

                total_cost_oracle += costs[torch.arange(len(labels)), labels].sum().item()
                total_cost_router += costs[torch.arange(len(labels)), a_router].sum().item()

                for a in a_router.cpu():
                    action_counts[a.item()] += 1

        if total_samples > 0:
            avg_reward_oracle = total_reward_oracle / total_samples
            avg_reward_router = total_reward_router / total_samples
            match_acc = total_match / total_samples
            avg_cost_oracle = total_cost_oracle / total_samples
            avg_cost_router = total_cost_router / total_samples
        else:
            avg_reward_oracle = avg_reward_router = 0.0
            match_acc = 0.0
            avg_cost_oracle = avg_cost_router = 0.0

        print(f"  [Val] avg_reward_oracle={avg_reward_oracle:.4f}, "
              f"avg_reward_router={avg_reward_router:.4f}")
        print(f"  [Val] match_acc={match_acc:.3f}, "
              f"avg_cost_oracle={avg_cost_oracle:.6f}, "
              f"avg_cost_router={avg_cost_router:.6f}")

        # CSV 저장
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            row = [
                epoch,
                avg_reward_oracle,
                avg_reward_router,
                match_acc,
                avg_cost_oracle,
                avg_cost_router,
                total_samples,
            ]
            row += action_counts.tolist()
            writer.writerow(row)

        # 베스트 router (avg_reward_router 기준) 업데이트
        if avg_reward_router > best_val_reward_router:
            best_val_reward_router = avg_reward_router
            best_epoch = epoch
            # state_dict를 CPU로 deep copy 해서 보관
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in router.state_dict().items()
            }

    print(f"[EXP] {exp_name} best avg_reward_router={best_val_reward_router:.4f} "
          f"(epoch={best_epoch})")

    return {
        "exp_name": exp_name,
        "lambda_cost": lambda_cost,
        "k_top": k_top,
        "best_val_reward_router": best_val_reward_router,
        "best_epoch": best_epoch,
        "state_dict": best_state_dict,
    }


########################################
# 메인: 여러 실험 돌리고, 베스트 하나 저장
########################################
def train_router():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_router] Using device: {device}")
    print(f"[train_router] N_EPOCHS={N_EPOCHS}, LR={LR}")
    print(f"[train_router] Experiments: {EXPERIMENTS}")

    # 1) SAE 로드 + freeze
    sae = SparseAutoencoder().to(device)
    sae.load_state_dict(torch.load("sae_model.pt", map_location=device))
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    print("[train_router] SAE loaded and frozen.")

    # 2) SBERT encoder (inference only)
    sbert = SBERTEncoder(device=device)

    # 3) Dataset / DataLoader (모든 실험에서 재사용)
    train_ds = RouterBenchSAEDataset(
        "data/routerbench_0shot.pkl",
        MODELS,
        sae=sae,
        sbert=sbert,
        split="train",
        device=device,
    )
    val_ds = RouterBenchSAEDataset(
        "data/routerbench_0shot.pkl",
        MODELS,
        sae=sae,
        sbert=sbert,
        split="val",
        device=device,
    )

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE_VAL)

    # 전체 실험 중 베스트
    global_best_reward = float("-inf")
    global_best_info: Dict[str, Any] = {}

    for exp in EXPERIMENTS:
        result = run_single_experiment(
            device=device,
            train_dl=train_dl,
            val_dl=val_dl,
            lambda_cost=exp["lambda_cost"],
            k_top=exp["k_top"],
            exp_name=exp["name"],
        )

        if result["best_val_reward_router"] > global_best_reward:
            global_best_reward = result["best_val_reward_router"]
            global_best_info = result

    # 베스트 실험 정보 출력
    print("\n[train_router] ===== Global Best Experiment =====")
    print(f"  name         : {global_best_info['exp_name']}")
    print(f"  lambda_cost  : {global_best_info['lambda_cost']}")
    print(f"  k_top        : {global_best_info['k_top']}")
    print(f"  best_reward  : {global_best_reward:.4f}")
    print(f"  best_epoch   : {global_best_info['best_epoch']}")

    # 베스트 state_dict로 RouterQNetwork 구성 후 저장
    best_router = RouterQNetwork(
        latent_dim=LATENT_DIM,
        n_models=len(MODELS)
    )
    best_router.load_state_dict(global_best_info["state_dict"])

    # 1) 순수 가중치만 저장
    torch.save(best_router.state_dict(), "router_qnet_best.pt")

    # 2) 메타정보 포함 checkpoint 저장
    checkpoint = {
        "state_dict": best_router.state_dict(),
        "latent_dim": LATENT_DIM,
        "n_models": len(MODELS),
        "models": MODELS,
        "k_top": global_best_info["k_top"],
        "lambda_cost": global_best_info["lambda_cost"],
        "exp_name": global_best_info["exp_name"],
        "best_epoch": global_best_info["best_epoch"],
        "best_val_reward_router": global_best_info["best_val_reward_router"],
    }
    torch.save(checkpoint, "router_qnet_best_checkpoint.pt")

    print("\n[train_router] Saved best router:")
    print("  - router_qnet_best.pt")
    print("  - router_qnet_best_checkpoint.pt")


if __name__ == "__main__":
    print("[main] starting Cost-aware Top-k Semantic Router sweeper...")
    train_router()
