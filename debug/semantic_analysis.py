# semantic_analysis.py
# --------------------
# RouterBench + SAE feature semantics 기반으로
#  - semantic cluster(top_task) 별
#    * 샘플 수
#    * oracle LLM 분포
#    * router가 선택한 LLM 분포
#    * router accuracy (oracle과 일치 비율)
#  를 요약해 주는 스크립트.
#
# 사용 예:
#   python semantic_analysis.py --split val --max_samples 2000 \
#       --ckpt ckpts_semantic_search/router_qnet_semantic_best_overall.pt \
#       --cluster_topk_feats 5 --cluster_min_ratio 0.5 \
#       --output_csv semantic_cluster_val.csv

import argparse
from collections import defaultdict, Counter

import torch
import pandas as pd

from src.sae_model import SparseAutoencoder
from src.sbert_encoder import SBERTEncoder
from src.router_model import RouterQNetwork
from src.train_router_topk import topk_mask_state
from src.dataset import RouterBenchSAEDataset

# -------------------------------------------------------------
# 기본 경로/파일 이름 (필요하면 수정해서 사용)
# -------------------------------------------------------------
DEFAULT_BEST_CKPT = "ckpts_semantic_search/router_qnet_semantic_best_overall.pt"
LEGACY_CKPT = "ckpts/router_qnet_checkpoint.pt"
FEATURE_CSV = "sae_feature_task_stats.csv"
SAE_WEIGHTS = "sae_model.pt"
ROUTERBENCH_PKL = "data/routerbench_0shot.pkl"


# -------------------------------------------------------------
# feature -> semantic task 통계 로드
# -------------------------------------------------------------
def load_feature_semantics(csv_path: str = FEATURE_CSV) -> dict:
    """
    sae_feature_task_stats.csv 에서
      feature -> (top_task, top_ratio, total)
    로 매핑을 만들어준다.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[load_feature_semantics] {csv_path} not found. "
              f"Feature semantics will not be used.")
        return {}

    feat_info = {}
    for _, row in df.iterrows():
        feat_idx = int(row["feature"])
        top_task = str(row["top_task"])
        top_ratio = float(row["top_ratio"])
        total = int(row["total"])
        feat_info[feat_idx] = (top_task, top_ratio, total)
    print(f"[load_feature_semantics] Loaded semantics for {len(feat_info)} features.")
    return feat_info


# -------------------------------------------------------------
# SAE / SBERT / Router Q-network 로드
# -------------------------------------------------------------
def load_models(device: str = "cpu",
                ckpt_path: str = DEFAULT_BEST_CKPT):
    """SAE, SBERT, Router Q-network, 모델 리스트, k_top 로드"""

    # 1) SAE
    sae = SparseAutoencoder().to(device)
    sae.load_state_dict(torch.load(SAE_WEIGHTS, map_location=device))
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    # 2) SBERT (여기서는 직접 사용하진 않지만 구조상 로드)
    sbert = SBERTEncoder(device=device)

    # 3) Router Q-network + 메타 정보
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except FileNotFoundError:
        print(f"[load_models] {ckpt_path} not found. "
              f"Trying legacy checkpoint: {LEGACY_CKPT}")
        ckpt = torch.load(LEGACY_CKPT, map_location=device)

    latent_dim = ckpt.get("latent_dim", 4096)
    n_models = ckpt.get("n_models", len(ckpt["models"]))

    router = RouterQNetwork(
        latent_dim=latent_dim,
        n_models=n_models
    ).to(device)
    router.load_state_dict(ckpt["state_dict"])
    router.eval()

    models = ckpt["models"]
    k_top = ckpt.get("k_top", None)
    exp_name = ckpt.get("exp_name", "unknown_exp")

    print("[load_models] ==============================")
    print(f"  exp_name          : {exp_name}")
    print(f"  latent_dim        : {latent_dim}")
    print(f"  n_models          : {n_models}")
    print(f"  k_top (semantic)  : {k_top}")
    print("===========================================\n")

    return sae, sbert, router, models, k_top


# -------------------------------------------------------------
# h(latent) -> semantic cluster 이름으로 매핑
# -------------------------------------------------------------
def get_semantic_cluster_from_h(
    h: torch.Tensor,
    feat_sem: dict,
    top_k_feats: int = 5,
    min_ratio: float = 0.0,
    default_cluster: str = "unknown"
) -> str:
    """
    h: (H,) SAE latent (dense)
    feat_sem: {feature_idx: (top_task, top_ratio, total)}

    - h에서 activation이 큰 feature 순서대로 보면서
      feat_sem에 등록된 feature의 top_task를 semantic cluster로 사용.
    - top_k_feats 개까지만 스캔.
    - top_ratio < min_ratio 이면 무시.

    return: cluster_name (string)
    """
    H = h.size(0)
    values, indices = torch.topk(h, k=min(top_k_feats, H))
    values = values.cpu().numpy()
    indices = indices.cpu().numpy()

    for feat_idx, v in zip(indices, values):
        if v <= 0:
            continue
        if feat_idx not in feat_sem:
            continue
        top_task, ratio, total = feat_sem[feat_idx]
        if ratio < min_ratio:
            continue
        return str(top_task)  # top_task를 cluster 이름으로 사용

    return default_cluster


# -------------------------------------------------------------
# semantic cluster별 LLM 성능/라우터 성능 비교
# -------------------------------------------------------------
def analyze_semantic_clusters(
    split: str = "val",
    device: str = "cpu",
    ckpt_path: str = DEFAULT_BEST_CKPT,
    top_k_feats_for_cluster: int = 5,
    min_feat_ratio_for_cluster: float = 0.0,
    max_samples: int | None = None,
):
    """
    semantic cluster별:
      - total 샘플 수
      - oracle Top-3 모델 + 비율
      - router Top-3 모델 + 비율
      - router accuracy
    를 모두 요약하는 표 생성
    """

    sae, sbert, router, MODELS, K_TOP = load_models(device=device,
                                                    ckpt_path=ckpt_path)
    feat_sem = load_feature_semantics()

    # 🔥 split 처리: "all"이면 train/val/test 전부
    if split == "all":
        split_list = ["train", "val", "test"]
    else:
        split_list = [split]

    print(f"[analyze_semantic_clusters] splits={split_list}, max_samples={max_samples}")
    print()

    # cluster별 통계 구조
    cluster_stats = defaultdict(lambda: {
        "n": 0,
        "router_correct": 0,
        "oracle_counts": Counter(),
        "router_choice_counts": Counter(),
    })

    total_seen = 0

    # ---- 여러 split 순회 ----
    for cur_split in split_list:
        ds = RouterBenchSAEDataset(
            ROUTERBENCH_PKL,
            MODELS,
            sae=sae,
            sbert=sbert,
            split=cur_split,
            device=device,
            return_meta=True,
        )

        # max_samples가 있으면 전체 합 기준으로 잘라주기
        if max_samples is None:
            cur_max = len(ds)
        else:
            remaining = max_samples - total_seen
            if remaining <= 0:
                break
            cur_max = min(len(ds), remaining)

        print(f"  -> processing split='{cur_split}' with {cur_max} samples (of {len(ds)})")

        for local_idx in range(cur_max):
            h, label, costs, meta = ds[local_idx]
            total_seen += 1

            oracle_idx = int(label.item())
            oracle_model = MODELS[oracle_idx]

            # semantic cluster
            cluster = get_semantic_cluster_from_h(
                h,
                feat_sem,
                top_k_feats=top_k_feats_for_cluster,
                min_ratio=min_feat_ratio_for_cluster,
                default_cluster="unknown"
            )

            # router forward
            h_batch = h.unsqueeze(0).to(device)
            with torch.no_grad():
                h_sparse = topk_mask_state(h_batch, k=K_TOP)
                q = router(h_sparse).squeeze(0)

            router_idx = int(torch.argmax(q).item())
            router_model = MODELS[router_idx]

            # update stats
            stat = cluster_stats[cluster]
            stat["n"] += 1
            stat["oracle_counts"][oracle_model] += 1
            stat["router_choice_counts"][router_model] += 1
            if router_idx == oracle_idx:
                stat["router_correct"] += 1

            if total_seen % 500 == 0:
                print(f"    processed {total_seen} samples in total...")




# -------------------------------------------------------------
# main
# -------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val",
                        help="Train/val/test split name used in RouterBenchSAEDataset")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_BEST_CKPT,
                        help="Router Q-network checkpoint path")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples to analyze (None = use all)")
    parser.add_argument("--cluster_topk_feats", type=int, default=5,
                        help="How many top features from h to consider when deciding cluster")
    parser.add_argument("--cluster_min_ratio", type=float, default=0.0,
                        help="Minimum top_ratio in sae_feature_task_stats to accept a feature as semantic")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="If set, save summary dataframe to this CSV path")

    args = parser.parse_args()

    df = analyze_semantic_clusters(
        split=args.split,
        device=device,
        ckpt_path=args.ckpt,
        top_k_feats_for_cluster=args.cluster_topk_feats,
        min_feat_ratio_for_cluster=args.cluster_min_ratio,
        max_samples=args.max_samples,
    )

    if args.output_csv is not None and not df.empty:
        df.to_csv(args.output_csv, index=False)
        print(f"[main] Saved summary to: {args.output_csv}")


if __name__ == "__main__":
    main()
