# scripts/compute_sae_feature_stats.py

import torch
import pickle
from collections import defaultdict, Counter
from tqdm import tqdm

from src.dataset import RouterBenchSAEDataset
from src.sae_model import SparseAutoencoder
from src.sbert_encoder import SBERTEncoder


SAVE_PATH = "data/sae_feature_stats.pkl"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[compute_sae_feature_stats] Using device: {device}")

    # ----------------------------------------------------
    # RouterBench 모델 리스트 (dataset과 동일해야 함)
    # ----------------------------------------------------
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

    # ----------------------------------------------------
    # 1) SAE & SBERT 로드
    # ----------------------------------------------------
    sae = SparseAutoencoder().to(device)
    sae.load_state_dict(torch.load("sae_model.pt", map_location=device))
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)

    sbert = SBERTEncoder(device=device)

    # ----------------------------------------------------
    # 2) RouterBench train split (return_meta=True)
    # ----------------------------------------------------
    ds = RouterBenchSAEDataset(
        "data/routerbench_0shot.pkl",
        MODELS,
        sae=sae,
        sbert=sbert,
        split="train",
        device=device,
        return_meta=True,         # <-- meta["task_name"] 사용
    )

    feature_buckets = defaultdict(list)  # feature_idx → list of (act, category)

    print("[compute_sae_feature_stats] Collecting activations...")
    for i in tqdm(range(len(ds))):
        h, _, _, meta = ds[i]

        category = meta.get("task_name", "unknown")

        # top-k feature activations
        values, indices = torch.topk(h, k=min(64, h.numel()))

        for v, j in zip(values, indices):
            v = float(v)
            if v <= 0.0:
                continue
            feature_buckets[int(j)].append((v, category))

    # ----------------------------------------------------
    # 3) feature → 카테고리 요약 통계 만들기
    # ----------------------------------------------------
    feature_stats = {}

    print("[compute_sae_feature_stats] Aggregating category stats...")
    for feat_idx, samples in feature_buckets.items():

        # category → (sum_activation, count)
        cat_map = defaultdict(lambda: [0.0, 0])

        for act, cat in samples:
            cat_map[cat][0] += act
            cat_map[cat][1] += 1

        # 평균 activation 계산
        cat_list = []
        for cat, (sum_act, cnt) in cat_map.items():
            avg_act = sum_act / cnt
            cat_list.append((cat, avg_act, cnt))

        # 평균 activation 기준 상위순
        cat_list.sort(key=lambda x: x[1], reverse=True)

        feature_stats[feat_idx] = cat_list

    # ----------------------------------------------------
    # 4) pickle로 저장
    # ----------------------------------------------------
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(feature_stats, f)

    print(f"[compute_sae_feature_stats] Saved: {SAVE_PATH}")
    print("[compute_sae_feature_stats] Done.")


if __name__ == "__main__":
    main()
