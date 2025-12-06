# src/analyze_sae_features_to_csv.py

import torch
import pandas as pd
from collections import defaultdict, Counter

from src.dataset import RouterBenchSAEDataset
from src.sae_model import SparseAutoencoder
from src.sbert_encoder import SBERTEncoder


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[analyze_sae_features_to_csv] Using device: {device}")

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

    # 1) SAE 로드
    sae = SparseAutoencoder(input_dim=384, hidden_dim=4096).to(device)
    sae.load_state_dict(torch.load("sae_model.pt", map_location=device))
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    print("[analyze_sae_features_to_csv] SAE loaded.")

    # 2) SBERT encoder
    sbert = SBERTEncoder(device=device)

    # 3) train split + meta
    ds = RouterBenchSAEDataset(
        "data/routerbench_0shot.pkl",
        MODELS,
        sae=sae,
        sbert=sbert,
        split="train",
        device=device,
        return_meta=True,
    )

    feat_top = defaultdict(list)  # feat_idx -> list of (activation, task_name)

    print("[analyze_sae_features_to_csv] Collecting activations...")
    for i in range(len(ds)):
        h, _, _, meta = ds[i]  # h: (4096,)
        task_name = meta["task_name"]

        values, indices = torch.topk(h, k=min(64, h.numel()))
        for v, j in zip(values, indices):
            v = float(v)
            if v <= 0.0:
                continue
            feat_top[int(j)].append((v, task_name))

    rows = []
    for feat_idx, samples in feat_top.items():
        tasks = [t for _, t in samples]
        cnt = Counter(tasks)
        total = sum(cnt.values())
        top_task, top_count = cnt.most_common(1)[0]
        top_ratio = top_count / total

        row = {
            "feature": feat_idx,
            "total": total,
            "top_task": top_task,
            "top_ratio": top_ratio,
        }

        # 상위 5개 task 분포도 같이 넣어둠
        for rank, (task, c) in enumerate(cnt.most_common(5), start=1):
            row[f"task{rank}_name"] = task
            row[f"task{rank}_count"] = c
            row[f"task{rank}_ratio"] = c / total

        rows.append(row)

    df_out = pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)
    out_path = "sae_feature_task_stats.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[analyze_sae_features_to_csv] Saved to {out_path}")


if __name__ == "__main__":
    main()
