# src/dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


def _parse_task_from_row(row) -> str:
    """
    RouterBench row에서 task / sample_id 기반으로 task_name을 추출한다.
    - 우선순위:
      1) row["task"] 또는 row["task_name"] 이 있으면 그걸 사용
      2) row["sample_id"] 가 있으면, 앞부분만 떼서 사용
         예: "mmlu-high-school-geography.val.91" -> "mmlu-high-school-geography"
    없으면 "unknown" 리턴.
    """
    for key in ["task", "task_name"]:
        if key in row and isinstance(row[key], str) and row[key]:
            return row[key]

    if "sample_id" in row and isinstance(row["sample_id"], str):
        sid = row["sample_id"]
        parts = sid.split(".")
        if len(parts) >= 2:
            return parts[0]
        return sid

    return "unknown"


def _make_stratified_split_indices(task_names, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    task_names: 길이 N의 리스트 (각 샘플의 task_name 문자열)
    각 task별로 8:1:1 비율로 stratified split을 만든다.
    """
    rng = np.random.RandomState(seed)
    task_to_indices = {}
    for idx, t in enumerate(task_names):
        task_to_indices.setdefault(t, []).append(idx)

    train_idx, val_idx, test_idx = [], [], []

    for t, idxs in task_to_indices.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)

        n = len(idxs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0

        train_idx.extend(idxs[:n_train].tolist())
        val_idx.extend(idxs[n_train:n_train + n_val].tolist())
        test_idx.extend(idxs[n_train + n_val:].tolist())

    return sorted(train_idx), sorted(val_idx), sorted(test_idx)


class RouterBenchSAEDataset(Dataset):
    """
    RouterBench + SAE + SBERT를 묶어서 쓰는 Dataset.

    - pkl_path: routerbench_0shot.pkl
    - models: 우리가 라우팅 대상으로 쓰려는 모델 리스트
    - sae: SparseAutoencoder 인스턴스
    - sbert: SBERTEncoder 인스턴스
    - split: "train" / "val" / "test"
    - device: "cuda" or "cpu"
    - return_meta: True면 (h, label, costs, meta) 반환
                   False면 (h, label, costs) 반환 (기본값, 기존 코드와 호환)
    """

    def __init__(
        self,
        pkl_path,
        models,
        sae,
        sbert,
        split="train",
        device="cpu",
        return_meta=False,
    ):
        super().__init__()
        assert split in ["train", "val", "test"], f"split must be train/val/test, got {split}"

        self.sae = sae
        self.sbert = sbert
        self.device = device

        self.models = models
        self.num_models = len(models)
        self.return_meta = return_meta

        print(f"[RouterBenchSAEDataset] Loading {pkl_path} ...")
        df = pd.read_pickle(pkl_path)

        # 0) oracle_model_to_route_to가 우리가 쓰려는 models 안에 있는 행만 사용
        if "oracle_model_to_route_to" not in df.columns:
            raise ValueError("routerbench pkl에 'oracle_model_to_route_to' 컬럼이 없습니다.")

        df = df[df["oracle_model_to_route_to"].isin(models)].reset_index(drop=True)

        # DataFrame을 나중에 meta 구성에 쓰기 위해 보관
        self.df = df

        # 1) prompt / label / costs
        if "prompt" not in df.columns:
            raise ValueError("routerbench pkl에 'prompt' 컬럼이 없습니다.")

        self.prompts = df["prompt"].tolist()

        model_to_idx = {m: i for i, m in enumerate(self.models)}
        labels = [model_to_idx[m] for m in df["oracle_model_to_route_to"]]
        self.labels = torch.tensor(labels, dtype=torch.long)

        # cost 컬럼들 모으기
        cost_cols = []
        for m in self.models:
            col = f"{m}|total_cost"
            if col not in df.columns:
                raise ValueError(f"routerbench pkl에 '{col}' 컬럼이 없습니다.")
            cost_cols.append(col)

        self.costs = torch.tensor(df[cost_cols].values, dtype=torch.float32)  # (N, M)

        # 1.5) 각 모델의 model_response 모으기 (디버그/분석용)
        #      예: "gpt-4-1106-preview|model_response"
        self.model_outputs = []
        model_resp_cols = {}
        for m in self.models:
            col = f"{m}|model_response"
            if col in df.columns:
                model_resp_cols[m] = col

        if model_resp_cols:
            for i in range(len(df)):
                row = df.iloc[i]
                d = {}
                for m, col in model_resp_cols.items():
                    val = row[col]
                    # 문자열이거나, 빈 값이 아닐 때만 저장
                    if isinstance(val, str) and val.strip():
                        d[m] = val
                self.model_outputs.append(d if d else None)
        else:
            # model_response 컬럼이 전혀 없으면 전부 None
            self.model_outputs = [None] * len(df)

        # 2) task_name / sample_id 메타 정보 추출
        if "sample_id" in df.columns:
            self.sample_ids = df["sample_id"].tolist()
        else:
            self.sample_ids = [None] * len(self.prompts)

        self.tasks = []
        for i in range(len(df)):
            row = df.iloc[i]
            t = _parse_task_from_row(row)
            self.tasks.append(t)

        # 3) split 정보 만들기
        if "split" in df.columns:
            df_split = df["split"].astype(str).str.lower()

            split_map = {
                "train": "train",
                "trn": "train",
                "training": "train",
                "valid": "val",
                "validation": "val",
                "dev": "val",
                "val": "val",
                "test": "test",
                "testing": "test",
            }
            mapped_split = df_split.map(lambda x: split_map.get(x, "train"))

            all_indices = list(range(len(df)))
            train_indices = [i for i in all_indices if mapped_split.iloc[i] == "train"]
            val_indices = [i for i in all_indices if mapped_split.iloc[i] == "val"]
            test_indices = [i for i in all_indices if mapped_split.iloc[i] == "test"]

            print(
                f"[RouterBenchSAEDataset] Using existing 'split' column: "
                f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
            )
        else:
            print("[RouterBenchSAEDataset] No 'split' column found. "
                  "Creating stratified 8:1:1 split by task_name...")
            train_indices, val_indices, test_indices = _make_stratified_split_indices(self.tasks)
            print(
                f"[RouterBenchSAEDataset] Stratified split sizes: "
                f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
            )

        if split == "train":
            self.indices = train_indices
        elif split == "val":
            self.indices = val_indices
        else:  # "test"
            self.indices = test_indices

        print(f"[RouterBenchSAEDataset] Final split='{split}' size={len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        prompt = self.prompts[real_idx]
        label = self.labels[real_idx]
        costs = self.costs[real_idx]          # (M,)
        sample_id = self.sample_ids[real_idx]
        task_name = self.tasks[real_idx]
        model_outputs = self.model_outputs[real_idx]

        # 1) SBERT 임베딩
        with torch.no_grad():
            x = self.sbert.encode(prompt)     # (D,)

        x = x.detach().clone().to(self.device)  # (D,)
        x = x.unsqueeze(0)                      # (1, D)

        # 2) SAE latent (inference only)
        with torch.no_grad():
            h = self.sae.encode(x)             # (1, H)
            h = h.squeeze(0).cpu()             # (H,)

        if self.return_meta:
            # df row에서 각 모델별 정답 여부(0/1)를 scores_by_model로 꺼내기
            row = self.df.iloc[real_idx]
            scores_by_model = {}
            for m in self.models:
                if m in row.index:
                    val = row[m]
                    try:
                        val = float(val)
                    except Exception:
                        val = 0.0
                else:
                    val = 0.0
                scores_by_model[m] = val

            meta = {
                "sample_id": sample_id,
                "task_name": task_name,
                "prompt": prompt,
                "model_outputs": model_outputs,
                "scores_by_model": scores_by_model,
            }
            return h, label, costs, meta

        return h, label, costs
