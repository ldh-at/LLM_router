# scripts/download_routerbench.py

from pathlib import Path
import shutil

import pandas as pd
from huggingface_hub import hf_hub_download

DATASET_REPO = "withmartian/routerbench"   # Hugging Face repo id
FILENAMES = [
    "routerbench_0shot.pkl",   # 0-shot 버전
    "routerbench_5shot.pkl",   # 5-shot 버전
]


def download_routerbench():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    local_files = []

    for fname in FILENAMES:
        print(f"[RouterBench] Downloading {fname} from Hugging Face...")
        # Hugging Face 캐시 위치로 다운로드
        src_path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=fname,
            repo_type="dataset",
        )

        dst_path = data_dir / fname

        # 이미 있으면 덮어쓰지 않고 스킵
        if dst_path.exists():
            print(f"[RouterBench] {dst_path} already exists, skipping copy.")
        else:
            shutil.copy(src_path, dst_path)
            print(f"[RouterBench] Saved to {dst_path}")

        local_files.append(dst_path)

    return local_files


def preview_pkl(path: Path, n: int = 5):
    print(f"\n[Preview] {path}")
    df = pd.read_pickle(path)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head(n))


if __name__ == "__main__":
    files = download_routerbench()

    # 0-shot 데이터 간단 프리뷰
    if files:
        preview_pkl(files[0])
