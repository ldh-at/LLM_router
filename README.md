# LLM Router Demo
SAE 기반 의미 피처 + contextual bandit 라우터로 프롬프트마다 최적 LLM을 선택하는 시스템입니다.

## 개요
- SBERT → SAE latent (sparse semantic state)
- Q-network 라우팅 (contextual bandit, RouterBench correctness reward 학습)
- 디버그 출력: 상위 SAE feature, 모델별 Q-value, 선택 근거

## 팀 정보
- 팀원: router (20231851/ 이도현)
- GitHub: https://github.com/ldh-at/LLM_router

## 설치
1) Conda 환경 생성/활성화
```
conda create -n llmrouter python=3.10 -y
conda activate llmrouter
```
2) 의존성 설치
```
pip install -r requirements.txt
```
CUDA별 torch가 필요하면 예시:
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 아티팩트 다운로드 및 배치
- SAE 가중치: `./sae_model.pt`
- 라우터 체크포인트(semantic): `./ckpts_semantic_search/router_qnet_semantic_best_overall.pt`
- 라우터 체크포인트(legacy): `./ckpts/router_qnet_checkpoint.pt`
- RouterBench 데이터: `./data/routerbench_0shot.pkl`
- SAE feature 통계: `./sae_feature_task_stats.csv`

배포 권장:
- GitHub Releases에 올리고 링크 기입  
  예) `https://github.com/ldh-at/LLM_router/releases/download/<tag>/router_qnet_semantic_best_overall.pt`
- 또는 Git LFS: `git lfs install` → `git lfs track "*.pt" "*.pkl"` → 커밋/푸시
- 외부 스토리지 사용 시 공개 링크 여부 확인

## 실행 예시
- 프롬프트 모드
```
python debug/debug_router_inference.py --mode prompt \
  --prompt "여기에 프롬프트" \
  --k_show 20 \
  --ckpt ckpts_semantic_search/router_qnet_semantic_best_overall.pt
```
- 데이터셋 모드
```
python debug/debug_router_inference.py --mode dataset \
  --idx 0 \
  --split val \
  --k_show 20 \
  --ckpt ckpts_semantic_search/router_qnet_semantic_best_overall.pt
```
- 주요 옵션: `--prompt`, `--idx`, `--split`, `--k_show`, `--ckpt`

## 프로젝트 구조
```
LLM_router/
├── debug/
│   └── debug_router_inference.py
├── ckpts_semantic_search/        # 라우터 체크포인트 (릴리스에서 다운로드)
├── data/                         # routerbench_0shot.pkl
├── sae_model.pt
├── sae_feature_task_stats.csv
├── requirements.txt
└── README.md
```

## 실험 결과 예시 (semantic reward)
| Config | Top-1 | Top-3 | Top-5 | Avg Cost |
|--------|-------|-------|-------|----------|
| 0      | 0.8223| 0.8908| 0.9118| 0.002668 |
| 1      | 0.8223| 0.8918| 0.9106| 0.002641 |
| 2      | 0.8213| 0.8929| 0.9134| 0.002706 |
| 3      | 0.8210| 0.8920| 0.9135| 0.002650 |

## 체크리스트
- conda env 생성 및 활성화
- `pip install -r requirements.txt`
- Releases(LFS)에서 모델/데이터 다운로드 후 지정 경로 배치
- prompt/dataset 모드로 실행 확인
- README에 실제 다운로드 링크, 보고서 파일(PPT) 경로 추가

## 트러블슈팅
- CUDA 인식 안 될 때: CUDA 버전에 맞춰 torch 재설치  
  `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- `dataset not found`: `data/routerbench_0shot.pkl` 경로 확인
- `sentence_transformers` import 오류: `pip install -r requirements.txt` 다시 실행
