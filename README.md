# LLM Router Demo

## 프로젝트 개요
- SAE + SBERT 기반 latent를 사용해 다중 모델 라우팅을 수행하는 contextual bandit 데모/디버그 스크립트.
- 프롬프트 모드: 임의 프롬프트에 대해 라우터가 선택한 모델/SAE 피처 상위값을 출력.
- 데이터셋 모드: RouterBench 샘플 기준으로 라우터의 선택과 모델별 답변 스니펫, 상위 Q-value를 비교.

## 팀 정보
- 팀원: router (20231851/ 이도현)
- GitHub 페이지: https://github.com/ldh-at/LLM_router

## 필요한 아티팩트(다운로드 후 경로에 배치)
- SAE 가중치: `sae_model.pt` → `./sae_model.pt`  
- 라우터 체크포인트(semantic): `router_qnet_semantic_best_overall.pt` → `./ckpts_semantic_search/router_qnet_semantic_best_overall.pt`  
- 라우터 체크포인트(legacy 대안): `router_qnet_checkpoint.pt` → `./ckpts/router_qnet_checkpoint.pt`  
- RouterBench 데이터: `routerbench_0shot.pkl` → `./data/routerbench_0shot.pkl`  
- SAE feature 통계: `sae_feature_task_stats.csv` → `./sae_feature_task_stats.csv`
- 위 파일들은 용량이 크면 Git LFS 또는 릴리스/외부 링크에 올리고, README에 실제 다운로드 링크를 채워 넣으세요.

## 환경 준비
- Python 3.10+ 권장.
- 패키지 설치: `pip install -r requirements.txt` (없다면 torch, pandas, sentence-transformers 등 의존성을 수동 설치)
- GPU 사용: CUDA가 있으면 자동으로 `cuda`, 없으면 `cpu`로 동작.

## 실행 방법
- 프롬프트 모드 예시:
  ```bash
  python debug/debug_router_inference.py --mode prompt --prompt "여기에 프롬프트" --k_show 20 --ckpt ckpts_semantic_search/router_qnet_semantic_best_overall.pt
  ```
- 데이터셋 모드 예시:
  ```bash
  python debug/debug_router_inference.py --mode dataset --idx 0 --split val --k_show 20 --ckpt ckpts_semantic_search/router_qnet_semantic_best_overall.pt
  ```
- 주요 옵션
  - `--prompt`: 프롬프트 모드에서 사용할 텍스트
  - `--idx`: 데이터셋 인덱스 (0 기반)
  - `--split`: `val`/`train` 등 데이터 스플릿
  - `--k_show`: 출력할 상위 SAE feature 개수
  - `--ckpt`: 사용할 라우터 체크포인트 경로

## 보고서(PPT)
- 첫 슬라이드에 팀원 학번/이름 + GitHub 링크를 명시.
- 내용: 주제, 설계, 구현, 실험 결과, 결론/한계/추가과제.
- 완성된 PPT를 저장소에도 포함(예: `docs/report.pptx`)하고, 사이버캠퍼스에는 팀원 1명이 제출.

## 제출/공유 체크리스트
1) 필요한 `.pt`/`.pkl` 다운로드 후 지정 경로에 배치.
2) `pip install -r requirements.txt`로 환경 준비.
3) 위 실행 예시로 프롬프트 모드/데이터셋 모드 검증.
4) README에 실제 다운로드 링크 채우기 + 보고서 파일 추가.
5) `git add .` → `git commit -m "Add router demo"` → `git push -u origin main`.
