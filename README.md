# LLM Router Demo
프롬프트를 분석해서 가장 올바른 답변을 하는 LLM으로 보내주는 router agent입니다.

## 개요
- 최근 여러 LLM 모델이 나오면서 기업, 사용자가 적절한 LLM의 사용을 원함.
- 사용자들이 어느 모델이 어느 분야의 대답을 잘 하는지 명확하지 않은 상태로 서비스를 이용함.
- SBERT → SAE latent (sparse semantic state)
- Q-network 라우팅 (contextual bandit, RouterBench correctness reward 학습)
- 데모 출력: 상위 SAE feature, 모델별 Q-value, 선택 근거

## 팀 정보
- 팀원: router (20231851/ 이도현)
- GitHub: https://github.com/ldh-at/LLM_router


## 설치
0) 저장소 클론
```
git clone https://github.com/ldh-at/LLM_router.git
cd LLM_router

mkdir -p ckpts_semantic_search data

```

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
- RouterBench 데이터: `./data/routerbench_0shot.pkl`
- SAE feature 통계: `./sae_feature_task_stats.csv`

데이터셋, pt 파일 다운:
- routerbench_0shot:'https://huggingface.co/datasets/withmartian/routerbench/tree/main'
- best_router : 'https://github.com/ldh-at/LLM_router/releases/download/v1.0/router_qnet_semantic_best_overall.pt'
- SAE : 'https://github.com/ldh-at/LLM_router/releases/download/v1.0/sae_model.pt'

## 프로젝트 구조(위의 파일들 정확히 보고 배치/ 디렉토리 없으면 mkdir로 추가해서 만들기)
```
LLM_router/
├── debug/
│   └── debug_router_inference.py
├── ckpts_semantic_search/router_qnet_semantic_best_overall.pt (릴리스에서 다운로드)
├── data/routerbench_0shot.pkl (릴리스에서 다운로드)
├── sae_model.pt (릴리스에서 다운로드)
├── sae_feature_task_stats.csv 
├── requirements.txt
└── README.md
```


## 실행 예시
- 데이터셋 모드(라우터 벤치 val 데이터에서 index로 프롬프트를 고름)
```
python -m debug.debug_router_inference --mode dataset --idx 350 

```

- 프롬프트 모드(실제 프롬프트를 입력하면 라우터가 모델을 골라줌)
```
python -m debug.debug_router_inference --mode prompt --prompt "*your prompt*" 

```

- 주요 옵션: `--prompt`, `--idx`, `--split`, `--k_show`, `--ckpt`




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


## Feature 의미 단위 설명  

- hellaswag — 일상 상황 서술의 다음 문장 상식 추론
- grade-school-math — GSM8K류 초중등 단계 서술형 수학 문제 풀이
- arc-challenge — ARC Challenge 난이도, 초중등 과학/상식 객관식 추론
- winogrande — 대명사/상황 기반 상식 코리퍼런스 판별
- bias_detection — 응답의 편향/차별 표현 탐지
- abstract2title — 논문 초록을 보고 적절한 제목 생성
- consensus_summary — 다중 응답 요약/합의 요약
- mbpp — 간단한 파이썬 프로그래밍 문제(입출력 예시 기반 코드 작성)
- chinese_zodiac — 띠/간지 관련 중국어 상식 질의
- Chinese_character_riddles — 한자 수수께끼 풀이
- chinese_tang_poetries — 당시(唐詩) 관련 질의/이해
- chinese_shi_jing — 시경 관련 질의/이해
- accounting_audit — 회계 감사 관련 질의
- chinese_ancient_poetry — 고대 중국 시문 해석/완성
- chinese_hard_translations — 난이도 높은 중국어 번역
- chinese_homonym — 동음이의어 관련 중국어 문제
- chinese_modern_poem_identification — 현대시 식별/분류
- chinese-lantern-riddles — 등불 수수께끼 풀이
- chinese_famous_novel — 중국 명작 소설 관련 질의
- mtbench — MT-Bench 일반 대화/지시 수행 혼합
- mtbench-math — MT-Bench 수학/문제풀이 하위셋
- mtbench-reference — MT-Bench 참고자료 활용형 프롬프트
- chinese_ancient_masterpieces_dynasty — 고전 명작+왕조 매칭
- chinese_song_ci — 송사(宋詞) 관련 질의
- chinese-remainder-theorem — 중국인 나머지 정리 문제
- chinese_chu_ci — 초사(楚辭) 관련 질의
- chinese_idioms — 성어/성구 의미·용법
- chinese_poem — 일반 중국 시문 질의
- test-match — 소수 샘플 테스트 태스크
- bias_detection — 편향 감지(텍스트)
- MMLU 계열(과목별 다지선다 상식/전문 지식):

 - mmlu-professional-law — 법률 전문 지식
- mmlu-moral-scenarios — 도덕 판단 상황
- mmlu-miscellaneous — 일반 상식 잡다 과목
- mmlu-professional-psychology — 심리학 전문
- mmlu-high-school-psychology — 고등 심리학
- mmlu-high-school-macroeconomics — 고등 거시경제
- mmlu-elementary-mathematics — 초등 수학
- mmlu-moral-disputes — 도덕 논쟁
- mmlu-prehistory — 선사 역사
- mmlu-philosophy — 철학
- mmlu-high-school-biology — 고등 생물
- mmlu-nutrition — 영양학
- mmlu-professional-accounting — 회계 전문
- mmlu-professional-medicine — 의학 전문
- mmlu-high-school-mathematics — 고등 수학(증명/계산)
- mmlu-clinical-knowledge — 임상의학
- mmlu-security-studies — 안보학
- mmlu-high-school-microeconomics — 고등 미시경제
- mmlu-high-school-world-history — 고등 세계사
- mmlu-conceptual-physics — 개념 물리
- mmlu-marketing — 마케팅
- mmlu-human-aging — 노화/노년학
- mmlu-high-school-statistics — 고등 통계
- mmlu-high-school-us-history — 고등 미국사
- mmlu-high-school-chemistry — 고등 화학
- mmlu-sociology — 사회학
- mmlu-high-school-geography — 고등 지리
- mmlu-high-school-government-and-politics — 고등 정치/정부
- mmlu-college-medicine — 대학 의학
- mmlu-world-religions — 세계 종교
- mmlu-virology — 바이러스학
- mmlu-high-school-european-history — 고등 유럽사
- mmlu-logical-fallacies — 논리적 오류 판별
- mmlu-astronomy — 천문학
- mmlu-high-school-physics — 고등 물리
- mmlu-electrical-engineering — 전기공학
- mmlu-college-biology — 대학 생물
- mmlu-anatomy — 해부학
- mmlu-human-sexuality — 성 건강/성학
- mmlu-formal-logic — 형식 논리
- mmlu-international-law — 국제법
- mmlu-econometrics — 계량경제
- mmlu-machine-learning — 머신러닝
- mmlu-public-relations — 홍보/PR
- mmlu-jurisprudence — 법학 이론
- mmlu-management — 경영학
- mmlu-college-physics — 대학 물리
- mmlu-abstract-algebra — 추상대수
- mmlu-business-ethics — 비즈니스 윤리
- mmlu-college-chemistry — 대학 화학
- mmlu-college-computer-science — 대학 컴공
- mmlu-college-mathematics — 대학 수학(해석/대수 혼합)
- mmlu-computer-security — 보안
- mmlu-global-facts — 세계 상식/국제 시사
- mmlu-high-school-computer-science — 고등 컴공
- mmlu-medical-genetics — 의학 유전학
- mmlu-us-foreign-policy — 미국 대외 정책
- 기타 두꺼운 상위 과목:

 - consensus_summary — 다중 답변 합성 요약
- abstract2title — 초록→제목 생성
- bias_detection — 편향 감지 (중복 표기)
- test-match — 샘플 테스트 소량