# pixellm — Project Plan

**Teaching small LLMs to draw pixel art via code generation, progressing from SFT to GRPO with multi-component rewards.**

> 이 문서는 `pixellm` 프로젝트의 통합 실행 계획입니다. 여러 차례 피드백과 리뷰를 거쳐 확정된 설계 결정과 단계별 로드맵을 담고 있습니다.

---

## 1. 프로젝트 정체성

### 한 줄 요약
16×16 픽셀아트를 Palette Index Grid DSL로 생성하는 소형(3B) 언어 모델을 SFT + GRPO로 학습.

### 포지셔닝
- **"AI 픽셀아트 생성기"가 아님** (상용 시장 포화)
- **"Controllable code-centric sprite generation"** — 텍스트에서 렌더 가능한 코드를 생성하는 더 큰 흐름(Chat2SVG, StarVector, SVGen, Code2World 등)의 픽셀아트 특화판

### 이 프로젝트가 증명하는 것
1. Open-source LLM fine-tuning 실무 역량 (SFT, QLoRA, TRL)
2. GRPO / RLVR 적용 경험 (2026 트렌드)
3. 검증 가능한 reward 설계 능력 (deterministic + semantic 보상 조합)
4. End-to-end 파이프라인 ownership (데이터 → 학습 → 평가 → 데모)

### 솔직한 프레이밍
이 프로젝트는 **"현업 LLM 파이프라인 경험과 별개로, open-source 모델 fine-tuning과 RL-based post-training을 직접 경험하기 위한 프로젝트"** 입니다. 현업 강점(multi-stage 파이프라인, LLM-as-Judge)과 억지로 엮지 않고, 새 영역 진입 + 이력서 약점 보강 프로젝트로 솔직하게 표현합니다.

---

## 2. 핵심 설계 결정

### 2.1 DSL 포맷 — Palette Index Grid

```text
<PALETTE>
0:transparent, 1:#2c1810, 2:#8b4513, 3:#daa520, 4:#ffffff
</PALETTE>
<GRID>
0000000000000000
0000001111110000
0000012222210000
...
</GRID>
```

**이 포맷을 선택한 이유:**
- **토큰 효율**: JSON 래핑 대비 2-3배 토큰 절약
- **패턴 학습 유리**: 연속된 같은 인덱스가 그대로 시퀀스로 표현되어 대칭·반복 패턴 학습에 유리
- **파싱 단순**: `<GRID>` 태그 추출 → 줄 단위 split → 문자 단위 인덱스 (정규식 한 줄)

**제약:**
- 크기 고정: 16×16
- 팔레트 크기: 최대 8색 (인덱스 0-7) — 첫 단계는 보수적으로
- `0` = transparent, 비투명 색상은 `1-7`만 사용
- `<GRID>`의 각 문자는 반드시 `0-7` 범위여야 함

**직렬화 원칙:**
- 외부 학습/평가 target은 항상 `<PALETTE>` / `<GRID>` 태그 기반 텍스트 DSL
- Pydantic 모델은 내부 표현과 검증용으로만 사용
- `src/pixellm/dsl.py`는 `parse_dsl(text) -> PixelArt`, `serialize_dsl(pixel_art) -> str`, `validate_pixel_art(pixel_art)`를 제공
- 학습 JSONL의 `dsl` 필드는 JSON 문자열이 아니라 태그 DSL 문자열

**팔레트 변환 규칙:**
- 이미지 전처리 시 alpha가 threshold 이하인 픽셀은 `0`으로 매핑
- 비투명 픽셀은 최대 7색으로 양자화해 `1-7`에 매핑
- 팔레트 선언에는 `0:transparent`를 반드시 포함하고, `1-7`은 실제 사용 색상만 선언 가능
- 파서/밸리데이터는 grid에 선언되지 않은 비투명 인덱스나 `8` 이상 문자가 있으면 invalid 처리

### 2.2 Base Model — Qwen2.5-Coder-3B-Instruct

**1차 선택 이유:**
- 코드/구조화 데이터 생성에 특화
- TRL, PEFT와의 호환성 검증 완료, 커뮤니티 사례 풍부
- 한국 시장 면접관 친숙도 높음

**대안 (2차, 시간 여유 시):**
- SmolLM3-3B — 트렌드 신호는 강하나 chat template 이슈 등 첫 셋업 리스크 있음

### 2.3 학습 전략 — SFT 먼저, GRPO 나중

**단계:**
1. **SFT**: Base model에게 DSL 형식을 가르침 (2-3 epoch, 2,000-3,000 샘플)
2. **GRPO**: SFT 체크포인트에서 reward 기반 강화 학습

**이 순서가 필수인 이유:**
- Base model이 DSL을 모르는 상태에서 GRPO → 모든 응답 invalid → reward = -1 → group variance = 0 → 학습 불가
- GRPO는 "이미 어느 정도 가능한 모델을 개선"하는 방법
- DeepSeek-R1도 cold-start SFT 후 RL 수행

### 2.4 Reward 함수 — 단계적 활성화

$$R_{total} = w_1 R_{syntax} + w_2 R_{symmetry} + w_3 R_{clip} + w_4 R_{vlm}$$

**Phase 1 — 형식 안정화 (GRPO 초기)**
```
w_syntax    = 0.7   # 형식 준수
w_symmetry  = 0.3   # (정면 프롬프트에만 조건부)
w_clip      = 0.0   # 아직 비활성
w_vlm       = 0.0
```

**Phase 2 — 의미적 품질 (parse rate 90%+ 달성 후)**
```
w_syntax    = 0.3
w_symmetry  = 0.1
w_clip      = 0.6   # 활성화
w_vlm       = 0.0
```

**Phase 3 — VLM judge (선택)**
```
w_syntax    = 0.2
w_symmetry  = 0.1
w_clip      = 0.3
w_vlm       = 0.4   # VLM judge 추가
```

**왜 단계적으로?**
- Phase 1에서 CLIP/VLM을 넣으면 형식을 깨면서 점수 올리는 reward hacking 발생
- Parse rate 안정 → semantic 품질 → 고급 평가 순으로 쌓아야 안정적

### 2.5 개별 Reward 설계 주의사항

- **R_symmetry**는 데이터 메타데이터의 `view == "front"`일 때만 활성화. 메타데이터가 없는 샘플은 프롬프트에 `"front view"` / `"frontal"` / `"정면"` 키워드가 명시된 경우에만 fallback 활성화
- **R_syntax**: 파싱 실패 = −1.0 (강한 페널티), 파싱 성공 + 제약 충족 정도에 따라 0~1
- **R_clip**: `clip_score(render(dsl), prompt)`, 보정 필요 (CLIP은 픽셀아트 도메인에서 노이지함)
- **R_vlm**: Qwen2.5-VL-7B 또는 Qwen3.5-VL 사용, 비용과 속도 trade-off 고려

### 2.6 GRPO 파라미터

- `num_generations` (group size): **8~16** (메모리 허용 범위 내에서)
- `max_prompt_length`: **256**
- `max_completion_length`: **1024** (16×16 그리드 + 팔레트 여유)
- `kl_coeff`: **0.1부터 시작** (TRL 기본 0.04보다 높게) → 학습 안정되면 내리면서 실험
- `learning_rate`: **1e-5** (SFT의 1e-4 ~ 2e-4보다 훨씬 낮게)

### 2.7 데이터셋 / 프롬프트 정책

**초기 데이터셋 우선순위:**
1. `m1guelpf/nouns`: train split, `image` + `text`, CC0. 정면 캐릭터 스타일이 일관적이므로 SFT v1의 주 데이터셋으로 사용
2. `Sc077y/pixel-art-synthetic-10k`: train split, `id` + `prompt` + `grid` + `preview`, MIT. 프롬프트와 구도가 다양하므로 보조 데이터 또는 일반화 평가에 사용

**학습 JSONL 스키마:**
```json
{
  "caption": "a pixel art character with square black glasses...",
  "prompt": "Draw 16x16 pixel art as Palette Index Grid DSL: a pixel art character with square black glasses...",
  "dsl": "<PALETTE>\n0:transparent, 1:#...\n</PALETTE>\n<GRID>\n...\n</GRID>",
  "source": "m1guelpf/nouns",
  "source_id": "train:000001",
  "category": "character",
  "view": "front",
  "license": "cc0-1.0"
}
```

**필터링 / 정규화 기본값:**
- SFT v1은 `category == "character"`와 `view == "front"`로 통일한다.
- `m1guelpf/nouns`는 모든 샘플을 `category="character"`, `view="front"`로 간주한다.
- `Sc077y/pixel-art-synthetic-10k`는 prompt에 `portrait`, `frontal`, `front view`, `frontal portrait` 중 하나가 있는 샘플만 SFT v1 후보로 포함하고, 나머지는 보조/평가 후보로 둔다.
- 모델 입력 prompt에는 항상 `Draw 16x16 pixel art as Palette Index Grid DSL: {caption}` 형태의 지시문을 붙인다.
- 원본 텍스트는 `caption`, 모델 입력 문자열은 `prompt`로 분리해 보존한다.

### 2.8 SFT Chat Template 정책

- 학습 데이터는 raw `messages` 형태로 구성하고, 학습 시 Qwen tokenizer의 `apply_chat_template()`를 적용한다.
- JSONL에는 가능하면 `{caption, prompt, dsl, source, source_id, category, view, license}`를 저장하고, SFT collator에서 `messages = [{"role": "user", ...}, {"role": "assistant", ...}]`로 변환한다.
- loss는 assistant completion token에만 적용한다.
- 평가/서빙도 같은 `generate_dsl(prompt)` 유틸을 사용해 chat template 적용 경로를 공유한다.

---

## 3. 기술 스택

| 영역 | 도구 |
|------|------|
| 언어 | Python 3.12 |
| 패키지 관리 | uv |
| 학습 | PyTorch, Transformers, TRL (≥1.0), PEFT |
| 양자화 | bitsandbytes (QLoRA 4bit) |
| 데이터 | datasets, Pillow, NumPy |
| 구조화 출력 | Pydantic |
| 평가 | CLIP (`openai/clip-vit-base-patch32`), Qwen2.5-VL-7B |
| 모니터링 | Weights & Biases |
| 서빙 | FastAPI + vLLM (선택) |
| 프론트 | Streamlit (기본), Next.js (향후 개선) |
| 인프라 | RunPod (RTX A6000 48GB) |

**의존성 관리 원칙:**
- 로컬 개발 기본 경로는 DSL/parser/render/test가 안정적으로 도는 것을 우선한다.
- GPU 학습 의존성(`bitsandbytes`, CUDA PyTorch, wandb, CLIP/VLM 관련 패키지)은 RunPod 환경 중심으로 관리한다.
- `pyproject.toml`은 가능하면 `core`, `train`, `eval`, `dev` 그룹으로 나눠 macOS 로컬 설치 실패를 줄인다.
- `.gitignore`에는 `data/raw/`, `data/processed/`, `wandb/`, `checkpoints/`를 포함한다.

---

## 4. 리포지토리 구조

```
pixellm/
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .python-version
├── src/
│   ├── pixellm/
│   │   ├── __init__.py
│   │   ├── dsl.py              # Palette Index Grid parser/validator
│   │   ├── render.py           # DSL → PIL Image
│   │   ├── data/
│   │   │   ├── prepare.py      # 공개 데이터셋 → DSL 변환
│   │   │   └── prompts.py      # 프롬프트 템플릿
│   │   ├── train/
│   │   │   ├── sft.py          # SFT 스크립트
│   │   │   └── grpo.py         # GRPO 스크립트
│   │   ├── eval/
│   │   │   ├── metrics.py      # parse rate, palette, symmetry, CLIP
│   │   │   ├── vlm_judge.py    # VLM 기반 평가
│   │   │   └── run_eval.py     # 체크포인트 일괄 평가
│   │   ├── rewards/
│   │   │   ├── syntactic.py
│   │   │   ├── symmetry.py
│   │   │   ├── clip_reward.py
│   │   │   └── vlm_reward.py
│   │   └── serve/
│   │       └── api.py          # FastAPI 추론 서버
├── tests/
│   ├── test_dsl_roundtrip.py
│   └── test_rewards.py
├── data/
│   ├── raw/                    # .gitignore (원본 데이터셋 캐시)
│   ├── processed/              # .gitignore (변환된 학습 데이터)
│   └── eval_prompts.json       # 고정 평가 프롬프트 100개
├── scripts/
│   ├── runpod_setup.sh
│   └── prepare_data.py
├── frontend/                   # Phase 5에서 추가
├── docs/
│   ├── dsl_design.md           # DSL 선택 이유, alternatives 비교
│   ├── reward_design.md        # Reward 함수 진화 스토리
│   ├── related_work.md         # 선행 연구 정리
│   └── results/
│       ├── gallery/            # 에폭별 결과 이미지
│       └── metrics.md          # 정량 평가 표
└── wandb/                      # .gitignore
```

---

## 5. 단계별 실행 로드맵 (6-8주)

### Week 1 — 기반 구축

**목표**: 데이터 파이프라인과 DSL 유틸리티 완성. 첫 commit.

**Day 1 — 레포 + 환경 셋업**
- [x] GitHub 레포 생성 (`pixellm`)
- [ ] `uv init --python 3.12` + 의존성 설치
- [ ] README 1줄 + LICENSE + .gitignore
- [ ] **첫 commit push**

```bash
mkdir pixellm && cd pixellm
git init
uv init --python 3.12
uv add pillow numpy pydantic datasets
uv add --group train transformers trl peft accelerate bitsandbytes wandb
uv add --group eval transformers pillow numpy
uv add --dev pytest ruff
echo "# pixellm" > README.md
git add . && git commit -m "initial commit"
```

**Day 2 — DSL 구현 + 라운드트립 검증**
- [ ] 현재 flat `src/` 코드를 계획 구조인 `src/pixellm/` 패키지로 재배치
- [ ] `src/pixellm/dsl.py`: 태그 DSL 파서/직렬화기, 내부 Pydantic 모델, 밸리데이터
- [ ] `src/pixellm/render.py`: DSL → PIL Image
- [ ] `tests/test_dsl_roundtrip.py`: 전처리 후 canonical 16×16 RGBA 이미지 → DSL → 이미지 픽셀 정확도 95%+

**Critical path**: 이 라운드트립이 정확해야 이후 모든 단계가 의미 있음.

**Day 3-4 — 데이터 파이프라인**
- [ ] `m1guelpf/nouns` 다운로드: `image` + `text`, CC0, SFT v1 주 데이터셋
- [ ] `Sc077y/pixel-art-synthetic-10k` 다운로드: `id` + `prompt` + `grid` + `preview`, MIT, 보조/평가 후보
- [ ] `scripts/prepare_data.py`: 이미지 → 16×16 resize → alpha threshold 적용 → 비투명 7색 양자화 → 태그 DSL 변환
- [ ] 최종 JSONL 생성: `{caption, prompt, dsl, source, source_id, category, view, license}` 형식 2,000-3,000 샘플
- [ ] SFT v1 데이터는 `category="character"`, `view="front"` 중심으로 구성
- [ ] 샘플 시각 검수: 랜덤 50개 뽑아 원본 vs 재렌더링 비교

**Day 5-6 — 평가 메트릭 기본**
- [ ] `src/pixellm/eval/metrics.py`:
  - `parse_rate(outputs)` — DSL 파싱 성공률
  - `palette_constraint_score(dsl)` — 선언 팔레트 vs 사용 색상 일치
  - `non_empty_score(dsl)` — 최소 픽셀 수
  - `symmetry_score(dsl)` — 좌우 대칭
  - `connected_component_score(dsl)` — 픽셀 응집도
- [ ] `data/eval_prompts.json`: 평가용 고정 100개 프롬프트

**Day 7 — 문서 + Week 1 마무리**
- [ ] `docs/dsl_design.md`: DSL 포맷 선택 이유, 고려한 alternatives
- [ ] README 업데이트: 프로젝트 개요, 아키텍처 개략도
- [ ] Week 1 성과 commit

**Week 1 성공 기준:**
- ✅ 레포 public, 2,000+ 학습 샘플 JSONL 완성
- ✅ DSL 라운드트립 테스트 통과 (95%+ 픽셀 정확도)
- ✅ 5개 평가 메트릭 함수 작동

---

### Week 2 — SFT v1

**목표**: Qwen2.5-Coder-3B가 valid DSL을 형식대로 출력.

**Day 1-2 — RunPod 셋업**
- [ ] RTX A6000 48GB 인스턴스
- [ ] Network Volume 50GB
- [ ] `scripts/runpod_setup.sh`: uv sync, HF 캐시 설정, wandb login

**Day 3-5 — SFT 학습**
- [ ] `src/pixellm/train/sft.py` 구현
- [ ] 하이퍼파라미터: QLoRA r=16, lr=2e-4, epoch=3, batch=2×8(accum)
- [ ] Qwen tokenizer의 `apply_chat_template()` 사용
- [ ] SFT collator에서 raw `{prompt, dsl}`를 messages로 변환:
  ```
  [{"role": "user", "content": "Draw 16x16 pixel art as Palette Index Grid DSL: {prompt}"},
   {"role": "assistant", "content": "{dsl}"}]
  ```
- [ ] loss는 assistant completion token에만 적용
- [ ] Wandb logging: loss, learning rate, 매 200 step마다 고정 프롬프트 5개 샘플 이미지

**Day 6-7 — 평가 + 분석**
- [ ] 100개 평가 프롬프트로 parse rate, palette, symmetry 측정
- [ ] base vs SFT v1 비교 스크린샷 갤러리
- [ ] 실패 샘플 분석 → `docs/sft_v1_analysis.md`

**Week 2 성공 기준:**
- ✅ Parse rate **70%+**
- ✅ "뭔가 형체가 보임" 수준의 출력 (본인 눈 검수)
- ✅ Base 대비 극적 차이 스크린샷 확보

**실패 대응:**
- Parse rate 낮음 → chat template 재확인, max_seq_length 점검
- 출력이 noise → learning rate 낮추기 (1e-4), epoch 늘리기
- OOM → batch 1, accum 16으로

---

### Week 3 — 평가 시스템 + GRPO 준비

**Day 1-3 — CLIP 평가 자동화**
- [ ] `src/pixellm/eval/clip_eval.py`: `openai/clip-vit-base-patch32` 기반
- [ ] `src/pixellm/eval/run_eval.py`: 체크포인트 → 100 프롬프트 × 4 샘플 → 전체 메트릭
- [ ] 결과 JSON 저장 + `docs/results/metrics.md` 표 생성

**Day 4-5 — Reward 함수 구현**
- [ ] `src/pixellm/rewards/`: 각 reward 모듈화
- [ ] `tests/test_rewards.py`: 엣지 케이스 (empty, invalid, perfect)
- [ ] Reward Phase 1 조합 테스트 (수작업 샘플로 sanity check)

**Day 6-7 — GRPO 파일럿**
- [ ] `src/pixellm/train/grpo.py` 구현 (TRL GRPOTrainer)
- [ ] Phase 1 가중치로 200 step 파일럿
- [ ] Reward 분포, KL, 출력 길이 wandb 모니터링
- [ ] **매 10 step 이미지 로깅** (고정 프롬프트 5개) ← 핵심

**Week 3 성공 기준:**
- ✅ 정량 평가 표 v1 완성 (base / SFT 비교)
- ✅ GRPO 파이프라인 작동 (200 step 에러 없이 완주)
- ✅ Reward 분포 group variance > 0 (학습 가능 상태)

---

### Week 4 — GRPO 본격 (Reward v1 → v2)

**Day 1-3 — Phase 1 학습 (Syntax + Symmetry)**
- [ ] 1,000-2,000 step 학습
- [ ] Parse rate 90%+ 달성 목표
- [ ] Reward hacking 패턴 모니터링 (출력 길이 급증, 특정 문자 반복 등)

**Day 4-7 — Phase 2 학습 (CLIP 추가)**
- [ ] Reward 가중치 전환 (Phase 2 설정)
- [ ] CLIP score 개선 추적
- [ ] 학습 전/후 A/B 갤러리 생성

**디버깅 체크리스트:**
- Mode collapse (출력이 한 가지 패턴) → `kl_coeff` 올리기
- Reward variance = 0 → group size 키우기, 프롬프트 다양성 점검
- CLIP score 개선되지만 육안으로 이상 → reward hacking 의심

**Week 4 성공 기준:**
- ✅ Parse rate **95%+**
- ✅ CLIP score SFT 대비 **+0.05 이상**
- ✅ Base / SFT / GRPO v1 / GRPO v2 4-way 비교 갤러리

---

### Week 5 — 프론트엔드 + 데모

**Day 1-3 — 추론 서버**
- [ ] `src/pixellm/serve/api.py`: FastAPI
  - `POST /generate` — prompt → DSL + rendered PNG
  - `GET /models` — 현재 로드된 체크포인트 목록
  - 기본은 단일 활성 모델 로드, base / SFT / GRPO 비교는 캐시된 결과 우선 사용
- [ ] vLLM 통합 검토 (선택, 복잡도 높으면 생략)

**Day 4-7 — 프론트엔드**
- [ ] Streamlit 기본 구현
- [ ] Next.js는 포트폴리오 개선 여력이 있을 때 향후 전환
- [ ] **메인 화면**: 프롬프트 입력 → 활성 모델 생성 → 렌더링 + 점수
- [ ] **Evolution Gallery**: 학습 단계별 같은 프롬프트 결과 비교
- [ ] **Code View**: 생성된 DSL 코드 표시 ("LLM이 코드로 그림을 그린다" 컨셉 강조)

**Week 5 성공 기준:**
- ✅ 로컬 데모 작동 (프롬프트 입력 → 10초 내 단일 활성 모델 결과)
- ✅ 캐시된 base / SFT / GRPO 3-way comparison 표시
- ✅ 30초 데모 영상 녹화 가능 상태

---

### Week 6 — 마무리 + 문서

**Day 1-3 — README + 문서 완성**
- [ ] README 구조:
  - 데모 GIF 30초
  - What / Why
  - Results 표 (4개 모델 × 5개 메트릭)
  - Architecture 다이어그램
  - Reward function design story
  - Related Work (Chat2SVG, StarVector, SVGen, Code2World)
  - Limitations & Future Work
- [ ] `docs/reward_design.md` 완성
- [ ] `docs/related_work.md`: 최소 3편 정리

**Day 4-5 — 데모 영상**
- [ ] 30-60초 영상:
  - 0-10s: 문제 정의
  - 10-30s: 웹 데모 시연
  - 30-50s: 학습 진화 비교
  - 50-60s: 메트릭 요약
- [ ] GitHub README, 블로그, 면접 자료에 활용

**Day 6-7 — 블로그 초안 (선택)**
- [ ] 한글/영문 병기 권장
- [ ] Dev.to 또는 본인 블로그

**Week 6 성공 기준:**
- ✅ README가 GitHub 첫 인상만으로 프로젝트 이해 가능 수준
- ✅ 데모 영상 준비 완료
- ✅ **응시 시작 가능** 상태

---

### Week 7-8 (선택) — 강화

- Reward Phase 3 (VLM judge) 추가
- SmolLM3-3B로 비교 실험
- 32×32 확장 시도
- 애니메이션 스프라이트시트 (4프레임)

이 단계는 면접 일정 여유에 따라 선택. **Week 6 끝에 이미 강한 포트폴리오.**

---

## 6. 정량 평가 계획 (최종 결과물)

### 평가 표 템플릿

| Model | Parse Rate | Palette OK | Symmetry | CLIP Score | VLM Score | Win Rate vs Base |
|-------|-----------|------------|----------|-----------|-----------|------------------|
| Base (Qwen2.5-Coder-3B) | ~5% | — | — | ~0.18 | — | — |
| SFT v1 | 87% | 92% | 0.74 | 0.22 | — | 78% |
| SFT + GRPO (Phase 1) | 95% | 98% | 0.81 | 0.23 | — | 81% |
| SFT + GRPO (Phase 2, +CLIP) | 93% | 95% | 0.79 | 0.31 | — | 89% |
| SFT + GRPO (Phase 3, +VLM) | 93% | 95% | 0.79 | 0.30 | 0.72 | 92% |

*(숫자는 가상 목표치 — 실제 학습 후 업데이트)*

### 시각 자료
- Evolution Gallery: 10개 프롬프트 × 4개 체크포인트 grid
- Reward hacking 사례 2-3개 (정직한 자기 평가 — 면접에서 좋은 신호)
- 베스트/워스트 샘플 페이지

---

## 7. 선행 연구 (면접 필수 사전 조사)

**최소 3편 반드시 읽기:**
1. **Chat2SVG** — 텍스트에서 SVG 생성, DSL 설계 참고
2. **StarVector** — 이미지/텍스트 → SVG 코드, multi-modal training
3. **SVGen / SVGThinker** — SVG 코드 생성 최적화

**GRPO/RLVR 배경:**
4. **DeepSeek-R1 paper** — GRPO 원조, RLVR 접근
5. **DAPO** — GRPO의 안정성 개선 (clip-higher, dynamic sampling 등)

**면접 예상 질문:**
- "이 분야 선행 연구 뭐 보셨어요?" — **무조건 나옴**. 답변 준비 필수
- "왜 픽셀아트를 택했나요?" — verifiable visual output, fine-tuning 경험 쌓기 좋음
- "왜 JSON이 아니라 Palette Index Grid인가요?" — 토큰 효율, 패턴 학습
- "Reward hacking 사례 있었나요?" — 정직하게 한두 개 공유
- "GRPO vs DPO 언제 뭐 써야 하나요?" — 검증 가능 도메인이면 GRPO, 선호 페어면 DPO

---

## 8. 병행 작업 (프로젝트 외)

### 응시는 Week 3-4부터 시작
프로젝트 완성을 기다리지 않습니다. 현재 이력서도 충분히 응시 가능한 수준입니다. 프로젝트는 면접까지의 기간 동안 완성되면 됩니다.

### 타겟 회사 (메모리 기반)
- **1순위**: Naver, Kakao, Samsung Research
- **2순위**: 한화시스템, 마키나락스, 올거나이즈, 포티투마루, 넥서스AI, 라피치, CJ올리브영, BCG X

### 이력서 업데이트 타이밍
- Week 4 (SFT + GRPO Phase 2 완료) 시점에 "Open-source LLM fine-tuning (QLoRA), GRPO" 프로젝트 한 줄 추가
- Week 6 (README + 데모 완성) 시점에 포트폴리오 링크 포함

---

## 9. 실행 원칙

1. **이번 주 commit 무조건**: 빈 README + 빈 dsl.py라도 push. 메모리 행동 패턴 (과도한 계획) 깨기
2. **Week별 성공 기준 명시 + 체크**: 시작일에 쓰고 종료일에 체크
3. **막히면 즉시 질문**: 혼자 끝없이 파지 않기. 특히 RunPod 셋업, chat template, GRPO reward 디버깅
4. **완성 우선 > 완벽 우선**: Week 3 SFT 완성 시점에 이미 응시 가능. GRPO는 보너스
5. **결과 스크린샷 매일 하나**: 학습 진행 스크린샷 모으면 README/블로그/영상 소스

---

## 10. 다음 단계 (본 프로젝트 완료 후)

**프로젝트 2: Math Reasoning + GRPO/RLVR**
- 본 프로젝트의 GRPO 경험을 **완전 verifiable** 도메인으로 확장
- SymPy 기반 단계별 풀이 검증
- DeepSeek-R1 축소 재현
- 평가 완전 깔끔 (GSM8K, MATH — 표준 벤치마크)

본 프로젝트에서 쌓은 GRPO 하이퍼파라미터 감각, reward 설계 경험, TRL 친숙도가 가속 연료가 됩니다.

---

## 부록 A: Day 1 즉시 실행 체크리스트

```bash
# 1. 레포 생성 (GitHub)
# 2. 로컬 셋업
mkdir pixellm && cd pixellm
git init
uv init --python 3.12

# 3. 의존성
uv add pillow numpy pydantic datasets
uv add --group train transformers trl peft accelerate bitsandbytes wandb
uv add --group eval transformers pillow numpy
uv add --dev pytest ruff

# 4. 최소 파일
cat > README.md << 'EOF'
# pixellm

Teaching small LLMs to draw pixel art via code generation, progressing from SFT to GRPO with multi-component rewards.

## Status
Week 1 — Foundation (in progress)

## Stack
Qwen2.5-Coder-3B-Instruct · TRL · QLoRA · GRPO · Palette Index Grid DSL

EOF

cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.venv/
data/raw/
data/processed/
wandb/
checkpoints/
.env
EOF

echo "3.12" > .python-version

# 5. 첫 commit
git add .
git commit -m "initial commit: project scaffolding"
git remote add origin git@github.com:pyrevine/pixellm.git
git push -u origin main
```

---

## 부록 B: 변경 이력 및 의사결정 기록

| 결정 사항 | 선택 | 검토한 대안 | 결정 근거 |
|----------|------|-----------|----------|
| DSL 포맷 | Palette Index Grid | ASCII, JSON, RLE | 토큰 효율, 패턴 학습 유리 |
| Base Model | Qwen2.5-Coder-3B | SmolLM3-3B, Qwen3-4B | 코드 특화, 호환성 안정 |
| 학습 순서 | SFT → GRPO | GRPO only, DPO | Cold-start 안정성 |
| Reward 활성화 | 단계별 | 한 번에 모두 | Reward hacking 방지 |
| 크기 | 16×16 | 32×32, 64×64 | 토큰 효율, 빠른 iteration |
| 팔레트 | 0=transparent, 1-7 비투명 색상 | 16색, 1-8 비투명 색상 | 학습 안정성, reward 검증 단순화 |
| 초기 범위 | 정면 캐릭터 1 카테고리 | 다카테고리 | 대칭성 보상 조건 충족 |
| SFT 데이터 포맷 | raw JSONL + 학습 시 chat template 적용 | 사전 템플릿 문자열 저장 | Qwen tokenizer와 추론 포맷 일치 |
| 프론트엔드 | Streamlit 우선 | Next.js 우선 | 구현 속도와 데모 완성 우선 |

---

**마지막 체크: 이 문서를 덮고, 이번 주 안에 첫 commit push.**
