---
plan_id: 014
version: 2.1 (simplification — §2.2 Out-of-scope 의 plan-004/006/012 재사용 negative blacklist 7 row → 1 row (positive "from-scratch 새 module" 통합). §1.5 마지막 bullet 동기화. §4~§8 STAGE-별 코드 spec 은 c3~c7 commit 에서 채움)
date: 2026-05-14 (Asia/Seoul)
status: spec
based_on:
  - 012
followed_by: []
scope: plan-012 (codebook bake-off + hybrid) 의 5-fold OOF 0.6350 plateau 의 root cause 는 *plan-004 selector + plan-006 F0 의 재사용 강박* — 이 명제를 **premise 로 채택** (본 plan 이 *검증* 하지 않음). 그 premise 위에서 plan-012 paradigm (codebook + classifier + regression hybrid) 을 **재사용 없이 from-scratch 재실험**. 즉 본 plan = "*제대로 된* plan-012 의 재실행" 이지 "재사용이 원인인가 falsify" 가 아님. baseline reproduce 없음 — plan-012 measured 5-fold OOF 0.6350 을 외부 reference 로만 사용. 본 파일은 *narrative 박제만* — 구체 spec 은 추후 채움.
exp_ids: []
lb_score: null
---

# plan-014 v1 — plan-012 Paradigm 의 From-Scratch 재실험 (premise: 재사용 강박, draft)

## §0. 한 줄 목적

> **F0 단일 선형 공식 (plan-006 `frenet_par120_perp_neg020`) 만으로 hit@1cm = 64%, hit@1.5cm = 84%.** 즉 84% 의 sample 은 F0 근방 *1.5cm 안에* 있고, **그 중 약 20%** (= 84% − 64%) 만 1cm zone 밖에 머무름. 본 plan 의 task essence = **그 20% 의 sample 을 평균 0.5cm 정도 적절한 방향으로 이동시켜 hit@1cm zone 으로 끌어당기기**.
>
> residual 을 MLP 로 *직접 회귀* (plan-005~007) 는 어렵다는 것이 실험으로 입증됨 → 점수 예측 (classification) 으로 풀되, **방향 후보 = Frenet local frame 위 7 방향** (±t / ±n / ±b / center, 또는 variant). 이게 plan-012 의 *codebook + classifier + regression hybrid* paradigm 의 본질.
>
> 본 plan = 그 paradigm 의 *제대로 된* 재실험. plan-012 가 실제로 한 것은 paradigm shift 가 아니라 **plan-004 의 `CandidateAttentionGRUSelector` 위에 reg_head 만 얹은 minimal patch** 였고, candidate-attention 부분이 본 task ("20% 끌어당김") 와 무관한 채로 logit 을 만드는 mismatch 가 paradigm 의 진짜 잠재력을 가렸다 (§1.2 사망 진단).
>
> **plan-004 참조 범위 (★ 본 plan 의 narrative anchor)** — 다음 2가지만 가져옴:
> - **input feature 가공 방식** (시계열 9-dim × 6-step 같은 input representation 의 *형식·전처리*)
> - **F0 sample cover 입증** (단일 선형 공식이 64%/84% cover — plan-006 LB 0.6692 의 hard evidence)
>
> **위 2가지 외 = 전부 새로 build** (`selector.py` import 없음). 본 plan 의 encoder + head 는 "**시계열 → 7 방향 점수 + 0.5cm 이동 offset**" task 에 직접 맞춘 새 module (§0.5 C1).
>
> 비교 baseline = plan-012.results.md 의 measured 5-fold OOF 0.6350 (외부 reference). 본 plan 의 OOF 가 §Target band 의 어느 위치에 들어가는지가 paradigm 의 진짜 잠재력 측정.
>
> **LB 제출 정책**: 미정 (narrative draft 단계).

---

## §0.5 Quick Reference

### 본 plan 의 task essence — "F0 64% cover + 남은 20% 끌어당김" (★ narrative anchor)

- **F0 단일 선형 공식** (plan-006 `frenet_par120_perp_neg020`): hit@1cm = **0.6320** (plan-012 G0 측정), hit@1.5cm = **0.8033** — *84% 가 F0 근방 1.5cm 안*.
- **남은 20%** (= 84% − 64%): F0 근방 *1.5cm 안* 이지만 *1cm zone 밖*. 평균 0.5cm 정도 적절한 방향으로 이동시키면 hit@1cm 진입.
- **residual 직접 회귀의 사망 진단** (plan-005~007): residual *vector regression* 은 매우 어렵다는 것이 실험으로 입증됨 → 본 plan = residual *direction* 만 점수 예측 (classification) 으로 풀고, magnitude 는 anchor scale (~0.5cm) prior + small offset head 로 처리.
- **방향 후보 = Frenet local frame 7 방향** (±t / ±n / ±b / center 또는 variant): trajectory-aligned 방향 분리 = "어느 방향으로 0.5cm 이동" task 와 직관 일치.
- → 본 plan = 이 task essence 에 직접 fit 한 encoder + classifier + regression hybrid 의 from-scratch 재실험 (= plan-012 paradigm 의 *제대로 된* 실행).

### plan-004 참조 범위 (★ 본 plan 의 narrative anchor) — 2가지만

- **(a) input feature 가공 방식** — 시계열 9-dim × 6-step 의 input representation 형식·전처리 (e.g. `selector.make_seq_features`). 검증된 데이터 표현이므로 reuse.
- **(b) F0 sample cover 입증** — 단일 선형 공식 64%/84% cover 의 *측정 결과* (= plan-006 LB 0.6692). 본 plan 의 task essence 가 정당화되는 hard evidence.

**위 2가지 외 = 전부 새로 build** — `selector.py` import 없음, 본 plan 의 새 module 안에서 from-scratch. 시계열 모듈 (GRU 등) 도 새 module 안에서 직접 생성 (`nn.GRU(...)` 등 표준 layer 사용은 OK, 단 plan-004 모듈 import 는 X).

### 본 plan 의 premise (← task essence 의 implied corollary, 검증 대상 아님)

- **Premise**: plan-012 의 0.6350 plateau 의 root cause = "plan-004 candidate-attention selector 위에 reg_head 만 얹은 minimal patch 가 본 plan 의 task essence ('20% 끌어당김') 와 mismatch" — task essence 에 맞는 *제대로 된 design* 으로 paradigm 잠재력은 0.66+ 까지 살아있음.
- 본 plan 은 premise 를 *검증* 하지 않는다. task essence 에 직접 맞춘 design 으로 paradigm 의 *measured 잠재력* 만 박제.
- premise 가 *틀렸을 경우* 의 표식 = §Target negative band — "premise 오류" vs "paradigm 자체 한계" 분리는 plan-013 join interpretation 으로만 가능 (§1.4).

### plan-012 가 "제대로" 가 아니었던 이유 (재사용 7 증상 → premise 의 근거)

plan-012 ring_classifier.py 는 paradigm shift 라고 self-label 했지만 실제 코드는 다음 *minimal patch*:

```
self.scorer  = base.CandidateAttentionGRUSelector(...)   # ← plan-004 의 27-way selector 그대로
self.reg_head = nn.Sequential(...)                       # ← 위에 작은 MLP 추가
F0 = f0_predict_frenet_par120_perp_neg020(...)           # ← plan-006 numpy 함수 그대로 (no grad)
anchor radius = 0.005m                                   # ← plan-004/006 era scale hardcode
```

7 observable failure mode → 단일 root cause:

| failure mode | 재사용 강박과의 인과 |
|---|---|
| F4 candidate-attention inductive-bias mismatch ★ v1.2 재정정 (+ v1.3 task-essence link) | **본 plan task = "20% sample 0.5cm 끌어당김"** (§task essence) ≠ **plan-004 task = "sample-별 27 후보 비교"**. 후자에 fit 된 모듈을 전자에 끼운 것이 F4 의 본질. 구조 상세: `CandidateAttentionGRUSelector` ([selector.py:697-726](src/pb_0_6822/selector.py#L697-L726)) 는 **2 부분** = (a) `self.gru` 시계열 인코더 + (b) `self.query`/`ctx_norm`/`event_norm`/`head` candidate-attention. (a) GRU 는 *task-neutral 시계열 처리 표준 도구* — 어떤 task 와도 묶이지 않음, 재사용 죄 아님. (b) candidate-attention 만 plan-004 task 에 fit. ring_classifier.py 의 *사용 패턴* = **두 갈래**: regression path 는 `self.scorer.gru(seq)` 로 GRU 만 떼서 씀 (= 정당), classifier path 는 `self.scorer(seq, cand_feat)` 통째 호출 → candidate-attention 까지 다 씀. plan-012 의 `cand_feat` 는 codebook *fixed* 7 anchor (sample-invariant) → sample-conditional query 의 효용 약함 = (b) 부분이 mismatch. (★ v1.0 "두 head 공유" / v1.1 "모듈 통째 mismatch" 폐기 — GRU 자체는 죄 없음. ★ v1.3: 본 plan task essence 와의 mismatch 가 root expression.) |
| F1 DCM collapse | F4 의 직접 결과 — encoder 가 plan-012 task 에 맞는 mode 분류 신호를 제공 못 함 → classifier head 가 신호 없이 학습 = safe minimum (mode 0 center) 수렴 |
| F7 frozen F0 path | plan-006 numpy F0 재사용 → gradient 없음, F0 학습 불가 |
| F3 F0 trivial dominance | F7 의 결과 — F0 hit sample 63% 가 학습 signal 에 무의미하게 남음 |
| F2 anchor scale mismatch | 재사용 시 plan-004/006 의 작은 scale (0.005m) 답습, task fit 재검토 없음 |
| F5 hard label CE noise | F2 의 결과 — anchor 가 hit zone 안에 갇혀 argmin label = noise |
| F6 codebook geometry uniformity | F1 의 결과 — encoder 가 어차피 center 만 고르니 anchor 위치 무관 |

→ 7 증상 = 1 root cause. 본 plan = 그 root cause 를 *제거한* 상태에서 paradigm 잠재력을 *측정* (= falsify 아님; 재설계 후 reading).

### 본 plan 의 single-path 설계 — plan-012 v3 (head-to-head 아님)

본 plan 의 실험 path = **1개** (= plan-012 paradigm 의 *제대로 된* 재실행):

- **plan-012 paradigm 유지** (= 본 plan 이 *변경하지 않는* 것):
  - codebook + classifier + regression hybrid (high-level paradigm idea)
  - 3D anchor / Frenet basis / hit@1cm metric / 5-fold OOF 평가 protocol
  - target = plan-012 와 동일 (5-fold OOF ≥ 0.66)

- **재사용 끊기 — 4 컴포넌트 from-scratch** (= 본 plan 이 *교체하는* 것):
  - **C1 encoder**: "시계열 → 7 방향 점수 + 0.5cm 이동 offset" task 에 직접 fit 한 from-scratch encoder. shared encoder + 2 head (classifier / regression) 가 자연스러운 default — encoder 개수는 §2 결정. 시계열 모듈은 본 plan 새 module 안에서 직접 build (e.g. `nn.GRU` / Transformer / 1D-CNN 중 §2 결정).
  - **C2 F0**: F0 coefficient 를 learnable parameter 로 (= plan-007 Step 4 통합) — F3/F7 inversion. plan-006 numpy `frenet_par120_perp_neg020` 재사용 폐기.
  - **C3 anchor**: anchor radius 를 task-fit scale 로 재정의 (hit radius 0.01m 동급 또는 학습 가능) — F2 inversion. plan-004/006 era scale 0.005m 답습 폐기.
  - **C4 label**: hard label CE → distance-weighted soft label — F5 inversion.

- 4 컴포넌트 모두 *동시에* 끊는다 (= "재사용 강박" 이라는 단일 root cause 의 totality 를 한 번에 제거). lever-by-lever ablation 아님. 컴포넌트별 attribution = plan-015 의 후속 과제.

- **A baseline (plan-012 minimal-patch reproduce) 별도 실행 안 함**. plan-012.results.md 의 GPU rerun 5-fold OOF 0.6350 (anchor 0.6344 / best 0.6350) 를 외부 reference 로 그대로 사용. 이유: (a) plan-012 GPU rerun 이 이미 spec-faithful (= epochs=50/batch=256/patience=5) 로 박제됨, (b) 재실행 비용 vs 정보 가치 trade-off — 동일 spec 재실행으로 얻을 정보 ≈ noise, (c) 본 plan 의 *측정 대상은 paradigm 의 잠재력* 이지 gap 자체가 아님.

### Target (judgement criteria) — 절대값 기반 (gap 기반 아님)

baseline reference = plan-012 measured 5-fold OOF **0.6350** (GPU rerun, plan-012.results.md `final_oof_5fold_hit_1cm`).

- **OOF ≥ 0.66 (= plan-012 의 원래 target)** ★ paradigm 부활 band: premise 가 옳다는 outcome-level 강한 신호. plan-015 = 본 plan 의 polish + LB 제출.
- **0.6350 + 0.010 ≤ OOF < 0.66** (= 0.6450~0.6600) — partial 회복 band: premise 약하게 지지 + paradigm 자체도 부분 한계. plan-013 의 corrector path 와 비교 필요. plan-015 = corrector + 본 plan 의 hybrid (plan-013 Candidate C 변형) 후보.
- **OOF < 0.6350 + 0.010** (= < 0.6450) — negative band: 재사용 끊어도 paradigm 한계. premise 자체 의심. plan-013 의 paradigm 폐기 정당화 + 더 deep path-pivot (plan-012 Candidate A KNN/GP/Diffusion 등 `notes/new-ideas.md` 후보).

### G-gates (정량 spec @ §3.4)

- G0: preflight artifact 산출 — F0 init reproduce (±0.005) + anchor 0.01m + soft label entropy ≥ 0.5 nat + plan-012 0.6350 ref confirm [TODO]
- G1: `plan014_paradigm.py` 새 module + 재사용 끊김 4가지 (selector import 0 / F0 grad ≠ 0 / anchor 0.01m / soft label entropy ≥ 0.5 nat) + smoke train pass [TODO]
- G2: 1-fold OOF (fold=0) ≥ 0.6511 (informational; <0.60 → severe) [TODO]
- G3: 5-fold concat OOF + §Target band 분류 박제 + submission [TODO]
- G_final: results.md 신규 + registry append + frontmatter sync + plan-015 후보 (band 별 분기) [TODO]

### Commit chain

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-014-plan012-failure-inversion.md` v0 narrative draft (검증 frame, head-to-head A vs B) | [DONE] 4657ff7 + 2a0f755 |
| c1.1 | docs | v1 narrative re-frame — "재사용 원인 검증" → "재사용 전제 + plan-012 재실험" (★ premise 채택, head-to-head 폐기, baseline reproduce 폐기, target 절대값 기반) | [DONE] c7cf5c8 |
| c1.2 | docs | v1.1 spot-fix — F4 진단 정확화: "shared encoder bottleneck (두 head 공유)" → "encoder inductive-bias mismatch (plan-004 candidate-attention 모듈을 candidate 없는 task 에 끼움)". §0.5 7-mode 표 / C1 encoder bullet / §1.3 trap chain 동기화 | [DONE] 5e98d6d |
| c1.3 | docs | v1.2 spot-fix — F4 진단 2차 정확화 (code-grounded): `CandidateAttentionGRUSelector` 가 GRU [task-neutral] + candidate-attention [plan-004 fit] 의 2 부분 구조임을 selector.py:697-726 + ring_classifier.py:309/315 read 로 확인. 폐기 범위 = candidate-attention 부분만, GRU 본체는 reuse 정당. §0.5 7-mode F4 / C1 encoder bullet / §1.3 trap chain 모두 sync | [DONE] c7fa9c8 |
| c1.4 | docs | v1.3 narrative reframe — task essence ("F0 64% cover + 남은 20% 끌어당김") 를 §0/§0.5 의 narrative anchor 로 박제. plan-004 참조 범위 = (a) input 가공 + (b) F0 sample cover 입증 2가지로 축소 (`selector.py` import 폐기). §0 한 줄 목적 + §0.5 새 sub-section (task essence + plan-004 참조 범위 + premise corollary) + §0.5 7-mode F4 / C1 encoder bullet task-essence link + §1.2 사망 진단 (residual vector regression 어려움) + §1.5 정직성 원칙 (plan-004 참조 2가지만) sync | [DONE] 3a7a26c |
| c1.5 | docs | **v1.4 simplification — plan-004 negative blacklist 제거.** positive whitelist 2가지만 박제 + "위 2가지 외 = 전부 새로 build" 한 줄. §0 한 줄 목적 / §0.5 plan-004 참조 sub-section / §0.5 C1 encoder bullet 의 v1.x audit 표현 (candidate-attention head 폐기 상세, v1.0~v1.2 변천사) 제거. 7-mode 표 / §1.2~§1.3 trap chain 은 historical reference 로 보존 (사용자 v1.3 선택). | [DONE] ab50cce |
| c2 | docs | **v2 spec drop — §2 Scope + §3 Pre-registration 채움.** narrative draft → executable spec 졸업. §2.1 In-scope (single path config: E0b Frenet-7Way / shared biGRU / learnable F0 / anchor 0.01m / Gaussian soft label / Adam 50ep) + §2.2 Out-of-scope (selector.py import 0 등 12 항목) + §3.1 fold split (`hash(sample_id, salt='plan-014-v1') % 5` 재구현) + §3.2 metric + §3.3 The Config (단일 path 명세) + §3.4 G-gate 정량 (G0 preflight artifact / G1 재사용 끊김 4가지 + smoke / G2 ≥0.6511 1-fold / G3 5-fold band 분류 / G_final 3파일 sync) + §3.5 plan-012/013 reference. §0.5 G-gates stub → §3.4 link. frontmatter version 1.4→2 + status draft→spec. §4~§8 STAGE-별 코드 spec 은 c3~c7 stub 박제 | [DONE] ad051e2 |
| c2.1 | docs | **v2.1 simplification — §2.2 Out-of-scope negative blacklist 통합.** plan-004/006/012 module·weight 재사용 7 row → 1 row (positive "from-scratch 새 module" 한 줄). §1.5 마지막 bullet 의 redundant negative phrasing 정리. 보존 = §2.1 positive spec (anchor 0.01m / Gaussian soft) / §3.3 The Config / §3.4 G1 import check (measurable G-gate) / 7-mode 표 / §1.2~§1.3 trap chain (historical reference). frontmatter version 2→2.1 | [DONE] 0a3c317 |
| c3 | code+exp | STAGE 0 (G0) — `analysis/plan-014/preflight.py` + `preflight.json` 산출. F0 init reproduce + anchor scale 박제 + soft label entropy check + plan-012 0.6350 ref grep. spec @ §4 | [TODO] |
| c4 | code | STAGE 1 (G1) — `src/pb_0_6822/plan014_paradigm.py` 새 module: 4 컴포넌트 from-scratch (encoder biGRU / learnable F0 / anchor 0.01m / Gaussian soft label) + 7 컴포넌트 self-contained (anchor gen, Frenet basis, F0 layer, encoder, 2 head, hybrid_predict, loss). smoke test (`tests/test_plan014_smoke.py`) + 재사용 끊김 4가지 박제. spec @ §5 | [TODO] |
| c5 | exp | STAGE 2 (G2) — `analysis/plan-014/g2_1fold.py` + `g2_1fold.json` + `runs/baseline/plan014_g2_1fold/`. fold=0 학습 + hit@1cm 측정 + ≥0.6511 check (informational). spec @ §6 | [TODO] |
| c6 | exp | STAGE 3 (G3) — `analysis/plan-014/g3_5fold.py` + `g3_5fold.json` + `runs/baseline/plan014_g3_5fold/submission.csv`. 5-fold concat hit@1cm + §3.2 band classifier + submission 생성. spec @ §7 | [TODO] |
| c7 | docs+sync | STAGE 4 (G_final) — `plans/plan-014-plan012-failure-inversion.results.md` 신규 + `registry.csv` append + frontmatter sync (`lb_score`, `exp_ids`, `status: spec→completed`) + plan-013 join interpretation table 의 activated row 박제 + plan-015 후보 ≥ 3 박제 (band 별 분기). spec @ §8 | [TODO] |

---

## §1. 배경 / 동기 (narrative)

### §1.1 v0 → v1 narrative inversion — "검증" 이 아니라 "재설계"

v0 narrative (commit 4657ff7 + 2a0f755) 는 "재사용 강박이 plan-012 의 진짜 원인인가" 를 A (minimal-patch reproduce) vs B (from-scratch) head-to-head 로 falsify 하는 *검증 plan* 으로 설계됨. 본 v1 은 사용자 지시에 따라 그 frame 을 폐기:

| 축 | v0 (검증 frame, 폐기) | **v1 (재설계 frame, 본 plan)** |
|---|---|---|
| premise 위치 | hypothesis (검증 대상) | **assumed (검증 안 함)** |
| 실험 path | A reproduce + B redesign (2 path) | **B redesign 단독 (1 path)** |
| baseline | A reproduce (in-plan 측정) | **plan-012 measured 0.6350 외부 reference** |
| Target | B − A gap 기반 (≥ +0.020 / +0.005 / < +0.005) | **OOF 절대값 기반 (≥ 0.66 / 0.6450~0.66 / < 0.6450)** |
| negative 해석 | "재사용이 원인 아님" 의 evidence | **"premise 오류 OR paradigm 한계" (분리 불가, plan-013 join 필요)** |
| 실험 의도 | 가설 falsify | **paradigm 잠재력 측정** |

→ 본 plan 의 정직한 self-label = "*plan-012 가 self-label = paradigm shift 였으나 minimal-patch 였던 부분을, 사용자 지시에 따라 paradigm 동일 + 컴포넌트 from-scratch 로 다시 실행* 하는 plan-012 v3."

### §1.2 plan-012 의 사망 진단 — premise 의 근거 (★ falsification spec 아님)

> 본 절은 §0.5 premise 의 *근거* 박제. premise 자체는 본 plan 에서 검증되지 않음. 본 절을 통과한 *옳음* 은 plan-014 의 outcome (= §Target 의 positive band 진입) 으로만 outcome-level 지지될 수 있음.

plan-012 results.md = "paradigm reframe 은 F0 raw hit 위 +0.002~0.003 만 추가 — paradigm 자체의 limit 확인". 그러나 plan-012 의 코드를 보면 *paradigm* 과 *재사용 강박* 이 구별되지 않은 채 같이 limit 으로 묶임:

- plan-012 는 self-label = "paradigm reframe (residual regression → classification + hybrid)"
- 실제 코드 = "plan-004 selector + plan-006 F0 위에 hybrid head 만 얹은 minimal patch"
- → plan-012 가 measured limit = **"minimal patch 의 limit"** 일 뿐, **"paradigm 의 limit"** 은 아직 측정 안 됨

본 plan 의 핵심 진단 (premise 의 토대):

> **plan-012 의 7 failure mode 는 7개 독립 문제가 아니라 1개 root cause (재사용 강박) 의 7가지 증상.**

따라서:
- 7-lever ablation (plan-012 의 G2/G3) = 증상 치료 ≠ 원인 치료 → 모든 lever 가 marginal 인 것은 합당
- 본 plan = 원인 (재사용) 제거 후 paradigm 의 *제대로 된* 측정 (= falsification 아닌 *재실험*)
- **본 plan 의 task essence 자체** (§0.5 박제): F0 단일 선형 공식이 64% / 84% cover 라는 plan-006 측정 위에서, residual 의 *vector regression* 직접 회귀 (plan-005~007) 가 어렵다는 입증 → residual *direction* 만 classification 으로 풀고 magnitude 는 anchor scale prior + small offset head 로. 이게 plan-012 의 codebook + classifier + regression hybrid paradigm 의 본질이며, 본 plan = 그 task essence 에 직접 fit 한 design 으로 다시 실험.

### §1.3 재사용 강박의 trap chain (구조적 인과 — premise 의 mechanism)

```
"plan-004 selector reuse"  ─┐
                            │
                            ▶ candidate-attention inductive bias mismatch (F4, v1.2 재정정).
                              `CandidateAttentionGRUSelector` = (a) GRU 시계열 인코더 [task-
                              neutral, 죄 없음] + (b) candidate-attention head [plan-004
                              sample-별 27-candidate task 에 fit]. ring_classifier 가
                              classifier path 에서 `self.scorer(seq, cand_feat)` 통째 호출
                              → (b) 까지 같이 호출. plan-012 의 cand_feat = codebook fixed
                              7 anchor (sample-invariant) → sample-conditional cross-
                              attention 의 효용 약함 = (b) 부분이 mismatch.
                              (※ "두 head 공유"[v1.0] / "모듈 통째 mismatch"[v1.1] 둘 다
                              over-statement — GRU 본체는 task-neutral 시계열 도구.)
                            │
                            ▶ classifier head 가 줄 신호 없이 학습 = safe minimum
                              (mode 0 center) collapse (F1)
                            │
                            ▶ codebook geometry 가 결과에 무관 (F6)

"plan-006 numpy F0 reuse" ──┐
                            │
                            ▶ F0 gradient 없음 (F7)
                            │
                            ▶ F0 정확 prediction 이 hit 63% 에서 이미 답 = head 가 학습할 거리 없음
                            │
                            ▶ F0 hit 63% sample 학습 signal 무의미 (F3)
                            │
                            ▶ hard label = noise (F5)

"plan-004/006 era scale"  ──┐
                            │
                            ▶ anchor 0.005m 답습 → hit zone 내부 갇힘 (F2)
```

→ 7 mode 의 *원인이 1개* 라는 진단이 옳다면, "재사용 끊기" 단일 path 의 from-scratch 재실험으로 7 mode 가 동시에 풀릴 잠재력이 있어야 함. 본 plan 의 outcome (§Target band) 이 그 *outcome-level* 신호 — formal falsification 은 아니지만 paradigm 잠재력의 직접 측정.

### §1.4 plan-013 과의 path 분기 — join interpretation 으로 premise 분리

- **plan-013** (직전): paradigm 폐기 + plan-004 framework 회귀 → G2 0/3 axis FAIL, G1 0.6381 fallback submission
- **plan-014** (본 plan): paradigm 부활 시도 — 단 *재사용 강박만 끊고* paradigm 자체는 유지 (= plan-012 v3)

본 plan 단독으로는 "premise 오류" 와 "paradigm 자체 한계" 를 분리할 수 없음 (§0.5 의 negative band 해석 참고). 두 path 의 결과를 join 해야 분리 가능 (plan-013 LB 0.68 thresh 는 가정값; plan-014 band 는 §0.5 절대값 기준):

| plan-013 LB | plan-014 5-fold OOF (§Target band) | 결합 해석 |
|---|---|---|
| ≥ 0.68 | < 0.6450 (negative) | paradigm 폐기 정당화 — plan-004 framework path 가 정답 |
| < 0.68 | ≥ 0.66 (positive) | paradigm 부활 — from-scratch redesign 이 정답, premise 옳음 |
| ≥ 0.68 | ≥ 0.66 (positive) | 둘 다 작동 — plan-015 = 두 path 의 ensemble/stacking |
| < 0.68 | < 0.6450 (negative) | 둘 다 실패 — 더 deep path-pivot (plan-012 Candidate A KNN/GP/Diffusion, `notes/new-ideas.md`) |
| 임의 | 0.6450~0.66 (partial) | plan-013 corrector + 본 plan hybrid 의 합체 (plan-013 Candidate C 변형) — plan-015 default |

### §1.5 본 plan 의 정직성 원칙

- **재실험 frame 명시**: 본 plan 은 falsification test 가 아닌 *premise-기반 재실험*. premise (재사용=원인) 가 옳은지 본 plan 으로는 *증명 불가*; 옳다는 *가정 아래* paradigm 잠재력만 측정.
- **negative band 해석의 한계**: §Target negative band (OOF < 0.6450) 가 나와도 "premise 오류" 와 "paradigm 자체 한계" 분리 불가 — plan-013 join 필수 (§1.4).
- **컴포넌트별 attribution 회피**: 4 컴포넌트 (C1~C4) 동시 swap — 본 plan 의 outcome 으로는 *어느 컴포넌트가 결정적이었는지* 알 수 없음. 그 attribution = plan-015 후속 ablation.
- **lever 마진 줍기 회피**: plan-012 의 7-lever ablation = 증상 치료. 본 plan = 원인 1개 (재사용 totality) 제거. 컴포넌트별 toggle X.
- **외부 reference 만 사용**: plan-012 measured 0.6350 을 외부 reference 로만. baseline reproduce 없음 → 본 plan 의 *모든 compute 는 B path 에 집중*.
- **plan-004 참조 범위 = 2가지만 (★ v1.3 narrative anchor, §0.5 박제)**: (a) input feature 가공 방식, (b) F0 64%/84% sample cover 입증. 그 외 = 본 plan 의 새 module (`src/pb_0_6822/plan014_paradigm.py`) 안에서 from-scratch build (시계열 모듈도 표준 layer 직접 생성). 코드 (§2 spec 단계) + plan body 양쪽 모두 이 narrative 를 유지.

---

## §2. Scope (명시적)

> 본 절 = §0.5 의 task essence + 4 컴포넌트 from-scratch 의 *executable spec 수준 enumeration*. 모든 수치는 §0.5 narrative anchor 유지 + 그 외 underspec 은 권장 default 채택 (decision-note 박제).

### §2.1 In-scope (= 본 plan 의 single path config)

| 항목 | 값 | 근거 |
|---|---|---|
| paradigm | codebook + classifier + regression hybrid (= plan-012 paradigm 유지) | §0.5 "본 plan 이 *변경하지 않는* 것" |
| codebook | **E0b Frenet-Orthogonal-7Way** (±t̂ / ±n̂ / ±b̂ + center) | §0.5 "Frenet local frame 7 방향", 사용자 방법 1 |
| K | 7 (fixed) | §0 task essence "7 방향 점수" |
| Frenet basis origin | F0 prediction point | trajectory-aligned local frame |
| C1 encoder | **새 module-local 2-layer biGRU, hidden=128** (`nn.GRU` 표준 layer), shared encoder + 2 head | §0.5 "from-scratch encoder, shared + 2 head 자연스러운 default". decision-note: 시계열 모듈 선택 = GRU (plan-004 era 검증된 시계열 처리 도구, weight reuse X) |
| Classifier head | linear → 7 logit (= 7 anchor mode 점수) | §0 task essence "7 방향 점수" |
| Regression head | linear → 7 × 3D offset (per-mode 3D offset), bound ±0.005m (= reg_scale prior) | §0 task essence "0.5cm 이동 offset" |
| C2 F0 | **learnable linear coef** `(α_par, α_perp_t, α_perp_n)` = `nn.Parameter`, init = plan-006 `frenet_par120_perp_neg020` = (1.20, −0.20, 0.0) | §0.5 "F0 coefficient 를 learnable parameter 로 (= plan-007 Step 4 통합)" — F3/F7 inversion |
| C3 anchor radius | **0.01m fixed** (= hit zone scale) | §0.5 "hit radius 0.01m 동급 또는 학습 가능" — F2 inversion. decision-note: fixed scalar (학습 가능 옵션은 plan-015 ablation 후보) |
| C4 soft label | **Gaussian kernel** `w_k ∝ exp(−d_k² / (2σ²))`, `d_k = ‖F0 + a_k − y_true‖`, σ = 0.01m (= anchor radius 동일) | §0.5 "distance-weighted soft label" — F5 inversion. decision-note: kernel form = Gaussian (Boltzmann softmax 동등; σ = anchor scale 통일이 단순) |
| Inference | **soft blend** `prob = softmax(logits / τ)`, `hybrid_pred = F0_pred + Σ_k prob_k × (a_k + reg_offset_k)`, τ = 0.03 | plan-012 default (= G1 anchor τ) |
| Input pipeline | **9-dim × 6-step 시계열** (plan-004 `make_seq_features` 형식 — *형식만 reuse*, 본 module 내 직접 재구현) | §0.5 plan-004 참조 (a). decision-note: 함수 import 폐기, 새 module 안에서 동일 형식 직접 build |
| Frenet basis vectors | F0 prediction point 근방 trajectory tangent → t̂, principal normal → n̂, binormal → b̂ = t̂ × n̂ | plan-006 frame def reuse (산식 reference, 코드 import X) |
| Loss | `L = α × CE(logits, soft_label) + β × Huber(reg_offset_blended, y_true − F0 − Σ_k prob_k × a_k)`, (α=1.0, β=1.0) | decision-note: 두 head 동등 weight 시작; β tuning 은 plan-015 ablation 후보 |
| Validation | **5-fold OOF**, fold = `hash(sample_id, salt='plan-014-v1') % 5` (새 module 내 직접 재구현) | §0.5 G3 spec. decision-note: plan-004 `stable_fold_id` 와 fold 분할 결과 *다를 수 있음* — but plan-014 의 measurement 는 plan-014 scheme 내 self-consistent (외부 비교는 plan-012 results.md `final_oof_5fold_hit_1cm = 0.6350` reference 만) |
| Training | Adam, lr=1e-3, batch=256, epochs=50, early-stop patience=5 (val loss) | plan-012 spec reuse (조정 = plan-015 후보) |
| Multi-seed | **single seed (= 20260514)** | decision-note: single seed = single config single measurement. multi-seed 안정성은 plan-015 ablation 후보. partial/negative band 진입 시 plan-015 에서 seed 분산 측정 |
| Target | 5-fold OOF band (≥0.66 positive / 0.6450~0.66 partial / <0.6450 negative) | §0.5 §Target band 그대로 |
| New module path | `src/pb_0_6822/plan014_paradigm.py` | decision-note: plan-014-specific module. ring_classifier.py 와 별개 파일 (재사용 끊기 가시화 — `selector.py` import 없음을 module 단위로 보장) |

### §2.2 Out-of-scope (= 본 plan 이 *하지 않는* 것)

| 항목 | 이유 |
|---|---|
| plan-004/006/012 의 module·weight 재사용 일체 (`selector.py` 함수/클래스, `CandidateAttentionGRUSelector` 통째·부분, plan-004 pretrained GRU weight, plan-006 numpy `frenet_par120_perp_neg020`, `ring_classifier.py` 함수·helper) | §0.5 plan-004 참조 2가지 (a input 가공 / b F0 cover 입증) *외* totality 제거 — 본 plan = 새 module `src/pb_0_6822/plan014_paradigm.py` 안에서 from-scratch build (시계열 모듈도 `nn.GRU` 등 표준 layer 직접 생성). anchor 0.005m / hard label CE 답습 = §2.1 positive spec (C3 0.01m / C4 Gaussian σ=0.01m soft) 으로 implicit 박제 — F2/F4/F5/F7 inversion 일괄 |
| Corrector path / `boundary.py` / `corrector_redesign*` | plan-005~011 paradigm 분리 (= plan-013 도 회귀했지만 plan-014 와 path 분기) |
| 27 후보 physics candidate / `candidates_extended.py` | scope X (= plan-008 산출) |
| K density swap (K=5/9/13 등) | scope X. K=7 단일 — plan-012 G2 E2 의 K 변형은 본 plan 잠재력 측정 후 plan-015 ablation 후보 |
| Temperature scan / r=0 logit prior tuning | scope X (= plan-012 G2 E3/E8). τ=0.03 단일 |
| Boundary sample weighting | scope X (= plan-012 G3 E6) |
| TTA / multi-parse inference | plan-015 후보 |
| Ensemble (with plan-013 fallback or plan-012 ring) | plan-015 후보 (band 별 분기 — §0.5 §Target 박제) |
| 컴포넌트별 ablation (C1 alone, C2 alone, ...) | §1.5 "lever 마진 줍기 회피" — 4 컴포넌트 *동시* swap. attribution = plan-015 |
| Baseline reproduce (plan-012 minimal-patch 재실행) | §0.5 박제 — plan-012.results.md `final_oof_5fold_hit_1cm = 0.6350` 외부 reference 만 사용 |
| LB 제출 | 미정 (band 결과 따라 plan-015 결정 — §0.5 §Target 별 분기) |

---

## §3. 사전 등록 (Pre-registration)

### §3.1 Fold split

- **5-fold OOF**: `fold_id = stable_hash(sample_id, salt='plan-014-v1') % 5` (새 module 내 재구현; SHA256(f"{salt}::{sample_id}").digest() → int.from_bytes(... [:8]) % 5)
- **G2 (1-fold sanity)**: fold=0 only, N_val ≈ 2020 (전체 ≈ 10100 sample 가정)
- **G3 (final 5-fold)**: 5-fold concat (모든 sample 이 정확히 1번 val 등장)
- decision-note: plan-004 `stable_fold_id` 와 분할 *다를 수 있음* — 외부 비교 = plan-012 `0.6350` *외부 reference* 만 (절대값 기반 §Target band)

### §3.2 평가 metric

- **main metric**: `hit@1cm = mean(‖hybrid_pred - y_true‖₂ ≤ 0.01m)` (5-fold concat OOF, G3) / 1-fold (G2)
- `hybrid_pred = F0_pred + Σ_{k=0..6} prob_k × (a_k + reg_offset_k)`
  - `F0_pred = α_par × F0_par_vec + α_perp_t × F0_perp_t_vec + α_perp_n × F0_perp_n_vec` (3 learnable scalar)
  - `prob = softmax(logits / τ)`, τ = 0.03
  - `a_k` = Frenet local anchor (k=0 center → 0; k=1..6 → ±0.01m × ±t̂ / ±n̂ / ±b̂)
  - `reg_offset_k` = regression head 의 k-th mode 3D offset (bound ±0.005m via tanh × scale)
- **secondary**: `hit@1.5cm` (= §0 64/84 hypothesis 의 tail boundary)
- **diagnostic**: `directional_commit_magnitude = mean(‖hybrid_pred - F0_pred‖₂)` — F0 으로부터 평균 이탈 크기 (= dilution_collapse 진단 척도)
- **band classifier** (§0.5 박제):
  - OOF ≥ 0.66 → **positive band** (paradigm 부활)
  - 0.6450 ≤ OOF < 0.66 → **partial 회복 band**
  - OOF < 0.6450 → **negative band** (premise 의심)

### §3.3 The Config (= 본 plan single path 의 anchor 정의)

★ plan-014 의 single path 는 ablation 후보가 아닌 **fixed config** — 본 절은 그 config 의 *완전 명세*:

- **codebook**: E0b Frenet-Orthogonal-7Way (±t̂/±n̂/±b̂/center)
- **K**: 7
- **frame**: Frenet local frame @ F0 prediction point
- **C1 encoder**: shared 2-layer biGRU (hidden=128, input=9, output=256 via concat bidir), 새 module-local `nn.GRU` 표준 layer
  - input: 9-dim × 6-step (plan-004 `make_seq_features` 형식 — 본 module 내 직접 build)
  - output: 256-dim feature → 2 head (cls 7-logit, reg 7×3 offset)
- **C2 F0**: learnable `(α_par, α_perp_t, α_perp_n)`, init = (1.20, −0.20, 0.0), Adam grad enabled
- **C3 anchor radius**: 0.01m fixed scalar
- **C4 soft label**: Gaussian σ = 0.01m, `w_k = exp(−d_k² / (2σ²))`, normalized over k=0..6
- **Loss**: `L = CE(logits, soft_label) + Huber(reg_offset_blended, y_true − F0_pred − Σ_k prob_k × a_k)`, (α=1.0, β=1.0)
- **Inference τ**: 0.03 (soft blend)
- **Training**: Adam lr=1e-3, batch=256, epochs=50, patience=5 (val loss), seed=20260514

### §3.4 G-gate quantitative criteria (= §0.5 G-gate stub 의 정량화)

#### G0 — preflight + reference confirm

- **artifact**: `analysis/plan-014/preflight.json`
- **(a) F0 init reproduce check**: `α=(1.20, −0.20, 0.0)` 으로 init 한 모든 train sample 의 hit@1cm 측정 → plan-006 reference (0.6320) ± 0.005 일치
- **(b) anchor scale 박제**: radius=0.01m, ±t̂/±n̂/±b̂/center 7 anchor 의 Frenet local coord 값 (= ±0.01 단위 직교 set + 원점) 박제
- **(c) soft label kernel check**: σ=0.01m Gaussian 의 sample-별 entropy 평균 ≥ 0.5 nat (= label 이 hard one-hot 아님 verify)
- **(d) plan-012 reference confirm**: `plans/plan-012-frenet-ring-classification.results.md` 에서 `final_oof_5fold_hit_1cm` grep → 0.6350 박제
- **fail trigger**: 위 (a)~(d) 중 1개 이상 누락 / 수치 mismatch → `preflight_artifact_missing` severe

#### G1 — 구현 완료 + 재사용 끊김 검증

- **artifact**: `src/pb_0_6822/plan014_paradigm.py` (새 module) + smoke test (`tests/test_plan014_smoke.py`)
- **(a) smoke train**: 1-fold (fold=0) 1-epoch 학습 — no NaN, val_loss < initial val_loss
- **(b) 재사용 끊김 4가지 박제** (= 4 컴포넌트 from-scratch 검증):
  1. **encoder import check**: `import_graph(src/pb_0_6822/plan014_paradigm.py)` 결과에 `selector` / `ring_classifier` / `boundary` 0 매치
  2. **F0 learnable check**: `α_par.grad`, `α_perp_t.grad`, `α_perp_n.grad` 모두 ≠ None + ≠ 0 (1 backward 후)
  3. **anchor scale check**: anchor `‖a_1‖ = 0.01m` (NOT 0.005m), 6 non-center anchor 의 ‖·‖ = 0.01 ± 1e-6
  4. **soft label entropy check**: train 1 epoch 의 batch 평균 soft label entropy ≥ 0.5 nat
- **fail trigger**: 위 (a)~(b) 4가지 중 1개 이상 fail → `reuse_cut_violation` severe (= premise 위배)

#### G2 — 1-fold OOF sanity (informational)

- **artifact**: `analysis/plan-014/g2_1fold.json` + `runs/baseline/plan014_g2_1fold/`
- **criterion**: 1-fold OOF (fold=0) hit@1cm ≥ **0.6511** (= plan-012 1-fold winner 0.6411 + 0.010)
- **fold mapping 가정 (§0.5 박제)**: 1-fold 0.6511 ≈ 5-fold 0.6450~0.6550 (fold noise 1폭 가정) → G2 통과 = §Target partial 회복 band floor 진입 후보
- **fail trigger**: hit@1cm < 0.60 → `g2_severe_underperform` severe (= 구현 버그 의심; 0.60 = "F0 raw hit 0.632 보다 *낮음* = forward path 망가짐"의 conservative threshold). 0.60 ≤ x < 0.6511 → informational continue (G3 진행)

#### G3 — 5-fold concat OOF + band classification

- **artifact**: `analysis/plan-014/g3_5fold.json` + `runs/baseline/plan014_g3_5fold/submission.csv`
- **measurement**: 5-fold concat hit@1cm 박제 + §3.2 band classifier 적용
- **band-별 분기 (informational, fail trigger 없음)**:
  - positive (≥0.66): plan-015 = polish + LB 제출
  - partial (0.6450~0.66): plan-015 = corrector + 본 plan hybrid (plan-013 Candidate C 변형)
  - negative (<0.6450): plan-015 = deep path pivot (plan-012 Candidate A KNN/GP/Diffusion, `notes/new-ideas.md`)

#### G_final — synthesis + 3 파일 sync + plan-015 후보

- **artifact**: `plans/plan-014-plan012-failure-inversion.results.md` (신규) + `registry.csv` append + plan-014 frontmatter sync (`lb_score`, `exp_ids`, `status: completed`)
- **content**:
  - G0~G3 박제 결과 narrative
  - §Target band 분류 결과
  - plan-013 join interpretation table (§1.4) 의 어느 row 가 activated 됐는지 박제
  - plan-015 instruction (band 별 후속 후보 ≥ 3 박제)
- **fail trigger**: 3 파일 sync 누락 → `final_sync_missing` severe

### §3.5 Plan-012 / plan-013 reference

| measure | plan-012 (외부 ref) | plan-013 (외부 ref) | plan-014 target |
|---|---|---|---|
| 5-fold OOF hit@1cm | 0.6350 (final, results.md) | 0.6381 (G1 fallback) | band 측정 (§3.2) |
| F0 raw hit@1cm | 0.6320 (plan-012 G0 측정) | — | G0 reproduce (±0.005) |
| 1-fold OOF (fold=0) | 0.6411 (plan-012 G1 winner) | — | G2 ≥ 0.6511 (+0.010) |

---

## §4. STAGE 0 (c3, G0) — preflight artifact 산출 [TODO]

(c2 spec drop 단계 — c3 commit 시 채움)

## §5. STAGE 1 (c4, G1) — `plan014_paradigm.py` 새 module + 재사용 끊김 검증 [TODO]

(c2 spec drop 단계 — c4 commit 시 채움)

## §6. STAGE 2 (c5, G2) — 1-fold OOF sanity [TODO]

(c2 spec drop 단계 — c5 commit 시 채움)

## §7. STAGE 3 (c6, G3) — 5-fold OOF + band classification + submission [TODO]

(c2 spec drop 단계 — c6 commit 시 채움)

## §8. STAGE 4 (c7, G_final) — synthesis + plan-015 후보 + 3 파일 sync [TODO]

(c2 spec drop 단계 — c7 commit 시 채움)
