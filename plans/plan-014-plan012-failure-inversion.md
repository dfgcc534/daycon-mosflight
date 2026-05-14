---
plan_id: 014
version: 1.3 (draft — narrative reframe: task essence = "F0 64% cover + 남은 20% 끌어당김"; plan-004 참조 범위 = input 가공 + F0 cover 입증 2가지만)
date: 2026-05-14 (Asia/Seoul)
status: draft
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
> 그 외 (`CandidateAttentionGRUSelector` 통째 / candidate-attention head / 27-way selector 구조 / pretrained weight 등) = **참조 안 함**. 본 plan 의 encoder + head 는 "**시계열 → 7 방향 점수 + 0.5cm 이동 offset**" task 에 직접 맞춘 from-scratch design (§0.5 C1).
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

그 외 plan-004 참조 = **금지**:
- `CandidateAttentionGRUSelector` 통째/부분 import 모두 X — 특히 candidate-attention head (`self.query` / `ctx_norm` / `event_norm` / `head`) 의 27-way selection 구조, pretrained weight init 등 plan-004 의 sample-별 27-후보-비교 task 에 fit 된 모든 부분.
- *시계열 인코딩 자체* (GRU 또는 equivalent) 는 task-neutral 표준 도구이므로 reuse 가 아니라 본 plan 의 새 encoder 내부에서 `nn.GRU(...)` 를 *직접 생성* (= `selector.py` import 하지 않음, 본 plan 의 새 module 안에서 build).

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
  - **C1 encoder** (★ v1.2 재정정 + v1.3 task-essence link): 본 plan 의 task = "시계열 → 7 방향 점수 + 0.5cm 이동 offset" (§task essence) 에 직접 fit 한 **from-scratch encoder**. plan-004 `CandidateAttentionGRUSelector` 는 import 안 함 — 단 GRU 자체는 task-neutral 시계열 모듈이므로 본 plan 의 새 encoder 내부에서 `nn.GRU(...)` 를 *직접 생성* (= "GRU reuse" 가 아니라 "GRU 를 처음부터 build"). 폐기 대상 = candidate-attention head 부분 (`self.query` / `ctx_norm` / `event_norm` / `head` 의 sample-별 candidate cross-attention 구조 + plan-004 fit weight). encoder 의 task 는 본질적으로 "시계열 → 7 방향 점수" 1개이므로 shared encoder + 2 head (classifier / regression) 가 자연스러운 default — **encoder 개수 (1 vs 2) 는 핵심 아님**. 새 arch 후보: (i) `nn.GRU(seq) → final hidden → 직접 7 logit + 3D offset` (cross-attention 없음), (ii) Transformer encoder over trajectory → 동일 head, 등. arch 후보는 §2 spec 단계에서 결정. ※ v1.0 narrative "별도 encoder vs 별도 projection branch 양자택일" + v1.1 "모듈 통째 폐기" + v1.2 의 "GRU reuse 정당" 표현 중 마지막은 *의미 보존* (GRU 자체는 죄 없음) 하되 본 plan 의 실제 *코드 정책* = "selector.py import 안 함, 새 module 안에서 GRU 직접 생성".
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

### G-gates (stub — 추후 채움)

- G0: preflight (F0 learnable 초기값 박제 + anchor scale 결정 + soft label kernel 선택 + plan-012 measured 0.6350 reference confirm via results.md read) [TODO]
- G1: 4 컴포넌트 (encoder/F0/anchor/label) 의 from-scratch 구현 완료 + smoke pass + DCM/loss curve 박제 (재사용 끊김 검증 = encoder weight diff > 0 from plan-004 pretrained, F0 coef grad != 0, anchor radius != 0.005m, label entropy > 0) [TODO]
- G2: 1-fold OOF 측정 + plan-012 1-fold winner OOF 0.6411 위 +0.010 (= 0.6511) [TODO]
  - **fold mapping (narrative 가정)**: 1-fold 0.6511 ≈ 5-fold 0.6450~0.6550 (fold noise 한 폭 가정). 즉 G2 통과 = §Target 의 partial 회복 band (≥ 0.6450) 의 *floor* 진입 후보. 정확한 mapping 은 §2/§3 spec 채움 단계에서 결정.
- G3: 5-fold OOF 측정 + §Target 의 어느 band 진입했는지 박제 + submission 후보 생성 [TODO]
- G_final: synthesis + plan-015 후보 (positive/partial/negative band 별 분기) + 3 파일 sync [TODO]

### Commit chain (stub — 추후 채움)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-014-plan012-failure-inversion.md` v0 narrative draft (검증 frame, head-to-head A vs B) | [DONE] 4657ff7 + 2a0f755 |
| c1.1 | docs | v1 narrative re-frame — "재사용 원인 검증" → "재사용 전제 + plan-012 재실험" (★ premise 채택, head-to-head 폐기, baseline reproduce 폐기, target 절대값 기반) | [DONE] c7cf5c8 |
| c1.2 | docs | v1.1 spot-fix — F4 진단 정확화: "shared encoder bottleneck (두 head 공유)" → "encoder inductive-bias mismatch (plan-004 candidate-attention 모듈을 candidate 없는 task 에 끼움)". §0.5 7-mode 표 / C1 encoder bullet / §1.3 trap chain 동기화 | [DONE] 5e98d6d |
| c1.3 | docs | v1.2 spot-fix — F4 진단 2차 정확화 (code-grounded): `CandidateAttentionGRUSelector` 가 GRU [task-neutral] + candidate-attention [plan-004 fit] 의 2 부분 구조임을 selector.py:697-726 + ring_classifier.py:309/315 read 로 확인. 폐기 범위 = candidate-attention 부분만, GRU 본체는 reuse 정당. §0.5 7-mode F4 / C1 encoder bullet / §1.3 trap chain 모두 sync | [DONE] c7fa9c8 |
| c1.4 | docs | v1.3 narrative reframe — task essence ("F0 64% cover + 남은 20% 끌어당김") 를 §0/§0.5 의 narrative anchor 로 박제. plan-004 참조 범위 = (a) input 가공 + (b) F0 sample cover 입증 2가지로 축소 (`selector.py` import 폐기). §0 한 줄 목적 + §0.5 새 sub-section (task essence + plan-004 참조 범위 + premise corollary) + §0.5 7-mode F4 / C1 encoder bullet task-essence link + §1.2 사망 진단 (residual vector regression 어려움) + §1.5 정직성 원칙 (plan-004 참조 2가지만) sync | [DONE] 3a7a26c |

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
- **plan-004 참조 범위 = 2가지만 (★ v1.3 narrative anchor, §0.5 박제)**: (a) input feature 가공 방식, (b) F0 64%/84% sample cover 입증. 그 외 (`CandidateAttentionGRUSelector` 통째/부분 import, pretrained weight, 27-way selector 구조 등) = **금지**. 시계열 모듈 (GRU 등) 은 본 plan 의 새 module 내부에서 *직접 build* (= `selector.py` import 없음). 코드 (§2 spec 단계) + plan body 양쪽 모두 이 narrative 를 유지.

---

## §2. 실험 설계 [TODO]

(narrative draft 단계 — 추후 채움)

## §3. 합격 기준 / G-gate 정량 spec [TODO]

(narrative draft 단계 — 추후 채움)

## §4~§N. 코드 / Phase 별 spec [TODO]

(narrative draft 단계 — 추후 채움)
