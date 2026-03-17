# RFT 예측 시스템 방법론

## 1. 시스템 개요

본 시스템은 드론 비행 중 실시간으로 **잔여 비행 시간(RFT, Remaining Flight Time)**을 예측하는 딥러닝 모델이다. 1D-CNN 인코더와 Temporal Attention, 배터리 물리 상태 멀티인풋을 결합한 구조로, 과거 비행 데이터로부터 시계열 패턴을 추출하고 배터리 상태를 직접 결합하여 RFT를 회귀 예측한다.

---

## 2. RFT 레이블 정의

```
RFT[i] = (비행 종료 타임스탬프 - 현재 타임스탬프) / 1,000,000  [초]
```

비행 종료 시점은 배터리 경고 임계값 도달 후 RTL(Return-to-Launch) 모드 진입 시점으로 정의한다.

- `nav_state == 18 AND battery_warning >= 2` 조건 충족 시 이후 데이터 제외
- RFT는 비행 시작 시 최대값에서 단조 감소하여 0에 수렴

---

## 3. 입력 피처

### 3.1 flight_features (과거 시계열, 16차원)

드론 텔레메트리로부터 5Hz로 리샘플링한 과거 5초(25스텝) 시계열 데이터.

| 인덱스 | 피처 | 설명 |
|--------|------|------|
| 0-2 | `local_vx, vy, vz` | 기체 속도 (m/s) |
| 3-5 | `ang_p, q, r` | 각속도 (rad/s) |
| 6-8 | `ang_dp, dq, dr` | 각가속도 (rad/s²) |
| 9 | `thrust_sp` | 추력 설정값 (0-1) |
| 10-13 | `mot0~3` | 모터 명령 (0-1) |
| 14 | `voltage_v` | 배터리 전압 (V) |
| 15 | `current_a` | 방전 전류 (A) |

**전처리:**
- 속도, 추력, 모터, 전압/전류: **ZOH(Zero-Order Hold)** 리샘플링
- 각속도, 각가속도: **EMA 필터(2.0Hz 컷오프)** 적용 후 리샘플링 (고주파 진동 제거)

### 3.2 battery_state (배터리 물리 상태, 5차원)

현재 시점의 배터리 상태를 요약한 멀티인풋 벡터. CNN 인코더 출력과 결합되어 RFT 예측에 직접 사용된다.

| 인덱스 | 피처 | 설명 |
|--------|------|------|
| 0 | `consumed_mah` | 누적 방전량 (mAh) |
| 1 | `V_ema` | EMA 필터링된 전압 (저주파 추세) |
| 2 | `I_ema` | EMA 필터링된 전류 (부하 수준) |
| 3 | `sag` | 전압 새그: V_now - V_ema (순간 부하 스트레스) |

원래 dV/dt 도 넣었으나, 뺀게 더 성능이 좋은 경향이 있어서 그냥 뺌.

**EMA 필터 공식 (half-life = 1.5s, 5Hz):**
```
alpha = exp(-ln(2) / (1.5 × 5)) ≈ 0.785
Y[i] = alpha × Y[i-1] + (1 - alpha) × X[i]
```

**배터리 상태 멀티인풋의 역할:**
- `consumed_mah`: 장기 누적 방전량 → CNN 시계열 창 밖의 장기 정보 보완
- `V_ema`: 배터리 방전 곡선상의 현재 위치 (저주파 트렌드)
- `I_ema`: 현재 부하 수준 (저주파 평균)
- `sag (= V_now - V_ema)`: 순간 부하로 인한 전압 드룹. CNN이 시계열에서 감지한 전압 강하 패턴이 실제 물리적 새그인지 노이즈인지 판별하는 보조 신호
- `dV_dt`: 평소에는 ≈0으로 정보량이 적지만, 방전 말기 전압 급강하 구간에서 핵심 신호로 작동하여 말기 예측 정확도에 기여

---

## 4. 모델 구조

```
input_past (25, 16)          input_state (5,)
      │                              │
  CNN Encoder                        │
  Conv1D(64, k=5) + BN + ReLU       │
  MaxPooling1D(2)  → (12, 64)       │
  Conv1D(128, k=3) + BN + ReLU      │
  MaxPooling1D(2)  → (6, 128)       │
  Conv1D(128, k=3) + BN + ReLU      │
  TemporalAttention           │
      z (128D)              │
      │                              │
      └──────── Concatenate ─────────┘
                     │
               Dense(128) + BN + ReLU + Dropout(0.3)
                     │
               Dense(64) + BN + ReLU + Dropout(0.2)
                     │
               Dense(1)
                     │
              RFT 예측값 (초)
```

### 4.1 CNN 인코더의 역할

1D-CNN은 과거 5초 시계열에서 **로컬 시간 패턴을 감지**한다. 각 Conv1D 레이어는 커널 크기만큼의 시간 창 내에서 특정 패턴(전류 스파이크, 전압 강하 패턴 등)을 탐지하도록 가중치를 학습한다.

- **MaxPooling의 작동 원리:** MaxPooling은 원시 데이터의 최댓값이 아니라, `Conv → BN → ReLU` 이후 **활성화 강도(패턴 발현 강도)의 최댓값**을 선택한다. 따라서 전압 새그가 발생한 시점의 활성화 값이 가장 크다면, MaxPooling은 이를 보존하여 다음 레이어로 전달한다.
- **Average Pooling 대비 MaxPooling 선택 이유:** Average Pooling은 국소적으로 큰 활성화 값을 주변의 0 값들과 평균화하여 희석시키므로, 순간적인 전압 새그와 같은 임펄스성 패턴 탐지에 불리하다.

### 4.2 Temporal Attention

CNN 레이어를 거친 후 각 타임스텝의 중요도를 학습 기반으로 가중합한다.

```
score = Dense(1)(x)              # (batch, 6, 1)
weight = softmax(score, axis=1)  # (batch, 6, 1) - 시간 축 정규화
output = Σ(x × weight)           # (batch, 128) - 가중합
```

MaxPooling으로 시퀀스가 6스텝으로 압축된 상태에서, Attention은 "어떤 시점의 패턴이 RFT 예측에 중요한가"를 학습한다. CNN이 패턴을 추출하고 Attention이 중요도를 부여하는 역할 분담 구조.

### 4.3 배터리 상태 멀티인풋의 역할

CNN 인코더는 **과거 5초 창** 내의 패턴만 볼 수 있다. 배터리 잔량처럼 비행 전 과정에 걸친 누적 정보는 시계열 창으로는 파악할 수 없다. 배터리 상태 멀티인풋은 이 장기 정보를 CNN 잠재 벡터(z)에 직접 결합하여 보완한다.

- **역할 분담:** CNN → 단기 비행 패턴 / battery_state → 장기 배터리 상태

---

## 5. 스케일링 전략

이기종(heterogeneous) 피처의 특성에 따라 차별화된 스케일러를 적용한다.

| 피처 그룹 | 스케일러 | 범위 |
|-----------|---------|------|
| 속도, 각속도, 전압/전류 | StandardScaler | 비제한 |
| 추력, 모터 명령 (0~1) | MinMaxScaler | [-2.0, 2.0] |
| battery_state | StandardScaler | 비제한 |
| RFT 레이블 | StandardScaler | 비제한 |

추력/모터는 하드웨어 물리 한계(0~1)가 명확하므로 MinMaxScaler로 극단값 발생을 방지하고 분산 스케일을 맞춘다.

---



## 7. 단조 제약 후처리 (Monotone Constraint)

RFT는 물리적으로 단조 감소해야 한다. 모델 출력에 스텝별 상승/하강 한계를 적용하여 물리적 타당성을 확보한다.

```
max_drop = 0.30 s/step  (전력 분포 분석 기반: P_p99 / P_median = 1.39×)
max_rise = 0.01 s/step  (순간적 under-prediction 보정 허용)

for t in range(1, T):
    y_fixed[t] = clip(y_raw[t], y_fixed[t-1] - 0.30, y_fixed[t-1] + 0.01)
```

- 비행 로그별(per-log) 독립 적용
- 학습/평가 지표(MAE, RMSE)는 후처리 전 raw 예측값 기준 계산

---

## 8. 학습 설정

| 항목 | 값 |
|------|-----|
| 입력 창 길이 | 25 스텝 (5초 @ 5Hz) |
| 배치 크기 | 64 |
| 최대 에폭 | 200 |
| 초기 학습률 | 0.001 (Adam) |
| 손실 함수 | Huber Loss (이상치 완화) |
| Early Stopping | patience=15, val_loss 기준 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |
| 데이터 분할 | GroupShuffleSplit (비행 로그 단위, 80/20) |

---

## 9. 실험 결과 요약

| 구성 | MAE |
|------|-----|
| Conv3+Pool2+Attn, battery 5개 (원본) | 7.178 s |
| Conv3+Pool2+Attn, battery 3개 (consumed/V_ema/I_ema) | 7.453 s |
| Conv3+Pool2+Attn, battery 4개 (consumed/V_ema/I_ema/sag) | **7.136 s** |
| Conv2+Pool1+Attn | 7.556 s |
| Conv3+Pool1+Attn | 7.862 s |
| Conv2+Pool2+Attn | 8.097 s |
| Conv3+Pool2+GAP | 8.330 s |
| Conv2+NoPool+Attn | 8.367 s |
| Conv3+NoPool+Attn | 8.649 s |
| 과거 창 10초 | 7.517 s |
| 과거 창 3초 | 8.575 s |

**Conv3+Pool2+TemporalAttention 구조가 일관되게 최적** (5초 창, MaxPooling 2회, Temporal Attention 조합)
