# Jetson 자세 코치 시스템 검증 계획

이 문서는 모델 정확도만이 아니라 **Camera → 전처리 → 자세 추론 → 자세 판정 → 알림/기록** 전체 시스템이 Jetson Nano 4GB의 시간·메모리·전력·발열 제약 안에서 동작하는지를 재현 가능하게 검증하기 위한 계획이다.

> 아래 수치는 첫 실측 전의 **설계 목표(acceptance target)** 이다. 측정 결과와 혼동하지 않도록 결과표에는 반드시 `target`과 `measured`를 분리해 기록한다.

## 1. 핵심 합격 기준

| ID | 범주 | 지표와 측정 범위 | 초기 합격 기준 | 이유 |
| --- | --- | --- | --- | --- |
| SYS-LAT-01 | 전체 지연 | 카메라 프레임 timestamp부터 자세 상태 생성까지 end-to-end latency | p50·p95·p99 보고, **p95 ≤ 200 ms** | 5 FPS 목표에서 한 프레임 처리 예산을 200 ms로 둠 |
| SYS-LAT-02 | 단계별 지연 | capture·preprocess·inference·postprocess·output 각각 | 단계별 p50·p95 기록 | 병목이 모델인지 I/O인지 구분 |
| SYS-THR-01 | 처리량 | 완료 프레임 수 / 정상상태 실행시간 | **평균 5 FPS 이상**, 목표 5–10 FPS | README의 실시간 처리 목표와 연결 |
| SYS-JIT-01 | 안정성 | end-to-end latency 표준편차와 p99−p50 | 값과 원인 보고 | 평균이 가리는 순간 지연 확인 |
| SYS-DROP-01 | 프레임 | 입력·처리·누락 프레임 수와 비율 | **비의도 누락률 ≤ 1%** | 처리된 프레임만 보고 성능을 과대평가하지 않음 |
| SYS-MEM-01 | 메모리 | 프로세스 RSS peak, 시스템 RAM·swap peak | **RSS peak ≤ 3.5 GB**, swap 지속 증가 없음 | Nano 4GB에서 OS와 함께 운용 |
| SYS-THERM-01 | 발열 | 30분 이상 실행 중 CPU/GPU/보드 온도 | 온도 시계열 보고, **thermal throttling 0회** | 장시간 성능 저하 확인 |
| SYS-PWR-01 | 전력 | idle·steady-state 평균/최대 전력, energy/frame | 전원 모드·측정 장비와 함께 보고 | 처리량만으로 알 수 없는 에너지 비용 비교 |
| ML-QUAL-01 | 분류 품질 | 사용자 분리 test set의 Macro-F1·class recall·confusion matrix | baseline 대비 악화 없음; 수치는 1차 데이터 후 확정 | 데이터 누출과 자세별 취약점 확인 |
| ML-SAFE-01 | 위험 오류 | 나쁜 자세를 GOOD으로 판단한 False Negative rate | 고정 기준보다 낮거나 같음 | 알림 누락을 우선 관리 |
| UX-ALERT-01 | 알림 품질 | 시간당 오경보, 자세 이탈 후 알림까지의 시간 | 개인 기준+지속 조건이 고정 기준보다 개선 | 연구 질문과 직접 연결 |
| OPS-OFFLINE-01 | 오프라인 | 네트워크 차단 중 추론·판정·로컬 기록 | 핵심 기능 100% 유지 | On-device 배치 타당성 검증 |
| OPS-REC-01 | 복구 | 카메라 단절·프로세스 종료·재부팅 후 복구 | 안전 상태 유지, 자동 재시작·복구시간 기록 | 운용 가능한 시스템 검증 |
| PRIV-01 | 개인정보 | 원본 프레임의 디스크/외부 전송 여부 | 기본 운용에서 원본 저장·외부 전송 0건 | On-device 개인정보 설계 검증 |

`p95 ≤ 200 ms`는 5 FPS 최소 목표에서 도출한 초깃값이다. 3초 지속 조건은 의도된 정책 지연이므로 프레임 처리 지연과 분리해 보고한다.

## 2. 실험 조건 고정

결과마다 다음 메타데이터를 함께 저장한다.

- Git commit SHA, 실행 날짜와 시험자
- Jetson 모델, L4T/JetPack, Python, runtime, 모델 파일 hash
- `nvpmodel` 모드, CPU/GPU/EMC clock 설정, 전원 어댑터
- 카메라 해상도·입력 FPS·처리 FPS, 모델 정밀도(FP32/FP16/INT8)
- 냉각 방식, 팬 속도, 실내 온도, warm-up 시간
- 실행 인자와 baseline 파일 hash
- 테스트 영상/참여자 ID(익명 ID만), 조명·거리·가림 조건

같은 비교 실험에서는 비교 변수 하나만 바꾸고 각 조건을 최소 3회 반복한다. 최초 warm-up 구간은 정상상태 통계에서 제외하되 시간은 별도 기록한다.

## 3. 측정 시나리오

### A. 성능·자원

1. 3분 warm-up 후 30분 연속 실행한다.
2. 각 프레임에 `capture_start`, `capture_end`, `infer_start`, `infer_end`, `decision_end`, `output_end`를 monotonic clock으로 기록한다.
3. 동시에 `tegrastats`로 RAM·CPU·GPU·EMC·온도·throttling 정보를 1초 간격으로 기록한다.
4. 첫 3분을 제외하고 p50·p95·p99, FPS, 누락률, peak RSS, 최고 온도를 계산한다.
5. 시작 5분과 마지막 5분의 p95/FPS를 비교해 열에 따른 성능 저하를 확인한다.

권장 원시 로그 필드:

```text
run_id,frame_id,captured_at,capture_ms,preprocess_ms,inference_ms,
postprocess_ms,output_ms,e2e_ms,state,dropped,rss_mb,temp_cpu_c,temp_gpu_c
```

### B. 모델·판정 품질

- 참여자 단위로 train/validation/test를 분리한다. 같은 참여자의 프레임이 서로 다른 split에 섞이지 않게 한다.
- GOOD·HEAD_FORWARD·TORSO_FORWARD·HEAD_AND_TORSO_FORWARD·NO_POSE별 precision/recall/F1과 confusion matrix를 보고한다.
- 고정 기준 vs 개인 기준, 즉시 판정 vs 3초 지속 조건을 같은 test set에서 비교한다.
- 조명(밝음/어두움), 거리, 부분 가림, 안경·의복 등 조건별로 False Negative와 판정 가능 비율을 나눠 기록한다.

### C. 실패·복구·개인정보

- 네트워크를 끊고 10분간 핵심 추론과 로컬 알림이 유지되는지 확인한다.
- 카메라 케이블 단절 또는 프레임 중단 시 crash 대신 `NO_POSE`/안전 상태가 되는지 확인한다.
- 프로세스를 강제 종료한 뒤 systemd 재시작 횟수와 복구시간을 측정한다.
- 재부팅 후 사용자 조작 없이 서비스·카메라·추론이 준비되는 시간을 측정한다.
- 생성 파일과 network connection을 점검해 원본 영상이 저장·외부 전송되지 않는지 확인한다.

## 4. 비교 실험

| 실험 | 독립 변수 | 고정 조건 | 주요 결과 |
| --- | --- | --- | --- |
| E01 | 처리 FPS 5/8/10 | 모델·해상도·전원 모드 | p95, FPS, RAM, 온도, 누락률 |
| E02 | 1280×720 vs 낮은 입력 해상도 | 판정 정책·runtime | 정확도, latency, power |
| E03 | TFLite vs TensorRT FP16 | 입력·데이터·판정 정책 | 정확도 차이, p95, FPS, RAM, 온도 |
| E04 | 고정 기준 vs 개인 기준 | 같은 사용자 test set | Macro-F1, FN, 시간당 오경보 |
| E05 | 즉시 vs 3초 지속 조건 | 같은 예측 시계열 | 오경보, 누락, 알림 지연 |
| E06 | 시작 시점 vs 30분 후 | 동일 실행 | p95/FPS 변화, 온도, throttling |
| E07 | 온라인 vs 네트워크 차단 | 동일 입력 | 기능 유지율, 오류·복구시간 |

## 5. Git에 남길 증거

커밋 가능한 비식별 결과만 다음 구조로 보관한다.

```text
benchmarks/
  README.md                  # 측정 장비·방법·실행 명령
  runs/<run-id>/config.json  # 환경, commit, 모델 hash, 인자
  runs/<run-id>/summary.json # percentile·FPS·RAM·온도·전력 요약
  comparison.csv             # E01~E07 비교표
  figures/                   # latency CDF, FPS/온도 시계열, confusion matrix
```

원본 영상, 얼굴 이미지, baseline 개인 데이터, IP·인증정보, 대용량 frame-level 로그는 Git에 커밋하지 않는다. 익명화한 집계 결과와 재현 명령만 남긴다.

## 6. 교수님 발표용 요약

- **실시간성:** 평균 FPS가 아니라 end-to-end p50/p95/p99와 deadline 초과율을 제시한다.
- **자원 제약:** peak RAM, swap, 온도, throttling, 전력을 같은 30분 실험에서 함께 측정한다.
- **AI 품질:** 사용자 분리 평가와 조건별 False Negative로 데이터 누출과 안전 오류를 관리한다.
- **시스템성:** 카메라 단절·네트워크 장애·재부팅 후 안전 상태와 복구시간을 검증한다.
- **재현성:** commit SHA·모델 hash·전원 모드·냉각·실행 인자까지 Git에 남긴다.

최종 결과는 “모델이 동작한다”가 아니라, **정확도–지연–메모리–전력–발열–가용성의 trade-off를 동일 장치에서 측정하고 요구조건 충족 여부를 설명한다**는 형태로 제시한다.
