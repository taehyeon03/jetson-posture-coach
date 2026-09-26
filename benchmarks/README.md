# 성능 벤치마크

[시스템 검증 계획](../docs/system_validation_plan.md)의 측정 항목을 재현 가능하게 실행하는 도구다. 실제 파이프라인(`web_mvp/server.py`의 `worker`)에 계측 지점만 넣어 측정하므로, 서비스에서 도는 코드와 벤치마크 코드가 갈라지지 않는다. 계측을 끄면(`recorder=None`) 서비스 동작은 그대로다.

## 실행 (Jetson)

```bash
# 1) 전체 파이프라인: 카메라 → 얼굴 검출 → TensorRT → 자세 판정 → JPEG
./scripts/nano_python.sh scripts/benchmark.py run --label fp16-baseline \
    --warmup 180 --duration 1800          # 검증 계획 기준. 빠른 점검은 기본값(15s/60s)

# 2) 모델만: 카메라·얼굴 검출 없이 엔진 속도/메모리/에너지 (합성 프레임)
./scripts/nano_python.sh scripts/benchmark.py model --label fp16

# 3) 왕복 지연: 1)을 --serve-port 8080 으로 띄워 부하를 건 채, 다른 터미널/PC에서
./scripts/nano_python.sh scripts/benchmark.py run --serve-port 8080 --label with-http
python3 scripts/benchmark.py rtt --url http://JETSON_IP:8080

# 4) 실행 결과 비교표 → benchmarks/comparison.csv
python3 scripts/benchmark.py compare
```

`rtt`·`compare`는 표준 라이브러리만 쓰므로 개발 PC에서도 실행된다. 전력 baseline을 위해 `run`/`model`은 시작 시 `--idle-seconds`(기본 10초) 동안 아무것도 돌리지 않고 idle 전력을 잰다. `sudo nvpmodel -q`로 전원 모드를 고정하고, 비교 실험(E01~E07)은 한 번에 변수 하나만 바꿔 최소 3회 반복한다.

## 측정 항목

| 요청 | 지표 | 어디서 나오나 |
| --- | --- | --- |
| Latency | 단계별 mean/std/p50/p95/p99/max: 카메라 대기, 얼굴 검출, 전처리, H2D, TensorRT 실행, D2H, 디코드, 판정, JPEG, 전체 | `summary.json` → `latency_ms` |
| Throughput | 정상상태 FPS, 5초 구간 최소/최대 FPS, 모델 단계 처리율(추론/s) | `throughput` |
| 안정화 | 지연 표준편차·p99−p50·CV, 초반 vs 후반 p95/FPS 변화(열 저하), 키포인트 떨림 raw vs 스무딩(px), 자세 상태 전환 횟수/분 | `stability`, `drift` |
| 모델 인식율 | 얼굴 검출률, 인물 잠금률, 유효 자세율, 관절별 평균 신뢰도·가시율, 상태 분포 | `recognition` |
| 에너지 | 평균/최대 전력(mW), idle 대비, 총 에너지(J), 추론 1회당 mJ | `energy` |
| 피크 메모리 | 프로세스 RSS peak, 시스템 RAM peak, swap peak, RSS 증가율(MB/분) | `memory` |
| 왕복 지연 | HTTP 요청→응답 RTT, 카메라 촬영→클라이언트 수신 | `rtt` 서브커맨드 |
| 안전성 | 데드라인(기본 200ms) 초과율, 카메라 타임아웃, 예외, 최고 온도·한계 초과 샘플, 카메라 프레임 스킵 | `reliability` |
| 추론 비용/속도/VRAM | 추론 1회당 에너지·시간, GPU 메모리 peak·증가분, 엔진 디바이스 메모리·파일 크기, (선택) 100만 회당 전력 비용 | `energy`, `memory`, `model` 모드 |

`summary.json`의 `targets`는 검증 계획 1절의 SYS-LAT-01(p95 ≤ 200ms), SYS-THR-01(≥ 5 FPS), SYS-MEM-01(RSS ≤ 3.5GB)을 자동 판정한다.

## 출력

```text
benchmarks/runs/<run-id>/
  config.json    # git SHA, 엔진/baseline hash, nvpmodel, L4T, TensorRT, 실행 인자
  summary.json   # 위 지표 요약 (커밋 대상)
  frames.csv     # 프레임별 단계 시간·판정 (gitignore)
  system.csv     # 1초 간격 RSS/RAM/swap/온도/GPU/전력 (gitignore)
```

## 해석할 때 주의

- **전력**은 `tegrastats`(없으면 INA3221 sysfs)가 보고하는 보드 입력 레일 값이다. 어댑터·주변장치 손실을 포함한 벽면 전력이 아니며, 외부 전력계로 교차 확인하지 않은 값은 그렇게 표기한다. `tegrastats`가 root를 요구하면 `sudo`로 실행한다.
- **VRAM**: Jetson은 CPU와 메모리를 공유하므로 별도 VRAM이 없다. GPU 메모리는 CUDA 드라이버가 보고하는 시스템 메모리 사용량이며 다른 프로세스 영향이 섞인다. 그래서 "기준 대비 증가분"과 엔진 디바이스 메모리를 함께 본다.
- **왕복/e2e 지연**의 카메라 구간은 GStreamer 버퍼 PTS와 파이프라인 클럭의 차이로 추정한 값(best-effort)이다. PTS를 못 얻으면 `age_ms`/`e2e_ms`가 비고 `total_ms`(프레임 수신 후 처리시간)만 남는다. `rtt`는 서버·클라이언트 시계를 요청 중간점으로 맞추므로 ±RTT/2 오차가 있고, `first_seen`은 폴링 간격 이하로 내려가지 않는다.
- **카메라 프레임 스킵**은 최신 프레임만 쓰는 설계상 정상이다(30fps 카메라를 ~10fps로 처리). 검증 계획의 "비의도 누락률"과 다르다.
- **키포인트 떨림·상태 전환**은 사용자가 가만히 앉아 있을 때만 의미 있다. 움직임이 섞이면 값이 커진다.
- **모델 인식율**은 검출·유효 판정 비율이다. 정답 라벨이 필요한 정확도(Macro-F1, False Negative)는 라벨링 데이터가 생긴 뒤 검증 계획 B절대로 별도 측정한다.
- RSS 증가율은 짧은 실행(초기 할당 포함)에서는 과대하게 나온다. 누수 판단은 30분 실행에서 한다.
