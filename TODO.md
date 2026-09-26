# 나중에 할 일

벤치마크 도구(`scripts/benchmark.py`)를 추가하면서 미뤄 둔 항목.

## 벤치마크 실측 (Jetson 필요)

- [ ] Jetson Nano에서 `run` / `model` / `rtt` 첫 실행. 개발 PC에는 TensorRT·OpenCV·카메라·tegrastats가 없어 가짜 worker로만 검증함.
- [ ] `tegrastats` 전력·GPU 사용률 파싱을 실제 출력으로 확인 (root 필요 여부, stdout 버퍼링).
- [ ] 카메라 PTS 기반 `age_ms`/`e2e_ms`가 실제로 나오는지 확인 (안 나오면 `total_ms`만 남음).
- [ ] `cuMemGetInfo` 기반 GPU 메모리 값이 통합 메모리에서 의미 있는지 확인.
- [ ] 검증 계획 기준 30분 실행 (`--warmup 180 --duration 1800`)과 E01~E07 비교 실험(조건당 3회 이상).
- [ ] 결과를 `benchmarks/runs/<run-id>/`에 남기고 `compare`로 `benchmarks/comparison.csv` 생성.
- [ ] 외부 전력계로 보드 레일 전력 교차 확인.

## 측정 항목 확장

- [ ] 정답 라벨 기반 정확도 (Macro-F1, 자세별 recall, confusion matrix, False Negative rate). 라벨링 데이터 수집 후.
- [ ] 조명·거리·가림 조건별 인식율 분해.
- [ ] 카메라 단절·프로세스 강제 종료·재부팅 복구시간 측정 (`OPS-REC-01`).
- [ ] 네트워크 차단 10분 오프라인 검증 (`OPS-OFFLINE-01`).
- [ ] 원본 프레임 저장·외부 전송 0건 점검 (`PRIV-01`).
- [ ] thermal throttling 감지를 온도 임계값이 아닌 실제 클럭 저하로 판정.
- [ ] MJPEG 스트림 자체의 지연 측정 (지금 `rtt`는 `/api/pose` 폴링만).

## 기존 개발 단계

- [ ] 오디오 동작 확인
- [ ] 자세 분류 → 음성 안내
- [ ] 동의 기반 데이터 수집·라벨링, 외부 PC 학습
- [ ] 개인 기준·신뢰도·시간 조건 비교
- [ ] 센서·LED·버튼 통합
- [ ] 시리얼·Bluetooth 및 기록 통합
- [ ] 전체 기능 동시 실행·장시간 검증
