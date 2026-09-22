# Jetson Posture Coach

Jetson Nano 4GB와 카메라를 활용하는 **개인화 자세 감지·음성 피드백 IoT 시스템** 연구 프로젝트입니다.

> 현재 단계: 카메라 동작과 MoveNet 기반 자세 추론 최소 기능을 구현했습니다. 자세 판정은 개인별 바른 자세를 먼저 보정한 뒤 고개 전진·상체 숙임의 지속 시간을 판정합니다. GPIO·음성·통신 통합과 정확도·피드백 효과 평가는 아직 진행 전입니다.

## 연구 목표

컴퓨터 작업 중 전방머리자세와 상체 기울어짐을 감지하고, 개인별 기준과 자세 지속 시간을 반영해 불필요한 알림을 줄이는 시스템을 개발합니다. 자세 변화 인지와 교정 행동을 지원하는 연구이며, 임상 진단 또는 치료 효과를 주장하지 않습니다.

**연구 질문:** 4GB 장치에서 개인별 기준과 시간 조건을 반영하면 감지 성능을 유지하면서 고정 기준 방식보다 오경보를 줄일 수 있는가?

## 시스템 구성안

```mermaid
flowchart TD
    S[착석 센서 / GPIO] --> G[모니터링 활성화]
    C[측면 RGB 카메라] --> P[경량 자세 추정]
    G --> P
    P --> F[귀·어깨·엉덩이 특징]
    F --> M[수집 데이터로 학습한 자세 분류 모델]
    M --> D[개인 기준·신뢰도·지속 시간 판단]
    B[버튼 입력] --> D
    D --> L[GPIO 상태 LED]
    D --> A[TTS 생성 음성 파일 재생]
    D --> R[로컬 자세 기록]
    R --> T[시리얼·Bluetooth 상태 전송]
```

- **학습용 PC:** 데이터 라벨링, 작은 자세 분류 모델 학습, 모델 변환.
- **Jetson Nano 4GB:** 카메라 추론, 센서·버튼·LED, 음성 재생, 로컬 기록과 통신.
- **독립 운용 목표:** 배포 이후 PC 연결 없이 전원 투입 시 자동 실행.
- **음성 안내:** 먼저 PC에서 TTS로 생성한 한국어 파일을 재생합니다. 장치에서의 실시간 TTS 합성은 후속 검토 범위입니다.
- **주변장치:** 카메라, 착석 센서, LED·저항, 버튼, USB 오디오 출력, 호환 통신 모듈. 부품과 배선은 미확정이며 GPIO 전압과 스마트폰 호환성을 확인한 후 선정합니다.

Jetson Nano와 Jetson Orin Nano는 다른 장치입니다. 본 프로젝트는 기존 Nano 4GB를 대상으로 하며 JetPack 4 계열 호환성을 고려합니다.

## 교과목과의 연결

| 학습 내용 | 계획한 실습 |
| --- | --- |
| Linux·Python | 카메라·추론·음성·기록 통합 및 자동 실행 |
| IoT 데이터 수집·AI 학습 | 직접 수집한 자세 특징으로 분류 모델 학습 |
| GPIO·디지털 입출력 | 착석 감지와 상태 LED |
| 인터럽트 입력 | 음소거·기준 재설정 버튼 |
| 시리얼·Bluetooth | 상태·알림 이력 스마트폰 전송 |
| 협업·발표 | CV / 임베디드 / 평가 / 기획 역할 분담 |

## 개발 단계

- [x] 연구 주제, 논문 3편, 배경·시장 조사
- [x] 연구 초록과 17쪽 발표 자료
- [ ] 보드·카메라·오디오 동작 확인 (카메라 확인 완료, 오디오 남음 — [기록](docs/hardware_notes.md))
- [x] 카메라 → 자세 추정 → 개인 기준 자세 분류 최소 기능 구현
- [ ] 자세 분류 → 음성 안내 최소 기능 구현
- [ ] 동의 기반 데이터 수집·라벨링 및 외부 PC 학습
- [ ] 개인 기준·신뢰도·시간 조건 비교
- [ ] 센서·LED·버튼 통합
- [ ] 시리얼·Bluetooth 및 기록 통합
- [ ] 전체 기능 동시 실행·장시간 검증

## 평가 계획

- 사용자 단위로 학습·검증·시험 데이터를 분리합니다.
- 고정 기준과 개인 기준, 즉시 알림과 지속 시간 조건을 비교합니다.
- Macro-F1, 재현율, 시간당 오경보, 판정 가능 비율을 측정합니다.
- FPS, 프레임 처리 지연 p95, OS를 포함한 최대 전체 RAM을 측정합니다.
- 초기 목표는 5–10 FPS, 최대 RAM 3.5GB 이하, 30분 이상 연속 실행입니다. **달성 결과가 아닌 설계 목표**입니다.
- 알림 유무에 따른 자세 복귀율, 이탈 지속 시간, 알림 피로도를 비교합니다.
- 귀–어깨 각도는 임상적 craniovertebral angle(CVA)와 구분합니다.

## 자세 판정 실행

Jetson Nano에서 최초 한 번 런타임과 MoveNet Lightning 모델을 설치합니다. 설치 스크립트는 Nano의 Python 3.6과 Cortex-A57에 맞는 TensorFlow Lite 런타임을 사용하며, 보드 기본 NumPy를 유지합니다.

```bash
cd ~/posture-coach-check
bash scripts/install_posture_ai.sh
```

카메라를 사람의 정확한 옆쪽, 어깨 또는 눈높이에 놓고 머리·어깨·엉덩이가 모두 보이게 합니다. 바르게 앉은 상태에서 다음 명령을 실행하면 3초 후 5초 동안 개인 기준을 수집하고, 이어서 30초 동안 자세를 판정합니다.

```bash
python3 scripts/posture_coach.py --calibrate
```

기준은 `data/posture_baseline.json`, 마지막 관절·판정 화면은 `captures/posture-latest.jpg`, 지속 경고 이력은 `logs/posture-events.jsonl`에 저장됩니다. 이후에는 `--calibrate` 없이 기존 기준을 사용합니다. 계속 실행하려면 `--duration 0`, 로컬 모니터에 미리보기를 표시하려면 `--display`를 추가합니다.

### 웹 대시보드

Jetson에서 다음 서버를 실행하면 실시간 영상, 관절 표시, 자세 상태, 고개·상체 변화량, 나쁜 자세 지속 시간, 보정 버튼과 경고 이력을 브라우저에서 볼 수 있습니다.

```bash
cd ~/posture-coach-check
python3 web_mvp/server.py --engine models/pose-resnet18-fp16.engine
```

같은 네트워크 또는 같은 Tailscale에 연결된 기기에서 `http://jetson-nano:8080`으로 접속합니다. 이름으로 접속되지 않으면 Jetson의 IP 주소를 사용합니다. 서버는 TensorRT GPU 엔진으로 자세를 추론하며, 기본적으로 모든 네트워크 인터페이스의 8080번 포트에서 대기합니다. 원본 영상은 파일로 저장하지 않고 브라우저에 실시간 JPEG 스트림으로 전달합니다.

부팅 시 자동 실행되는 서비스로 등록하려면 한 번만 다음 명령을 실행합니다.

```bash
bash scripts/install_web_service.sh
```

웹 서버가 실행 중일 때는 카메라를 전담합니다. 카메라 단독 테스트나 명령행 자세 판정을 실행하려면 `sudo systemctl stop posture-coach-web`으로 잠시 중지하고, 완료 후 `sudo systemctl start posture-coach-web`으로 다시 시작합니다.

## 발표 자료

발표 자료는 2026-09-13 연구 기획 초안입니다. 이후 논의된 GPIO·통신·음성 기능을 포함한 최신 범위는 이 README를 기준으로 합니다. 학과·학번·이름·팀원은 입력란으로 남아 있습니다.

- [PowerPoint .ppt](자세교정_연구제안.ppt)
- [편집용 .pptx](자세교정_연구제안.pptx)
- [PDF](자세교정_연구제안.pdf)
- [논문 조사·발표 원고](논문조사_초록_발표원고.md)
- [연구 초록](연구초록.txt)

### 발표 자료 재생성

개발 PC에서 실행합니다. 아래 과정은 보드용 AI 프로그램 실행 방법이 아닙니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-docs.txt
python build_presentation.py
```

스크립트는 같은 폴더의 PPTX·발표 원고·초록을 덮어씁니다. PPT와 PDF는 LibreOffice로 별도 변환합니다. 한국어 표시에는 Noto Sans CJK KR 글꼴을 사용합니다.

```bash
libreoffice --headless --convert-to 'ppt:MS PowerPoint 97' --outdir . 자세교정_연구제안.pptx
libreoffice --headless --convert-to pdf --outdir . 자세교정_연구제안.ppt
```

## 관련 논문과 공식 자료

| 자료 | 역할 |
| --- | --- |
| [Lee et al., Applied Sciences 2023](https://doi.org/10.3390/app131910935) | 측면 영상 기반 목·상체 자세 분류 |
| [Lite-HRNet, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Yu_Lite-HRNet_A_Lightweight_High-Resolution_Network_CVPR_2021_paper.html) | 경량 자세 추정 후보 |
| [Lite Pose, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Lite_Pose_Efficient_Architecture_Design_for_2D_Human_Pose_Estimation_CVPR_2022_paper.html) | 엣지 모델 효율 비교 |
| [Jetson Nano 공식 사양](https://developer.nvidia.com/embedded/jetson-nano) | 대상 하드웨어 |
| [JetPack 4 최종 릴리스 공지](https://forums.developer.nvidia.com/t/announcing-end-of-life-for-nvidia-jetpack-4-with-the-release-of-jetpack-4-6-6/314300) | 소프트웨어 호환성 |

논문에 보고된 타 장치 성능은 본 프로젝트의 Nano 실측 결과가 아닙니다. 원본 논문과 사전학습 모델은 해당 권리자의 조건을 따릅니다.

## 데이터 관리

원본 사용자 영상은 기본적으로 저장·외부 전송하지 않는 운용을 목표로 합니다. 연구용 수집은 참여자 동의를 받아 별도로 수행하며, 원본 영상·개인 식별정보·인증정보는 공개 저장소에 포함하지 않습니다. 자세 기록도 개인별 행동 데이터이므로 로컬에서 관리합니다.
