# 보드·카메라 동작 확인 (2026-09-17)

실측 결과만 기록한다. 목표치가 아니다.

## 보드

- Jetson Nano, L4T R32.7.6 (JetPack 4.6 계열), Ubuntu 18.04.6 LTS, aarch64
- Python 3.6.9 (시스템 기본)
- **커스텀 캐리어보드** — 공식 Jetson Nano Developer Kit이 아님. `Jetson.GPIO` import 시 다음 경고 발생:
  > Carrier board is not from a Jetson Developer Kit. Jetson.GPIO library has not been verified with this carrier board, and in fact is unlikely to work correctly.
  → FR-10/FR-11 (LED·버튼 GPIO) 작업 전에 실제 핀맵을 캐리어보드 문서로 별도 확인해야 한다. Jetson.GPIO의 기본 핀 번호 매핑을 그대로 신뢰하지 않는다.
- 확인 시점 디스크: 14G 중 8G 여유 (전체 시스템 기준, 패키지 설치 후 7.7G). NFR-01(4GB 운용) 관점에서 디스크도 여유가 크지 않아 모델·의존성 설치 시 용량을 매번 확인한다.
- RAM 3.9G, swap 1.9G(미사용)
- `nvidia-l4t-cuda`만 설치돼 있고 JetPack 메타패키지(cuDNN, TensorRT, OpenCV-CUDA, PyTorch)는 없음. NVIDIA apt 저장소(r32.7 common/t210)는 등록돼 있어 필요한 구성요소를 개별 설치할 수 있다.

## 카메라

- `/dev/video0` 존재. CSI 센서, raw Bayer **RG10** 포맷 (IMX219 계열로 추정).
- 지원 해상도/프레임: 3264x2464@21, 3264x1848@28, 1920x1080@30, 1640x1232@30, 1280x720@60, 1280x720@120 (v4l2-ctl 확인).
- Bayer raw라 `cv2.VideoCapture(0)`으로 직접 못 읽는다. `nvarguscamerasrc`(NVIDIA ISP) GStreamer 파이프라인이 필요하다.
- apt로 설치한 `python3-opencv`(Ubuntu 18.04 표준 빌드, 3.2.0)는 **GStreamer 지원 없이 빌드**돼 있어 `cv2.VideoCapture("... ! appsink")` 문자열 파이프라인도 못 쓴다.
- 우회: `python3-gi` (PyGObject, 이미 이미지에 포함)로 GStreamer appsink에서 직접 버퍼를 당겨 numpy 배열로 변환. `src/camera.py` 참고. cv2는 저장(`imwrite`)·후처리에만 사용.
- 실측: 1280x720 요청 시 실제 협상 모드는 1280x720@120 스트림, 캡처 루프 처리량 **26.56 FPS** (`scripts/camera_smoke_test.py`, appsink `try-pull-sample` + numpy reshape 기준, GPU 인코딩·모델 추론 없이 순수 캡처 경로만). NFR-02(5–10 FPS 목표) 대비 캡처 단계는 여유 있음 — 병목은 이후 추정·분류 단계에서 잴 것.
- 스냅샷 색감·노출 정상 확인 (ISP 자동 노출/화이트밸런스 동작).

## 설치한 apt 패키지

`scripts/jetson_setup.sh` 참고: `python3-pip`, `python3-venv`, `python3-opencv`, `v4l-utils`.

`python3-gi` / gstreamer1.0-* 플러그인은 이미지에 기본 포함돼 있어 별도 설치 안 함 (`python3 -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst"` 로 확인).

## 남은 것

- 오디오 출력(USB 스피커) 미확인 — 다음 확인 항목.
- 착석 센서·LED·버튼 GPIO — 캐리어보드 핀맵 확인 먼저 필요.
- 자세 추정 모델이 이 환경(cuDNN/TensorRT 없음, RAM 3.5GB 목표)에서 실제로 돌아가는지는 별도 검증 필요.
